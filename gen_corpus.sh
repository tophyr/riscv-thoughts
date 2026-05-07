#!/bin/bash

# Generate an on-disk RVT corpus on a remote generator host (odin).
#
# Same orchestration as multinode_gen.sh — ship sources, spawn a
# many-worker mux_batches pool — but the corpus lands on the remote's
# disk; nothing is streamed back. Defaults are tuned for the T2
# pipeline (multi-instruction chunks, T2 opcode mix, no invalid
# injection).
#
# Usage:
#   gen_corpus.sh TOTAL_BATCHES REMOTE_OUTPUT_PATH
#   gen_corpus.sh 100000 corpora/t2_100k.rvt
#
# REMOTE_OUTPUT_PATH is interpreted relative to $HOME on the remote.
# TOTAL_BATCHES must be a multiple of WORKERS.
#
# Override generation knobs via env vars:
#   WORKERS         (default 100)
#   BATCH_SIZE      (default 128)
#   TWINS           (default 3)
#   PARTNERS        (default 20)
#   N_STATES        (default 8)
#   RULE            (default 'branch+cap=8' — T2 chunks)
#   CONFIG          (default configs/instr_t2.json)
#   INJECT_INVALID  (default 0 — clean corpus)
#   INJECT_EQUIV    (default unset — uses config's equivalences.rate)

set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '3,/^set -/p' "$0" | sed '$d' | sed 's/^# \?//' >&2
    exit 1
fi

TOTAL=$1
OUT=$2

WORKERS=${WORKERS:-100}
BATCH_SIZE=${BATCH_SIZE:-128}
TWINS=${TWINS:-3}
PARTNERS=${PARTNERS:-20}
N_STATES=${N_STATES:-8}
RULE=${RULE:-branch+cap=8}
CONFIG=${CONFIG:-configs/instr_t2.json}
INJECT_INVALID=${INJECT_INVALID:-0}
INJECT_EQUIV=${INJECT_EQUIV:-}

if (( TOTAL % WORKERS != 0 )); then
    echo "ERROR: TOTAL ($TOTAL) must be divisible by WORKERS ($WORKERS)" >&2
    exit 1
fi
PER_WORKER=$((TOTAL / WORKERS))

REMOTE=chrissarbora@odin
REMOTE_PYTHON='$HOME/Projects/riscv-thoughts/.venv/bin/python'
PIDFILE=/tmp/riscv-thoughts-corpus-remote.pid

REMOTE_DIR=$(ssh "${REMOTE}" 'mktemp -d -t riscv-thoughts.XXXXXX')
if [ -z "${REMOTE_DIR}" ]; then
    echo "ERROR: failed to create remote temp dir" >&2
    exit 1
fi

cleanup() {
    # Mirror multinode_gen.sh's cleanup, plus an explicit gen_batches
    # sweep — those workers reparent to init when their parent dies and
    # can survive ssh teardown for many minutes (CLAUDE.md: "orphan
    # gen_batches workers").
    ssh "${REMOTE}" "
        if [ -s '${PIDFILE}' ]; then
            P=\$(cat '${PIDFILE}' 2>/dev/null);
            if [ -n \"\$P\" ]; then
                pkill -P \"\$P\" 2>/dev/null;
                kill \"\$P\" 2>/dev/null;
            fi
        fi
        PIDS=\$(pgrep -f 'scripts/gen_batches\\.py' || true)
        if [ -n \"\$PIDS\" ]; then kill -9 \$PIDS 2>/dev/null || true; fi
        rm -rf '${REMOTE_DIR}' 2>/dev/null
        rm -f '${PIDFILE}' 2>/dev/null
    " 2>/dev/null || true
}
trap cleanup EXIT

echo "Shipping sources to ${REMOTE}:${REMOTE_DIR}" >&2
tar -czf - --exclude='__pycache__' --exclude='*.pyc' --exclude='tests' \
    datagen emulator tokenizer scripts configs \
  | ssh "${REMOTE}" "tar -xzf - -C ${REMOTE_DIR}" \
  || { echo "ERROR: failed to ship sources" >&2; exit 1; }

echo "Generating ${TOTAL} batches (${WORKERS} workers × ${PER_WORKER}) → ${REMOTE}:~/${OUT}" >&2
echo "Remote stderr/log: /tmp/gen-corpus-remote.log" >&2

# All knobs are passed positionally to the remote heredoc so we don't
# have to deal with quoting envs through ssh.
ssh "${REMOTE}" bash -s -- \
    "${REMOTE_DIR}" "${REMOTE_PYTHON}" "${OUT}" "${PIDFILE}" \
    "${WORKERS}" "${PER_WORKER}" "${BATCH_SIZE}" \
    "${TWINS}" "${PARTNERS}" "${N_STATES}" \
    "${RULE}" "${CONFIG}" "${INJECT_INVALID}" "${INJECT_EQUIV:-_NONE_}" <<'REMOTE_SCRIPT'
set -uo pipefail

REMOTE_DIR=$1; REMOTE_PYTHON_TPL=$2; OUT=$3; PIDFILE=$4
WORKERS=$5; PER_WORKER=$6; BATCH_SIZE=$7
TWINS=$8; PARTNERS=$9; N_STATES=${10}
RULE=${11}; CONFIG=${12}; INJECT_INVALID=${13}; INJECT_EQUIV=${14}
[ "$INJECT_EQUIV" = "_NONE_" ] && INJECT_EQUIV=""

# Pidfile-based cleanup of any prior corpus run, mirroring
# multinode_remote.sh.
if [ -s "$PIDFILE" ]; then
    OLD=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "$OLD" ]; then
        pkill -P "$OLD" 2>/dev/null || true
        kill "$OLD" 2>/dev/null || true
        sleep 1
    fi
fi
echo $$ > "$PIDFILE"
trap 'pkill -P $$ 2>/dev/null; rm -f "$PIDFILE"' EXIT

# Resolve $HOME-relative output path under the remote's home.
case "$OUT" in
    /*) ABS_OUT="$OUT" ;;
    *)  ABS_OUT="$HOME/$OUT" ;;
esac
mkdir -p "$(dirname "$ABS_OUT")"

# Expand the python-path template ($HOME inside it) on the remote.
eval "REMOTE_PYTHON=$REMOTE_PYTHON_TPL"

cd "$REMOTE_DIR"

# Same arena-clamp as multinode_remote.sh — without it, glibc allocates
# 8*ncpus arenas per worker and clear_page_erms dominates wall time.
export MALLOC_ARENA_MAX=2

LOG=/tmp/gen-corpus-remote.log
echo "=== corpus-gen started $(date -u +%FT%TZ) pid=$$ out=$ABS_OUT ===" >> "$LOG"

EXTRA=( --inject-invalid "$INJECT_INVALID" )
if [ -n "$INJECT_EQUIV" ]; then
    EXTRA+=( --inject-equiv "$INJECT_EQUIV" )
fi

"$REMOTE_PYTHON" scripts/mux_batches.py \
    --gen-count "$WORKERS" \
    --rule "$RULE" \
    --batch-size "$BATCH_SIZE" \
    --twins "$TWINS" --partners "$PARTNERS" --n-states "$N_STATES" \
    --n-batches "$PER_WORKER" \
    --config "$CONFIG" \
    "${EXTRA[@]}" \
    > "$ABS_OUT" 2>>"$LOG"
EC=$?
SIZE=$(stat -c %s "$ABS_OUT" 2>/dev/null || echo "?")
echo "=== corpus-gen ended $(date -u +%FT%TZ) exit=$EC bytes=$SIZE ===" >> "$LOG"
exit $EC
REMOTE_SCRIPT
