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

mkdir -p "$(dirname "$OUT")"

# Same arena-clamp as multinode_remote.sh — without it, glibc allocates
# 8*ncpus arenas per worker and clear_page_erms dominates wall time.
export MALLOC_ARENA_MAX=2

EXTRA=( --inject-invalid "$INJECT_INVALID" )
if [ -n "$INJECT_EQUIV" ]; then
    EXTRA+=( --inject-equiv "$INJECT_EQUIV" )
fi

.venv/bin/python scripts/mux_batches.py \
    --gen-count "$WORKERS" \
    --rule "$RULE" \
    --batch-size "$BATCH_SIZE" \
    --twins "$TWINS" --n-states "$N_STATES" \
    --n-batches "$PER_WORKER" \
    --config "$CONFIG" \
    "${EXTRA[@]}" \
    > "$OUT"
