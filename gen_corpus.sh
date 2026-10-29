#!/bin/bash

# Generate an on-disk RVT corpus on a remote generator host (odin).
#
# Same orchestration as multinode_gen.sh — ship sources, spawn an
# all-cores datagen pool — but the corpus lands on the remote's
# disk; nothing is streamed back. Defaults are tuned for the T2
# pipeline (multi-instruction chunks, T2 opcode mix, no invalid
# injection).
#
# Usage:
#   gen_corpus.sh TOTAL_BATCHES REMOTE_OUTPUT_PATH
#   gen_corpus.sh 100000 corpora/t2_100k.rvt
#
# REMOTE_OUTPUT_PATH is interpreted relative to $HOME on the remote.
# Exactly TOTAL_BATCHES are written (datagen --count); the worker pool is
# reaped once the count is reached.
#
# Override generation knobs via env vars:
#   WORKERS         (default: all CPU cores)
#   BATCH_SIZE      (default 384)
#   N_STATES        (default 8)
#   RULE            (default 'branch+cap=8' — T2 chunks)
#   CONFIG          (REQUIRED — the corpus config. A corpus is defined by
#                    that one file: opcode mix, exclude_x0_dest, and the
#                    equivalence / invalidity / chain rates. No default, and
#                    no env override for any rate.)

set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '3,/^set -/p' "$0" | sed '$d' | sed 's/^# \?//' >&2
    exit 1
fi

TOTAL=$1
OUT=$2

WORKERS=${WORKERS:-}
BATCH_SIZE=${BATCH_SIZE:-384}
N_STATES=${N_STATES:-8}
RULE=${RULE:-branch+cap=8}
CONFIG=${CONFIG:?set CONFIG to a corpus config under configs/ — a corpus is one reviewable file, and there is no default}

mkdir -p "$(dirname "$OUT")"

# Same arena-clamp as multinode_remote.sh — without it, glibc allocates
# 8*ncpus arenas per worker and clear_page_erms dominates wall time.
export MALLOC_ARENA_MAX=2

# WORKERS unset → datagen defaults to all CPU cores. Set it to override.
WORKER_ARGS=()
if [ -n "$WORKERS" ]; then
    WORKER_ARGS=( --gen-count "$WORKERS" )
fi

# datagen --count emits exactly TOTAL batches then reaps the pool.
.venv/bin/python -m datagen \
    --rule "$RULE" \
    --batch-size "$BATCH_SIZE" \
    --n-states "$N_STATES" \
    --config "$CONFIG" \
    --count "$TOTAL" \
    "${WORKER_ARGS[@]}" \
    </dev/null > "$OUT"   # pure generator — don't auto-mux an inherited stdin
