#!/bin/bash
# Remote half of multinode_gen: runs datagen | nc on the generator
# host. Owns its lifecycle via a pidfile so a prior run that died
# abruptly can be cleaned up deterministically by the next run.
#
# Args: REMOTE_DIR REMOTE_PYTHON PORT <args to datagen...>
#
# The pidfile is keyed by PORT so concurrent runs on different ports do not
# reap each other — the stale-run cleanup below is per-listener, not global.

set -uo pipefail

PORT=$3
PIDFILE=/tmp/riscv-thoughts-multinode-remote.${PORT}.pid

if [ -s "$PIDFILE" ]; then
    OLD=$(cat "$PIDFILE")
    pkill -P "$OLD" 2>/dev/null
    kill "$OLD" 2>/dev/null
    sleep 1
fi

echo $$ > "$PIDFILE"
trap 'pkill -P $$ 2>/dev/null; rm -f "$PIDFILE"' EXIT

REMOTE_DIR=$1
REMOTE_PYTHON=$2
shift 3

cd "$REMOTE_DIR"
# Persist the pipeline's stderr to a file outside REMOTE_DIR (which the
# local-side cleanup deletes). Survives ssh teardown so post-mortem
# diagnosis is possible even when the failure is exactly the disconnect.
# Markers must go to fd 2, not fd 1 — fd 1 is the data path to nc.
LOG=/tmp/multinode-remote.log
echo "=== remote-half started $(date -u +%FT%TZ) pid=$$ ===" >> "$LOG"
# lz4 on the wire: RVT batches are padded numeric arrays and the link is
# gigabit (measured 111 MB/s), which one trainer alone half-fills. Must be
# matched by `lz4 -dc` on the receiving end. `-1` is the fast mode — anything
# slower becomes the new bottleneck at these rates.
#
# MALLOC_ARENA_MAX=2 cuts glibc malloc's per-thread arena count to 2;
# the default is 8 * num_cpus, which on a 64-core box means hundreds
# of arenas per worker process and massive page-zeroing churn (the
# `clear_page_erms` kernel function was the #1 hotspot before this).
export MALLOC_ARENA_MAX=2
# </dev/null: this is a pure generator — never auto-mux an inherited ssh stdin.
"$REMOTE_PYTHON" -m datagen "$@" </dev/null 2>>"$LOG" \
    | lz4 -1 -c 2>>"$LOG" \
    | nc -l -q 0 "$PORT" 2>>"$LOG"
EC=$?
echo "=== remote-half ended $(date -u +%FT%TZ) exit=$EC ===" >> "$LOG"
