#!/bin/bash
# Remote half of multinode_gen: runs mux_batches | lz4 | nc on the
# generator host. Owns its lifecycle via a pidfile so a prior run that
# died abruptly can be cleaned up deterministically by the next run.
#
# Args: REMOTE_DIR REMOTE_PYTHON -- <args to mux_batches.py...>

set -uo pipefail

PIDFILE=/tmp/riscv-thoughts-multinode-remote.pid

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
shift 2

cd "$REMOTE_DIR"
# Persist the pipeline's stderr to a file outside REMOTE_DIR (which the
# local-side cleanup deletes). Survives ssh teardown so post-mortem
# diagnosis is possible even when the failure is exactly the disconnect.
# Markers must go to fd 2, not fd 1 — fd 1 is the data path to nc.
LOG=/tmp/multinode-remote.log
echo "=== remote-half started $(date -u +%FT%TZ) pid=$$ ===" >> "$LOG"
# lz4 removed during failure-hunting: removing lz4 makes failures
# happen ~5x faster (the failure is bytes-of-data dependent), which is
# what we want when iterating on diagnostics.
#
# MALLOC_ARENA_MAX=2 cuts glibc malloc's per-thread arena count to 2;
# the default is 8 * num_cpus, which on a 64-core box means hundreds
# of arenas per worker process and massive page-zeroing churn (the
# `clear_page_erms` kernel function was the #1 hotspot before this).
export MALLOC_ARENA_MAX=2
"$REMOTE_PYTHON" scripts/mux_batches.py "$@" 2>>"$LOG" \
    | nc -l -q 0 6464 2>>"$LOG"
EC=$?
echo "=== remote-half ended $(date -u +%FT%TZ) exit=$EC ===" >> "$LOG"
