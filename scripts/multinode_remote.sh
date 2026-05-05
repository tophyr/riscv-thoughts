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
"$REMOTE_PYTHON" scripts/mux_batches.py "$@" | lz4 | nc -l -q 0 6464
