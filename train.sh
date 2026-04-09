#!/bin/bash

# This script is a training aid for Chris Sarbora; it is not something intended for use by others.
# Use it as an example if you like: This uses a second networked machine to generate extra data,
# along with pregenerated data on disk, to ensure that the GPU is always fed to maximum capacity.

PYTHON=.venv/bin/python

BATCH_SIZE=4096
NUM_BATCHES=$1

MUX_BATCHES="${PYTHON} scripts/mux_batches.py --batch-size ${BATCH_SIZE} --n-batches ${NUM_BATCHES}"

REMOTE=chrissarbora@odin
ssh ${REMOTE} "cd Projects/riscv-thoughts && ${MUX_BATCHES} --gen 128 | lz4 | nc -l -q 0 6464" &
SSH_PID=$!
trap "kill $SSH_PID 2>/dev/null; ssh ${REMOTE} 'pkill -f \"mux_batches.*6464\"; pkill -f \"nc -l.*6464\"' 2>/dev/null; wait $SSH_PID 2>/dev/null" EXIT

# Wait for remote workers to start and nc to begin listening.
sleep 5
kill -0 $SSH_PID 2>/dev/null || { echo "ERROR: remote process died" >&2; exit 1; }

${MUX_BATCHES} --gen 16 <(nc odin 6464 -d | unlz4) /mnt/d/batch_data/70k_4k.dat |
  ${PYTHON} scripts/batch_slice.py --count ${NUM_BATCHES} |
  ${PYTHON} scripts/train_compressor.py --lr-schedule ${NUM_BATCHES}
