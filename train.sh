#!/bin/bash

# This script is a training aid for Chris Sarbora; it is not something intended for use by others.
# Use it as an example if you like: This uses a second networked machine to generate extra data,
# along with focused branch-heavy batches, to ensure that the GPU is always fed to maximum capacity.

PYTHON=.venv/bin/python
BATCH_SIZE=4096
NUM_BATCHES=$1
D_OUT=${2:-128}
N_LAYERS=${3:-2}

REMOTE=chrissarbora@odin
# The remote is expected to have a usable Python venv with numpy + tinyfive
# already installed here. The Python *sources* are shipped per-run (below),
# so the remote does not need an in-sync checkout of this repo.
REMOTE_PYTHON='$HOME/Projects/riscv-thoughts/.venv/bin/python'

REMOTE_DIR=$(ssh ${REMOTE} 'mktemp -d -t riscv-thoughts.XXXXXX')
if [ -z "${REMOTE_DIR}" ]; then
  echo "ERROR: failed to create remote temp dir" >&2
  exit 1
fi

cleanup() {
  if [ -n "${SSH_PID:-}" ]; then
    kill "${SSH_PID}" 2>/dev/null
  fi
  ssh ${REMOTE} "pkill -f 'mux_batches.*6464' 2>/dev/null; pkill -f 'nc -l.*6464' 2>/dev/null; rm -rf ${REMOTE_DIR}" 2>/dev/null
  if [ -n "${SSH_PID:-}" ]; then
    wait "${SSH_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT

# Ship Python sources over the ssh pipe as a compressed tar stream. We only
# need the datagen side of the pipeline (mux_batches + the generator worker
# it spawns), which pulls in datagen/, emulator/, tokenizer/, and the JSON
# configs. Total payload is well under 100KB.
echo "Shipping sources to ${REMOTE}:${REMOTE_DIR}" >&2
tar -czf - --exclude='__pycache__' --exclude='*.pyc' --exclude='tests' \
    datagen emulator tokenizer scripts configs \
  | ssh ${REMOTE} "tar -xzf - -C ${REMOTE_DIR}" \
  || { echo "ERROR: failed to ship sources" >&2; exit 1; }

ssh ${REMOTE} "cd ${REMOTE_DIR} && \
  ${REMOTE_PYTHON} scripts/mux_batches.py --gen instr --gen-count 80 --batch-size ${BATCH_SIZE} --n-batches ${NUM_BATCHES} --config configs/instr_default.json \
  | lz4 | nc -l -q 0 6464" &
SSH_PID=$!

# Wait for remote workers to start and nc to begin listening.
sleep 5
kill -0 $SSH_PID 2>/dev/null || { echo "ERROR: remote process died" >&2; exit 1; }

${PYTHON} scripts/mux_batches.py --gen instr --gen-count 16 --batch-size ${BATCH_SIZE} --n-batches ${NUM_BATCHES} --config configs/instr_default.json \
    <(nc odin 6464 -d | unlz4) | \
  ${PYTHON} scripts/batch_slice.py --count ${NUM_BATCHES} | \
  ${PYTHON} scripts/train_compressor.py --mode instr --n-steps ${NUM_BATCHES} --equiv-weight 0.05 --d-out ${D_OUT} --n-layers ${N_LAYERS}
