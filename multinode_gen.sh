#!/bin/bash

# Cluster-style RVT corpus generator.
#
# Combines a remote 96-worker gen_batches pool (over a compressed netcat
# link) with a local 8-worker pool. Each worker generates
# BATCHES_PER_WORKER batches, then exits; the script ends when all
# workers have drained. Pipe stdout to a file (or to a downstream
# consumer like batch_repeat / train_*).
#
# Generation knobs are applied identically to both mux_batches
# invocations so local and remote workers produce the same shape of
# data. Worker counts (96 odin, 8 local) are tuned to the specific
# machines and stay hardcoded.
#
# This is a personal aid for Chris Sarbora's setup; treat it as an
# example pattern rather than a turnkey tool.
#
# Usage:
#   multinode_gen.sh BATCHES_PER_WORKER > /tmp/corpus.rvt
#   multinode_gen.sh 2500              # → 104 * 2500 = 260,000 batches
#
# Override generation knobs via env vars:
#   BATCH_SIZE      (default 128)
#   TWINS           (default 3)
#   PARTNERS        (default 20)
#   N_STATES        (default 8)
#   RULE            (default 'single' — single-instruction chunks; the
#                    old "--gen instr" equivalent)
#   CONFIG          (default configs/instr_default.json)
#   INJECT_INVALID  (default 0 — overrides the config's invalidity.rate
#                    so the corpus is clean by default. Set to e.g. 0.2
#                    to re-enable validity-training augmentation.)
#   INJECT_EQUIV    (default unset — uses config's equivalences.rate)

set -euo pipefail

PYTHON=.venv/bin/python
BATCHES_PER_WORKER=$1
BATCH_SIZE=${BATCH_SIZE:-128}
TWINS=${TWINS:-3}
PARTNERS=${PARTNERS:-20}
N_STATES=${N_STATES:-8}
RULE=${RULE:-single}
CONFIG=${CONFIG:-configs/instr_default.json}
INJECT_INVALID=${INJECT_INVALID:-0}
INJECT_EQUIV=${INJECT_EQUIV:-}

REMOTE=chrissarbora@odin
# The remote is expected to have a usable Python venv with numpy + scipy
# already installed at this path. The Python *sources* are shipped
# per-run (below), so the remote does not need an in-sync checkout.
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
  ssh ${REMOTE} 'PID=$(cat /tmp/riscv-thoughts-multinode-remote.pid 2>/dev/null); if [ -n "$PID" ]; then pkill -P "$PID" 2>/dev/null; kill "$PID" 2>/dev/null; fi; rm -rf '"${REMOTE_DIR}"' 2>/dev/null' 2>/dev/null
  if [ -n "${SSH_PID:-}" ]; then
    wait "${SSH_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT

# Ship Python sources over the ssh pipe as a compressed tar stream. We
# only need the datagen side of the pipeline (mux_batches + the
# gen_batches workers it spawns), which pulls in datagen/, emulator/,
# tokenizer/, and the JSON configs. Total payload is well under 100KB.
echo "Shipping sources to ${REMOTE}:${REMOTE_DIR}" >&2
tar -czf - --exclude='__pycache__' --exclude='*.pyc' --exclude='tests' \
    datagen emulator tokenizer scripts configs \
  | ssh ${REMOTE} "tar -xzf - -C ${REMOTE_DIR}" \
  || { echo "ERROR: failed to ship sources" >&2; exit 1; }

# Spawn remote 96-worker pool. Each worker is a gen_batches.py with
# matching knobs; mux interleaves their output and lz4-compresses it
# over netcat to us.
EXTRA_ARGS="--inject-invalid ${INJECT_INVALID}"
if [ -n "${INJECT_EQUIV}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --inject-equiv ${INJECT_EQUIV}"
fi

ssh ${REMOTE} "bash ${REMOTE_DIR}/scripts/multinode_remote.sh \
    ${REMOTE_DIR} ${REMOTE_PYTHON} \
    --gen-count 96 \
    --rule ${RULE} \
    --batch-size ${BATCH_SIZE} \
    --twins ${TWINS} --partners ${PARTNERS} --n-states ${N_STATES} \
    --n-batches ${BATCHES_PER_WORKER} \
    --config ${CONFIG} \
    ${EXTRA_ARGS}" &
SSH_PID=$!

# Wait for remote workers to start and nc to begin listening.
sleep 5
kill -0 $SSH_PID 2>/dev/null || { echo "ERROR: remote process died" >&2; exit 1; }

# Local mux: 8 local workers plus the remote netcat stream as an
# additional input. mux exits on its own once all inputs (8 workers +
# remote stream) have closed, so total batches = 104 * BATCHES_PER_WORKER.
${PYTHON} scripts/mux_batches.py \
    --gen-count 8 \
    --rule ${RULE} \
    --batch-size ${BATCH_SIZE} \
    --twins ${TWINS} --partners ${PARTNERS} --n-states ${N_STATES} \
    --n-batches ${BATCHES_PER_WORKER} \
    --config ${CONFIG} \
    ${EXTRA_ARGS} \
    <(nc odin 6464 -d | unlz4)
