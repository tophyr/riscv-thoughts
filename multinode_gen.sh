#!/bin/bash

# Cluster-style RVT corpus generator.
#
# Combines a remote datagen pool (over a netcat link) with a local
# datagen pool. Both generate infinitely; the local side bounds the
# joined stream to TOTAL_BATCHES with --count. When that count is reached,
# BrokenPipe cascades local→nc→remote and every worker (pdeathsig-armed)
# is reaped. Pipe stdout to a file (or to a downstream consumer like
# train_*).
#
# Generation knobs are applied identically to both datagen invocations
# so local and remote workers produce the same shape of data.
#
# Worker split (REMOTE_GEN / LOCAL_GEN) sets the gen mix ratio, not the
# total (which --count fixes). LOCAL_GEN defaults to 0: when this stream
# feeds a local GPU trainer, local datagen workers contend with the
# trainer's per-step H2D prep and starve the GPU (measured: ~54% util /
# 16.6ms/step with local gen, vs ~90% / 9ms with ALL generation on odin).
# Offloading every worker to odin is therefore the default, and is what
# makes this the natural front-end for a real training run. For a pure
# on-disk corpus with no local trainer, set LOCAL_GEN>0 to use local
# cores too.
#
# This is a personal aid for Chris Sarbora's setup; treat it as an
# example pattern rather than a turnkey tool.
#
# Usage:
#   multinode_gen.sh TOTAL_BATCHES > /tmp/corpus.rvt
#   multinode_gen.sh 260000            # → exactly 260,000 batches
#
# Override generation knobs via env vars:
#   BATCH_SIZE      (default 128)
#   TWINS           (default 3)
#   N_STATES        (default 8)
#   RULE            (default 'single' — single-instruction chunks; the
#                    old "--gen instr" equivalent)
#   CONFIG          (REQUIRED — the corpus config. A corpus is defined by
#                    that one file: opcode mix, exclude_x0_dest, and the
#                    equivalence / invalidity / chain rates. There is no
#                    default and no env override for any rate, so the file a
#                    run names is the whole corpus definition and
#                    scripts/record_corpus.py can copy it into the run dir.)
#   ANCHOR_SEED     (default 0 — passed to BOTH pools; MUST equal the
#                    trainer's --anchor-seed, else value_predict targets
#                    silently mismatch, INVARIANT 1.)
#   GEN_SEED        (default 0 — the instruction-stream seed; worker i draws
#                    seed+i. Change it to draw a DIFFERENT corpus from the
#                    same distribution, which is how a run-to-run noise floor
#                    is measured. Unrelated to ANCHOR_SEED, which must not
#                    move.)
#   REMOTE_GEN      (default 96 — odin worker count; it is a 128-core box)
#   LOCAL_GEN       (default 0 — see the worker-split note above)
#   PORT            (default 6464 — the nc listener port. Give concurrent
#                    runs distinct ports: the remote pidfile is keyed by it,
#                    so sharing one makes the second run reap the first.)

set -euo pipefail

PYTHON=.venv/bin/python
TOTAL=$1
BATCH_SIZE=${BATCH_SIZE:-128}
TWINS=${TWINS:-3}
N_STATES=${N_STATES:-8}
RULE=${RULE:-single}
CONFIG=${CONFIG:?set CONFIG to a corpus config under configs/ — a corpus is one reviewable file, and there is no default}
ANCHOR_SEED=${ANCHOR_SEED:-0}
GEN_SEED=${GEN_SEED:-0}
REMOTE_GEN=${REMOTE_GEN:-96}
LOCAL_GEN=${LOCAL_GEN:-0}
PORT=${PORT:-6464}

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

# Teardown must not invent an exit status. Reaping the ssh carrying the remote
# pool yields 255, and under `set -e` a nonzero step here also abandons the
# rest of the teardown — so every step absorbs its own failure and the status
# the script was already exiting with is restored at the end. Without that, a
# corpus run that wrote every batch it was asked for still reports failure.
cleanup() {
  local rc=$?
  if [ -n "${SSH_PID:-}" ]; then
    kill "${SSH_PID}" 2>/dev/null || true
  fi
  ssh ${REMOTE} 'PID=$(cat /tmp/riscv-thoughts-multinode-remote.'"${PORT}"'.pid 2>/dev/null); if [ -n "$PID" ]; then pkill -P "$PID" 2>/dev/null; kill "$PID" 2>/dev/null; fi; rm -rf '"${REMOTE_DIR}"' 2>/dev/null' 2>/dev/null || true
  if [ -n "${SSH_PID:-}" ]; then
    wait "${SSH_PID}" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT

# Ship Python sources over the ssh pipe as a compressed tar stream. We
# only need the datagen side of the pipeline (datagen + the worker
# processes it forks), which pulls in datagen/, emulator/, tokenizer/,
# and the JSON configs. Total payload is well under 100KB.
echo "Shipping sources to ${REMOTE}:${REMOTE_DIR}" >&2
tar -czf - --exclude='__pycache__' --exclude='*.pyc' --exclude='tests' \
    datagen emulator tokenizer scripts configs \
  | ssh ${REMOTE} "tar -xzf - -C ${REMOTE_DIR}" \
  || { echo "ERROR: failed to ship sources" >&2; exit 1; }

# Spawn the remote 48-worker pool. datagen forks the workers and
# interleaves their output over netcat to us. Generation is infinite;
# the local --count bound (below) drives shutdown.
ssh ${REMOTE} "bash ${REMOTE_DIR}/scripts/multinode_remote.sh \
    ${REMOTE_DIR} ${REMOTE_PYTHON} ${PORT} \
    --gen-count ${REMOTE_GEN} \
    --rule ${RULE} \
    --batch-size ${BATCH_SIZE} \
    --twins ${TWINS} --n-states ${N_STATES} \
    --anchor-seed ${ANCHOR_SEED} --seed ${GEN_SEED} \
    --config ${CONFIG}" &
SSH_PID=$!

# Wait for remote workers to start and nc to begin listening.
sleep 5
kill -0 $SSH_PID 2>/dev/null || { echo "ERROR: remote process died" >&2; exit 1; }

# Local pool: LOCAL_GEN local workers (default 0 = pure mux, all
# generation on odin) plus the remote netcat stream as a mux input.
# --count TOTAL bounds the joined stream and, on reaching it, reaps the
# local pool and (via BrokenPipe over nc) the remote pool.
# </dev/null: mux only the explicit <(nc …) input, never an inherited stdin.
${PYTHON} -m datagen \
    --gen-count ${LOCAL_GEN} \
    <(nc odin ${PORT} -d | lz4 -dc) \
    --rule ${RULE} \
    --batch-size ${BATCH_SIZE} \
    --twins ${TWINS} --n-states ${N_STATES} \
    --anchor-seed ${ANCHOR_SEED} --seed ${GEN_SEED} \
    --config ${CONFIG} \
    --count ${TOTAL} </dev/null
