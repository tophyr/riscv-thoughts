#!/bin/bash
# Train ONE T2 cell and score it. Every knob is an env var with the adopted
# config as its default, so a cell is named by what it changes:
#
#   LEN=2 STEPS=1000000 GEN_SEED=7 TAG=s7 bash scripts/run_t2.sh
#
# The equivariant core's settings are no longer knobs: read positions are
# canonical ranks, register slots are seeded from those ranks, and the pool
# reads each slot's write step. Those are the only values any tier uses.
#
# The run directory encodes the knobs, so two cells that differ in any of them
# cannot overwrite each other; TAG separates a re-run of an already-recorded
# cell from the record itself.
#
# Scored on vp (encode-upward fidelity), the binding heads, the GVN collapse
# rows, and the vp error crossed with dataflow depth — a vp win that craters
# GVN or the deep slots is a regression, not a result.
set -u
cd /home/tophy/Projects/riscv-thoughts
source .venv/bin/activate
D=/home/tophy/Projects/riscv-thoughts/logs
# The frozen lower tier. A T2 checkpoint is only interchangeable with others
# trained on the SAME T1, so a cell that changes this must be named for it.
T1=${T1:-runs/t1_writers_250k_rank/encoder.pt}
# VP = vp-head hidden layers. The head must EVALUATE the chunk's function at the
# anchor points, so its capacity is a confound in every vp number; refitting it
# on frozen essences cannot test that, because depth-2 slots are 2.5% of the
# data and a fresh head sees too few.
LEN=${LEN:-2} DOUT=${DOUT:-512} STEPS=${STEPS:-250000} OC=${OC:-4} VP=${VP:-1}
NPOOL=${NPOOL:-16}
# Distinguishes a re-run of an already-recorded cell (the run dir is otherwise a
# pure function of the knobs, so it would silently overwrite it).
TAG=${TAG:-}

cell() {
  local OUT=t2_len${LEN}_d${DOUT}_s${STEPS}_dep${OC}_vp${VP}_pool${NPOOL}${TAG:+_$TAG}_writers
  echo "######## $(date '+%m-%d %H:%M:%S')  -> $OUT"
  # Generation runs on odin (multinode_gen.sh, LOCAL_GEN=0) so local cores stay
  # free for the trainer's per-step host work. ANCHOR_SEED must equal the
  # trainer's --anchor-seed (INVARIANT 1). The config carries no invalidity
  # block, so multinode's INJECT_INVALID=0 default matches the local corpus.
  RULE="cap=$LEN" CONFIG=configs/instr_alu_writers.json \
  TWINS=2 BATCH_SIZE=128 N_STATES=8 ANCHOR_SEED=0 \
    ./multinode_gen.sh "$STEPS" \
    | python scripts/train_t2_encoder.py --t1-model "$T1" \
      --d-out "$DOUT" --d-event 16 --op-cell-depth "$OC" --vp-head-depth "$VP" \
      --n-pool-heads "$NPOOL" --route binding \
      --n-steps "$STEPS" --anchor-seed 0 --n-anchor-states 8 --out-dir "$OUT" \
      > "$D/t2_${OUT}.log" 2>&1
  local RC=$?
  if [ ! -f "runs/$OUT/t2.pt" ]; then
    echo "   FAILED rc=$RC"; tail -12 "$D/t2_${OUT}.log"; return 1
  fi
  echo "   final: $(grep "step $STEPS " "$D/t2_${OUT}.log" | tail -1)"
  echo "   --- binding + GVN (eval.py) ---"
  # --gvn-chunk-len must match the cell's chunk length: gvn_collapse builds its
  # OWN pairs and defaults to 4, so a len=2 cell would otherwise be scored
  # entirely out of distribution.
  python scripts/eval.py --t1-model "$T1" --t2 "runs/$OUT" \
      --t2-corpus "/tmp/t2_eval_${LEN}.rvt" --route binding --max-batches 20 \
      --gvn-chunk-len "$LEN" 2>&1 \
    | grep -vE "Warning|readline|history" | sed 's/^/   /'
  echo "   --- faithfulness / on-sphere ---"
  python probes/t2_faithfulness.py "runs/$OUT" "$T1" "$LEN" 2>&1 \
    | grep -vE "Warning|readline|history" | sed 's/^/   /'
  # vp_budget, not vp_imm_share: the criterion is exact 32-bit match and the
  # axes have to be crossed. Per-bit flatters a broken subset (96% per-bit has
  # been 74% exact), and either axis alone hides the cells that carry the error.
  echo "   --- vp error budget: sources x immediate x depth ---"
  python probes/vp_budget.py "runs/$OUT" "$T1" "$LEN" 300 2>&1 \
    | grep -vE "Warning|readline|history" | sed 's/^/   /'
}

# One shared held-out corpus so every cell is scored on identical data.
if [ ! -f "/tmp/t2_eval_${LEN}.rvt" ]; then
  echo "generating shared eval corpus..."
  python -m datagen --rule "cap=$LEN" --twins 2 --batch-size 128 --n-states 8 \
      --anchor-seed 0 --config configs/instr_alu_writers.json --count 200 \
      > "/tmp/t2_eval_${LEN}.rvt" 2>/dev/null
fi

cell
echo "######## COMPLETE $(date '+%m-%d %H:%M:%S') ########"
