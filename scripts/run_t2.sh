#!/bin/bash
# Train ONE T2 cell and score it. Every knob is an env var with the current
# best config as its default, so a cell is named by what it changes:
#
#   LEN=2 STEPS=250000 DEPTHW=8 TAG=depthw8 bash scripts/run_t2.sh
#
# The run directory encodes the knobs, so two cells that differ in any of them
# cannot overwrite each other; TAG separates a re-run of an already-recorded
# cell from the record itself. MODES/PEXTRAS accept several values and the
# script iterates their product.
#
# Scored on vp (encode-upward fidelity), the binding heads, the GVN collapse
# rows, and the vp error crossed with dataflow depth — a vp win that craters
# GVN or the deep slots is a regression, not a result.
set -u
cd /home/tophy/Projects/riscv-thoughts
source .venv/bin/activate
D=/home/tophy/Projects/riscv-thoughts/logs
T1=runs/t1_writers_250k/encoder.pt
# VP = vp-head hidden layers. The head must EVALUATE the chunk's function at the
# anchor points, so its capacity is a confound in every vp number; refitting it
# on frozen essences cannot test that, because depth-2 slots are 2.5% of the
# data and a fresh head sees too few.
LEN=${LEN:-2} DOUT=${DOUT:-256} STEPS=${STEPS:-250000} OC=${OC:-4} VP=${VP:-1}
NPOOL=${NPOOL:-16}
# compete is only meaningful for k>1 (k=1 degenerates to a uniform mean).
MODES=${MODES:-"pma compete"}
PEXTRAS=${PEXTRAS:-"write_time"}
RTMODE=${RTMODE:-stamp}
OCRANK=${OCRANK:-0}   # feed the op-cell each operand's canonical read position
SEEDMODE=${SEEDMODE:-tags}  # leaf identity: random tag, or canonical read rank
# Weight an output slot at dataflow depth d by DEPTHW^(d-1). 1 = uniform.
DEPTHW=${DEPTHW:-1}
# RMSNorm every value entering a register slot (leaf and computed alike).
VALNORM=${VALNORM:-0}
# Direct linear path from an instruction's content to the slot it writes.
SKIP=${SKIP:-0}
# Distinguishes a re-run of an already-recorded cell (the run dir is otherwise a
# pure function of the knobs, so it would silently overwrite it).
TAG=${TAG:-}

cell() {  # MODE POOL_EXTRA
  local MODE=$1 PE=$2
  local OUT=t2_len${LEN}_d${DOUT}_s${STEPS}_dep${OC}_vp${VP}_pool${NPOOL}_${MODE}_pe${PE}_rt${RTMODE}${OCRANK:+$([ "$OCRANK" = 1 ] && echo _ocrank)}$([ "$SEEDMODE" != tags ] && echo _seed$SEEDMODE)$([ "$DEPTHW" != 1 ] && echo _dw$DEPTHW)$([ "$VALNORM" = 1 ] && echo _vnorm)$([ "$SKIP" = 1 ] && echo _cskip)${TAG:+_$TAG}_writers
  echo "######## $(date '+%m-%d %H:%M:%S')  mode=$MODE  pool_extra=$PE  -> $OUT"
  # Generation runs on odin (multinode_gen.sh, LOCAL_GEN=0) so local cores stay
  # free for the trainer's per-step host work. ANCHOR_SEED must equal the
  # trainer's --anchor-seed (INVARIANT 1). The config carries no invalidity
  # block, so multinode's INJECT_INVALID=0 default matches the local corpus.
  RULE="cap=$LEN" CONFIG=configs/instr_alu_writers.json \
  TWINS=2 BATCH_SIZE=128 N_STATES=8 ANCHOR_SEED=0 \
    ./multinode_gen.sh "$STEPS" \
    | python scripts/train_t2_encoder.py --t1-model "$T1" \
      --d-out "$DOUT" --d-event 16 --op-cell-depth "$OC" --vp-head-depth "$VP" \
      --n-pool-heads "$NPOOL" --pool-mode "$MODE" --pool-extra "$PE" \
      --read-time-mode "$RTMODE" $([ "$OCRANK" = 1 ] && echo --op-cell-rank) \
      --seed-mode "$SEEDMODE" --vp-depth-weight "$DEPTHW" \
      $([ "$VALNORM" = 1 ] && echo --value-norm) \
      $([ "$SKIP" = 1 ] && echo --content-skip) --route binding \
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

for MODE in $MODES; do for PE in $PEXTRAS; do cell "$MODE" "$PE"; done; done
echo "######## 2x2 COMPLETE $(date '+%m-%d %H:%M:%S') ########"
