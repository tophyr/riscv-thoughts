#!/bin/bash
# Train ONE T2 cell and score it. Every knob is an env var with the adopted
# config as its default, so a cell is named by what it changes:
#
#   LEN=2 GEN_SEED=7 TAG=s7 bash scripts/run_t2.sh
#   CONFIG=configs/instr_alu_writers.json bash scripts/run_t2.sh
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
T1=${T1:-runs/t1_writers_500k_vp3/encoder.pt}
# VP = vp-head hidden layers. The head must EVALUATE the chunk's function at the
# anchor points, so its capacity is a confound in every vp number; at T2 its
# shape is free until vp = 0 at len=2 (METRICS.md).
LEN=${LEN:-2} DOUT=${DOUT:-512} STEPS=${STEPS:-1000000} OC=${OC:-4} VP=${VP:-4}
NPOOL=${NPOOL:-16}
# Local generator workers. Per-row anchors make a batch ~2.7x larger on the
# wire, so generation runs on this machine rather than over the link to odin.
GEN=${GEN:-26}
# The corpus is ONE file: opcode mix, exclude_x0_dest, and the equivalence /
# invalidity / chain rates all live in it, and no flag can override any of
# them. A sweep over corpus composition is therefore a config per arm, and the
# run directory is named after the config so two arms cannot collide.
CONFIG=${CONFIG:-configs/instr_alu_writers_chain010.json}
# The instruction-stream seed. Re-running a cell without changing it redraws
# the SAME corpus and measures initialization variance only, so a replicate
# needs GEN_SEED (METRICS.md, null interventions).
GEN_SEED=${GEN_SEED:-0}
# Distinguishes a re-run of an already-recorded cell (the run dir is otherwise a
# pure function of the knobs, so it would silently overwrite it).
TAG=${TAG:-}
CORPUS_SLUG=$(basename "$CONFIG" .json); CORPUS_SLUG=${CORPUS_SLUG#instr_}

cell() {
  local OUT=t2_len${LEN}_d${DOUT}_s${STEPS}_dep${OC}_vp${VP}_pool${NPOOL}${TAG:+_$TAG}_${CORPUS_SLUG}
  echo "######## $(date '+%m-%d %H:%M:%S')  -> $OUT"
  # What the run trained on, recorded before it starts: hparams.json is the
  # trainer's own arguments and says nothing about the data. The config's
  # contents are copied in, so editing the file later cannot rewrite history.
  python scripts/record_corpus.py "runs/$OUT" --config "$CONFIG" \
    --rule "cap=$LEN" --twins 2 --batch-size 128 --n-states 8 \
    --gen-seed "$GEN_SEED" --count "$STEPS" \
    --generator "datagen --gen-count $GEN" || return 1
  python -m datagen --rule "cap=$LEN" --twins 2 --batch-size 128 --n-states 8 \
      --config "$CONFIG" --seed "$GEN_SEED" --gen-count "$GEN" \
      --count "$STEPS" 2> "$D/gen_${OUT}.log" \
    | python scripts/train_t2_encoder.py --t1-model "$T1" \
      --d-out "$DOUT" --d-event 16 --op-cell-depth "$OC" --vp-head-depth "$VP" \
      --n-pool-heads "$NPOOL" --route binding \
      --n-steps "$STEPS" --out-dir "$OUT" \
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
      --t2-corpus "$EVAL" --route binding --max-batches 20 \
      --gvn-chunk-len "$LEN" 2>&1 \
    | grep -vE "Warning|readline|history" | sed 's/^/   /'
  # The criterion is exact 32-bit match with the axes crossed: per-bit flatters
  # a broken subset (96% per-bit has been 74% exact), and either axis alone
  # hides the cells that carry the error.
  echo "   --- vp error budget: sources x immediate x depth ---"
  python probes/vp_budget.py "runs/$OUT" "$T1" "$LEN" 100 \
      configs/instr_alu_writers.json "$EVAL" 64 2>&1 \
    | grep -vE "Warning|readline|history" | sed 's/^/   /'
}

# One shared held-out corpus so every cell is scored on identical data, at all
# 64 states of the fixed held-out anchor set no model trains on: with fewer, one
# hard value pair decides a whole bucket. It is drawn from the UNMODIFIED
# writers corpus whatever a cell trained on, because a cell that raised the
# density of some chunk shape has to be graded on the population the model will
# actually meet, not on its own training mix.
EVAL=/tmp/t2_eval_heldout64_${LEN}.rvt
if [ ! -f "$EVAL" ]; then
  echo "generating shared eval corpus..."
  python -m datagen --rule "cap=$LEN" --twins 2 --batch-size 128 --n-states 64 \
      --heldout-anchors --config configs/instr_alu_writers.json --count 100 \
      > "$EVAL.part" 2>/dev/null && mv "$EVAL.part" "$EVAL"
fi

cell
echo "######## COMPLETE $(date '+%m-%d %H:%M:%S') ########"
