#!/bin/bash
# Sweep: pool_mode {pma, compete} x pool_extra {none, write_time, order,
# event}, at len=2 in the
# known-good regime (op_cell=4, vp_head=1, n_pool_heads=16, d_out=256, 250k).
#
# Why len=2: it is the smallest squash, and the probe (vp_pool_localize.py) put
# the per-slot ceiling there at 98.3% bit accuracy vs the essence's ~80% — so
# ~18 points are lost in the pool, ~13 of them recoverable from write_time
# alone. This asks whether the ENCODER can exploit that, not just a readout.
#
# Why compete is in the grid: it has only ever been run with the DEEP vp_head
# (op_cell=1, vp_head=4), i.e. the regime that made multi-head pma look useless
# too. Its cell in the weak-read-back regime is empty, not lost.
#
# Scored on BOTH vp (encode-upward fidelity) and the GVN collapse rows
# (rename / commutative / distinct) — a vp win that craters GVN is a regression.
set -u
cd /home/tophy/Projects/riscv-thoughts
source .venv/bin/activate
D=/home/tophy/Projects/riscv-thoughts/logs
T1=runs/t1_writers_250k/encoder.pt
LEN=${LEN:-2} DOUT=256 STEPS=${STEPS:-250000} OC=4 VP=1
NPOOL=${NPOOL:-16}
# compete is only meaningful for k>1 (k=1 degenerates to a uniform mean).
MODES=${MODES:-"pma compete"}
PEXTRAS=${PEXTRAS:-"write_time"}
RTMODE=${RTMODE:-stamp}

cell() {  # MODE POOL_EXTRA
  local MODE=$1 PE=$2
  local OUT=t2_len${LEN}_d${DOUT}_s${STEPS}_dep${OC}_vp${VP}_pool${NPOOL}_${MODE}_pe${PE}_rt${RTMODE}_writers
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
      --read-time-mode "$RTMODE" --route binding \
      --n-steps "$STEPS" --anchor-seed 0 --n-anchor-states 8 --out-dir "$OUT" \
      > "$D/t2_${OUT}.log" 2>&1
  local RC=$?
  if [ ! -f "runs/$OUT/t2.pt" ]; then
    echo "   FAILED rc=$RC"; tail -12 "$D/t2_${OUT}.log"; return 1
  fi
  echo "   final: $(grep "step $STEPS " "$D/t2_${OUT}.log" | tail -1)"
  echo "   --- binding + GVN (eval.py) ---"
  python scripts/eval.py --t1-model "$T1" --t2 "runs/$OUT" \
      --t2-corpus "/tmp/t2_eval_${LEN}.rvt" --route binding --max-batches 20 2>&1 \
    | grep -vE "Warning|readline|history" | sed 's/^/   /'
  echo "   --- faithfulness / on-sphere ---"
  python probes/t2_faithfulness.py "runs/$OUT" "$T1" "$LEN" 2>&1 \
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
