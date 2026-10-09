#!/bin/bash
# Measure how much T2's score moves when only the frozen T1 beneath it is
# retrained — the floor any comparison between T1s has to clear (EXPERIMENT_LOG
# phase-15, finding 25, which put it at 0.023 points of exact match against
# 0.002 for retraining the T2 alone). Each arm is one T1 at a distinct corpus
# seed, then one T2 over it at the standard config; T2's own corpus seed is held
# at 0 so the T1 is the only variable. Edit the train_t1 body to replicate a
# different cell.
#
# Two arms at a time: a lone len=2 trainer leaves the GPU ~50% idle, while two
# concurrent ones hold it at 84-99% for ~1.3x aggregate throughput. Distinct
# PORTs are mandatory — the remote pidfile is keyed by it, so sharing one makes
# the second run reap the first's worker pool.
set -u
cd /home/tophy/Projects/riscv-thoughts
D=logs
train_t1() {  # SEED PORT
  local S=$1 P=$2
  BATCH_SIZE=128 TWINS=2 N_STATES=8 RULE=single \
  CONFIG=configs/instr_alu_writers.json ANCHOR_SEED=0 GEN_SEED=$S \
  PORT=$P REMOTE_GEN=56 \
    ./multinode_gen.sh 250000 \
    | .venv/bin/python scripts/train_encoder.py --d-model 128 --n-heads 4 \
      --n-layers 2 --d-out 64 --max-window 32 --imm-weight 0.3 \
      --read-time-mode rank --seed-mode tags --pool-extra write_time \
      --n-steps 250000 --anchor-seed 0 --n-anchor-states 8 \
      --out-dir t1_rttags_s$S > $D/t1_rttags_s$S.log 2>&1
}
run_t2() {   # SEED PORT
  local S=$1 P=$2
  T1=runs/t1_rttags_s$S/encoder.pt \
  LEN=2 STEPS=1000000 DOUT=512 OC=4 VP=1 NPOOL=16 MODES=pma \
  PEXTRAS=write_time RTMODE=rank SEEDMODE=rank PORT=$P REMOTE_GEN=56 \
  TAG=rttagsS$S \
    bash scripts/run_t2.sh > $D/run_t2_rttags_s$S.log 2>&1
}
for PAIR in "1 2" "3 4"; do
  set -- $PAIR; A=$1; B=$2
  echo "=== $(date '+%m-%d %H:%M:%S') T1 arms $A,$B ==="
  train_t1 $A 6464 & P1=$!
  train_t1 $B 6465 & P2=$!
  wait $P1 $P2
  for S in $A $B; do
    [ -f runs/t1_rttags_s$S/encoder.pt ] || { echo "  T1 s$S FAILED"; tail -5 $D/t1_rttags_s$S.log; }
  done
  echo "=== $(date '+%m-%d %H:%M:%S') T2 over arms $A,$B ==="
  run_t2 $A 6464 & P1=$!
  run_t2 $B 6465 & P2=$!
  wait $P1 $P2
  for S in $A $B; do
    printf "  s%s: %s\n" "$S" "$(grep -E 'OVERALL' $D/run_t2_rttags_s$S.log | sed 's/^ *//' || echo 'NO RESULT')"
  done
done
echo "=== $(date '+%m-%d %H:%M:%S') ALL DONE ==="
grep -H "OVERALL" $D/run_t2_rttags_s[1-4].log | sed 's/^ *//'
