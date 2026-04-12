#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "=== Generating training corpus ==="
python scripts/gen_seq_batches.py --n-batches 5000 --batch-size 256 -v \
    > /tmp/context_exp_corpus.bin

echo ""
echo "=== Training window_size=1 (baseline, no context) ==="
python scripts/batch_repeat.py --forever /tmp/context_exp_corpus.bin | \
    python scripts/train_compressor.py \
        --window-size 1 --n-steps 5000 --log-every 500 \
        --lr-schedule 5000 \
        --save /tmp/context_exp_w1

echo ""
echo "=== Training window_size=2 (one instruction of context) ==="
python scripts/batch_repeat.py --forever /tmp/context_exp_corpus.bin | \
    python scripts/train_compressor.py \
        --window-size 2 --n-steps 5000 --log-every 500 \
        --lr-schedule 5000 \
        --save /tmp/context_exp_w2

echo ""
echo "=== Evaluating window_size=1 ==="
python scripts/gen_seq_batches.py --n-batches 100 --batch-size 64 --seed 9999 | \
    python scripts/eval_compressor.py \
        --model /tmp/context_exp_w1/model.pt --window-size 1

echo ""
echo "=== Evaluating window_size=2 ==="
python scripts/gen_seq_batches.py --n-batches 100 --batch-size 64 --seed 9999 | \
    python scripts/eval_compressor.py \
        --model /tmp/context_exp_w2/model.pt --window-size 2

echo ""
echo "=== Done ==="
