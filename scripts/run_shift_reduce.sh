#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

CORPUS=/tmp/sr_corpus.bin

echo "=== Generating corpus ==="
python scripts/gen_seq_batches.py --n-batches 2000 --batch-size 32 -v \
    > "$CORPUS"

echo ""
echo "=== Training shift-reduce compressor with REINFORCE (500 steps) ==="
python scripts/batch_repeat.py --forever "$CORPUS" | \
    python scripts/train_compressor.py \
        --mode streaming \
        --n-steps 500 --log-every 25 \
        --lr 3e-4 \
        --pairwise-weight 1.0 \
        --save /tmp/sr_reinforce_run

echo ""
echo "=== Done ==="
