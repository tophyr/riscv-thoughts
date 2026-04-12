#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

CORPUS=/tmp/overnight_corpus.bin

echo "=== Generating large corpus ==="
python scripts/gen_seq_batches.py --n-batches 5000 --batch-size 256 -v \
    > "$CORPUS"

echo ""
echo "=== Training streaming compressor + decoder + round-trip (10000 steps) ==="
python scripts/batch_repeat.py --forever "$CORPUS" | \
    python scripts/train_compressor.py \
        --mode streaming \
        --n-steps 10000 --log-every 200 \
        --lr 3e-4 --lr-schedule 10000 \
        --recon-weight 1.0 \
        --pairwise-weight 1.0 \
        --roundtrip-weight 1.0 \
        --save /tmp/overnight_run

echo ""
echo "=== Done ==="
