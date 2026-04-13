#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

CORPUS=/tmp/overnight_corpus.bin

if [ ! -f "$CORPUS" ]; then
    echo "=== Generating corpus ==="
    python scripts/gen_seq_batches.py --n-batches 5000 --batch-size 256 -v \
        > "$CORPUS"
fi

echo ""
echo "=== Training with emit cost (5000 steps, warmup 2000) ==="
python scripts/batch_repeat.py --forever "$CORPUS" | \
    python scripts/train_compressor.py \
        --mode streaming \
        --n-steps 5000 --log-every 200 \
        --lr 3e-4 --lr-schedule 5000 \
        --recon-weight 1.0 \
        --pairwise-weight 1.0 \
        --roundtrip-weight 1.0 \
        --emit-cost 1.0 \
        --emit-cost-warmup 2000 \
        --save /tmp/emit_cost_run

echo ""
echo "=== Done ==="
