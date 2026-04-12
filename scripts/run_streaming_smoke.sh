#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "=== Generating corpus ==="
python scripts/gen_seq_batches.py --n-batches 2000 --batch-size 256 -v \
    > /tmp/streaming_smoke_corpus.bin

echo ""
echo "=== Training streaming compressor (1000 steps) ==="
python scripts/batch_repeat.py --forever /tmp/streaming_smoke_corpus.bin | \
    python scripts/train_compressor.py \
        --mode streaming --n-steps 1000 --log-every 100 \
        --lr-schedule 1000 \
        --save /tmp/streaming_smoke

echo ""
echo "=== Done ==="
