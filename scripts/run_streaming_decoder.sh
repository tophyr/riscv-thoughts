#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

CORPUS=/tmp/streaming_smoke_corpus.bin

if [ ! -f "$CORPUS" ]; then
    echo "=== Generating corpus ==="
    python scripts/gen_seq_batches.py --n-batches 2000 --batch-size 256 -v \
        > "$CORPUS"
fi

echo ""
echo "=== Training streaming compressor + decoder (1000 steps) ==="
python scripts/batch_repeat.py --forever "$CORPUS" | \
    python scripts/train_compressor.py \
        --mode streaming --n-steps 1000 --log-every 100 \
        --lr-schedule 1000 \
        --save /tmp/streaming_decoder

echo ""
echo "=== Done ==="
