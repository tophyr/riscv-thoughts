#!/usr/bin/env python3
"""Evaluate a trained T0→T1 compressor model.

Reads held-out batches from stdin for correlation evaluation.

Usage:
    gen_batches.py --n-batches 10 --batch-size 64 | eval_compressor.py runs/20260407_184124
    gen_batches.py --n-batches 10 --batch-size 64 > eval_data.bin
    eval_compressor.py runs/20260407_184124 < eval_data.bin
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compressor.eval import evaluate
from datagen import BatchReader


def main():
    p = argparse.ArgumentParser(description='Evaluate a trained compressor.')
    p.add_argument('run_dir', type=str, help='Path to a saved run directory')
    args = p.parse_args()

    batches = None
    if not sys.stdin.isatty():
        batches = list(BatchReader(sys.stdin.buffer))
        print(f'Read {len(batches)} held-out batches from stdin')

    evaluate(args.run_dir, batches=batches)


if __name__ == '__main__':
    main()
