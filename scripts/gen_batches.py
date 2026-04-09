#!/usr/bin/env python3
"""Generate training batches and write them to stdout in binary format.

Single-threaded. For parallelism, use mux_batches.py to run multiple
instances and interleave their output.

Modes:
    normal:  random instructions, structured random registers (default)
    focused: branch-heavy batches with equality-rich registers

Usage:
    python scripts/gen_batches.py --n-batches 1000 > corpus.bin
    python scripts/gen_batches.py --n-batches 100 --focused --batch-size 256 > focused.bin
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from datagen import produce_batch, produce_focused_batch, write_batch, write_stream_header
from scripts._batch_util import binary_stdout


def main():
    p = argparse.ArgumentParser(description='Generate training batches.')
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--n-inputs', type=int, default=32)
    p.add_argument('--n-batches', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--focused', action='store_true',
                   help='Generate branch-focused batches with equal registers')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    out = binary_stdout()

    rng = np.random.default_rng(args.seed)
    write_stream_header(out)

    producer = produce_focused_batch if args.focused else produce_batch

    try:
        for i in range(args.n_batches):
            batch = producer(args.batch_size, args.n_inputs, rng)
            write_batch(out, batch)
            if args.verbose and (i + 1) % 100 == 0:
                print(f'{i + 1}/{args.n_batches} batches')
    except BrokenPipeError:
        pass

    out.close()
    if args.verbose:
        print(f'Done: {args.n_batches} batches, '
              f'{args.n_batches * args.batch_size} instructions')


if __name__ == '__main__':
    main()
