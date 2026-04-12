#!/usr/bin/env python3
"""Generate sequence training batches and write them to stdout in binary format.

Single-threaded. For parallelism, use mux_batches.py to run multiple
instances and interleave their output.

Usage:
    python scripts/gen_seq_batches.py --n-batches 1000 > corpus.bin
    python scripts/gen_seq_batches.py --n-batches 100 --batch-size 256 --max-block-len 5 > seqs.bin
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from datagen import produce_seq_batch, write_seq_batch, write_seq_stream_header
from scripts._batch_util import binary_stdout


def main():
    p = argparse.ArgumentParser(description='Generate sequence training batches.')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--n-inputs', type=int, default=4)
    p.add_argument('--max-block-len', type=int, default=5)
    p.add_argument('--n-batches', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    out = binary_stdout()

    rng = np.random.default_rng(args.seed)
    write_seq_stream_header(out)

    try:
        for i in range(args.n_batches):
            batch = produce_seq_batch(
                args.batch_size, args.n_inputs,
                args.max_block_len, rng)
            write_seq_batch(out, batch)
            if args.verbose and (i + 1) % 100 == 0:
                print(f'{i + 1}/{args.n_batches} batches',
                      file=sys.stderr)
    except BrokenPipeError:
        pass

    out.close()
    if args.verbose:
        print(f'Done: {args.n_batches} batches, '
              f'{args.n_batches * args.batch_size} sequences',
              file=sys.stderr)


if __name__ == '__main__':
    main()
