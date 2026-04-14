#!/usr/bin/env python3
"""Generate single-instruction training batches (RVB format).

Each batch contains batch_size random instructions, each executed
on n_inputs shared random register states.

Usage:
    python scripts/gen_instr_batches.py --n-batches 1000 > corpus.bin
    python scripts/gen_instr_batches.py --config branch_heavy.json --n-batches 1000 > focused.bin
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from datagen import produce_instruction_batch, load_distribution
from datagen.batchgen import write_stream_header, write_batch
from scripts._batch_util import binary_stdout


def main():
    p = argparse.ArgumentParser(
        description='Generate single-instruction training batches.')
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--n-inputs', type=int, default=32)
    p.add_argument('--n-batches', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--config', type=str, default=None,
                   help='JSON config file for instruction type distribution')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    dist = load_distribution(args.config) if args.config else None

    out = binary_stdout()
    rng = np.random.default_rng(args.seed)
    write_stream_header(out)

    try:
        for i in range(args.n_batches):
            batch = produce_instruction_batch(
                args.batch_size, args.n_inputs, rng, dist=dist)
            write_batch(out, batch)
            if args.verbose and (i + 1) % 100 == 0:
                print(f'{i + 1}/{args.n_batches} batches',
                      file=sys.stderr)
    except BrokenPipeError:
        pass

    out.close()
    if args.verbose:
        print(f'Done: {args.n_batches} batches, '
              f'{args.n_batches * args.batch_size} instructions',
              file=sys.stderr)


if __name__ == '__main__':
    main()
