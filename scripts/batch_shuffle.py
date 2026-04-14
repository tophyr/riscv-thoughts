#!/usr/bin/env python3
"""Shuffle batch order in a batch file (RVS or RVB)."""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._batch_util import binary_stdout, detect_format


def _build_index(f, fmt):
    index = []
    while True:
        offset = f.tell()
        header = f.read(fmt.batch_header.size)
        if len(header) == 0:
            break
        if len(header) < fmt.batch_header.size:
            raise EOFError(f'Truncated header at offset {offset}')
        vals = fmt.validate(header)
        body_size = fmt.body_size(*vals)
        total_size = fmt.batch_header.size + body_size
        index.append((offset, total_size))
        f.seek(offset + total_size)
    return index


def shuffle_seekable(f, out, fmt, seed, verbose=False):
    index = _build_index(f, fmt)
    if verbose:
        print(f'Indexed {len(index)} {fmt.name} batches', file=sys.stderr)
    rng = random.Random(seed)
    rng.shuffle(index)
    fmt.write_header(out)
    try:
        for i, (offset, size) in enumerate(index):
            f.seek(offset)
            data = f.read(size)
            if len(data) < size:
                raise EOFError(f'Batch {i}: short read')
            out.write(data)
            if verbose and (i + 1) % 100 == 0:
                print(f'{i + 1}/{len(index)} batches', file=sys.stderr)
    except BrokenPipeError:
        pass
    if verbose:
        print(f'Done: {len(index)} batches shuffled', file=sys.stderr)


def shuffle_buffered(f, out, fmt, seed, verbose=False):
    batches = []
    while True:
        data = fmt.read_bytes(f)
        if data is None:
            break
        batches.append(data)
    if verbose:
        print(f'Buffered {len(batches)} {fmt.name} batches', file=sys.stderr)
    rng = random.Random(seed)
    rng.shuffle(batches)
    fmt.write_header(out)
    try:
        for data in batches:
            out.write(data)
    except BrokenPipeError:
        pass
    if verbose:
        print(f'Done: {len(batches)} batches shuffled', file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description='Shuffle batch order.')
    p.add_argument('file', nargs='?', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    out = binary_stdout()
    if args.file:
        with open(args.file, 'rb') as f:
            fmt = detect_format(f)
            shuffle_seekable(f, out, fmt, args.seed, verbose=args.verbose)
    else:
        fmt = detect_format(sys.stdin.buffer)
        shuffle_buffered(sys.stdin.buffer, out, fmt, args.seed,
                         verbose=args.verbose)
    out.close()


if __name__ == '__main__':
    main()
