#!/usr/bin/env python3
"""Shuffle the order of batches in a batch file.

For seekable files (regular files), builds an index of batch offsets
in a first pass, shuffles the index, then writes batches in shuffled
order. Memory usage is O(n_batches), not O(file_size).

For stdin, buffers all batches in memory. Only suitable for small streams.

Usage:
    batch_shuffle.py corpus.bin > shuffled.bin
    batch_shuffle.py --seed 123 corpus.bin > shuffled.bin
    cat corpus.bin | batch_shuffle.py > shuffled.bin  # memory-intensive
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen import read_stream_header, read_batch_bytes, write_stream_header
from scripts._batch_util import binary_stdout, validate_batch_header, \
    BATCH_HEADER, batch_body_size


def _build_index(f):
    """Scan a batch stream and return list of (offset, size) tuples."""
    index = []
    while True:
        offset = f.tell()
        header = f.read(BATCH_HEADER.size)
        if len(header) == 0:
            break
        if len(header) < BATCH_HEADER.size:
            raise EOFError(f'Truncated header at offset {offset}')
        vals = validate_batch_header(header)
        body = batch_body_size(*vals)
        total_size = BATCH_HEADER.size + body
        index.append((offset, total_size))
        f.seek(offset + total_size)
    return index


def shuffle_seekable(f, out, seed, verbose=False):
    """Shuffle using seek-based random access. O(n_batches) memory."""
    index = _build_index(f)

    if verbose:
        print(f'Indexed {len(index)} batches', file=sys.stderr)

    rng = random.Random(seed)
    rng.shuffle(index)

    write_stream_header(out)
    try:
        for i, (offset, size) in enumerate(index):
            f.seek(offset)
            data = f.read(size)
            if len(data) < size:
                raise EOFError(
                    f'Batch {i}: read {len(data)} bytes, expected {size}')
            out.write(data)
            if verbose and (i + 1) % 100 == 0:
                print(f'{i + 1}/{len(index)} batches', file=sys.stderr)
    except BrokenPipeError:
        pass

    if verbose:
        print(f'Done: {len(index)} batches shuffled', file=sys.stderr)


def shuffle_buffered(f, out, seed, verbose=False):
    """Shuffle by buffering all batches in memory. For stdin."""
    batches = []
    while True:
        data = read_batch_bytes(f)
        if data is None:
            break
        batches.append(data)

    if verbose:
        print(f'Buffered {len(batches)} batches', file=sys.stderr)

    rng = random.Random(seed)
    rng.shuffle(batches)

    write_stream_header(out)
    try:
        for data in batches:
            out.write(data)
    except BrokenPipeError:
        pass

    if verbose:
        print(f'Done: {len(batches)} batches shuffled', file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description='Shuffle batch order.')
    p.add_argument('file', nargs='?', default=None,
                   help='Input file (default: stdin, buffered)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    out = binary_stdout()

    if args.file:
        with open(args.file, 'rb') as f:
            read_stream_header(f)
            shuffle_seekable(f, out, args.seed, verbose=args.verbose)
    else:
        read_stream_header(sys.stdin.buffer)
        shuffle_buffered(sys.stdin.buffer, out, args.seed,
                         verbose=args.verbose)

    out.close()


if __name__ == '__main__':
    main()
