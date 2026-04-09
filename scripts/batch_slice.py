#!/usr/bin/env python3
"""Slice and inspect binary batch streams.

Like dd but for batches. Reads a batch stream from a file or stdin,
writes a batch stream to stdout.

Usage:
    # First 100 batches:
    batch_slice.py --count 100 < corpus.bin > first100.bin

    # Skip 50, take 100:
    batch_slice.py --skip 50 --count 100 < corpus.bin > middle.bin

    # Last 10 batches:
    batch_slice.py --tail 10 corpus.bin > last10.bin

    # Count and validate (no output):
    batch_slice.py --info corpus.bin

    # Salvage a truncated file:
    batch_slice.py --lenient corpus.bin > clean.bin
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen import read_stream_header, read_batch_bytes, write_stream_header
from scripts._batch_util import binary_stdout, validate_batch_header


def _read_one_batch(f, lenient=False):
    """Read one batch as raw bytes with validation.

    Returns (bytes, None) on success, (None, None) on clean EOF,
    (None, error_str) on error. In lenient mode, truncation returns
    clean EOF instead of error.
    """
    try:
        data = read_batch_bytes(f)
    except EOFError as e:
        if lenient:
            return None, None
        return None, str(e)
    if data is None:
        return None, None

    try:
        validate_batch_header(data)
    except ValueError as e:
        return None, str(e)

    return data, None


def do_info(f, lenient=False):
    """Count and validate batches. Print summary to stderr."""
    try:
        read_stream_header(f)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1

    count = 0
    total_instructions = 0
    last_B = None
    last_n_inputs = None

    while True:
        data, err = _read_one_batch(f, lenient=lenient)
        if data is None and err is None:
            break
        if data is None:
            print(f'ERROR at batch {count}: {err}', file=sys.stderr)
            if not lenient:
                return 1
            break
        B, _, n_inputs = validate_batch_header(data)
        count += 1
        total_instructions += B
        last_B = B
        last_n_inputs = n_inputs

    print(f'Batches:      {count}', file=sys.stderr)
    print(f'Instructions: {total_instructions}', file=sys.stderr)
    if last_B is not None:
        print(f'Batch size:   {last_B}', file=sys.stderr)
        print(f'Input states: {last_n_inputs}', file=sys.stderr)
    return 0


def do_slice(f, out, skip=0, count=None, lenient=False):
    """Copy a range of batches from f to out."""
    try:
        read_stream_header(f)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1

    write_stream_header(out)

    idx = 0
    written = 0

    try:
        while True:
            if count is not None and written >= count:
                break

            data, err = _read_one_batch(f, lenient=lenient)
            if data is None and err is None:
                break
            if data is None:
                print(f'ERROR at batch {idx}: {err}', file=sys.stderr)
                if not lenient:
                    return 1
                break

            if idx >= skip:
                out.write(data)
                written += 1
            idx += 1
    except BrokenPipeError:
        pass

    print(f'Wrote {written} batches (skipped {skip}, scanned {idx})',
          file=sys.stderr)
    return 0


def do_tail(f, out, n, lenient=False):
    """Copy the last n batches from f to out."""
    from collections import deque

    try:
        read_stream_header(f)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1

    ring = deque(maxlen=n)
    total = 0

    while True:
        data, err = _read_one_batch(f, lenient=lenient)
        if data is None and err is None:
            break
        if data is None:
            print(f'ERROR at batch {total}: {err}', file=sys.stderr)
            if not lenient:
                return 1
            break
        ring.append(data)
        total += 1

    write_stream_header(out)
    try:
        for data in ring:
            out.write(data)
    except BrokenPipeError:
        pass

    print(f'Wrote {len(ring)} batches (tail of {total} total)',
          file=sys.stderr)
    return 0


def main():
    p = argparse.ArgumentParser(
        description='Slice and inspect binary batch streams.')
    p.add_argument('file', nargs='?', default=None,
                   help='Input file (default: stdin)')
    p.add_argument('--info', action='store_true',
                   help='Count and validate batches (no output)')
    p.add_argument('--skip', type=int, default=0,
                   help='Skip first N batches')
    p.add_argument('--count', type=int, default=None,
                   help='Output at most N batches')
    p.add_argument('--tail', type=int, default=None,
                   help='Output the last N batches')
    p.add_argument('--lenient', action='store_true',
                   help='Tolerate truncated/corrupted batches')
    args = p.parse_args()

    if args.file:
        f = open(args.file, 'rb')
    else:
        f = sys.stdin.buffer

    if args.info:
        rc = do_info(f, lenient=args.lenient)
    elif args.tail is not None:
        out = binary_stdout()
        rc = do_tail(f, out, args.tail, lenient=args.lenient)
        out.close()
    else:
        out = binary_stdout()
        rc = do_slice(f, out, skip=args.skip, count=args.count,
                      lenient=args.lenient)
        out.close()

    if args.file:
        f.close()
    sys.exit(rc)


if __name__ == '__main__':
    main()
