#!/usr/bin/env python3
"""Concatenate multiple batch files into one stream.

Each input file must have a valid stream header. The output gets one
stream header followed by all batches from all files in order.

Usage:
    batch_cat.py part1.bin part2.bin part3.bin > combined.bin
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen import read_stream_header, read_batch_bytes, write_stream_header
from scripts._batch_util import binary_stdout


def main():
    p = argparse.ArgumentParser(description='Concatenate batch files.')
    p.add_argument('inputs', nargs='+', help='Input batch files')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--lenient', action='store_true',
                   help='Skip truncated batches instead of failing')
    args = p.parse_args()

    out = binary_stdout()
    write_stream_header(out)

    total = 0
    try:
        for path in args.inputs:
            with open(path, 'rb') as f:
                try:
                    read_stream_header(f)
                except ValueError as e:
                    print(f'ERROR: {path}: {e}', file=sys.stderr)
                    sys.exit(1)
                count = 0
                while True:
                    try:
                        data = read_batch_bytes(f)
                    except EOFError as e:
                        print(f'WARNING: {path}: {e}', file=sys.stderr)
                        if not args.lenient:
                            sys.exit(1)
                        break
                    if data is None:
                        break
                    out.write(data)
                    count += 1
                total += count
                if args.verbose:
                    print(f'{path}: {count} batches')
    except BrokenPipeError:
        pass

    out.close()
    if args.verbose:
        print(f'Done: {total} batches from {len(args.inputs)} files')


if __name__ == '__main__':
    main()
