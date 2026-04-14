#!/usr/bin/env python3
"""Concatenate multiple batch files into one stream (RVS or RVB)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._batch_util import binary_stdout, detect_format


def main():
    p = argparse.ArgumentParser(description='Concatenate batch files.')
    p.add_argument('inputs', nargs='+')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--lenient', action='store_true')
    args = p.parse_args()

    out = binary_stdout()
    fmt = None
    total = 0
    try:
        for path in args.inputs:
            with open(path, 'rb') as f:
                try:
                    file_fmt = detect_format(f)
                except ValueError as e:
                    print(f'ERROR: {path}: {e}', file=sys.stderr)
                    sys.exit(1)
                if fmt is None:
                    fmt = file_fmt
                    fmt.write_header(out)
                elif file_fmt.name != fmt.name:
                    print(f'ERROR: {path}: format {file_fmt.name} != '
                          f'{fmt.name}', file=sys.stderr)
                    sys.exit(1)
                count = 0
                while True:
                    try:
                        data = fmt.read_bytes(f)
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
