#!/usr/bin/env python3
"""Repeat a batch file for multi-epoch training.

Reads a seekable batch file and emits its batches N times (or forever)
as a single stream. Requires a file argument (cannot repeat stdin).

Usage:
    batch_repeat.py --epochs 10 corpus.bin | train_compressor.py
    batch_repeat.py --forever corpus.bin | train_compressor.py --n-steps 1000000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen import read_stream_header, read_batch_bytes, write_stream_header
from scripts._batch_util import binary_stdout, validate_batch_header, \
    BATCH_HEADER, batch_body_size


def _count_batches(f):
    """Count batches by scanning headers only (no body reads)."""
    count = 0
    while True:
        offset = f.tell()
        header = f.read(BATCH_HEADER.size)
        if len(header) == 0:
            break
        if len(header) < BATCH_HEADER.size:
            break
        vals = validate_batch_header(header)
        body = batch_body_size(*vals)
        f.seek(offset + BATCH_HEADER.size + body)
        count += 1
    return count


def main():
    p = argparse.ArgumentParser(description='Repeat a batch file.')
    p.add_argument('file', help='Input batch file (must be seekable)')
    p.add_argument('--epochs', type=int, default=None,
                   help='Number of times to repeat')
    p.add_argument('--forever', action='store_true',
                   help='Repeat indefinitely (until reader closes pipe)')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    if args.epochs is None and not args.forever:
        print('Specify --epochs N or --forever', file=sys.stderr)
        sys.exit(1)

    out = binary_stdout()

    with open(args.file, 'rb') as f:
        read_stream_header(f)
        data_start = f.tell()

        n_batches = _count_batches(f)

        if args.verbose:
            print(f'{n_batches} batches per epoch')

        write_stream_header(out)

        epoch = 0
        total = 0
        try:
            while args.forever or epoch < args.epochs:
                f.seek(data_start)
                for _ in range(n_batches):
                    data = read_batch_bytes(f)
                    if data is None:
                        break
                    out.write(data)
                    total += 1
                epoch += 1
                if args.verbose:
                    print(f'Epoch {epoch}: {total} batches total')
        except BrokenPipeError:
            pass

    out.close()
    if args.verbose:
        print(f'Done: {epoch} epochs, {total} batches')


if __name__ == '__main__':
    main()
