#!/usr/bin/env python3
"""Repeat a batch file for multi-epoch training (RVS or RVB)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._batch_util import binary_stdout, detect_format


def main():
    p = argparse.ArgumentParser(description='Repeat a batch file.')
    p.add_argument('file', help='Input batch file (must be seekable)')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--forever', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    if args.epochs is None and not args.forever:
        print('Specify --epochs N or --forever', file=sys.stderr)
        sys.exit(1)

    out = binary_stdout()

    with open(args.file, 'rb') as f:
        fmt = detect_format(f)
        data_start = f.tell()

        n_batches = 0
        while True:
            offset = f.tell()
            header = f.read(fmt.batch_header.size)
            if len(header) == 0:
                break
            if len(header) < fmt.batch_header.size:
                break
            vals = fmt.validate(header)
            body_size = fmt.body_size(*vals)
            f.seek(offset + fmt.batch_header.size + body_size)
            n_batches += 1

        if args.verbose:
            print(f'{n_batches} {fmt.name} batches per epoch')

        fmt.write_header(out)
        epoch = 0
        total = 0
        try:
            while args.forever or epoch < args.epochs:
                f.seek(data_start)
                for _ in range(n_batches):
                    data = fmt.read_bytes(f)
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
