#!/usr/bin/env python3
"""Slice and inspect binary batch streams (RVS or RVB)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._batch_util import binary_stdout, detect_format, read_batch_or_error


def do_info(f, fmt, lenient=False):
    count = 0
    total_items = 0
    last_B = None
    while True:
        data, err = read_batch_or_error(fmt, f, lenient=lenient)
        if data is None and err is None:
            break
        if data is None:
            print(f'ERROR at batch {count}: {err}', file=sys.stderr)
            if not lenient:
                return 1
            break
        vals = fmt.validate(data)
        B = vals[0]
        count += 1
        total_items += B
        last_B = B

    print(f'Format:       {fmt.name}', file=sys.stderr)
    print(f'Batches:      {count}', file=sys.stderr)
    print(f'Items:        {total_items}', file=sys.stderr)
    if last_B is not None:
        print(f'Batch size:   {last_B}', file=sys.stderr)
    return 0


def do_slice(f, out, fmt, skip=0, count=None, lenient=False):
    fmt.write_header(out)
    idx = 0
    written = 0
    try:
        while True:
            if count is not None and written >= count:
                break
            data, err = read_batch_or_error(fmt, f, lenient=lenient)
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


def do_tail(f, out, fmt, n, lenient=False):
    from collections import deque
    ring = deque(maxlen=n)
    total = 0
    while True:
        data, err = read_batch_or_error(fmt, f, lenient=lenient)
        if data is None and err is None:
            break
        if data is None:
            print(f'ERROR at batch {total}: {err}', file=sys.stderr)
            if not lenient:
                return 1
            break
        ring.append(data)
        total += 1
    fmt.write_header(out)
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
    p.add_argument('file', nargs='?', default=None)
    p.add_argument('--info', action='store_true')
    p.add_argument('--skip', type=int, default=0)
    p.add_argument('--count', type=int, default=None)
    p.add_argument('--tail', type=int, default=None)
    p.add_argument('--lenient', action='store_true')
    args = p.parse_args()

    f = open(args.file, 'rb') if args.file else sys.stdin.buffer
    try:
        fmt = detect_format(f)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    if args.info:
        rc = do_info(f, fmt, lenient=args.lenient)
    elif args.tail is not None:
        out = binary_stdout()
        rc = do_tail(f, out, fmt, args.tail, lenient=args.lenient)
        out.close()
    else:
        out = binary_stdout()
        rc = do_slice(f, out, fmt, skip=args.skip, count=args.count,
                      lenient=args.lenient)
        out.close()

    if args.file:
        f.close()
    sys.exit(rc)


if __name__ == '__main__':
    main()
