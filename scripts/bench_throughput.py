#!/usr/bin/env python3
"""Read RVT batches from stdin and report throughput.

Counts batches per second and bytes per second over a rolling window.
Used as the terminal stage in pipeline benchmarks: any pipeline that
produces a binary batch stream can be measured by piping into this.

Periodic stderr output every --log-every-sec; final summary on EOF.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._batch_util import detect_format


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log-every-sec', type=float, default=5.0,
                   help='Periodic status interval (seconds).')
    p.add_argument('--max-batches', type=int, default=None,
                   help='Stop after this many batches.')
    args = p.parse_args()

    fmt = detect_format(sys.stdin.buffer)

    n = 0
    bytes_total = 0
    t0 = time.perf_counter()
    last_log = t0
    last_n = 0
    last_bytes = 0

    try:
        while True:
            data = fmt.read_bytes(sys.stdin.buffer)
            if data is None:
                break
            n += 1
            bytes_total += len(data)

            now = time.perf_counter()
            if now - last_log >= args.log_every_sec:
                window = now - last_log
                window_n = n - last_n
                window_bytes = bytes_total - last_bytes
                elapsed = now - t0
                print(f'[{elapsed:6.1f}s] '
                      f'total={n:6d} batches  '
                      f'window={window_n/window:5.2f} batch/s  '
                      f'avg={n/elapsed:5.2f} batch/s  '
                      f'mb/s={window_bytes/1e6/window:6.1f} (window) '
                      f'{bytes_total/1e6/elapsed:6.1f} (avg)',
                      file=sys.stderr)
                last_log = now
                last_n = n
                last_bytes = bytes_total

            if args.max_batches and n >= args.max_batches:
                break
    except KeyboardInterrupt:
        pass

    elapsed = time.perf_counter() - t0
    if elapsed > 0:
        print(f'Final: {n} batches in {elapsed:.1f}s '
              f'= {n/elapsed:.2f} batch/s, '
              f'{bytes_total/1e6/elapsed:.1f} MB/s '
              f'({bytes_total/1e6:.1f} MB total)',
              file=sys.stderr)
    else:
        print(f'Final: {n} batches', file=sys.stderr)


if __name__ == '__main__':
    main()
