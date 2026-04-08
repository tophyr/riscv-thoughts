#!/usr/bin/env python3
"""Multiplex batch streams from parallel gen_batches.py workers.

Spawns N gen_batches.py processes with distinct seeds and interleaves
their binary batch output into a single stream on stdout.

Usage:
    python scripts/mux_batches.py --n-batches 10000 --n-workers 8 | python scripts/train_compressor.py
    python scripts/mux_batches.py --n-batches 100000 --n-workers 24 > corpus.bin
"""

import argparse
import os
import subprocess
import sys
import threading
import queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen import (
    read_batch_bytes, read_stream_header, write_stream_header,
)


def _reader(pipe, q):
    """Read batches from a worker's stdout and put them on the queue."""
    try:
        read_stream_header(pipe)
        while True:
            data = read_batch_bytes(pipe)
            if data is None:
                break
            q.put(data)
    except (EOFError, ValueError) as e:
        q.put(e)
    q.put(None)  # sentinel


def main():
    p = argparse.ArgumentParser(description='Mux parallel batch generators.')
    p.add_argument('--n-batches', type=int, required=True,
                   help='Total number of batches to produce')
    p.add_argument('--n-workers', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--n-inputs', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('-v', '--verbose', action='count', default=0,
                   help='-v for mux status, -vv for worker status too')
    args = p.parse_args()

    out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb')
    sys.stdout = sys.stderr

    # Divide batches among workers.
    base, extra = divmod(args.n_batches, args.n_workers)
    counts = [base + (1 if i < extra else 0) for i in range(args.n_workers)]

    script = str(Path(__file__).resolve().parent / 'gen_batches.py')
    q = queue.Queue(maxsize=args.n_workers * 2)

    worker_args = [sys.executable, script,
                   '--batch-size', str(args.batch_size),
                   '--n-inputs', str(args.n_inputs)]
    if args.verbose >= 2:
        worker_args.append('-v')

    workers = []
    for i, count in enumerate(counts):
        if count == 0:
            continue
        proc = subprocess.Popen(
            worker_args + ['--n-batches', str(count),
                           '--seed', str(args.seed + i)],
            stdout=subprocess.PIPE,
            stderr=sys.stderr if args.verbose >= 2 else subprocess.DEVNULL,
        )
        t = threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True)
        t.start()
        workers.append(proc)

    write_stream_header(out)

    written = 0
    done = 0
    while done < len(workers):
        data = q.get()
        if data is None:
            done += 1
            continue
        if isinstance(data, Exception):
            print(f'Worker error: {data}', file=sys.stderr)
            done += 1
            continue
        out.write(data)
        written += 1
        if args.verbose >= 1 and written % 100 == 0:
            print(f'{written}/{args.n_batches} batches')

    out.close()
    failed = []
    for i, proc in enumerate(workers):
        rc = proc.wait()
        if rc != 0:
            failed.append(i)
    if failed:
        print(f'WARNING: workers {failed} exited with errors', file=sys.stderr)
    if args.verbose >= 1:
        print(f'Done: {written} batches, '
              f'{written * args.batch_size} instructions')


if __name__ == '__main__':
    main()
