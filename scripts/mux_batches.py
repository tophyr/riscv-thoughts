#!/usr/bin/env python3
"""Multiplex multiple binary batch streams into one.

Inputs can be files, pipes, process substitution, or spawned
gen_batches.py workers. All go through the same muxing plumbing.

Muxing modes:
  fifo:        output whichever batch is ready first (default)
  round-robin: cycle through inputs in order
  shuffle:     pick a random input each time

Usage:
    # Spawn 16 generators, 1000 batches each:
    mux_batches.py --gen 16 --n-batches 1000 > corpus.bin

    # Mix spawned generators with a network stream:
    mux_batches.py --gen 8 --n-batches 500 <(nc threadripper 9000) > corpus.bin

    # File inputs only:
    mux_batches.py input1.bin input2.bin > combined.bin

    # Round-robin mode:
    mux_batches.py --mode round-robin --gen 4 --n-batches 100 > corpus.bin
"""

import argparse
import random
import subprocess
import sys
import threading
import queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen import (
    read_batch_bytes, read_stream_header, write_stream_header,
)
from scripts._batch_util import binary_stdout


def _reader(f, q, idx):
    """Read batches from a stream and put (idx, bytes) on the queue."""
    try:
        read_stream_header(f)
        while True:
            data = read_batch_bytes(f)
            if data is None:
                break
            q.put((idx, data))
    except (EOFError, ValueError) as e:
        q.put((idx, e))
    q.put((idx, None))  # sentinel


def _mux_write(out, written_ref, data, verbose):
    """Write one batch and update counter. Returns False on BrokenPipe."""
    try:
        out.write(data)
    except BrokenPipeError:
        return False
    written_ref[0] += 1
    if verbose and written_ref[0] % 100 == 0:
        print(f'{written_ref[0]} batches', file=sys.stderr)
    return True


def mux_fifo(files, out, verbose=False):
    """Output whichever batch is ready first."""
    q = queue.Queue(maxsize=len(files) * 2)

    for i, f in enumerate(files):
        t = threading.Thread(target=_reader, args=(f, q, i), daemon=True)
        t.start()

    written = [0]
    done = 0
    while done < len(files):
        idx, data = q.get()
        if data is None:
            done += 1
            continue
        if isinstance(data, Exception):
            print(f'Input {idx} error: {data}', file=sys.stderr)
            done += 1
            continue
        if not _mux_write(out, written, data, verbose):
            break

    return written[0]


def mux_round_robin(files, out, verbose=False):
    """Cycle through inputs in order, one batch per input per cycle."""
    # One queue per input so we can pull from a specific input.
    queues = []
    for i, f in enumerate(files):
        q = queue.Queue(maxsize=2)
        t = threading.Thread(target=_reader, args=(f, q, i), daemon=True)
        t.start()
        queues.append(q)

    written = [0]
    active = list(range(len(files)))

    while active:
        next_active = []
        for i in active:
            idx, data = queues[i].get()
            if data is None:
                continue
            if isinstance(data, Exception):
                print(f'Input {i} error: {data}', file=sys.stderr)
                continue
            if not _mux_write(out, written, data, verbose):
                return written[0]
            next_active.append(i)
        active = next_active

    return written[0]


def mux_shuffle(files, out, verbose=False, seed=42):
    """Pick a random input each time."""
    # One queue per input so we can pull from a specific input.
    queues = []
    for i, f in enumerate(files):
        q = queue.Queue(maxsize=2)
        t = threading.Thread(target=_reader, args=(f, q, i), daemon=True)
        t.start()
        queues.append(q)

    written = [0]
    active = list(range(len(files)))
    rng = random.Random(seed)

    while active:
        pick = rng.choice(active)
        idx, data = queues[pick].get()
        if data is None:
            active.remove(pick)
            continue
        if isinstance(data, Exception):
            print(f'Input {pick} error: {data}', file=sys.stderr)
            active.remove(pick)
            continue
        if not _mux_write(out, written, data, verbose):
            break

    return written[0]


def main():
    p = argparse.ArgumentParser(description='Mux binary batch streams.')
    p.add_argument('inputs', nargs='*', help='Input files/pipes to mux')
    p.add_argument('--mode', choices=['fifo', 'round-robin', 'shuffle'],
                   default='fifo')
    p.add_argument('--shuffle-seed', type=int, default=42,
                   help='Random seed for shuffle mode (default: 42)')
    p.add_argument('-v', '--verbose', action='count', default=0,
                   help='-v for mux status, -vv for worker status too')

    g = p.add_argument_group('generator spawning')
    g.add_argument('--gen', type=int, default=0, metavar='N',
                   help='Spawn N gen_batches.py workers')
    g.add_argument('--n-batches', type=int, default=1000,
                   help='Batches per spawned worker (default: 1000)')
    g.add_argument('--batch-size', type=int, default=4096)
    g.add_argument('--n-inputs', type=int, default=32)
    g.add_argument('--seed', type=int, default=42)

    args = p.parse_args()

    if not args.inputs and args.gen == 0:
        p.error('Provide input files and/or --gen N')

    out = binary_stdout()

    files = []
    procs = []
    file_handles = []

    # Open file/pipe inputs.
    for path in args.inputs:
        f = open(path, 'rb')
        files.append(f)
        file_handles.append(f)

    # Spawn generator workers.
    if args.gen > 0:
        script = str(Path(__file__).resolve().parent / 'gen_batches.py')
        base_cmd = [sys.executable, script,
                    '--batch-size', str(args.batch_size),
                    '--n-inputs', str(args.n_inputs)]
        if args.verbose >= 2:
            base_cmd.append('-v')

        for i in range(args.gen):
            cmd = base_cmd + ['--n-batches', str(args.n_batches),
                              '--seed', str(args.seed + i)]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=sys.stderr if args.verbose >= 2
                else subprocess.DEVNULL)
            procs.append(proc)
            files.append(proc.stdout)

    if args.verbose >= 1:
        n_file = len(file_handles)
        n_gen = len(procs)
        parts = []
        if n_file:
            parts.append(f'{n_file} file inputs')
        if n_gen:
            parts.append(f'{n_gen} spawned workers')
        print(f'Muxing {" + ".join(parts)}, mode={args.mode}',
              file=sys.stderr)

    write_stream_header(out)

    if args.mode == 'shuffle':
        written = mux_shuffle(files, out, verbose=(args.verbose >= 1),
                              seed=args.shuffle_seed)
    else:
        mux_fn = {'fifo': mux_fifo, 'round-robin': mux_round_robin}[args.mode]
        written = mux_fn(files, out, verbose=(args.verbose >= 1))

    out.close()
    for f in file_handles:
        f.close()

    failed = []
    for i, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failed.append(i)
    if failed:
        print(f'WARNING: workers {failed} exited with errors',
              file=sys.stderr)
    if args.verbose >= 1:
        print(f'Done: {written} batches', file=sys.stderr)


if __name__ == '__main__':
    main()
