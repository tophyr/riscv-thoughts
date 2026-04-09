#!/usr/bin/env python3
"""Multiplex multiple binary batch streams into one.

Inputs can be files, pipes, process substitution, or spawned
gen_batches.py workers. All go through the same muxing plumbing.

Muxing modes:
  fifo:        output whichever batch is ready first (default when
               no weights given). Ignores weights.
  round-robin: cycle through inputs in order. Ignores weights.
  weighted:    pick proportional to weight from available inputs
               (default when any input has a weight prefix).
               Weights affect interleaving, not throughput.

Inputs can have weights via prefix syntax: "128:/dev/fd/63" or
"0.5:corpus.bin". Default weight is 1.0. Spawned workers each
get --gen-weight (default 1.0).

Usage:
    # Spawn 16 generators, 1000 batches each:
    mux_batches.py --gen 16 --n-batches 1000 > corpus.bin

    # Mix local generators with weighted network stream:
    mux_batches.py --gen 16 128:<(nc threadripper 9000 -d | unlz4) > corpus.bin

    # File inputs with weights:
    mux_batches.py 1:local.bin 8:remote.bin > combined.bin

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


def _reader_per_q(f, per_q, ready, idx):
    """Read batches from a stream into a per-input queue."""
    try:
        read_stream_header(f)
        while True:
            data = read_batch_bytes(f)
            if data is None:
                break
            per_q.put(data)
            ready.release()
    except (EOFError, ValueError) as e:
        per_q.put(e)
        ready.release()
    per_q.put(None)  # sentinel
    ready.release()


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


def mux_fifo(files, weights, out, verbose=False):
    """Output whichever batch is ready first.

    Weights are ignored — all inputs compete equally for queue space.
    """
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


def mux_round_robin(files, weights, out, verbose=False):
    """Cycle through inputs in order, one batch per input per cycle.

    Weights are ignored — each input gets one batch per cycle regardless
    of weight. Blocks on the slowest input each cycle.
    """
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


def mux_weighted(files, weights, out, verbose=False, seed=42):
    """Pick from available inputs proportional to weight."""
    ready = threading.Semaphore(0)
    queues = []
    for i, f in enumerate(files):
        per_q = queue.Queue(maxsize=2)
        t = threading.Thread(target=_reader_per_q,
                             args=(f, per_q, ready, i), daemon=True)
        t.start()
        queues.append(per_q)

    written = [0]
    active = set(range(len(files)))
    rng = random.Random(seed)

    while active:
        ready.acquire()

        # Find non-empty queues among active inputs.
        available = [i for i in active if not queues[i].empty()]
        if not available:
            continue

        w = [weights[i] for i in available]
        pick = rng.choices(available, weights=w, k=1)[0]
        data = queues[pick].get_nowait()

        if data is None:
            active.discard(pick)
            continue
        if isinstance(data, Exception):
            print(f'Input {pick} error: {data}', file=sys.stderr)
            active.discard(pick)
            continue
        if not _mux_write(out, written, data, verbose):
            break

    return written[0]


def _parse_weighted_input(s):
    """Parse 'weight:path' or plain 'path'. Returns (path, weight)."""
    if ':' in s:
        parts = s.split(':', 1)
        try:
            weight = float(parts[0])
            return parts[1], weight
        except ValueError:
            pass
    return s, 1.0


def main():
    p = argparse.ArgumentParser(description='Mux binary batch streams.')
    p.add_argument('inputs', nargs='*',
                   help='Input files/pipes, optionally weight-prefixed '
                        '(e.g. "128:file.bin")')
    p.add_argument('--mode', choices=['fifo', 'round-robin', 'weighted'],
                   default=None,
                   help='Muxing mode (default: weighted if any input has '
                        'a weight prefix, fifo otherwise). fifo and '
                        'round-robin ignore weights.')
    p.add_argument('--shuffle-seed', type=int, default=42,
                   help='Random seed for weighted mode (default: 42)')
    p.add_argument('-v', '--verbose', action='count', default=0,
                   help='-v for mux status, -vv for worker status too')

    g = p.add_argument_group('generator spawning')
    g.add_argument('--gen', type=int, default=0, metavar='N',
                   help='Spawn N gen_batches.py workers')
    g.add_argument('--gen-weight', type=float, default=1.0,
                   help='Weight per spawned worker (default: 1.0)')
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
    weights = []
    procs = []
    file_handles = []

    # Open file/pipe inputs with optional weights.
    has_explicit_weight = False
    for raw in args.inputs:
        path, weight = _parse_weighted_input(raw)
        if weight != 1.0:
            has_explicit_weight = True
        f = open(path, 'rb')
        files.append(f)
        weights.append(weight)
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
            weights.append(args.gen_weight)

    # Auto-select mode if not specified.
    mode = args.mode
    if mode is None:
        mode = 'weighted' if has_explicit_weight else 'fifo'

    if args.verbose >= 1:
        n_file = len(file_handles)
        n_gen = len(procs)
        parts = []
        if n_file:
            w_strs = [f'{w:.3g}' for w in weights[:n_file]]
            parts.append(f'{n_file} file inputs (weights: {", ".join(w_strs)})')
        if n_gen:
            parts.append(f'{n_gen} spawned workers (weight: {args.gen_weight})')
        print(f'Muxing {" + ".join(parts)}, mode={mode}',
              file=sys.stderr)

    write_stream_header(out)

    if mode == 'weighted':
        written = mux_weighted(files, weights, out,
                               verbose=(args.verbose >= 1),
                               seed=args.shuffle_seed)
    else:
        mux_fn = {'fifo': mux_fifo,
                   'round-robin': mux_round_robin}[mode]
        written = mux_fn(files, weights, out,
                         verbose=(args.verbose >= 1))

    out.close()
    for f in file_handles:
        f.close()

    for proc in procs:
        proc.terminate()
    for proc in procs:
        proc.wait()
    if args.verbose >= 1:
        print(f'Done: {written} batches', file=sys.stderr)


if __name__ == '__main__':
    main()
