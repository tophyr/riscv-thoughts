#!/usr/bin/env python3
"""Multiplex multiple RVT batch streams into one.

Spawns gen_batches.py workers and/or reads from files/pipes. All
inputs must be the same format (currently only RVT exists). Worker
configuration is NOT redeclared here: everything after `--` is forwarded
verbatim to each spawned gen_batches worker (mux only appends a per-worker
--seed so the workers diverge).

Usage:
    # Spawn N parallel gen_batches workers (worker flags after `--`):
    mux_batches.py --gen-count 16 -- \\
        --rule branch+cap=8 --twins 3 -n 1000 > corpus.rvt

    # File inputs (auto-detected):
    mux_batches.py 1:local.rvt 8:remote.rvt > combined.rvt

    # Mix spawned workers + a file/pipe input:
    mux_batches.py --gen-count 8 8:remote.rvt -- --rule cap=8 -n 500
"""

import argparse
import random
import subprocess
import sys
import threading
import queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._streamfmt import binary_stdout, peek_format, RVT


def _reader(f, q, idx, fmt):
    try:
        while True:
            data = fmt.read_bytes(f)
            if data is None:
                break
            q.put((idx, data))
    except (EOFError, ValueError) as e:
        q.put((idx, e))
    q.put((idx, None))


def _reader_per_q(f, per_q, idx, fmt):
    try:
        while True:
            data = fmt.read_bytes(f)
            if data is None:
                break
            per_q.put(data)
    except (EOFError, ValueError) as e:
        per_q.put(e)
    per_q.put(None)


def _mux_write(out, written_ref, data, verbose):
    try:
        out.write(data)
    except BrokenPipeError:
        return False
    written_ref[0] += 1
    if verbose and written_ref[0] % 100 == 0:
        print(f'{written_ref[0]} batches', file=sys.stderr)
    return True


def mux_fifo(files, weights, out, fmt, verbose=False):
    q = queue.Queue(maxsize=len(files) * 2)
    for i, f in enumerate(files):
        t = threading.Thread(target=_reader, args=(f, q, i, fmt),
                             daemon=True)
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


def mux_round_robin(files, weights, out, fmt, verbose=False):
    queues = []
    for i, f in enumerate(files):
        q = queue.Queue(maxsize=2)
        t = threading.Thread(target=_reader, args=(f, q, i, fmt),
                             daemon=True)
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


def mux_weighted(files, weights, out, fmt, verbose=False, seed=42):
    queues = []
    for i, f in enumerate(files):
        per_q = queue.Queue(maxsize=2)
        t = threading.Thread(target=_reader_per_q,
                             args=(f, per_q, i, fmt), daemon=True)
        t.start()
        queues.append(per_q)
    written = [0]
    active = list(range(len(files)))
    rng = random.Random(seed)
    while active:
        w = [weights[i] for i in active]
        pick = rng.choices(active, weights=w, k=1)[0]
        data = queues[pick].get()
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


def _parse_weighted_input(s):
    if ':' in s:
        parts = s.split(':', 1)
        try:
            weight = float(parts[0])
            return parts[1], weight
        except ValueError:
            pass
    return s, 1.0


def main():
    # Everything after `--` is forwarded verbatim to each spawned
    # gen_batches worker. Split it off before argparse, otherwise the
    # `inputs` positional (nargs='*') would swallow the forwarded flags.
    argv = sys.argv[1:]
    if '--' in argv:
        sep = argv.index('--')
        mux_argv, gen_extra = argv[:sep], argv[sep + 1:]
    else:
        mux_argv, gen_extra = argv, []

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='*',
                   help='Input files, optionally weight-prefixed (W:path)')
    p.add_argument('--mode', choices=['fifo', 'round-robin', 'weighted'],
                   default=None,
                   help='How to interleave inputs: fifo (as-ready), '
                        'round-robin (one batch each), or weighted. To '
                        'concatenate files in order, just `cat` them.')
    p.add_argument('--shuffle-seed', type=int, default=42)
    p.add_argument('-v', '--verbose', action='count', default=0)

    g = p.add_argument_group(
        'gen_batches spawning',
        'Configure spawned workers with their own flags after `--` '
        '(e.g. `--gen-count 16 -- --rule branch+cap=8 -n 1000`).')
    g.add_argument('--gen-count', type=int, default=0, metavar='N',
                   help='Spawn N gen_batches workers')
    g.add_argument('--gen-weight', type=float, default=1.0,
                   help='Mux weight for each spawned worker stream')
    g.add_argument('--seed', type=int, default=42,
                   help='Base RNG seed; worker i gets --seed (base + i) so '
                        'workers produce distinct streams')

    args = p.parse_args(mux_argv)

    if not args.inputs and args.gen_count == 0:
        p.error('Provide input files and/or --gen-count N')
    if gen_extra and args.gen_count == 0:
        p.error('Arguments after `--` configure spawned workers and require '
                '--gen-count N')

    fmt = RVT  # Single format; mux preserves it.

    out = binary_stdout()
    files = []
    weights = []
    procs = []
    file_handles = []

    has_explicit_weight = False
    for raw in args.inputs:
        path, weight = _parse_weighted_input(raw)
        if weight != 1.0:
            has_explicit_weight = True
        f = open(path, 'rb')
        file_fmt = peek_format(f)
        if file_fmt.name != fmt.name:
            print(f'ERROR: {path}: format {file_fmt.name} != {fmt.name}',
                  file=sys.stderr)
            sys.exit(1)
        files.append(f)
        weights.append(weight)
        file_handles.append(f)

    if args.gen_count > 0:
        script = str(Path(__file__).resolve().parent / 'gen_batches.py')
        base_cmd = [sys.executable, script] + gen_extra

        for i in range(args.gen_count):
            # --seed appended last so the per-worker offset overrides any
            # --seed the caller passed after `--`.
            cmd = base_cmd + ['--seed', str(args.seed + i)]
            # Inherit stderr unconditionally — silent workers hide
            # crashes (Python tracebacks land here), and that's exactly
            # what you want to see if a worker dies mid-stream.
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=sys.stderr)
            procs.append(proc)
            files.append(proc.stdout)
            weights.append(args.gen_weight)

    if fmt is None:
        print('ERROR: cannot determine format', file=sys.stderr)
        sys.exit(1)

    mode = args.mode
    if mode is None:
        mode = 'weighted' if has_explicit_weight else 'fifo'

    if args.verbose >= 1:
        n_file = len(file_handles)
        n_gen = len(procs)
        parts = []
        if n_file:
            parts.append(f'{n_file} file inputs')
        if n_gen:
            parts.append(f'{n_gen} spawned gen_batches workers')
        print(f'Muxing {" + ".join(parts)}, mode={mode}, '
              f'format={fmt.name}', file=sys.stderr)

    if mode == 'weighted':
        written = mux_weighted(files, weights, out, fmt,
                               verbose=(args.verbose >= 1),
                               seed=args.shuffle_seed)
    else:
        mux_fn = {'fifo': mux_fifo, 'round-robin': mux_round_robin}[mode]
        written = mux_fn(files, weights, out, fmt,
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
