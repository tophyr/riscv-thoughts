#!/usr/bin/env python3
"""Repeat batches from a file or stdin in a loop.

Two source modes, unified by a pool abstraction:

  file mode  (positional `file` argument given): read the whole file
             into the pool once. The pool size equals the file's
             batch count. Cycle through the pool indefinitely or for
             a fixed `--epochs` count.

  stdin mode (no `file` argument): read from stdin into a sliding
             pool of `--pool-size` batches. Once the pool fills, start
             emitting; subsequent arrivals evict the oldest entry to
             keep size constant. Useful when an upstream producer is
             slower than the consumer — the consumer can keep
             cycling on the buffered pool while the producer refreshes
             it.

Either mode supports cycle (in-order) or shuffle emission.

Pipeline shapes:
  file   :  batch_repeat corpus.rvc --epochs 5 | train
  stdin  :  gen+chunk+precompute | batch_repeat --pool-size 300 | train

Behavior summary (stdin mode):
  Cold start (pool not yet full)     consumer waits on condition var
  Pool fills                         consumer starts emitting (one-shot
                                       "pool full" event under -v)
  Subsequent arrivals                evict oldest, keep pool at size
  Producer EOF before pool fills     emit whatever made it in
  Producer EOF after pool fills      pool stays static; cycles forever
  Consumer closes pipe               clean exit on BrokenPipeError
"""

import argparse
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._batch_util import binary_stdout, detect_format


class _Stats:
    def __init__(self, pool_size, source_label):
        self.pool_size = pool_size  # mutable: in file mode, set after load
        self.source_label = source_label
        self.received = 0
        self.eof = False
        self.fill_announced = False


def _file_reader(path, fmt, pool, cv, stats, verbose):
    """Read all batches from a file into the pool, then signal EOF.
    Pool size is updated to the actual count once the file is fully read.
    """
    try:
        with open(path, 'rb') as f:
            file_fmt = detect_format(f)
            if file_fmt.name != fmt.name:
                # Shouldn't happen — we already detected from this path.
                raise RuntimeError(
                    f'file format {file_fmt.name} != expected {fmt.name}')
            while True:
                data = fmt.read_bytes(f)
                if data is None:
                    break
                with cv:
                    pool.append(data)
                    stats.received += 1
                    cv.notify()
    finally:
        with cv:
            stats.eof = True
            stats.pool_size = len(pool)
            # fill_announced gates the epoch-stop condition; in file
            # mode "fill" = file fully loaded. Independent of --verbose.
            if stats.pool_size > 0 and not stats.fill_announced:
                stats.fill_announced = True
                if verbose:
                    print(f'[batch_repeat] file loaded: {stats.pool_size} '
                          f'batches', file=sys.stderr)
            cv.notify_all()


def _stdin_reader(stdin_buf, fmt, pool, cv, stats, verbose):
    """Read batches from stdin into a sliding pool. Sets fill_announced
    the first time the pool reaches its target size — gates the
    epoch-stop condition. Verbose only gates the printed announcement."""
    try:
        while True:
            data = fmt.read_bytes(stdin_buf)
            if data is None:
                break
            with cv:
                pool.append(data)
                stats.received += 1
                size = len(pool)
                if not stats.fill_announced and size >= stats.pool_size:
                    stats.fill_announced = True
                    if verbose:
                        print(f'[batch_repeat] pool filled ({size}); '
                              f'writer enabled, evicting oldest hereafter',
                              file=sys.stderr)
                cv.notify()
    finally:
        with cv:
            stats.eof = True
            cv.notify_all()


def _logger(pool, cv, stats, interval, stop_event):
    while not stop_event.wait(interval):
        with cv:
            size = len(pool)
            received = stats.received
            eof = stats.eof
        print(f'[batch_repeat] pool={size}/{stats.pool_size} '
              f'received={received} eof={eof} src={stats.source_label}',
              file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('file', nargs='?', default=None,
                   help='Input batch file. If omitted, read from stdin.')
    p.add_argument('--pool-size', type=int, default=300,
                   help='Pool capacity (stdin mode only). Writer waits '
                        'until pool fills before emitting; subsequent '
                        'arrivals evict oldest. Ignored in file mode '
                        '(pool size = file batch count).')
    p.add_argument('--epochs', type=int, default=None,
                   help='Stop after pool_size * EPOCHS emissions. '
                        'Default: emit indefinitely.')
    p.add_argument('--final-passes', type=int, default=None,
                   help='After stdin EOF, emit FINAL_PASSES additional '
                        'full passes through the pool, then close. '
                        'Default: emit indefinitely after EOF.')
    p.add_argument('--forever', action='store_true',
                   help='Explicit infinite emission. Default behavior; '
                        'kept for backward compatibility.')
    p.add_argument('--mode', choices=['cycle', 'shuffle'], default='cycle',
                   help='cycle: round-robin through pool; shuffle: '
                        'random pick each emit (uniform over pool).')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for shuffle mode.')
    p.add_argument('--log-every-sec', type=float, default=0.0,
                   help='Emit pool-size status to stderr every N seconds.')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    if args.epochs is not None and args.forever:
        sys.exit('--epochs and --forever are mutually exclusive')
    if args.epochs is not None and args.final_passes is not None:
        sys.exit('--epochs and --final-passes are mutually exclusive')
    if args.epochs is not None and args.epochs < 1:
        sys.exit('--epochs must be >= 1')
    if args.final_passes is not None and args.final_passes < 1:
        sys.exit('--final-passes must be >= 1')
    if args.pool_size < 1:
        sys.exit('--pool-size must be >= 1')

    # Detect format. In file mode we peek at the file; in stdin mode
    # we consume the stream header from stdin.
    if args.file is not None:
        with open(args.file, 'rb') as f:
            fmt = detect_format(f)
        # Pool size for file mode is determined by file content.
        pool = deque()
        source_label = args.file
    else:
        fmt = detect_format(sys.stdin.buffer)
        pool = deque(maxlen=args.pool_size)
        source_label = 'stdin'

    out = binary_stdout()
    fmt.write_header(out)

    stats = _Stats(args.pool_size, source_label)
    cv = threading.Condition()
    rng = np.random.default_rng(args.seed)

    if args.file is not None:
        reader_thread = threading.Thread(
            target=_file_reader,
            args=(args.file, fmt, pool, cv, stats, args.verbose),
            daemon=True)
    else:
        reader_thread = threading.Thread(
            target=_stdin_reader,
            args=(sys.stdin.buffer, fmt, pool, cv, stats, args.verbose),
            daemon=True)
    reader_thread.start()

    logger_stop = threading.Event()
    if args.log_every_sec > 0:
        logger_thread = threading.Thread(
            target=_logger,
            args=(pool, cv, stats, args.log_every_sec, logger_stop),
            daemon=True)
        logger_thread.start()

    cursor = 0
    n_emitted = 0
    n_emitted_post_eof = 0
    final_pool_size = None
    try:
        while True:
            with cv:
                # Wait until pool has reached its target size, or the
                # producer hits EOF (which finalizes pool size).
                while len(pool) < stats.pool_size:
                    if stats.eof:
                        break
                    cv.wait()
                if not pool:
                    return

                # In file mode, reader_thread sets stats.pool_size to
                # the actual file batch count after EOF; cap emission
                # count accordingly.
                pool_size = len(pool)
                if args.mode == 'cycle':
                    idx = cursor % pool_size
                    cursor += 1
                else:
                    idx = int(rng.integers(0, pool_size))
                data = pool[idx]
                eof_now = stats.eof

            try:
                out.write(data)
                n_emitted += 1
                if eof_now:
                    if final_pool_size is None:
                        final_pool_size = pool_size
                    n_emitted_post_eof += 1
            except BrokenPipeError:
                return

            # In file mode with --epochs, stop after pool_size * epochs.
            # In stdin mode, --epochs is meaningful only after the first
            # full fill; we treat it the same way: stop after
            # pool_size * epochs emissions.
            if args.epochs is not None and stats.fill_announced:
                if n_emitted >= stats.pool_size * args.epochs:
                    return

            if (args.final_passes is not None
                    and final_pool_size is not None
                    and n_emitted_post_eof >= final_pool_size * args.final_passes):
                return
    finally:
        logger_stop.set()
        if args.verbose:
            print(f'[batch_repeat] received={stats.received} '
                  f'emitted={n_emitted} final_pool={len(pool)} '
                  f'eof={stats.eof}', file=sys.stderr)


if __name__ == '__main__':
    main()
