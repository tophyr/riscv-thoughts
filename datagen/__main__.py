#!/usr/bin/env python3
"""`python -m datagen` — generate RVT training batches in parallel, optionally
muxing external streams.

This package IS the generator tool. By DEFAULT it spawns one generation worker
per CPU core and interleaves their output into a single RVT stream on stdout. It
can ALSO mux in external RVT streams (a disk corpus, a netcat link) given as
positional inputs — and a piped or redirected stdin is auto-muxed too
(`cat corpus.rvt | python -m datagen …`; a tty or /dev/null is skipped). By
default generation runs forever — the stream ends when a downstream consumer
closes the pipe; `--count N` bounds it to N batches and reaps the pool. `-v`
reports throughput (batch/s, MB/s).

    python -m datagen --rule single --count 250000 | train_encoder.py ...
    python -m datagen 8:<(nc odin 6464) --rule branch+cap=8 | train_t2_encoder.py ...
    cat corpus.rvt | python -m datagen --rule single --count 250000 | train_encoder.py ...

There is intentionally NO single-worker mode and NO --n-batches (per-worker
counts). A lone single-core generator cannot keep a GPU fed (generation is
CPU-bound Python, one process pins one core), so it is the project's standard
way to silently starve a training run. Workers here are an INTERNAL function
with no command line of their own — you cannot run one by hand, by design.
`--gen-count 0` disables generation entirely (pure mux of the external inputs).

Worker lifecycle is OS-enforced: each worker arms PR_SET_PDEATHSIG so the
kernel SIGKILLs it the instant this parent dies (even on `kill -9`), and all
workers share one process group reaped on exit. Orphan workers cannot happen.

A malformed batch from any input (a non-RVT stream, a truncated transfer, a
schema/dtype mismatch) is FATAL: datagen reaps the pool and exits non-zero
rather than silently dropping the source — fail early, fail loud. A clean
end-of-stream is not malformed; it just drains the source.
"""

import argparse
import ctypes
import os
import queue
import random
import signal
import stat
import sys
import threading
import time
import multiprocessing

import numpy as np

from . import (
    RVT_FORMAT, generate_chunks, collect_into_batches,
    make_anchor_states,
    DEFAULT_DISTRIBUTION, build_opcode_table, validate_distribution,
    load_distribution,
    single, until_branch, until_transformation, length_cap, either,
    DEFAULT_TYPE_WEIGHTS, build_type_table, generate_invalid,
)


def binary_stdout():
    """Return a binary fd for stdout, redirecting Python stdout to stderr.

    Force blocking mode on the fd: GNU `tee` (and other stream-pipe consumers)
    flip pipes non-blocking on their side as a performance optimization, and the
    flag is shared across the kernel's open file description with the writing
    peer. Without this guard, Python's BufferedWriter.write() raises
    BlockingIOError mid-batch when the downstream consumer is briefly slow,
    leaving the reader with a truncated body.
    """
    fd = os.dup(sys.stdout.fileno())
    os.set_blocking(fd, True)
    out = os.fdopen(fd, 'wb')
    sys.stdout = sys.stderr
    return out


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

# --rule is a '+'-joined set of components, composed via either() so chunk
# termination fires when ANY matches. 'single' is a standalone shorthand.
# Every rule must stay length-bounded (the set must include a cap=N).
_TERMINATOR_COMPONENTS = {
    'branch': until_branch,
    'transform': until_transformation,
}


def _parse_rule(spec):
    """Parse a '+'-joined --rule into a TerminationRule. 'single' is a
    standalone shorthand. Unknown components and unbounded rules (no cap)
    raise here, eagerly — the worker reads rule.max_len to size batch arrays,
    so a bare 'branch'/'transform' must be caught here or it surfaces
    downstream as an opaque None-arithmetic TypeError."""
    s = spec.strip()
    if s == 'single':
        return single()
    rules = []
    for part in s.split('+'):
        part = part.strip()
        if part.startswith('cap='):
            rules.append(length_cap(int(part[len('cap='):])))
        elif part in _TERMINATOR_COMPONENTS:
            rules.append(_TERMINATOR_COMPONENTS[part]())
        else:
            raise ValueError(f'Unknown rule component {part!r} in {spec!r}')
    rule = rules[0] if len(rules) == 1 else either(*rules)
    if rule.max_len is None:
        raise ValueError(
            f'Rule {spec!r} is unbounded (no length cap); it can build an '
            f'infinite chunk from the instruction stream. Add a cap, '
            f'e.g. {spec.strip()}+cap=8.')
    return rule


# ---------------------------------------------------------------------------
# Generation worker — INTERNAL ONLY. Spawned via multiprocessing.Process under
# a fork context; never exposed as a CLI. Writes framed RVT to its pipe fd and
# loops until the parent goes away or the read end closes.
# ---------------------------------------------------------------------------

_PR_SET_PDEATHSIG = 1  # <linux/prctl.h>


def _arm_pdeathsig():
    """Ask the kernel to SIGKILL this process when its parent dies. The single
    load-bearing fix for the orphan-worker hazard: it fires even when the
    parent is `kill -9`'d, which the old terminate()-on-clean-exit path could
    not cover."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass  # non-Linux / no prctl: fall back to parent-side reaping only


def _build_batches_iter(cfg, seed):
    """Build the (chunks → batches) generator for one worker. Every worker
    rebuilds its own rng / opcode table / anchor states so forked workers
    never share rng state (a shared, pre-fork rng would emit identical
    streams)."""
    rule = _parse_rule(cfg.rule)
    rng = np.random.default_rng(seed)
    anchor_states = make_anchor_states(cfg.n_states, cfg.anchor_seed)

    if cfg.config:
        dist = load_distribution(cfg.config)
    else:
        dist = dict(DEFAULT_DISTRIBUTION)
        validate_distribution(dist)
    opcode_table = build_opcode_table(dist)

    eq_cfg = dist.get('equivalences', {})
    eq_rate = (cfg.inject_equiv if cfg.inject_equiv is not None
               else eq_cfg.get('rate', 0.0))
    chain_cfg = dist.get('chain', {})
    chain_rate = (cfg.chain_rate if cfg.chain_rate is not None
                  else chain_cfg.get('rate', 0.0))
    inv_cfg = dist.get('invalidity', {})
    inv_rate = (cfg.inject_invalid if cfg.inject_invalid is not None
                else inv_cfg.get('rate', 0.0))
    inv_types = inv_cfg.get('types', DEFAULT_TYPE_WEIGHTS)

    invalid_provider = None
    if inv_rate > 0:
        type_table = build_type_table(inv_types)
        def invalid_provider():
            tokens, _ = generate_invalid(
                rng, opcode_table, cfg.max_invalid_window, type_table)
            return tokens

    # Optional corpus restriction: exclude write-to-x0 (rd drawn from 1..31).
    dest_regs = list(range(1, 32)) if dist.get('exclude_x0_dest') else None

    chunks_iter = generate_chunks(
        rule, rng, opcode_table=opcode_table, dest_regs=dest_regs,
        eq_rate=eq_rate,
        eq_max_per_class=eq_cfg.get('max_per_class', 8),
        eq_min_per_class=eq_cfg.get('min_per_class', 0),
        eq_boost=eq_cfg.get('boost', None),
        chain_rate=chain_rate,
    )
    return collect_into_batches(
        chunks_iter,
        batch_size=cfg.batch_size,
        twins=cfg.twins,
        anchor_states=anchor_states, rng=rng,
        invalid_rate=inv_rate, invalid_provider=invalid_provider,
        max_invalid_window=cfg.max_invalid_window,
        max_chunk_len=rule.max_len,
    )


def _worker_loop(write_fd, cfg, seed):
    """Worker entry point (target= of a forked Process). Generates batches
    forever and writes framed RVT to write_fd until the read end closes."""
    _arm_pdeathsig()
    # Race window: if the parent died between fork() and _arm_pdeathsig(), the
    # signal disposition was reset by the death — bail if we've been reparented.
    if os.getppid() == 1:
        os._exit(0)

    # fd hygiene: detach from inherited stdin/stdout so we never hold the
    # downstream data pipe open (that would delay its EOF). stderr stays for
    # logging. The worker writes ONLY to write_fd.
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    if devnull > 2:
        os.close(devnull)

    out = os.fdopen(write_fd, 'wb', buffering=1 << 16)
    pid = os.getpid()
    print(f'[pid={pid}] worker_start seed={seed} rule={cfg.rule}',
          file=sys.stderr, flush=True)
    written = 0
    try:
        for batch in _build_batches_iter(cfg, seed):
            RVT_FORMAT.write_batch(out, batch)
            out.flush()  # per-batch so the muxer sees whole batches promptly
            written += 1
            if cfg.worker_log_every and written % cfg.worker_log_every == 0:
                print(f'[pid={pid}] batches={written}',
                      file=sys.stderr, flush=True)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            out.close()
        except BrokenPipeError:
            pass
    print(f'[pid={pid}] worker_end written={written}',
          file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Mux core (reader threads → queue → single writer). The readers are
# source-agnostic — a file, an nc pipe, and a worker pipe all look like a
# BufferedReader of framed RVT.
# ---------------------------------------------------------------------------

def _reader(f, q, idx, fmt):
    try:
        while True:
            data = fmt.read_batch_bytes(f)
            if data is None:
                break
            q.put((idx, data))
    except (EOFError, ValueError) as e:
        q.put((idx, e))
    q.put((idx, None))


def _reader_per_q(f, per_q, idx, fmt):
    try:
        while True:
            data = fmt.read_batch_bytes(f)
            if data is None:
                break
            per_q.put(data)
    except (EOFError, ValueError) as e:
        per_q.put(e)
    per_q.put(None)


class _Writer:
    """The single output chokepoint. Every mux mode funnels batches through
    here, so the two producer-native features live in ONE place: the `--count`
    cap (stop after N batches) and, under `-v`, windowed throughput reporting
    (folds in the old standalone bench tool — under backpressure datagen's emit
    rate IS the delivered rate, so this answers "is generation the
    bottleneck?")."""

    def __init__(self, out, *, count=None, verbose=False, log_every_sec=5.0):
        self.out = out
        self.count = count
        self.verbose = verbose
        self.log_every_sec = log_every_sec
        self.written = 0
        self.nbytes = 0
        self._t0 = time.perf_counter()
        self._last_log = self._t0
        self._last_n = 0
        self._last_bytes = 0

    def write(self, data):
        """Write one batch. Returns False when the muxer should STOP — on a
        closed downstream pipe OR once `--count` batches have been emitted."""
        try:
            self.out.write(data)
        except BrokenPipeError:
            return False
        self.written += 1
        self.nbytes += len(data)
        if self.verbose:
            self._maybe_log()
        if self.count is not None and self.written >= self.count:
            return False
        return True

    def _maybe_log(self):
        now = time.perf_counter()
        if now - self._last_log < self.log_every_sec:
            return
        window = now - self._last_log
        elapsed = now - self._t0
        wn = self.written - self._last_n
        wb = self.nbytes - self._last_bytes
        print(f'[{elapsed:6.1f}s] total={self.written:7d} batches  '
              f'window={wn / window:6.1f} batch/s  '
              f'avg={self.written / elapsed:6.1f} batch/s  '
              f'mb/s={wb / 1e6 / window:6.1f} (window) '
              f'{self.nbytes / 1e6 / elapsed:6.1f} (avg)', file=sys.stderr)
        self._last_log = now
        self._last_n = self.written
        self._last_bytes = self.nbytes

    def final(self):
        elapsed = time.perf_counter() - self._t0
        if elapsed > 0:
            print(f'datagen: {self.written} batches in {elapsed:.1f}s = '
                  f'{self.written / elapsed:.2f} batch/s, '
                  f'{self.nbytes / 1e6 / elapsed:.1f} MB/s '
                  f'({self.nbytes / 1e6:.1f} MB total)', file=sys.stderr)
        else:
            print(f'datagen: {self.written} batches', file=sys.stderr)


class _InputParseError(Exception):
    """A source emitted bytes that are not a valid RVT batch. Fatal: a
    malformed batch means the stream is corrupt (a truncated transfer, a
    schema/dtype mismatch, a non-RVT input), so we fail the whole run loudly
    rather than silently drop the source and train on a partial/wrong corpus.
    A CLEAN end-of-stream (`read_bytes` → None) is NOT this — it just drains
    the source."""


def mux_fifo(files, weights, writer, fmt, labels):
    q = queue.Queue(maxsize=len(files) * 2)
    for i, f in enumerate(files):
        threading.Thread(target=_reader, args=(f, q, i, fmt),
                         daemon=True).start()
    done = 0
    while done < len(files):
        idx, data = q.get()
        if data is None:
            done += 1
            continue
        if isinstance(data, Exception):
            raise _InputParseError(f'{labels[idx]}: {data}')
        if not writer.write(data):
            break
    return writer.written


def mux_round_robin(files, weights, writer, fmt, labels):
    queues = []
    for i, f in enumerate(files):
        q = queue.Queue(maxsize=2)
        threading.Thread(target=_reader, args=(f, q, i, fmt),
                         daemon=True).start()
        queues.append(q)
    active = list(range(len(files)))
    while active:
        next_active = []
        for i in active:
            idx, data = queues[i].get()
            if data is None:
                continue
            if isinstance(data, Exception):
                raise _InputParseError(f'{labels[i]}: {data}')
            if not writer.write(data):
                return writer.written
            next_active.append(i)
        active = next_active
    return writer.written


def mux_weighted(files, weights, writer, fmt, labels, seed=42):
    queues = []
    for i, f in enumerate(files):
        per_q = queue.Queue(maxsize=2)
        threading.Thread(target=_reader_per_q, args=(f, per_q, i, fmt),
                         daemon=True).start()
        queues.append(per_q)
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
            raise _InputParseError(f'{labels[pick]}: {data}')
        if not writer.write(data):
            break
    return writer.written


def _parse_weighted_input(s):
    if ':' in s:
        parts = s.split(':', 1)
        try:
            weight = float(parts[0])
            return parts[1], weight
        except ValueError:
            pass
    return s, 1.0


# ---------------------------------------------------------------------------
# Worker pool — fork-spawned, pdeathsig-armed, one shared process group.
# ---------------------------------------------------------------------------

class WorkerPool:
    """Spawns the generation workers and owns their reaping. The pool lives in
    its OWN process group (distinct from the parent's) so a single killpg
    reaps every worker without touching the parent."""

    def __init__(self):
        self.procs = []
        self.read_ends = []   # parent-side BufferedReaders, one per worker
        self.pgid = None

    def spawn(self, gen_count, cfg, base_seed):
        """Fork `gen_count` workers. MUST be called before binary_stdout() so
        workers never inherit the downstream data fd. Returns the list of
        parent-side read-end file objects (framed RVT)."""
        ctx = multiprocessing.get_context('fork')
        for i in range(gen_count):
            r_fd, w_fd = os.pipe()
            p = ctx.Process(target=_worker_loop, args=(w_fd, cfg, base_seed + i),
                            daemon=True)
            p.start()
            os.close(w_fd)  # parent keeps only the read end
            self.procs.append(p)
            self.read_ends.append(os.fdopen(r_fd, 'rb', buffering=1 << 16))
            # Put every worker in one group led by worker 0. setpgid on a
            # forked (not exec'd) child is race-free from the parent side.
            if self.pgid is None:
                self.pgid = p.pid
            try:
                os.setpgid(p.pid, self.pgid)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        return self.read_ends

    def reap(self):
        """Tear the pool down. killpg the whole group, then SIGKILL/join each
        proc as a backstop. pdeathsig already covers parent-death; this is the
        clean-exit / BrokenPipe path."""
        if self.pgid is not None:
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            try:
                p.join(timeout=5)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.kill()
                except Exception:
                    pass
        for f in self.read_ends:
            try:
                f.close()
            except Exception:
                pass


class _GenCfg:
    """Plain container of generation params handed to each forked worker.
    (Picklable-irrelevant under fork, but kept simple and primitive anyway.)"""
    __slots__ = ('rule', 'batch_size', 'twins', 'n_states', 'anchor_seed',
                 'inject_invalid', 'inject_equiv', 'chain_rate',
                 'max_invalid_window',
                 'config', 'worker_log_every')

    def __init__(self, args):
        self.rule = args.rule
        self.batch_size = args.batch_size
        self.twins = args.twins
        self.n_states = args.n_states
        self.anchor_seed = args.anchor_seed
        self.inject_invalid = args.inject_invalid
        self.inject_equiv = args.inject_equiv
        self.chain_rate = args.chain_rate
        self.max_invalid_window = args.max_invalid_window
        self.config = args.config
        self.worker_log_every = args.worker_log_every


def main():
    p = argparse.ArgumentParser(
        description='Generate RVT batches across all CPU cores; optionally '
                    'mux in external RVT streams.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Generation flags — first-class (no `--` separator); one worker per core
    # generates with these knobs.
    g = p.add_argument_group('generation (per-worker)')
    g.add_argument('--rule', default='branch+cap=8',
                   help='Chunk-termination rule. Default: branch+cap=8.')
    g.add_argument('--batch-size', type=int, default=128,
                   help='Source rows per batch (twins extend the batch).')
    g.add_argument('--twins', type=int, default=3,
                   help='Relabeled equivalents per valid source chunk.')
    g.add_argument('--n-states', type=int, default=8,
                   help='Anchor states for value-prediction / aux precompute.')
    g.add_argument('--anchor-seed', type=int, default=0,
                   help='Anchor-states RNG seed (must match the trainer).')
    g.add_argument('--inject-invalid', type=float, default=None,
                   help='Invalid-window injection rate. Overrides config.')
    g.add_argument('--chain-rate', type=float, default=None,
                   help='Override the config\'s chain.rate. Fraction of chunks '
                        'rewritten so each instruction '
                        'reads its predecessor\'s result. A uniform register '
                        'draw composes only ~4.7%% of two-instruction chunks, '
                        'and the cells pairing composition with an immediate '
                        'carry most of T2\'s value-prediction error. Opcodes '
                        'are untouched, so the config\'s opcode distribution '
                        'is unchanged. 0 = draw registers freely.')
    g.add_argument('--inject-equiv', type=float, default=None,
                   help='Post-chunk MANIFEST equivalence rate. Overrides config.')
    g.add_argument('--max-invalid-window', type=int, default=32,
                   help='Token-length cap on injected invalid windows.')
    g.add_argument('--config', type=str, default=None,
                   help='JSON config: opcode mix + equivalences + invalidity.')

    # Pool + mux flags.
    pool = p.add_argument_group('pool / mux')
    pool.add_argument('--gen-count', type=int, default=None, metavar='N',
                      help='Generation workers to spawn. Default: all CPU cores '
                           f'({os.cpu_count()} here). 0 = pure mux of the '
                           'external inputs (no generation).')
    pool.add_argument('--gen-weight', type=float, default=1.0,
                      help='Mux weight for each spawned worker stream.')
    pool.add_argument('--seed', type=int, default=0,
                      help='Base RNG seed; worker i gets seed+i so workers '
                           'produce distinct streams.')
    pool.add_argument('inputs', nargs='*',
                      help='External RVT inputs, optionally weight-prefixed '
                           '(W:path). Files or pipes (e.g. <(nc host port)).')
    pool.add_argument('--mode', choices=['fifo', 'round-robin', 'weighted'],
                      default=None,
                      help='How to interleave sources: fifo (as-ready; '
                           'default), round-robin, or weighted.')
    pool.add_argument('--count', type=int, default=None, metavar='N',
                      help='Emit exactly N batches total, then stop and reap '
                           'the pool. Default: stream forever (until the '
                           'downstream consumer closes the pipe).')
    pool.add_argument('--shuffle-seed', type=int, default=42)
    pool.add_argument('--worker-log-every', type=int, default=0,
                      help='Per-worker stderr heartbeat every N batches '
                           '(0 = only start/end). Workers are noisy at scale.')
    pool.add_argument('--log-every-sec', type=float, default=5.0,
                      help='Throughput report interval under -v (seconds).')
    pool.add_argument('-v', '--verbose', action='count', default=0,
                      help='Report windowed throughput (batch/s, MB/s) every '
                           '--log-every-sec, plus a final summary.')

    args = p.parse_args()

    # Auto-mux stdin when something is piped or redirected in — a FIFO/pipe
    # (`cat corpus.rvt | datagen`) or a regular file (`datagen < corpus.rvt`).
    # A tty or /dev/null is a character device and is skipped, so a bare
    # `datagen` (or a generator run with stdin=/dev/null in a script) is
    # unaffected. Added later as a LAZY source (no eager peek): an empty stdin
    # contributes nothing, and a silent/never-closing one cannot stall the
    # run — generation is infinite and ends on --count or downstream close.
    stdin_is_input = False
    try:
        st_mode = os.fstat(sys.stdin.fileno()).st_mode
        stdin_is_input = stat.S_ISFIFO(st_mode) or stat.S_ISREG(st_mode)
    except (OSError, ValueError, AttributeError):
        pass  # no usable stdin fd — nothing to mux

    gen_count = os.cpu_count() if args.gen_count is None else args.gen_count
    if gen_count < 0:
        p.error('--gen-count must be >= 0')
    if gen_count == 0 and not args.inputs and not stdin_is_input:
        p.error('--gen-count 0 needs an input to mux (a positional file or a '
                'piped stdin); otherwise drop it and let generation default '
                'to all cores.')

    # Validate the rule eagerly in the parent (it is re-parsed per worker) so
    # an unbounded/unknown rule fails fast with a clear message and non-zero
    # exit, instead of surfacing as a traceback inside every forked worker.
    if gen_count > 0:
        try:
            _parse_rule(args.rule)
        except ValueError as e:
            p.error(str(e))

    fmt = RVT_FORMAT  # the one and only format

    # 1. Spawn workers FIRST (before binary_stdout) so they never inherit the
    #    downstream data fd. Their read-ends become mux sources.
    pool_obj = WorkerPool()
    worker_files = []
    worker_weights = []
    if gen_count > 0:
        cfg = _GenCfg(args)
        worker_files = pool_obj.spawn(gen_count, cfg, args.seed)
        worker_weights = [args.gen_weight] * gen_count

    # 2. Now claim the binary stdout fd for the muxer.
    out = binary_stdout()

    # 3. Open external inputs.
    file_handles = []
    ext_files = []
    ext_weights = []
    ext_labels = []
    has_explicit_weight = False
    for raw in args.inputs:
        path, weight = _parse_weighted_input(raw)
        if weight != 1.0:
            has_explicit_weight = True
        f = open(path, 'rb')
        try:
            fmt.peek_magic(f)  # fail fast on a non-RVT file, before generating
        except ValueError as e:
            print(f'ERROR: {path}: {e}', file=sys.stderr)
            pool_obj.reap()
            sys.exit(1)
        ext_files.append(f)
        ext_weights.append(weight)
        ext_labels.append(path)
        file_handles.append(f)

    # Add stdin (detected above) as a lazy source. Read it through a PRIVATE
    # dup of fd 0, never `sys.stdin.buffer`: under `--count` the main thread can
    # exit while the daemon reader is blocked mid-read, and if that reader held
    # `sys.stdin`'s buffer lock, interpreter shutdown would deadlock on
    # `_enter_buffered_busy` (it force-deallocates sys.stdin). A private object
    # referenced only by the daemon frame is never finalized, so no lock
    # contention. Not added to file_handles — left to leak; process exit
    # reclaims the fd.
    if stdin_is_input:
        stdin_in = os.fdopen(os.dup(sys.stdin.fileno()), 'rb',
                             buffering=1 << 16)
        ext_files.append(stdin_in)
        ext_weights.append(1.0)
        ext_labels.append('stdin')

    files = ext_files + worker_files
    weights = ext_weights + worker_weights
    labels = ext_labels + [f'worker[{i}]' for i in range(len(worker_files))]

    mode = args.mode
    if mode is None:
        mode = 'weighted' if has_explicit_weight else 'fifo'

    if args.verbose >= 1:
        parts = []
        if ext_files:
            label = f'{len(ext_files)} external input'
            label += 's' if len(ext_files) != 1 else ''
            if stdin_is_input:
                label += ' (incl. stdin)'
            parts.append(label)
        if pool_obj.procs:
            parts.append(f'{len(pool_obj.procs)} generation workers')
        print(f'datagen: {" + ".join(parts)}, mode={mode}, '
              f'format={fmt.name}', file=sys.stderr)

    writer = _Writer(out, count=args.count, verbose=(args.verbose >= 1),
                     log_every_sec=args.log_every_sec)
    fatal = None
    try:
        if mode == 'weighted':
            mux_weighted(files, weights, writer, fmt, labels,
                         seed=args.shuffle_seed)
        else:
            mux_fn = {'fifo': mux_fifo, 'round-robin': mux_round_robin}[mode]
            mux_fn(files, weights, writer, fmt, labels)
    except _InputParseError as e:
        # Fail early, fail loud: a malformed batch from ANY source kills the
        # whole tree. The finally below reaps the worker pool (killpg).
        fatal = e
    finally:
        try:
            out.close()
        except BrokenPipeError:
            pass
        for f in file_handles:
            f.close()
        pool_obj.reap()

    if fatal is not None:
        print(f'datagen: FATAL — malformed batch from {fatal}; killed the '
              f'pool and aborted.', file=sys.stderr)
        sys.exit(1)

    if args.verbose >= 1:
        writer.final()


if __name__ == '__main__':
    main()
