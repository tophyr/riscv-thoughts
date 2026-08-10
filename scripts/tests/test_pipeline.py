"""Integration tests for the batch pipeline scripts.

Runs `python -m datagen` and the trainers via subprocess.
All tools share the single RVT format.
"""

import glob
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable


def _children_of(pid):
    """Direct child PIDs of `pid`. datagen forks its workers, so they
    share the parent's argv and can't be told apart by a cmdline scan;
    identify them by parentage instead (pgrep -P)."""
    try:
        r = subprocess.run(['pgrep', '-P', str(pid)],
                           capture_output=True, text=True)
    except FileNotFoundError:
        return set()
    return {int(x) for x in r.stdout.split()}


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def run(cmd, stdin=None, timeout=60):
    kwargs = dict(capture_output=True, timeout=timeout, cwd=str(ROOT))
    if stdin is None:
        # datagen auto-muxes a piped/redirected stdin; give it /dev/null
        # (a char device, skipped) so an inherited pipe under some test
        # runner can't perturb generation-only runs.
        kwargs['stdin'] = subprocess.DEVNULL
    else:
        kwargs['input'] = stdin
    result = subprocess.run(cmd, **kwargs)
    return result.stdout, result.stderr.decode(), result.returncode


def gen(n_batches=2, batch_size=8, twins=1, n_states=4,
        rule='cap=2', seed=42):
    """A bounded RVT fixture: a single-worker datagen pool capped to exactly
    n_batches via --count. One worker keeps the bytes deterministic under
    `seed`."""
    out, err, rc = run([
        PYTHON, '-m', 'datagen', '--gen-count', '1', '--count', str(n_batches),
        '--batch-size', str(batch_size), '--rule', rule,
        '--twins', str(twins), '--n-states', str(n_states),
        '--seed', str(seed)], timeout=120)
    assert rc == 0, f'datagen failed: {err}'
    assert len(out) > 0
    return out


def _count_batches(data):
    """Count batches in an RVT blob, in-process (replaces the deleted
    `batch_slice --info`). A clean parse also confirms the bytes are valid
    RVT — the reader raises otherwise."""
    import io
    sys.path.insert(0, str(ROOT))
    from datagen import RVT_FORMAT, Batch
    return sum(1 for _ in RVT_FORMAT.reader(io.BytesIO(data), Batch))


def _first_header(data):
    """Unpack the first batch's RVT dimension header:
    (B, max_tokens, max_n_instrs, n_anchors, OB)."""
    sys.path.insert(0, str(ROOT))
    from datagen import RVT_FORMAT
    off = RVT_FORMAT.batch_prefix_size
    return RVT_FORMAT.batch_header.unpack(
        data[off:off + RVT_FORMAT.batch_header.size])


class TestGenBatches:
    def test_emits_valid_rvt(self):
        data = gen()
        assert _count_batches(data) == 2

    def test_out_regs_mode_header(self):
        """Every rule — single (T1) and multi-instruction (T2) — ships the
        unified CANONICAL out_regs oracle (OB == B). Both carry
        n_anchors == --n-states."""
        out = gen(n_batches=1, batch_size=4, twins=0, n_states=4,
                  rule='single')
        B, _, _, n_anchors, OB = _first_header(out)
        assert B == 4, B
        assert OB == B, f'single must ship OB==B (canonical), got OB={OB} B={B}'
        assert n_anchors == 4, n_anchors

        # A multi-instruction rule ships the same out_regs payload.
        multi = gen(n_batches=1, batch_size=4, twins=0, n_states=4,
                    rule='cap=2')
        B_m, _, _, n_anchors_m, OB_m = _first_header(multi)
        assert OB_m == B_m, f'T2 mode must ship OB==B, got OB={OB_m} B={B_m}'
        assert n_anchors_m == 4, n_anchors_m


class TestGenRules:
    """datagen --rule parsing: capped composite rules emit valid RVT
    respecting the length cap; unbounded / unknown rules fail clearly."""

    @pytest.mark.parametrize('component,cap', [('branch', 3),
                                               ('transform', 5)])
    def test_capped_rule_respects_cap(self, component, cap):
        rule = f'{component}+cap={cap}'
        # twins=0 keeps the batch to source rows; large batch so several
        # chunks hit the cap.
        data = gen(n_batches=3, batch_size=16, twins=0, n_states=4,
                   rule=rule, seed=7)
        assert _count_batches(data) == 3
        # Every chunk's instruction count is bounded by the cap. Read the
        # per-row instr_lens column counts back out of the stream.
        sys.path.insert(0, str(ROOT))
        from datagen import RVT_FORMAT, Batch
        import io
        max_instrs = 0
        f = io.BytesIO(data)
        for b in RVT_FORMAT.reader(f, Batch):
            counts = (b.instr_lens > 0).sum(axis=1)
            if counts.size:
                max_instrs = max(max_instrs, int(counts.max()))
        assert 0 < max_instrs <= cap, (
            f'{rule}: max instrs/chunk {max_instrs} violates cap {cap}')

    def test_bare_branch_rejected(self):
        """A bare terminator (no length cap) can build an unbounded chunk;
        datagen must reject it eagerly in the parent (before forking
        workers), with a clear message and non-zero exit — not a traceback
        deep inside a worker."""
        out, err, rc = run([
            PYTHON, '-m', 'datagen', '--batch-size', '4',
            '--rule', 'branch', '--twins', '0', '--n-states', '4'])
        assert rc != 0, 'bare branch must fail'
        assert 'unbounded' in err.lower() and 'cap' in err.lower(), err
        assert len(out) == 0

    def test_unknown_component_rejected(self):
        out, err, rc = run([
            PYTHON, '-m', 'datagen', '--batch-size', '4',
            '--rule', 'frobnicate', '--twins', '0', '--n-states', '4'])
        assert rc != 0, 'unknown rule component must fail'
        assert 'Unknown rule component' in err and 'frobnicate' in err, err
        assert len(out) == 0


class TestCount:
    """`--count N` (folded in from the retired batch_slice): emit exactly N
    batches total across the pool, then stop and reap."""

    def test_count_caps_pool_output(self):
        out, err, rc = run([
            PYTHON, '-m', 'datagen', '--gen-count', '4', '--count', '10',
            '--batch-size', '8', '--rule', 'cap=2', '--twins', '1',
            '--n-states', '4'], timeout=120)
        assert rc == 0, err
        assert _count_batches(out) == 10

    def test_count_on_pure_mux(self):
        """--count also bounds a pure-mux (--gen-count 0) pass over inputs —
        the old `batch_slice --count` on a stream."""
        with tempfile.NamedTemporaryFile(suffix='.rvt') as f1, \
             tempfile.NamedTemporaryFile(suffix='.rvt') as f2:
            f1.write(gen(n_batches=3, seed=1))
            f1.flush()
            f2.write(gen(n_batches=3, seed=2))
            f2.flush()
            out, err, rc = run([
                PYTHON, '-m', 'datagen', '--gen-count', '0', '--count', '4',
                f1.name, f2.name])
            assert rc == 0, err
            assert _count_batches(out) == 4


class TestMuxBatches:
    def test_file_inputs(self):
        """Pure mux (--gen-count 0) of two file inputs, conserving total."""
        with tempfile.NamedTemporaryFile(suffix='.rvt') as f1, \
             tempfile.NamedTemporaryFile(suffix='.rvt') as f2:
            f1.write(gen(n_batches=2, seed=1))
            f1.flush()
            f2.write(gen(n_batches=3, seed=2))
            f2.flush()
            out, err, rc = run([PYTHON, '-m', 'datagen',
                                '--gen-count', '0', f1.name, f2.name])
            assert rc == 0, err
            assert _count_batches(out) == 5

    def test_spawned_workers(self):
        """Two generation workers, bounded to 4 batches by --count."""
        out, err, rc = run([
            PYTHON, '-m', 'datagen', '--gen-count', '2', '--count', '4',
            '--batch-size', '8', '--rule', 'cap=2', '--twins', '1',
            '--n-states', '4'], timeout=120)
        assert rc == 0, err
        assert _count_batches(out) == 4

    def test_round_robin_unequal(self):
        """round-robin over two file inputs of UNEQUAL length: it
        interleaves one batch from each, then drains the longer input
        alone. Total count is conserved (2 + 3 = 5)."""
        with tempfile.NamedTemporaryFile(suffix='.rvt') as f1, \
             tempfile.NamedTemporaryFile(suffix='.rvt') as f2:
            f1.write(gen(n_batches=2, seed=1))
            f1.flush()
            f2.write(gen(n_batches=3, seed=2))
            f2.flush()
            out, err, rc = run([
                PYTHON, '-m', 'datagen', '--gen-count', '0',
                '--mode', 'round-robin', f1.name, f2.name])
            assert rc == 0, err
            assert _count_batches(out) == 5

    def test_weighted_mode(self):
        """weighted mode honors per-input weights (W:path syntax) and
        terminates cleanly once every input drains: total is conserved
        regardless of the weighting (the weights bias ORDER, not count).
        Both the explicit --mode weighted and the implicit `W:path`
        auto-selection are exercised."""
        with tempfile.NamedTemporaryFile(suffix='.rvt') as f1, \
             tempfile.NamedTemporaryFile(suffix='.rvt') as f2:
            f1.write(gen(n_batches=2, seed=1))
            f1.flush()
            f2.write(gen(n_batches=4, seed=2))
            f2.flush()
            out, err, rc = run([
                PYTHON, '-m', 'datagen', '--gen-count', '0',
                '--mode', 'weighted', '1:' + f1.name, '5:' + f2.name])
            assert rc == 0, err
            assert _count_batches(out) == 6

            # `W:path` with a non-1.0 weight auto-selects weighted mode
            # even without --mode.
            out2, err2, rc2 = run([
                PYTHON, '-m', 'datagen', '--gen-count', '0',
                '1:' + f1.name, '5:' + f2.name])
            assert rc2 == 0, err2
            assert _count_batches(out2) == 6

    def test_no_orphan_workers_after_kill(self):
        """The project's #1 recurring hazard, now OS-enforced. datagen
        forks generation workers that arm PR_SET_PDEATHSIG, so a `kill -9`
        of the parent (no clean shutdown, no reaping code runs) takes the
        whole pool with it. Start a pool, observe its worker children,
        SIGKILL the parent, assert no worker survives — identifying workers
        by parentage (fork children share the parent's argv)."""
        proc = subprocess.Popen([
            PYTHON, '-m', 'datagen', '--gen-count', '4',
            '--rule', 'branch+cap=8', '--batch-size', '64',
            '--twins', '3', '--n-states', '8'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, cwd=str(ROOT))
        workers = set()
        try:
            deadline = time.time() + 30
            while time.time() < deadline:
                workers = _children_of(proc.pid)
                if len(workers) >= 4:
                    break
                assert proc.poll() is None, (
                    f'parent exited early rc={proc.returncode}')
                time.sleep(0.1)
            assert len(workers) >= 4, (
                f'expected 4 workers, observed {sorted(workers)}')

            # kill -9 the parent: NO clean shutdown path runs. Only
            # pdeathsig can save us here.
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)

            deadline = time.time() + 15
            survivors = {w for w in workers if _alive(w)}
            while time.time() < deadline and survivors:
                time.sleep(0.2)
                survivors = {w for w in workers if _alive(w)}
            assert not survivors, (
                f'orphan workers survived kill -9 of parent: '
                f'{sorted(survivors)}')
        finally:
            for w in workers:
                try:
                    os.kill(w, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


class TestThroughput:
    """`-v` throughput reporting (folded in from the retired bench tool):
    windowed batch/s + MB/s lines and a final summary, on a --count run."""

    def test_final_summary(self):
        _, err, rc = run([
            PYTHON, '-m', 'datagen', '--gen-count', '1', '--count', '4',
            '--rule', 'cap=2', '--batch-size', '32', '--twins', '1',
            '--n-states', '4', '-v'], timeout=60)
        assert rc == 0, err
        assert 'datagen: 4 batches' in err, err

    def test_emits_periodic_log(self):
        """A small --log-every-sec emits at least one windowed status line
        over a --count-bounded run. stdout → /dev/null so the (large) data
        stream isn't buffered into the test; only stderr is captured."""
        r = subprocess.run([
            PYTHON, '-m', 'datagen', '--gen-count', '2', '--count', '3000',
            '--rule', 'cap=2', '--batch-size', '32', '--twins', '2',
            '--n-states', '4', '-v', '--log-every-sec', '0.05'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, cwd=str(ROOT), timeout=60)
        err = r.stderr.decode()
        assert r.returncode == 0, err
        # At least one periodic line of the form `[  N.Ns] total=...`.
        import re
        assert re.search(r'^\[\s*\d+\.\d+s\] total=', err, re.MULTILINE), err
        assert 'datagen: 3000 batches' in err, err


class TestStdin:
    """A piped/redirected stdin (FIFO or regular file) is auto-muxed as an
    input; a tty or /dev/null (char devices) is skipped so bare generation is
    unaffected. `run(stdin=bytes)` feeds the child via a pipe (a FIFO)."""

    def test_stdin_passthrough_pure_mux(self):
        """`cat corpus | datagen --gen-count 0` passes the piped corpus
        straight through (no generation) — the old corpus-replay path."""
        corpus = gen(n_batches=4, seed=3)
        out, err, rc = run([PYTHON, '-m', 'datagen', '--gen-count', '0'],
                            stdin=corpus)
        assert rc == 0, err
        assert _count_batches(out) == 4

    def test_stdin_muxed_with_workers(self):
        """A piped stdin coexists with generation workers; --count bounds the
        joined stream (the run completes and reaps regardless of stdin)."""
        corpus = gen(n_batches=4, seed=3)
        out, err, rc = run([
            PYTHON, '-m', 'datagen', '--gen-count', '2', '--count', '12',
            '--rule', 'cap=2', '--batch-size', '8', '--twins', '1',
            '--n-states', '4'], stdin=corpus, timeout=120)
        assert rc == 0, err
        assert _count_batches(out) == 12

    def test_count_bounds_piped_stdin(self):
        """`--count` on a pure-mux stdin pass bounds it (the retired
        `batch_slice --count` on a stream)."""
        corpus = gen(n_batches=10, seed=5)
        out, err, rc = run([PYTHON, '-m', 'datagen', '--gen-count', '0',
                            '--count', '3'], stdin=corpus)
        assert rc == 0, err
        assert _count_batches(out) == 3


class TestMalformedInputFatal:
    """A malformed batch from ANY source aborts the whole run loudly (fail
    early, fail loud) — never a silently-dropped source. A CLEAN EOF is not
    malformed and just drains the source (covered by the mux/stdin tests)."""

    def test_garbage_stdin_is_fatal(self):
        out, err, rc = run([PYTHON, '-m', 'datagen', '--gen-count', '0'],
                           stdin=b'NOTAVALIDRVTSTREAMXXXX')
        assert rc != 0, 'malformed input must fail'
        assert 'FATAL' in err and 'stdin' in err, err

    def test_garbage_after_good_is_fatal(self):
        """Valid batches THEN garbage (mid-stream corruption) still aborts."""
        good = gen(n_batches=3, seed=7)
        out, err, rc = run([PYTHON, '-m', 'datagen', '--gen-count', '0'],
                           stdin=good + b'GARBAGE!')
        assert rc != 0, err
        assert 'FATAL' in err, err

    def test_fatal_reaps_worker_pool(self):
        """A malformed input mid-generation kills the whole tree — no worker
        survives the fatal abort."""
        proc = subprocess.Popen([
            PYTHON, '-m', 'datagen', '--gen-count', '4', '--count', '100000',
            '--rule', 'branch+cap=8', '--batch-size', '64', '--twins', '3',
            '--n-states', '8'],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, cwd=str(ROOT))
        # Observe workers, then feed garbage to trigger the fatal path.
        workers = set()
        deadline = time.time() + 30
        while time.time() < deadline:
            workers = _children_of(proc.pid)
            if len(workers) >= 4:
                break
            time.sleep(0.1)
        assert len(workers) >= 4, f'expected 4 workers, saw {sorted(workers)}'
        # Feed garbage now that workers are up. Read stderr directly (not
        # communicate(), which would re-close the stdin we close here).
        try:
            proc.stdin.write(b'GARBAGE!!')
            proc.stdin.flush()
        except BrokenPipeError:
            pass  # datagen may already be aborting
        proc.stdin.close()
        err = proc.stderr.read()
        proc.stderr.close()
        proc.wait(timeout=30)
        assert proc.returncode != 0, 'fatal must exit non-zero'
        assert b'FATAL' in err, err
        deadline = time.time() + 15
        survivors = {w for w in workers if _alive(w)}
        while time.time() < deadline and survivors:
            time.sleep(0.2)
            survivors = {w for w in workers if _alive(w)}
        assert not survivors, f'workers survived the fatal abort: {sorted(survivors)}'


def _train_t1(tmp, name='t1'):
    """Produce a tiny 1-step T1 encoder checkpoint under tmp/<name>.
    Returns the encoder.pt path. --n-anchor-states 4 matches the corpus's
    --n-states 4 (canonical vp runs on the shipped out_regs)."""
    run_dir = str(Path(tmp) / name)
    data = gen(n_batches=2, batch_size=8, rule='cap=2', twins=1, n_states=4)
    out, err, rc = run([
        PYTHON, 'scripts/train_encoder.py',
        '--n-steps', '1', '--log-every', '1', '--out-dir', run_dir,
        '--d-model', '32', '--n-heads', '2', '--n-layers', '1',
        '--d-out', '32', '--max-window', '32', '--n-anchor-states', '4',
        '--no-compile'],
        stdin=data, timeout=120)
    assert rc == 0, f'T1 train failed: {err}'
    enc = Path(run_dir) / 'encoder.pt'
    assert enc.exists(), f'no encoder.pt in {run_dir}'
    return enc


def _train_t2(tmp, t1_enc, name='t2', extra_args=()):
    """Produce a tiny 1-step T2 checkpoint on top of a frozen T1.
    Returns the t2 run dir. branch+cap=8 → multi-instruction chunks."""
    run_dir = str(Path(tmp) / name)
    data = gen(n_batches=2, batch_size=8, rule='branch+cap=8', twins=1,
               n_states=4)
    out, err, rc = run([
        PYTHON, 'scripts/train_t2_encoder.py', '--t1-model', str(t1_enc),
        '--n-steps', '1', '--log-every', '1', '--out-dir', run_dir,
        '--d-out', '32', '--d-event', '16',
        '--n-anchor-states', '4', '--warmup-steps', '0', '--no-compile',
        *extra_args],
        stdin=data, timeout=180)
    assert rc == 0, f'T2 train failed: {err}'
    assert (Path(run_dir) / 't2.pt').exists(), f'no t2.pt in {run_dir}'
    return Path(run_dir)


class TestTraining:
    def test_encoder_train_smoke(self):
        """datagen | train_encoder smoke (single step). T1 now runs the
        unified canonical vp on out_regs (present for every rule), so
        --n-anchor-states must match gen's --n-states (4) per invariant 1."""
        data = gen(n_batches=2, batch_size=8)
        with tempfile.TemporaryDirectory() as tmp:
            out, err, rc = run([
                PYTHON, 'scripts/train_encoder.py',
                '--n-steps', '1', '--log-every', '1',
                '--out-dir', tmp,
                '--d-model', '32', '--n-heads', '2', '--n-layers', '1',
                '--d-out', '32', '--max-window', '32', '--n-anchor-states', '4',
                '--no-compile'],
                stdin=data, timeout=120)
        assert rc == 0, f'encoder training failed: {err}'

    def test_t2_encoder_train_smoke(self):
        """train_t2_encoder needs a T1 checkpoint: build one with a 1-step
        train_encoder run, then pipe a branch+cap=8 corpus into
        train_t2_encoder for one step. Exits 0 and writes t2.pt."""
        with tempfile.TemporaryDirectory() as tmp:
            t1_enc = _train_t1(tmp)
            t2_dir = _train_t2(tmp, t1_enc)
            assert (t2_dir / 't2.pt').exists()

    @pytest.mark.parametrize('mode,s', [
        ('regression', 0.0),    # exact-match indicator (binary control)
        ('regression', 7.0),    # graded, knee at |Δ|~1e5-1e6
        ('infonce', 7.0),       # soft-weighted NT-Xent fallback
    ])
    def test_t2_contrastive_train_smoke(self, mode, s):
        """T2 with the behavioral-distance contrastive loss ON and vp demoted
        to 0: exercises out_regs shipping under contrastive-only, the d_beh +
        cosine matrices, and the regression/infonce heads (incl. s=0 indicator).
        One step, exits 0, writes t2.pt — and the logged contrastive loss is a
        finite number (no NaN from the masked softmax / empty-footprint guards)."""
        with tempfile.TemporaryDirectory() as tmp:
            t1_enc = _train_t1(tmp)
            t2_dir = _train_t2(
                tmp, t1_enc, name=f't2_co_{mode}_{s}',
                extra_args=['--value-predict-weight', '0',
                            '--contrastive-weight', '1.0',
                            '--contrastive-loss', mode,
                            '--contrastive-s', str(s)])
            losses = json.loads((t2_dir / 'losses.json').read_text())
            co = losses[-1]['contrastive']
            assert co == co, 'contrastive loss is NaN'     # NaN != NaN
            assert co >= 0.0


class TestEval:
    """scripts/eval.py pulls many compressor internals that drift. Guard
    import + --help, then run end-to-end (T1 + T2, binding routing) on a
    1-step checkpoint + tiny corpus."""

    def test_eval_help(self):
        _, err, rc = run([PYTHON, 'scripts/eval.py', '--help'])
        assert rc == 0, err

    def test_eval_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            t1_enc = _train_t1(tmp)
            t2_dir = _train_t2(tmp, t1_enc)
            t1_corpus = Path(tmp) / 't1.rvt'      # single-instruction (T1)
            t1_corpus.write_bytes(gen(n_batches=2, batch_size=8, rule='single',
                                      twins=1, n_states=4, seed=98))
            t2_corpus = Path(tmp) / 't2.rvt'      # multi-instruction (T2)
            t2_corpus.write_bytes(gen(n_batches=2, batch_size=8,
                                      rule='branch+cap=8', twins=1, n_states=4,
                                      seed=99))

            out, err, rc = run([
                PYTHON, 'scripts/eval.py',
                '--t1-model', str(t1_enc), '--t2', str(t2_dir),
                '--t1-corpus', str(t1_corpus), '--t2-corpus', str(t2_corpus),
                '--max-batches', '2', '--route', 'binding',
                '--gvn-pairs', '8', '--gvn-chunk-len', '3',
                '--device', 'cpu'], timeout=300)
            assert rc == 0, f'eval failed: {err}'
            # Equivariance is exact by construction even on a 1-step model.
            text = out.decode() if isinstance(out, bytes) else out
            assert 'state_max_err=' in text, text
            assert 'GVN collapse' in text, text

# Wire-format unit tests (magic peek, dtype guard, header-bounds validator)
# live with the format itself in datagen/tests/test_batch.py.
