"""Integration tests for the batch pipeline scripts.

Runs gen_batches, batch_slice, and mux_batches via subprocess.
All tools share the single RVT format.
"""

import glob
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


def _gen_worker_pids():
    """PIDs of live gen_batches.py worker processes, by /proc cmdline
    scan (psutil-free). Used by the orphan-cleanup test.

    Matches only processes RUNNING gen_batches.py as a script — i.e. an
    argv element whose basename is exactly 'gen_batches.py'. A plain
    substring scan would also match shell wrappers and helper processes
    that merely mention the path in their command line (e.g. the pytest
    launcher), which would make the orphan check spuriously fire.
    """
    pids = set()
    for d in glob.glob('/proc/[0-9]*'):
        try:
            with open(os.path.join(d, 'cmdline'), 'rb') as f:
                cmdline = f.read()
        except OSError:
            continue  # process exited between glob and open
        argv = cmdline.split(b'\x00')
        if any(os.path.basename(a) == b'gen_batches.py' for a in argv):
            pids.add(int(os.path.basename(d)))
    return pids


def run(cmd, stdin=None, timeout=60):
    result = subprocess.run(
        cmd, input=stdin, capture_output=True, timeout=timeout,
        cwd=str(ROOT))
    return result.stdout, result.stderr.decode(), result.returncode


def gen(n_batches=2, batch_size=8, twins=1, n_states=4,
        rule='cap=2', seed=42):
    out, err, rc = run([
        PYTHON, 'scripts/gen_batches.py',
        '-n', str(n_batches),
        '--batch-size', str(batch_size),
        '--rule', rule,
        '--twins', str(twins),
        '--n-states', str(n_states),
        '--seed', str(seed)])
    assert rc == 0, f'gen_batches failed: {err}'
    assert len(out) > 0
    return out


def slice_info(data):
    _, err, rc = run([PYTHON, 'scripts/batch_slice.py', '--info'],
                     stdin=data)
    assert rc == 0, f'batch_slice --info failed: {err}'
    return err


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
        info = slice_info(data)
        assert 'Format:       rvt' in info
        assert 'Batches:      2' in info

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
    """gen_batches --rule parsing: capped composite rules emit valid RVT
    respecting the length cap; unbounded / unknown rules fail clearly."""

    @pytest.mark.parametrize('component,cap', [('branch', 3),
                                               ('transform', 5)])
    def test_capped_rule_respects_cap(self, component, cap):
        rule = f'{component}+cap={cap}'
        # twins=0 keeps the batch to source rows; large batch so several
        # chunks hit the cap.
        data = gen(n_batches=3, batch_size=16, twins=0, n_states=4,
                   rule=rule, seed=7)
        info = slice_info(data)
        assert 'Format:       rvt' in info, info
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
        gen_batches must reject it eagerly with a clear message, not crash
        downstream with an opaque None-arithmetic error."""
        out, err, rc = run([
            PYTHON, 'scripts/gen_batches.py', '-n', '1', '--batch-size', '4',
            '--rule', 'branch', '--twins', '0', '--n-states', '4'])
        assert rc != 0, 'bare branch must fail'
        assert 'unbounded' in err.lower() and 'cap' in err.lower(), err
        assert len(out) == 0

    def test_unknown_component_rejected(self):
        out, err, rc = run([
            PYTHON, 'scripts/gen_batches.py', '-n', '1', '--batch-size', '4',
            '--rule', 'frobnicate', '--twins', '0', '--n-states', '4'])
        assert rc != 0, 'unknown rule component must fail'
        assert 'Unknown rule component' in err and 'frobnicate' in err, err
        assert len(out) == 0


class TestBatchSlice:
    def test_count(self):
        data = gen(n_batches=5)
        out, err, rc = run(
            [PYTHON, 'scripts/batch_slice.py', '--count', '2'],
            stdin=data)
        assert rc == 0
        assert 'Batches:      2' in slice_info(out)

    def test_skip(self):
        data = gen(n_batches=5)
        out, _, rc = run(
            [PYTHON, 'scripts/batch_slice.py', '--skip', '3'],
            stdin=data)
        assert rc == 0
        assert 'Batches:      2' in slice_info(out)

    def test_tail(self):
        with tempfile.NamedTemporaryFile(suffix='.rvt') as f:
            f.write(gen(n_batches=5))
            f.flush()
            out, _, rc = run([
                PYTHON, 'scripts/batch_slice.py', '--tail', '2', f.name])
            assert rc == 0
            assert 'Batches:      2' in slice_info(out)


class TestMuxBatches:
    def test_file_inputs(self):
        with tempfile.NamedTemporaryFile(suffix='.rvt') as f1, \
             tempfile.NamedTemporaryFile(suffix='.rvt') as f2:
            f1.write(gen(n_batches=2, seed=1))
            f1.flush()
            f2.write(gen(n_batches=3, seed=2))
            f2.flush()
            out, err, rc = run([PYTHON, 'scripts/mux_batches.py',
                                f1.name, f2.name])
            assert rc == 0, err
            assert 'Batches:      5' in slice_info(out)

    def test_spawned_workers(self):
        out, err, rc = run([
            PYTHON, 'scripts/mux_batches.py',
            '--gen-count', '2', '--',
            '--n-batches', '2', '--batch-size', '8', '--rule', 'cap=2',
            '--twins', '1', '--n-states', '4'],
            timeout=120)
        assert rc == 0, f'mux spawned failed: {err}'
        assert 'Batches:      4' in slice_info(out)

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
                PYTHON, 'scripts/mux_batches.py', '--mode', 'round-robin',
                f1.name, f2.name])
            assert rc == 0, err
            assert 'Batches:      5' in slice_info(out)

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
                PYTHON, 'scripts/mux_batches.py', '--mode', 'weighted',
                '1:' + f1.name, '5:' + f2.name])
            assert rc == 0, err
            assert 'Batches:      6' in slice_info(out)

            # `W:path` with a non-1.0 weight auto-selects weighted mode
            # even without --mode.
            out2, err2, rc2 = run([
                PYTHON, 'scripts/mux_batches.py',
                '1:' + f1.name, '5:' + f2.name])
            assert rc2 == 0, err2
            assert 'Batches:      6' in slice_info(out2)

    def test_no_orphan_workers_after_exit(self):
        """The project's recurring hazard: gen_batches workers spawned by
        mux via subprocess.Popen must NOT survive mux's exit. Spawn 2
        long-running workers, force mux to exit while they're still
        producing (a downstream consumer closes the pipe after a few
        batches), then assert no gen_batches child PID survives.

        psutil is not a hard dep — scan /proc for the worker cmdlines.
        """
        before = _gen_worker_pids()
        # Long -n so the workers keep producing indefinitely; THIS test
        # owns the read end of mux's pipe, so it controls precisely when
        # the pipe closes. No early-finishing downstream consumer races
        # the PID poll: we block until we positively observe the workers,
        # and only THEN close our read end to trigger mux's BrokenPipe
        # cleanup path. That makes "workers were observed" deterministic.
        mux = subprocess.Popen([
            PYTHON, 'scripts/mux_batches.py', '--gen-count', '2', '--',
            '--n-batches', '100000', '--batch-size', '64',
            '--rule', 'branch+cap=8', '--twins', '3', '--n-states', '8'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(ROOT))
        spawned = set()
        try:
            # Block until both workers are observed alive (or time out).
            # We deliberately do NOT drain mux.stdout here — letting the
            # pipe buffer fill keeps the workers running/visible.
            deadline = time.time() + 30
            while time.time() < deadline:
                spawned |= (_gen_worker_pids() - before)
                if len(spawned) >= 2:
                    break
                assert mux.poll() is None, (
                    f'mux exited early rc={mux.returncode} before workers '
                    f'were observed')
                time.sleep(0.1)
            assert len(spawned) >= 2, (
                f'expected 2 gen_batches workers, observed {sorted(spawned)}')

            # Close our read end → mux's writes hit BrokenPipeError; it
            # drains its loop and reaps (terminate+wait) every worker.
            mux.stdout.close()
            mux.wait(timeout=60)
            assert mux.returncode == 0, f'mux rc {mux.returncode}'

            # Every spawned worker must be reaped — no orphan survives.
            deadline = time.time() + 15
            survivors = _gen_worker_pids() & spawned
            while time.time() < deadline and survivors:
                time.sleep(0.2)
                survivors = _gen_worker_pids() & spawned
            assert not survivors, (
                f'orphan gen_batches workers survived mux exit: '
                f'{sorted(survivors)}')
        finally:
            # Defensive: never let this test itself leak workers.
            for pid in (_gen_worker_pids() & spawned):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if mux.poll() is None:
                mux.kill()
                mux.wait(timeout=10)


class TestBenchThroughput:
    def test_final_summary(self):
        """Pipe N batches; expect `Final: N batches` on stderr."""
        data = gen(n_batches=4)
        _, err, rc = run([PYTHON, 'scripts/bench_throughput.py',
                          '--log-every-sec', '5'], stdin=data)
        assert rc == 0, err
        assert 'Final: 4 batches' in err, err

    def test_max_batches_caps(self):
        """--max-batches stops the count even with more available."""
        data = gen(n_batches=10)
        _, err, rc = run([
            PYTHON, 'scripts/bench_throughput.py',
            '--log-every-sec', '5', '--max-batches', '3'],
            stdin=data)
        assert rc == 0, err
        assert 'Final: 3 batches' in err, err

    def test_errors_on_bad_input(self):
        """Non-RVT input fails with a clear error."""
        _, err, rc = run([PYTHON, 'scripts/bench_throughput.py'],
                         stdin=b'NOT-A-VALID-STREAM-HEADER')
        assert rc != 0
        assert 'Unknown format' in err, err

    def test_emits_periodic_log(self):
        """Live pipe with small --log-every-sec should emit at least
        one periodic status line. A large -n gives enough wall time to
        fire the 0.05s timer."""
        gen_proc = __import__('subprocess').Popen([
            PYTHON, 'scripts/gen_batches.py',
            '--rule', 'cap=2', '--batch-size', '32',
            '--twins', '2', '--n-states', '4',
            '-n', '2000', '--seed', '42'],
            stdout=__import__('subprocess').PIPE,
            stderr=__import__('subprocess').DEVNULL,
            cwd=str(ROOT))
        try:
            r = subprocess.run([
                PYTHON, 'scripts/bench_throughput.py',
                '--log-every-sec', '0.05'],
                stdin=gen_proc.stdout, capture_output=True,
                cwd=str(ROOT), timeout=60)
        finally:
            gen_proc.wait(timeout=10)
        err = r.stderr.decode()
        assert r.returncode == 0, err
        # At least one periodic line of the form `[N.Ns] total=...`.
        import re
        assert re.search(r'^\[\s*\d+\.\d+s\] total=', err, re.MULTILINE), err
        assert 'Final: 2000 batches' in err, err


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


def _train_t2(tmp, t1_enc, name='t2'):
    """Produce a tiny 1-step T2 checkpoint on top of a frozen T1.
    Returns the t2 run dir. branch+cap=8 → multi-instruction chunks."""
    run_dir = str(Path(tmp) / name)
    data = gen(n_batches=2, batch_size=8, rule='branch+cap=8', twins=1,
               n_states=4)
    out, err, rc = run([
        PYTHON, 'scripts/train_t2_encoder.py', '--t1-model', str(t1_enc),
        '--n-steps', '1', '--log-every', '1', '--out-dir', run_dir,
        '--d-out', '32', '--d-event', '16',
        '--n-anchor-states', '4', '--warmup-steps', '0', '--no-compile'],
        stdin=data, timeout=180)
    assert rc == 0, f'T2 train failed: {err}'
    assert (Path(run_dir) / 't2.pt').exists(), f'no t2.pt in {run_dir}'
    return Path(run_dir)


class TestTraining:
    def test_encoder_train_smoke(self):
        """gen_batches | train_encoder smoke (single step). T1 now runs the
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


class TestStreamFmt:
    """_streamfmt header validation + format detection rejection paths.
    The RVT header is 5 fields: (B, max_tokens, max_n_instrs, n_anchors, OB)."""

    @staticmethod
    def _prefix():
        sys.path.insert(0, str(ROOT))
        from datagen import RVT_FORMAT
        return RVT_FORMAT._prefix_struct.pack(
            RVT_FORMAT.magic, RVT_FORMAT.version, RVT_FORMAT.dtype_chars)

    @staticmethod
    def _batch_bytes(B, max_tokens, max_n_instrs, n_anchors, OB=0,
                     magic=None, version=None):
        sys.path.insert(0, str(ROOT))
        from datagen import RVT_FORMAT
        m = RVT_FORMAT.magic if magic is None else magic
        v = RVT_FORMAT.version if version is None else version
        prefix = RVT_FORMAT._prefix_struct.pack(
            m, v, RVT_FORMAT.dtype_chars)
        header = RVT_FORMAT.batch_header.pack(
            B, max_tokens, max_n_instrs, n_anchors, OB)
        return prefix + header

    def test_validate_rejects_out_of_bounds_header(self):
        sys.path.insert(0, str(ROOT))
        from scripts._streamfmt import _validate_rvt, _MAX_B
        # B == 0 (must be > 0).
        with pytest.raises(ValueError, match='Invalid RVT header'):
            _validate_rvt(self._batch_bytes(0, 32, 1, 4))
        # B > _MAX_B.
        with pytest.raises(ValueError, match='Invalid RVT header'):
            _validate_rvt(self._batch_bytes(_MAX_B + 1, 32, 1, 4))
        # n_anchors > 256.
        with pytest.raises(ValueError, match='Invalid RVT header'):
            _validate_rvt(self._batch_bytes(4, 32, 1, 300))

    def test_validate_accepts_good_header(self):
        sys.path.insert(0, str(ROOT))
        from scripts._streamfmt import _validate_rvt
        vals = _validate_rvt(self._batch_bytes(4, 32, 1, 4, OB=4))
        assert vals == (4, 32, 1, 4, 4)

    def test_detect_format_unknown_magic(self):
        import io
        sys.path.insert(0, str(ROOT))
        from scripts._streamfmt import detect_format
        buf = b'XYZ\x00' + self._batch_bytes(4, 32, 1, 4)[4:]
        f = io.BufferedReader(io.BytesIO(buf))
        with pytest.raises(ValueError, match='Unknown format'):
            detect_format(f)

    def test_detect_format_wrong_version(self):
        import io
        sys.path.insert(0, str(ROOT))
        from scripts._streamfmt import detect_format
        buf = self._batch_bytes(4, 32, 1, 4, 4, version=99)
        f = io.BufferedReader(io.BytesIO(buf))
        with pytest.raises(ValueError, match='version'):
            detect_format(f)

    def test_detect_format_peeks(self):
        """detect_format must PEEK, not consume: after detection the next
        read still yields batch 0 byte-for-byte."""
        import io
        sys.path.insert(0, str(ROOT))
        from scripts._streamfmt import detect_format
        from datagen import RVT_FORMAT
        data = gen(n_batches=2, batch_size=4, twins=0, n_states=4,
                   rule='cap=2')
        f = io.BufferedReader(io.BytesIO(data))
        fmt = detect_format(f)
        assert fmt.name == 'rvt'
        after_detect = fmt.read_bytes(f)
        # Same bytes a fresh reader would see for batch 0.
        g = io.BufferedReader(io.BytesIO(data))
        fresh = RVT_FORMAT.read_batch_bytes(g)
        assert after_detect == fresh, 'detect_format consumed the header'
