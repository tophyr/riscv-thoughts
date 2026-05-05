"""Integration tests for the batch pipeline scripts.

Runs gen_batches, batch_slice, batch_cat, batch_repeat, batch_shuffle,
and mux_batches via subprocess. All tools share the single RVT format.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable


def run(cmd, stdin=None, timeout=60):
    result = subprocess.run(
        cmd, input=stdin, capture_output=True, timeout=timeout,
        cwd=str(ROOT))
    return result.stdout, result.stderr.decode(), result.returncode


def gen(n_batches=2, batch_size=8, twins=1, partners=2, n_states=4,
        rule='cap=2', seed=42):
    out, err, rc = run([
        PYTHON, 'scripts/gen_batches.py',
        '-n', str(n_batches),
        '--batch-size', str(batch_size),
        '--rule', rule,
        '--twins', str(twins),
        '--partners', str(partners),
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


class TestGenBatches:
    def test_emits_valid_rvt(self):
        data = gen()
        info = slice_info(data)
        assert 'Format:       rvt' in info
        assert 'Batches:      2' in info

    def test_no_pairs_mode(self):
        out, err, rc = run([
            PYTHON, 'scripts/gen_batches.py',
            '-n', '1', '--batch-size', '4',
            '--rule', 'single', '--twins', '0', '--partners', '0',
            '--n-states', '4'])
        assert rc == 0, f'no-pairs gen failed: {err}'
        assert len(out) > 0


class TestBatchSlice:
    def test_info(self):
        data = gen(n_batches=5)
        info = slice_info(data)
        assert 'Batches:      5' in info

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


class TestBatchCat:
    def test_concatenates(self):
        with tempfile.NamedTemporaryFile(suffix='.rvt') as f1, \
             tempfile.NamedTemporaryFile(suffix='.rvt') as f2:
            f1.write(gen(n_batches=2, seed=1))
            f1.flush()
            f2.write(gen(n_batches=3, seed=2))
            f2.flush()
            out, err, rc = run([PYTHON, 'scripts/batch_cat.py',
                                f1.name, f2.name])
            assert rc == 0, err
            assert 'Batches:      5' in slice_info(out)


class TestBatchRepeat:
    def test_epochs(self):
        with tempfile.NamedTemporaryFile(suffix='.rvt') as f:
            f.write(gen(n_batches=2))
            f.flush()
            out, _, rc = run([
                PYTHON, 'scripts/batch_repeat.py', '--epochs', '3', f.name])
            assert rc == 0
            assert 'Batches:      6' in slice_info(out)


class TestBatchShuffle:
    def test_preserves_count(self):
        with tempfile.NamedTemporaryFile(suffix='.rvt') as f:
            f.write(gen(n_batches=5))
            f.flush()
            out, _, rc = run([PYTHON, 'scripts/batch_shuffle.py', f.name])
            assert rc == 0
            assert 'Batches:      5' in slice_info(out)

    def test_buffered_from_stdin(self):
        data = gen(n_batches=3)
        out, _, rc = run([PYTHON, 'scripts/batch_shuffle.py'], stdin=data)
        assert rc == 0
        assert 'Batches:      3' in slice_info(out)


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
            '--gen-count', '2', '--n-batches', '2',
            '--batch-size', '8', '--rule', 'cap=2',
            '--twins', '1', '--partners', '2', '--n-states', '4'],
            timeout=120)
        assert rc == 0, f'mux spawned failed: {err}'
        assert 'Batches:      4' in slice_info(out)


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
        assert 'Unknown format' in err or 'Missing' in err, err

    def test_emits_periodic_log(self):
        """Live pipe with small --log-every-sec should emit at least
        one periodic status line. gen_batches' behavioral_distance gives
        enough wall time to fire the timer."""
        gen_proc = __import__('subprocess').Popen([
            PYTHON, 'scripts/gen_batches.py',
            '--rule', 'cap=2', '--batch-size', '32',
            '--twins', '2', '--partners', '4', '--n-states', '4',
            '-n', '20', '--seed', '42'],
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
        assert 'Final: 20 batches' in err, err


class TestTraining:
    def test_encoder_train_smoke(self):
        """gen_batches | train_encoder smoke (single step)."""
        data = gen(n_batches=2, batch_size=8)
        out, err, rc = run([
            PYTHON, 'scripts/train_encoder.py',
            '--n-steps', '1', '--log-every', '1', '--no-save',
            '--d-model', '32', '--n-heads', '2', '--n-layers', '1',
            '--d-out', '32', '--max-window', '32'],
            stdin=data, timeout=120)
        assert rc == 0, f'encoder training failed: {err}'
