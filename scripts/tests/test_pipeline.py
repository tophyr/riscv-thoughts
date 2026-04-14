"""Integration tests for the batch pipeline scripts.

Runs gen_instr_batches, gen_seq_batches, batch_slice, batch_cat,
batch_repeat, batch_shuffle, and mux_batches via subprocess.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable


def run(cmd, stdin=None, timeout=30):
    result = subprocess.run(
        cmd, input=stdin, capture_output=True, timeout=timeout,
        cwd=str(ROOT))
    return result.stdout, result.stderr.decode(), result.returncode


def gen_instr(n_batches=3, batch_size=8, seed=42):
    out, err, rc = run([
        PYTHON, 'scripts/gen_instr_batches.py',
        '--n-batches', str(n_batches),
        '--batch-size', str(batch_size),
        '--seed', str(seed)])
    assert rc == 0, f'gen_instr_batches failed: {err}'
    assert len(out) > 0
    return out


def gen_seqs(n_batches=3, batch_size=8, seed=42):
    out, err, rc = run([
        PYTHON, 'scripts/gen_seq_batches.py',
        '--n-batches', str(n_batches),
        '--batch-size', str(batch_size),
        '--seed', str(seed)])
    assert rc == 0, f'gen_seq_batches failed: {err}'
    assert len(out) > 0
    return out


def slice_info(data):
    _, err, rc = run(
        [PYTHON, 'scripts/batch_slice.py', '--info'],
        stdin=data)
    assert rc == 0, f'batch_slice --info failed: {err}'
    return err


class TestGenBatches:
    def test_generates_valid_stream(self):
        data = gen_instr()
        info = slice_info(data)
        assert 'Format:       rvb' in info
        assert 'Batches:      3' in info
        assert 'Batch size:   8' in info


class TestGenSeqBatches:
    def test_generates_valid_stream(self):
        data = gen_seqs()
        info = slice_info(data)
        assert 'Format:       rvs' in info
        assert 'Batches:      3' in info


class TestBatchSlice:
    def test_info(self):
        data = gen_instr(n_batches=5)
        info = slice_info(data)
        assert 'Items:        40' in info

    def test_count(self):
        data = gen_instr(n_batches=5)
        out, err, rc = run(
            [PYTHON, 'scripts/batch_slice.py', '--count', '2'],
            stdin=data)
        assert rc == 0
        info = slice_info(out)
        assert 'Batches:      2' in info

    def test_skip(self):
        data = gen_instr(n_batches=5)
        out, err, rc = run(
            [PYTHON, 'scripts/batch_slice.py', '--skip', '3'],
            stdin=data)
        assert rc == 0
        info = slice_info(out)
        assert 'Batches:      2' in info

    def test_tail(self):
        with tempfile.NamedTemporaryFile(suffix='.bin') as f:
            f.write(gen_instr(n_batches=5))
            f.flush()
            out, err, rc = run(
                [PYTHON, 'scripts/batch_slice.py', '--tail', '2', f.name])
            assert rc == 0
            info = slice_info(out)
            assert 'Batches:      2' in info

    def test_works_with_rvs(self):
        data = gen_seqs(n_batches=3)
        info = slice_info(data)
        assert 'Format:       rvs' in info


class TestBatchCat:
    def test_concatenates(self):
        with tempfile.NamedTemporaryFile(suffix='.bin') as f1, \
             tempfile.NamedTemporaryFile(suffix='.bin') as f2:
            f1.write(gen_instr(n_batches=2, seed=1))
            f1.flush()
            f2.write(gen_instr(n_batches=3, seed=2))
            f2.flush()
            out, err, rc = run(
                [PYTHON, 'scripts/batch_cat.py', f1.name, f2.name])
            assert rc == 0
            info = slice_info(out)
            assert 'Batches:      5' in info

    def test_rejects_mixed_formats(self):
        with tempfile.NamedTemporaryFile(suffix='.bin') as f1, \
             tempfile.NamedTemporaryFile(suffix='.bin') as f2:
            f1.write(gen_instr(n_batches=1))
            f1.flush()
            f2.write(gen_seqs(n_batches=1))
            f2.flush()
            _, err, rc = run(
                [PYTHON, 'scripts/batch_cat.py', f1.name, f2.name])
            assert rc != 0
            assert 'format' in err.lower() or 'ERROR' in err


class TestBatchRepeat:
    def test_repeats(self):
        with tempfile.NamedTemporaryFile(suffix='.bin') as f:
            f.write(gen_instr(n_batches=2))
            f.flush()
            out, err, rc = run(
                [PYTHON, 'scripts/batch_repeat.py', '--epochs', '3', f.name])
            assert rc == 0
            info = slice_info(out)
            assert 'Batches:      6' in info


class TestBatchShuffle:
    def test_shuffles_preserving_count(self):
        with tempfile.NamedTemporaryFile(suffix='.bin') as f:
            f.write(gen_instr(n_batches=5))
            f.flush()
            out, err, rc = run(
                [PYTHON, 'scripts/batch_shuffle.py', f.name])
            assert rc == 0
            info = slice_info(out)
            assert 'Batches:      5' in info

    def test_buffered_from_stdin(self):
        data = gen_instr(n_batches=3)
        out, err, rc = run(
            [PYTHON, 'scripts/batch_shuffle.py'],
            stdin=data)
        assert rc == 0
        info = slice_info(out)
        assert 'Batches:      3' in info


class TestMuxBatches:
    def test_mux_file_inputs(self):
        with tempfile.NamedTemporaryFile(suffix='.bin') as f1, \
             tempfile.NamedTemporaryFile(suffix='.bin') as f2:
            f1.write(gen_instr(n_batches=2, seed=1))
            f1.flush()
            f2.write(gen_instr(n_batches=3, seed=2))
            f2.flush()
            out, err, rc = run(
                [PYTHON, 'scripts/mux_batches.py', f1.name, f2.name])
            assert rc == 0
            info = slice_info(out)
            assert 'Batches:      5' in info

    def test_mux_spawned_instr_workers(self):
        out, err, rc = run([
            PYTHON, 'scripts/mux_batches.py',
            '--gen', 'instr', '--gen-count', '2',
            '--n-batches', '2', '--batch-size', '4'],
            timeout=60)
        assert rc == 0, f'mux instr failed: {err}'
        info = slice_info(out)
        assert 'Batches:      4' in info

    def test_mux_spawned_seq_workers(self):
        out, err, rc = run([
            PYTHON, 'scripts/mux_batches.py',
            '--gen', 'seq', '--gen-count', '2',
            '--n-batches', '2', '--batch-size', '4'],
            timeout=60)
        assert rc == 0, f'mux seq failed: {err}'
        info = slice_info(out)
        assert 'Batches:      4' in info

    def test_mux_instr_with_config(self):
        """Mux passes --config through to instr generator workers."""
        out, err, rc = run([
            PYTHON, 'scripts/mux_batches.py',
            '--gen', 'instr', '--gen-count', '2',
            '--n-batches', '2', '--batch-size', '4',
            '--config', 'configs/instr_branch_heavy.json'],
            timeout=60)
        assert rc == 0, f'mux instr with config failed: {err}'
        info = slice_info(out)
        assert 'Batches:      4' in info

    def test_mux_reads_nonseekable_fifo(self):
        """Mux reads from bash <(...) process substitution (non-seekable FIFO)."""
        cmd = (
            f'{PYTHON} scripts/mux_batches.py '
            f'<({PYTHON} scripts/gen_instr_batches.py '
            f'--n-batches 3 --batch-size 8)'
        )
        result = subprocess.run(
            ['bash', '-c', cmd],
            capture_output=True, cwd=str(ROOT), timeout=30)
        assert result.returncode == 0, \
            f'mux via FIFO failed: {result.stderr.decode()}'
        info = slice_info(result.stdout)
        assert 'Batches:      3' in info


class TestTraining:
    def test_rvb_generation_and_slice(self):
        """gen_instr_batches produces valid RVB that batch_slice can read."""
        out, err, rc = run([
            PYTHON, 'scripts/gen_instr_batches.py',
            '--n-batches', '3', '--batch-size', '16', '--n-inputs', '32'])
        assert rc == 0, f'gen_instr_batches failed: {err}'
        info = slice_info(out)
        assert 'Format:       rvb' in info
        assert 'Batches:      3' in info
        assert 'Batch size:   16' in info

    def test_rvb_batch_training(self):
        """gen_instr_batches | train_compressor --mode instr works."""
        gen_out = gen_instr(n_batches=10, batch_size=32)
        out, err, rc = run(
            [PYTHON, 'scripts/train_compressor.py',
             '--mode', 'instr',
             '--n-steps', '5', '--log-every', '5', '--no-save'],
            stdin=gen_out, timeout=120)
        assert rc == 0, f'instr training failed: {err}'
        assert 'Done:' in err or 'Done:' in out.decode()
