"""Tests for module-level import ordering.

The emulator package and the tokenizer package have a circular
dependency (gpu_emulator needs tokenizer for token ID constants;
tokenizer needs emulator for Instruction and opcode types). The
cycle is broken by lazy-loading gpu_emulator via module-level
__getattr__ in emulator/__init__.py.

These tests run fresh Python subprocesses to verify the cycle
doesn't re-emerge. An in-process test wouldn't catch regressions,
because once any test loads the emulator package the imports are
cached and subsequent imports don't trigger the cycle.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run(code):
    """Run a Python snippet in a fresh subprocess."""
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT))
    return result


def test_import_tokenizer_first():
    """Importing tokenizer in a fresh process must not fail.

    This is the order that previously broke: tokenizer loads
    emulator, which in the old non-lazy init eagerly loaded
    gpu_emulator, which re-imported the half-loaded tokenizer.
    """
    result = _run('from tokenizer import VOCAB_SIZE, encode_instruction')
    assert result.returncode == 0, (
        f'tokenizer import failed:\n{result.stderr}')


def test_import_emulator_first():
    """Importing emulator CPU names in a fresh process must work."""
    result = _run('from emulator import run, Instruction, random_regs')
    assert result.returncode == 0, (
        f'emulator CPU import failed:\n{result.stderr}')


def test_import_gpu_emulator_via_package():
    """Importing GPU names via the emulator package must work.

    This triggers the lazy __getattr__ path in emulator/__init__.py.
    """
    result = _run(
        'from emulator import batch_execute, batch_parse_tokens, '
        'random_regs_gpu, instructions_to_tensors')
    assert result.returncode == 0, (
        f'emulator GPU import failed:\n{result.stderr}')


def test_import_gpu_emulator_direct():
    """Importing GPU names from emulator.gpu_emulator must work."""
    result = _run(
        'from emulator.gpu_emulator import batch_execute, '
        'batch_parse_tokens')
    assert result.returncode == 0, (
        f'direct gpu_emulator import failed:\n{result.stderr}')


def test_import_compressor_train_first():
    """Importing compressor.train in a fresh process must work.

    compressor.train pulls in tokenizer and emulator transitively.
    This is the import chain that originally revealed the bug.
    """
    result = _run('from compressor.train import load_checkpoint')
    assert result.returncode == 0, (
        f'compressor.train import failed:\n{result.stderr}')
