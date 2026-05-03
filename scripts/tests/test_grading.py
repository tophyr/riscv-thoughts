"""End-to-end "from scratch" grading suite.

Codifies the manual grading run as five subprocess phases:

  1. gen_batches — produce a small RVT corpus
  2. batch_slice --info — verify the corpus
  3. batch_repeat | train_encoder — train an encoder from random init
  4. batch_repeat | train_decoder — train a decoder on the frozen encoder
  5. pytest test_acceptance.py — invoke the model-acceptance suite

Each phase writes artifacts to a session-scoped tmpdir; later phases
read what earlier phases produced. If a phase fails, all subsequent
phases will also fail with a clear "missing artifact" assertion.

The phase-5 acceptance suite WILL fail its metric thresholds for the
toy run (500 steps, tiny model). The grading suite only verifies that
the diagnostic harness ran end-to-end and produced a metrics table —
not that the toy run met production thresholds.

Slow (~3-5 minutes on GPU). Skipped by default. Run with:
    pytest scripts/tests/test_grading.py --run-grading -v
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable


# Hparams shared across phases. Small for speed.
N_BATCHES   = 200
BATCH_SIZE  = 32
TWINS       = 2
PARTNERS    = 4
N_STATES    = 4
RULE        = 'cap=2'
INJECT_INVALID = 0.1
GEN_SEED    = 1

D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 2
D_OUT      = 32
MAX_WINDOW = 32

N_STEPS    = 500
LOG_EVERY  = 100


@pytest.fixture(scope='session')
def workspace(tmp_path_factory):
    """One tmpdir for the whole grading session."""
    return tmp_path_factory.mktemp('grading')


def _shell(cmd, *, timeout=600):
    """Run a bash pipeline. Returns (stdout, stderr_str, returncode)."""
    r = subprocess.run(
        ['bash', '-c', cmd], capture_output=True,
        cwd=str(ROOT), timeout=timeout)
    return r.stdout, r.stderr.decode(), r.returncode


@pytest.mark.grading
class TestEndToEndGrading:
    """Five sequential phases. Tests must run in declaration order
    (default pytest behavior — do not invoke with pytest-randomly)."""

    def test_phase1_generate_corpus(self, workspace):
        """`gen_batches.py ...` → corpus.rvt."""
        corpus = workspace / 'corpus.rvt'
        cmd = (
            f'{PYTHON} scripts/gen_batches.py '
            f'--rule {RULE} '
            f'--batch-size {BATCH_SIZE} '
            f'--twins {TWINS} '
            f'--partners {PARTNERS} '
            f'--n-states {N_STATES} '
            f'--inject-invalid {INJECT_INVALID} '
            f'-n {N_BATCHES} '
            f'--seed {GEN_SEED} '
            f'> {corpus}'
        )
        _, err, rc = _shell(cmd, timeout=300)
        assert rc == 0, f'gen_batches failed:\n{err}'
        assert corpus.exists()
        assert corpus.stat().st_size > 0

    def test_phase2_inspect_corpus(self, workspace):
        """`batch_slice.py --info corpus.rvt` reports format=rvt + B=N_BATCHES."""
        corpus = workspace / 'corpus.rvt'
        assert corpus.exists(), 'phase 1 must run first'
        out, err, rc = _shell(
            f'{PYTHON} scripts/batch_slice.py --info {corpus}',
            timeout=30)
        assert rc == 0, err
        info = err  # batch_slice writes info to stderr
        assert 'Format:       rvt' in info, info
        assert f'Batches:      {N_BATCHES}' in info, info

    def test_phase3_train_encoder(self, workspace):
        """`batch_repeat | train_encoder` → encoder run dir with checkpoints."""
        corpus = workspace / 'corpus.rvt'
        run_dir = workspace / 'encoder'
        assert corpus.exists(), 'phase 1 must run first'
        cmd = (
            f'{PYTHON} scripts/batch_repeat.py --forever {corpus} '
            f'2>/dev/null | '
            f'{PYTHON} scripts/train_encoder.py '
            f'--d-model {D_MODEL} --n-heads {N_HEADS} --n-layers {N_LAYERS} '
            f'--d-out {D_OUT} --max-window {MAX_WINDOW} '
            f'--n-steps {N_STEPS} --log-every {LOG_EVERY} '
            f'--pair-weight 1.0 --valid-weight 0.5 '
            f'--save {run_dir}'
        )
        out, err, rc = _shell(cmd, timeout=600)
        assert rc == 0, f'train_encoder failed:\n{err}'
        assert (run_dir / 'encoder.pt').exists()
        assert (run_dir / 'hparams.json').exists()
        assert (run_dir / 'losses.json').exists()
        # The training output should reach the documented final line.
        assert b'Done:' in out
        assert b'Saved to' in out

    def test_phase4_train_decoder(self, workspace):
        """`batch_repeat | train_decoder --model encoder.pt` → decoder.pt."""
        corpus = workspace / 'corpus.rvt'
        encoder = workspace / 'encoder' / 'encoder.pt'
        decoder = workspace / 'decoder.pt'
        assert encoder.exists(), 'phase 3 must run first'
        cmd = (
            f'{PYTHON} scripts/batch_repeat.py --forever {corpus} '
            f'2>/dev/null | '
            f'{PYTHON} scripts/train_decoder.py --model {encoder} '
            f'--d-model {D_MODEL} --n-heads {N_HEADS} --n-layers {N_LAYERS} '
            f'--d-out {D_OUT} --max-window {MAX_WINDOW} '
            f'--dec-d-model {D_MODEL} --dec-n-heads {N_HEADS} '
            f'--dec-n-layers {N_LAYERS} '
            f'--n-steps {N_STEPS} --log-every {LOG_EVERY} '
            f'--save-decoder {decoder}'
        )
        out, err, rc = _shell(cmd, timeout=600)
        assert rc == 0, f'train_decoder failed:\n{err}'
        assert decoder.exists()
        assert decoder.with_suffix('.hparams.json').exists()
        assert b'Saved decoder' in out

    def test_phase5_acceptance_suite(self, workspace):
        """Invoke the acceptance suite. Verify the diagnostic table is
        produced for all four metrics. Threshold pass/fail is irrelevant
        for this toy run."""
        encoder = workspace / 'encoder' / 'encoder.pt'
        decoder = workspace / 'decoder.pt'
        assert encoder.exists() and decoder.exists(), \
            'phases 3 and 4 must run first'
        cmd = (
            f'{PYTHON} -m pytest compressor/tests/test_acceptance.py '
            f'--encoder {encoder} --decoder {decoder} '
            f'--tb=no --no-header -q -rN'
        )
        out, err, rc = _shell(cmd, timeout=300)
        # rc != 0 is expected (toy training will fail thresholds); we
        # only require the suite ran and emitted its metrics table.
        text = out.decode() + err
        assert 'model evaluation' in text, text
        assert 'pair_distance_correlation' in text, text
        assert 'equivalence_collapse' in text, text
        assert 'validity_separation' in text, text
        assert 'decoder_accuracy' in text, text
