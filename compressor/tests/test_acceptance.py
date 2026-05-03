"""Trained-model acceptance tests.

Each test asserts a Step 1 / Step 2 success criterion from the plan and
records the actual metric values via the `metrics` fixture so they show
up in the post-run table regardless of pass/fail.

Skipped by default. Run with:
    pytest compressor/tests/test_acceptance.py \\
        --encoder runs/<stamp>/encoder.pt \\
        [--decoder /path/to/decoder.pt]

Use --tb=no -q -rN for a clean report (just the metrics table).
"""

import numpy as np
import pytest

from compressor.eval import (
    pair_distance_correlation, equivalence_collapse,
    validity_separation, decoder_accuracy,
)
from datagen.batch import generate_chunks, collect_into_batches
from datagen.compare import make_anchor_states
from datagen.generate import length_cap


# Fixed eval seeds — deterministic data across runs so metrics are comparable.
_EVAL_SEED = 1001
_ANCHOR_SEED = 1002


@pytest.fixture(scope='session')
def eval_batches(device):
    """A small, fixed set of RVT batches for acceptance evaluation.
    Mirrors the training-time generation but with frozen seeds."""
    rng = np.random.default_rng(_EVAL_SEED)
    anchors = make_anchor_states(8, seed=_ANCHOR_SEED)
    gen = collect_into_batches(
        generate_chunks(length_cap(2), rng),
        batch_size=64, twins=2, partners=8,
        anchor_states=anchors, rng=rng,
    )
    return [next(gen) for _ in range(20)]


# ---------------------------------------------------------------------------
# Step 1 — encoder
# ---------------------------------------------------------------------------

@pytest.mark.acceptance
def test_pair_distance_correlation(encoder, eval_batches, device, metrics):
    """Pearson correlation between encoder pairwise distance and target
    distance over the batch's pair structure. Step 1 acceptance: > 0.85."""
    out = pair_distance_correlation(encoder, iter(eval_batches),
                                    device=device)
    metrics['pearson'] = out['pearson']
    metrics['spearman'] = out['spearman']
    metrics['n_pairs'] = out['n_pairs']
    assert out['pearson'] > 0.85, f'pearson={out["pearson"]:.3f}'


@pytest.mark.acceptance
def test_equivalence_collapse(encoder, device, metrics):
    """Per-MANIFEST-class intra/inter ratio on direction-normalized
    T1 vectors. Step 1 acceptance: every class ratio < 0.3."""
    out = equivalence_collapse(encoder, device=device, n_samples=50, seed=0)
    metrics['mean_ratio'] = out['mean_ratio']
    failing = {name: c['ratio'] for name, c in out['classes'].items()
               if c['ratio'] is not None and c['ratio'] >= 0.3}
    metrics['n_classes'] = len(out['classes'])
    metrics['n_failing'] = len(failing)
    if failing:
        worst = max(failing.items(), key=lambda kv: kv[1])
        metrics['worst'] = f'{worst[0]}({worst[1]:.2f})'
    assert not failing, f'{len(failing)} classes have ratio >= 0.3'


@pytest.mark.acceptance
def test_validity_separation(encoder, device, metrics):
    """‖T1‖ should separate valid windows from invalid ones. Acceptance:
    magnitude_acc beats the majority baseline by at least 0.10."""
    out = validity_separation(encoder, device=device, n_per_class=2000,
                              seed=0)
    metrics['magnitude_acc'] = out['magnitude_acc']
    metrics['baseline'] = out['majority_baseline']
    metrics['valid_norm'] = out['class_stats']['valid']['norm_mean']
    metrics['multi_norm'] = out['class_stats']['multi']['norm_mean']
    margin = out['magnitude_acc'] - out['majority_baseline']
    metrics['margin'] = margin
    assert margin >= 0.10, f'margin {margin:.3f} below 0.10 threshold'


# ---------------------------------------------------------------------------
# Step 2 — decoder
# ---------------------------------------------------------------------------

@pytest.mark.acceptance
@pytest.mark.needs_decoder
def test_decoder_accuracy(encoder, decoder, eval_batches, device, metrics):
    """Teacher-forced reconstruction. Step 2 acceptance: instr_acc > 0.95."""
    out = decoder_accuracy(encoder, decoder, iter(eval_batches),
                           device=device)
    metrics['tok_acc'] = out['tok_acc']
    metrics['instr_acc'] = out['instr_acc']
    metrics['n_instrs'] = out['n_instrs']
    assert out['instr_acc'] > 0.95, f'instr_acc={out["instr_acc"]:.3f}'
