"""Smoke tests for compressor.eval diagnostic functions.

These run each diagnostic against an untrained model on a small fixture.
They verify:
  - Each function executes end-to-end without crashing.
  - The returned dict has the documented keys/shapes.

They do NOT assert on metric values: an untrained encoder won't satisfy
the Step 1 / Step 2 acceptance thresholds. Acceptance assertions belong
in test_acceptance.py with --encoder/--decoder fixtures.
"""

import numpy as np
import pytest
import torch

from compressor.model import T1Compressor, Decoder
from compressor.eval import (
    pair_distance_correlation, equivalence_collapse,
    validity_separation, decoder_accuracy,
)
from datagen.batch import (
    Chunk, RVT_FORMAT, generate_chunks, collect_into_batches,
    _make_valid_chunk,
)
from datagen.compare import make_anchor_states
from datagen.generate import single, random_instruction
from tokenizer import VOCAB_SIZE


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    return T1Compressor(VOCAB_SIZE, d_model=32, n_heads=2, n_layers=1,
                        d_out=32, max_window=72)


@pytest.fixture
def decoder():
    torch.manual_seed(0)
    return Decoder(VOCAB_SIZE, d_model=32, n_heads=2, n_layers=1, d_emb=32)


@pytest.fixture
def batches():
    rng = np.random.default_rng(0)
    anchors = make_anchor_states(4, seed=0)
    gen = collect_into_batches(
        generate_chunks(single(), rng),
        batch_size=8, twins=1, partners=2,
        anchor_states=anchors, rng=rng,
    )
    return [next(gen) for _ in range(2)]


def test_pair_distance_correlation(encoder, batches):
    out = pair_distance_correlation(encoder, iter(batches), device='cpu',
                                    max_batches=2)
    assert set(out) == {'pearson', 'spearman', 'n_pairs'}
    assert out['n_pairs'] > 0


def test_equivalence_collapse(encoder):
    out = equivalence_collapse(encoder, device='cpu', n_samples=2, seed=0)
    assert 'classes' in out
    assert 'mean_ratio' in out
    has_ratio = any(c['ratio'] is not None for c in out['classes'].values())
    assert has_ratio


def test_validity_separation(encoder):
    out = validity_separation(
        encoder, device='cpu', n_per_class=20, max_window=32, seed=0,
        batch_size=20)
    assert set(out) == {'class_stats', 'magnitude_acc',
                        'majority_baseline', 'magnitude_threshold'}
    assert set(out['class_stats']) == {
        'valid', 'partial', 'spanning', 'multi', 'bogus'}
    for stats in out['class_stats'].values():
        assert stats['n'] == 20
        assert stats['norm_mean'] >= 0
    assert 0.0 <= out['magnitude_acc'] <= 1.0
    assert 0.5 <= out['majority_baseline'] <= 1.0


def test_decoder_accuracy(encoder, decoder, batches):
    out = decoder_accuracy(encoder, decoder, iter(batches), device='cpu',
                           max_batches=2)
    assert set(out) == {'tok_acc', 'instr_acc', 'n_tokens',
                        'n_instrs', 'n_batches'}
    assert out['n_batches'] == 2
    assert out['n_instrs'] > 0
    assert 0.0 <= out['tok_acc'] <= 1.0
    assert 0.0 <= out['instr_acc'] <= 1.0
