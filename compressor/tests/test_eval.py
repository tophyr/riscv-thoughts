"""Smoke tests for compressor.eval diagnostic functions.

Runs each diagnostic against an untrained model on a small fixture, verifying
it executes end-to-end and returns the documented keys/shapes (no metric-value
assertions — an untrained model means nothing about the numbers). The
equivariant measures (equivariance_error / tag_invariance / binding accuracy /
gvn_collapse) are exercised end-to-end by scripts/tests/test_pipeline.py via
the scripts/eval.py CLI.
"""

import numpy as np
import pytest
import torch

from compressor.model import T1Compressor, Decoder
from compressor.eval import decoder_accuracy
from datagen.batch import generate_chunks, collect_into_batches
from datagen.compare import make_anchor_states
from datagen.generate import single
from tokenizer import VOCAB_SIZE


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    return T1Compressor(vocab_size=VOCAB_SIZE, d_model=32, n_heads=2,
                        n_layers=1, max_window=72, d_out=32, d_event=16)


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
        batch_size=8, twins=1,
        anchor_states=anchors, rng=rng,
        max_chunk_len=1,         # single-instr T1 corpus (canonical out_regs)
    )
    return [next(gen) for _ in range(2)]


def test_decoder_accuracy(encoder, decoder, batches):
    out = decoder_accuracy(encoder, decoder, iter(batches), device='cpu',
                           max_batches=2)
    assert set(out) == {'tok_acc', 'instr_acc', 'n_tokens',
                        'n_instrs', 'n_batches'}
    assert out['n_batches'] == 2
    assert out['n_instrs'] > 0
    assert 0.0 <= out['tok_acc'] <= 1.0
    assert 0.0 <= out['instr_acc'] <= 1.0
