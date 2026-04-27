"""Tests for the trainer-side chunker (compressor/chunker.py).

The boundary detection / augmentation tests live in
datagen/tests/test_chunkgen.py. This file covers the small bridge
that runs T1 encoder on an RVC ChunkBatch's per-instruction tokens
to produce a T2Batch ready for training.
"""

import numpy as np
import pytest
import torch

from datagen.seqgen import produce_sequence_batch
from datagen.chunkgen import chunk_rvs, augment_chunkbatch_with_invalid
from compressor.model import T1Compressor
from compressor.chunker import (
    encode_chunkbatch, concat_t2_batches, T2Batch,
)
from tokenizer import VOCAB_SIZE


@pytest.fixture(scope='module')
def t1_encoder():
    torch.manual_seed(0)
    m = T1Compressor(VOCAB_SIZE, d_model=32, n_heads=2, n_layers=1,
                     d_out=8, max_window=32)
    m.eval()
    return m


def test_encode_shape(t1_encoder):
    rng = np.random.default_rng(42)
    rvs = produce_sequence_batch(8, n_inputs=2, max_block_len=8, rng=rng)
    cb = chunk_rvs(rvs, max_chunk_len=16)
    t2 = encode_chunkbatch(cb, t1_encoder)
    assert t2.chunk_emissions.shape == (
        cb.token_ids.shape[0], cb.token_ids.shape[1], 8)
    assert t2.chunk_lens.shape == (cb.token_ids.shape[0],)
    assert t2.valid_mask.shape == (cb.token_ids.shape[0],)


def test_encode_zeros_beyond_chunk_len(t1_encoder):
    """Positions beyond chunk_lens[c] are zeroed."""
    rng = np.random.default_rng(43)
    rvs = produce_sequence_batch(8, n_inputs=2, max_block_len=8, rng=rng)
    cb = chunk_rvs(rvs, max_chunk_len=16)
    t2 = encode_chunkbatch(cb, t1_encoder)
    for c in range(t2.chunk_emissions.shape[0]):
        L = int(t2.chunk_lens[c])
        beyond = t2.chunk_emissions[c, L:]
        assert (beyond == 0).all()


def test_encode_preserves_metadata(t1_encoder):
    rng = np.random.default_rng(44)
    rvs = produce_sequence_batch(4, n_inputs=2, max_block_len=8, rng=rng)
    cb = chunk_rvs(rvs, max_chunk_len=16)
    t2 = encode_chunkbatch(cb, t1_encoder)
    assert np.array_equal(
        t2.chunk_lens.cpu().numpy(), cb.chunk_lens.astype(np.int64))
    assert np.array_equal(t2.valid_mask.cpu().numpy(), cb.valid_mask)
    assert np.array_equal(t2.chunk_type.cpu().numpy(), cb.chunk_type)
    assert np.array_equal(t2.reg_delta.cpu().numpy(), cb.reg_delta)


def test_encode_with_augmented(t1_encoder):
    """Augmented chunks (with invalids) encode cleanly."""
    rng = np.random.default_rng(45)
    rvs = produce_sequence_batch(8, n_inputs=2, max_block_len=8, rng=rng)
    cb = chunk_rvs(rvs, max_chunk_len=16)
    aug = augment_chunkbatch_with_invalid(
        cb, invalidity_rate=0.3, storage_max_chunk_len=24,
        rng=np.random.default_rng(0))
    t2 = encode_chunkbatch(aug, t1_encoder)
    assert t2.chunk_emissions.shape[0] == aug.token_ids.shape[0]
    assert t2.chunk_emissions.shape[1] == 24
    n_valid = int(t2.valid_mask.sum())
    assert n_valid == int(aug.valid_mask.sum())


def test_concat_two_t2_batches(t1_encoder):
    """concat_t2_batches stacks chunks correctly."""
    rng = np.random.default_rng(46)
    rvs1 = produce_sequence_batch(4, n_inputs=2, max_block_len=8, rng=rng)
    rvs2 = produce_sequence_batch(4, n_inputs=2, max_block_len=8, rng=rng)
    cb1 = chunk_rvs(rvs1, max_chunk_len=16)
    cb2 = chunk_rvs(rvs2, max_chunk_len=16)
    t2_a = encode_chunkbatch(cb1, t1_encoder)
    t2_b = encode_chunkbatch(cb2, t1_encoder)
    combined = concat_t2_batches([t2_a, t2_b])
    assert combined.chunk_emissions.shape[0] == (
        t2_a.chunk_emissions.shape[0] + t2_b.chunk_emissions.shape[0])
    assert combined.chunk_lens.shape[0] == combined.chunk_emissions.shape[0]
