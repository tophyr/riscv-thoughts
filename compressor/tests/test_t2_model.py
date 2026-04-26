"""Tests for T2Compressor.

Verifies forward shape, ball geometry (no F.normalize), padding-mask
correctness (output independent of padding contents), and end-to-end
integration with the chunker.
"""

import numpy as np
import pytest
import torch

from compressor.t2_model import T2Compressor


@pytest.fixture
def model():
    torch.manual_seed(0)
    return T2Compressor(d_in=64, d_model=64, n_heads=2, n_layers=1,
                        d_out=32, max_chunk_len=16)


def test_forward_shape(model):
    B, T = 4, 8
    chunks = torch.randn(B, T, 64)
    lens = torch.tensor([T, T, T, T], dtype=torch.int64)
    out = model(chunks, lens)
    assert out.shape == (B, 32)


def test_no_normalize(model):
    """Output magnitudes are not constrained to unit (T2 lives in the ball)."""
    torch.manual_seed(1)
    B, T = 16, 8
    chunks = torch.randn(B, T, 64)
    lens = torch.full((B,), T, dtype=torch.int64)
    with torch.no_grad():
        out = model(chunks, lens)
    norms = out.norm(dim=-1)
    # Random init produces varied magnitudes; not all close to 1.0.
    assert norms.std() > 1e-4, 'magnitudes look pinned to a constant'


def test_padding_invariance(model):
    """Output depends only on positions within chunk_lens, not padding contents."""
    torch.manual_seed(2)
    B, T = 4, 8
    chunks = torch.randn(B, T, 64)
    lens = torch.tensor([5, 3, 7, 2], dtype=torch.int64)

    # Run with the original padding values.
    with torch.no_grad():
        out1 = model(chunks, lens)

    # Replace padding positions with garbage and re-run.
    chunks_garbage = chunks.clone()
    for b in range(B):
        L = int(lens[b])
        chunks_garbage[b, L:] = torch.randn(T - L, 64) * 1000.0

    with torch.no_grad():
        out2 = model(chunks_garbage, lens)

    torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)


def test_determinism(model):
    """Same inputs produce same output (deterministic)."""
    torch.manual_seed(3)
    B, T = 4, 8
    chunks = torch.randn(B, T, 64)
    lens = torch.tensor([5, 3, 7, 2], dtype=torch.int64)
    with torch.no_grad():
        out1 = model(chunks, lens)
        out2 = model(chunks, lens)
    torch.testing.assert_close(out1, out2)


def test_different_inputs_different_outputs(model):
    """Distinct inputs produce distinct outputs (probably)."""
    torch.manual_seed(4)
    B, T = 4, 8
    chunks_a = torch.randn(B, T, 64)
    chunks_b = torch.randn(B, T, 64)
    lens = torch.full((B,), T, dtype=torch.int64)
    with torch.no_grad():
        out_a = model(chunks_a, lens)
        out_b = model(chunks_b, lens)
    # Very unlikely to coincide.
    assert (out_a - out_b).norm() > 1e-3


def test_d_in_mismatch_raises(model):
    """Wrong d_in raises a clear error."""
    chunks = torch.randn(2, 4, 32)  # d_in=32, model expects 64
    lens = torch.tensor([4, 4], dtype=torch.int64)
    with pytest.raises(ValueError, match='d_in'):
        model(chunks, lens)


def test_seq_len_over_max_raises(model):
    """T > max_chunk_len raises."""
    chunks = torch.randn(2, 32, 64)  # T=32, model max=16
    lens = torch.tensor([32, 32], dtype=torch.int64)
    with pytest.raises(ValueError, match='max_chunk_len'):
        model(chunks, lens)


def test_zero_length_does_not_crash(model):
    """Defensive: zero-length chunks don't crash and produce zero output."""
    chunks = torch.randn(2, 4, 64)
    lens = torch.tensor([0, 3], dtype=torch.int64)
    with torch.no_grad():
        out = model(chunks, lens)
    # Row 0 (zero-length) should be zeroed.
    assert (out[0] == 0).all()
    # Row 1 (length 3) should be nonzero in general.
    assert (out[1] != 0).any()


def test_end_to_end_with_chunker():
    """Smoke test: chunker output flows into T2Compressor cleanly."""
    from datagen.seqgen import produce_sequence_batch
    from compressor.model import T1Compressor
    from compressor.chunker import chunk_rvs_to_t2_batch
    from tokenizer import VOCAB_SIZE

    rng = np.random.default_rng(42)
    rvs = produce_sequence_batch(4, n_inputs=2, max_block_len=8, rng=rng)
    torch.manual_seed(0)
    t1 = T1Compressor(VOCAB_SIZE, d_model=32, n_heads=2, n_layers=1,
                      d_out=8, max_window=32)
    t1.eval()
    t2_batch = chunk_rvs_to_t2_batch(rvs, t1, max_chunk_len=16)

    torch.manual_seed(1)
    t2_model = T2Compressor(d_in=8, d_model=32, n_heads=2, n_layers=1,
                            d_out=16, max_chunk_len=16)
    t2_model.eval()
    with torch.no_grad():
        t2_vecs = t2_model(t2_batch.chunk_emissions, t2_batch.chunk_lens)

    assert t2_vecs.shape == (t2_batch.chunk_emissions.shape[0], 16)
    # Magnitudes should be finite, nonzero on valid chunks.
    assert torch.isfinite(t2_vecs).all()
    norms = t2_vecs.norm(dim=-1)
    assert (norms > 0).all()
