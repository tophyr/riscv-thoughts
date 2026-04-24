"""Regression tests for T1Compressor.forward() shift-reduce loop.

Checks structural invariants of the forward pass output:
- All returned lists have consistent iteration counts.
- Per-iteration tensors have shape (B, ...).
- emit_counts equals the number of emission_info entries per batch.
- Deterministic output given a fixed RNG seed.
"""

import numpy as np
import pytest
import torch

from compressor.model import T1Compressor
from tokenizer import VOCAB_SIZE


def _build_inputs(B=4, L=12, n_instr=3, device='cpu', seed=0):
    """Build a small synthetic batch for forward()."""
    rng = np.random.default_rng(seed)
    token_ids = torch.from_numpy(
        rng.integers(4, VOCAB_SIZE, size=(B, L)).astype(np.int64))
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    # Last 2 positions of each sequence are padding
    padding_mask[:, -2:] = True
    # Token-to-instruction index: split tokens roughly evenly
    tokens_per_instr = (L - 2) // n_instr
    idx = torch.zeros(B, L, dtype=torch.int32)
    for t in range(L - 2):
        idx[:, t] = min(t // tokens_per_instr, n_instr - 1)
    idx[:, -2:] = -1  # padding positions
    n_instructions = torch.full((B,), n_instr, dtype=torch.int32)
    return (token_ids.to(device), padding_mask.to(device),
            idx.to(device), n_instructions.to(device))


@pytest.fixture
def model():
    torch.manual_seed(42)
    return T1Compressor(VOCAB_SIZE, d_model=32, n_heads=2, n_layers=1,
                        d_out=16, max_iterations_per_token=3)


def test_forward_returns_six_things(model):
    inputs = _build_inputs()
    out = model(*inputs)
    assert len(out) == 9
    (iter_t1s, iter_window_tokens, iter_gate_log_probs,
     iter_gate_decisions, iter_gate_logits,
     iter_window_buf, iter_window_lens,
     emission_info, emit_counts) = out


def test_forward_shapes(model):
    B, L = 4, 12
    inputs = _build_inputs(B=B, L=L)
    (iter_t1s, iter_window_tokens, iter_gate_log_probs,
     iter_gate_decisions, iter_gate_logits,
     iter_window_buf, iter_window_lens,
     emission_info, emit_counts) = model(*inputs)

    n_iters = len(iter_t1s)
    assert n_iters > 0
    assert len(iter_window_tokens) == n_iters
    assert len(iter_gate_log_probs) == n_iters
    assert len(iter_gate_decisions) == n_iters
    assert len(iter_gate_logits) == n_iters

    for t1 in iter_t1s:
        assert t1.shape == (B, model.d_out)
    for lp in iter_gate_log_probs:
        assert lp.shape == (B, 3)
    for dec in iter_gate_decisions:
        assert dec.shape == (B, 3)
    for lg in iter_gate_logits:
        assert lg.shape == (B, 3)
    for wt in iter_window_tokens:
        assert isinstance(wt, list) and len(wt) == B
        for tokens in wt:
            assert isinstance(tokens, list)
    assert iter_window_buf.shape == (n_iters, B, model.runtime_max_window)
    assert iter_window_lens.shape == (n_iters, B)
    assert emit_counts.shape == (B,)


def test_emit_counts_match_emission_info(model):
    inputs = _build_inputs()
    (_, _, _, _, _, _, _, emission_info, emit_counts) = model(*inputs)

    per_batch = np.zeros(emit_counts.shape[0], dtype=np.int64)
    for info in emission_info:
        per_batch[info['batch_idx']] += 1
    assert np.array_equal(per_batch, emit_counts.cpu().numpy())


def test_emission_info_keys(model):
    inputs = _build_inputs()
    (_, _, _, _, _, _, _, emission_info, _) = model(*inputs)
    required = {'batch_idx', 'instr_start', 'instr_end',
                'target_tokens', 'window_size', 'has_complete',
                'iteration'}
    for info in emission_info:
        assert required.issubset(info.keys())
        assert info['window_size'] == len(info['target_tokens'])
        if info['has_complete']:
            assert info['instr_start'] >= 0
            assert info['instr_end'] >= info['instr_start']


def test_emit_counts_plausible(model):
    """Each sequence should emit at least once given enough iterations."""
    inputs = _build_inputs()
    (_, _, _, _, _, _, _, _, emit_counts) = model(*inputs)
    # With randomized gates and fallback logic, at least some emissions
    # should happen across the batch.
    assert emit_counts.sum().item() > 0


def test_deterministic_given_seed(model):
    torch.manual_seed(42)
    inputs = _build_inputs(seed=0)
    out1 = model(*inputs)

    torch.manual_seed(42)
    out2 = model(*inputs)

    # T1s and decisions should match exactly across runs with same seed.
    assert len(out1[0]) == len(out2[0])
    for a, b in zip(out1[0], out2[0]):
        torch.testing.assert_close(a, b)
    # emit_counts is the last element.
    assert out1[-1].tolist() == out2[-1].tolist()
