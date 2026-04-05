"""Unit tests for the T0→T1 compressor."""

import numpy as np
import torch
import pytest

from emulator import Instruction
from tokenizer import VOCAB_SIZE
from compressor import (
    T1Compressor,
    tokenize_batch,
    compute_exec_distance_matrix,
    correlation_loss,
    random_instruction,
)


class TestTokenizeBatch:
    def test_shapes(self):
        instrs = [
            Instruction('ADD', 5, 3, 7),       # 4 tokens
            Instruction('ADDI', 5, 3, 42),      # 6 tokens
            Instruction('ADDI', 5, 3, -1),      # 7 tokens (with NEG)
        ]
        ids, mask = tokenize_batch(instrs)
        assert ids.shape == (3, 7)  # padded to longest
        assert mask.shape == (3, 7)

    def test_padding_mask(self):
        instrs = [
            Instruction('ADD', 5, 3, 7),       # 4 tokens
            Instruction('ADDI', 5, 3, 42),      # 6 tokens
        ]
        ids, mask = tokenize_batch(instrs)
        # First instruction: 4 real + 2 padding
        assert not mask[0, :4].any()
        assert mask[0, 4:].all()
        # Second instruction: 6 real + 0 padding
        assert not mask[1, :6].any()

    def test_no_padding_when_same_length(self):
        instrs = [
            Instruction('ADD', 5, 3, 7),
            Instruction('SUB', 1, 2, 3),
        ]
        ids, mask = tokenize_batch(instrs)
        assert not mask.any()  # no padding needed

    def test_token_ids_valid(self):
        instrs = [random_instruction(np.random.default_rng(i)) for i in range(10)]
        ids, _ = tokenize_batch(instrs)
        assert (ids >= 0).all()
        assert (ids < VOCAB_SIZE).all()


class TestT1Compressor:
    def test_output_shape(self):
        model = T1Compressor(d_out=32)
        instrs = [Instruction('ADD', 5, 3, 7), Instruction('ADDI', 1, 0, 42)]
        ids, mask = tokenize_batch(instrs)
        out = model(ids, mask)
        assert out.shape == (2, 32)

    def test_output_shape_no_padding(self):
        model = T1Compressor(d_out=16)
        instrs = [Instruction('ADD', 5, 3, 7), Instruction('SUB', 1, 2, 3)]
        ids, mask = tokenize_batch(instrs)
        out = model(ids, mask)
        assert out.shape == (2, 16)

    def test_deterministic(self):
        model = T1Compressor(d_out=32)
        model.eval()
        instrs = [Instruction('ADD', 5, 3, 7)]
        ids, mask = tokenize_batch(instrs)
        with torch.no_grad():
            out1 = model(ids, mask)
            out2 = model(ids, mask)
        assert torch.allclose(out1, out2)

    def test_different_instructions_differ(self):
        model = T1Compressor(d_out=32)
        model.eval()
        ids1, m1 = tokenize_batch([Instruction('ADD', 5, 3, 7)])
        ids2, m2 = tokenize_batch([Instruction('SUB', 5, 3, 7)])
        with torch.no_grad():
            out1 = model(ids1, m1)
            out2 = model(ids2, m2)
        # Random init should produce different outputs
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_gradients_flow(self):
        model = T1Compressor(d_out=32)
        instrs = [random_instruction(np.random.default_rng(i)) for i in range(8)]
        ids, mask = tokenize_batch(instrs)
        out = model(ids, mask)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            assert not torch.all(p.grad == 0)


class TestExecDistanceMatrix:
    def test_shape(self):
        instrs = [
            Instruction('ADDI', 5, 0, 10),
            Instruction('ADDI', 5, 0, 20),
            Instruction('ADD', 5, 3, 7),
        ]
        dist = compute_exec_distance_matrix(instrs, n_inputs=4,
                                            rng=np.random.default_rng(42))
        assert dist.shape == (3, 3)

    def test_diagonal_zero(self):
        instrs = [
            Instruction('ADDI', 5, 0, 10),
            Instruction('ADDI', 5, 0, 20),
        ]
        dist = compute_exec_distance_matrix(instrs, n_inputs=4,
                                            rng=np.random.default_rng(42))
        assert dist[0, 0] == 0.0
        assert dist[1, 1] == 0.0

    def test_symmetric(self):
        instrs = [random_instruction(np.random.default_rng(i)) for i in range(5)]
        dist = compute_exec_distance_matrix(instrs, n_inputs=8,
                                            rng=np.random.default_rng(42))
        assert np.allclose(dist, dist.T)

    def test_equivalent_instructions_zero_distance(self):
        # ADD x5, x3, x3 and SLLI x5, x3, 1 are execution-equivalent
        instrs = [
            Instruction('ADD', 5, 3, 3),
            Instruction('SLLI', 5, 3, 1),
        ]
        dist = compute_exec_distance_matrix(instrs, n_inputs=16,
                                            rng=np.random.default_rng(42))
        assert dist[0, 1] == 0.0

    def test_different_instructions_nonzero_distance(self):
        instrs = [
            Instruction('ADDI', 5, 0, 10),
            Instruction('ADDI', 5, 0, 100),
        ]
        dist = compute_exec_distance_matrix(instrs, n_inputs=4,
                                            rng=np.random.default_rng(42))
        assert dist[0, 1] > 0

    def test_distance_ordering(self):
        # ADDI x5, x0, 10 should be closer to ADDI x5, x0, 11
        # than to ADDI x5, x0, 100
        instrs = [
            Instruction('ADDI', 5, 0, 10),
            Instruction('ADDI', 5, 0, 11),
            Instruction('ADDI', 5, 0, 100),
        ]
        dist = compute_exec_distance_matrix(instrs, n_inputs=4,
                                            rng=np.random.default_rng(42))
        assert dist[0, 1] < dist[0, 2]


class TestCorrelationLoss:
    def test_perfect_correlation(self):
        """When T1 distances are proportional to exec distances, loss ≈ 0."""
        # T1 vectors at 0, 1, 3 → pairwise dists are 1, 3, 2
        # Exec dists match the same proportions.
        t1_vecs = torch.tensor([[0.0], [1.0], [3.0]])
        exec_dists = torch.tensor([
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ])
        loss = correlation_loss(t1_vecs, exec_dists)
        assert loss.item() < 0.01

    def test_no_correlation(self):
        """When T1 distances are uncorrelated with exec, loss ≈ 1."""
        torch.manual_seed(42)
        t1_vecs = torch.randn(32, 4)
        # Random exec distances unrelated to T1 positions
        exec_dists = torch.rand(32, 32)
        exec_dists = (exec_dists + exec_dists.T) / 2
        exec_dists.fill_diagonal_(0)
        loss = correlation_loss(t1_vecs, exec_dists)
        assert 0.5 < loss.item() < 1.5

    def test_inverted_distances(self):
        """When T1 distances are anti-correlated with exec, loss > 1."""
        t1_vecs = torch.tensor([[0.0], [1.0], [3.0]])
        # Exec distances are inverted: close in T1 = far in exec
        exec_dists = torch.tensor([
            [0.0, 3.0, 1.0],
            [3.0, 0.0, 2.0],
            [1.0, 2.0, 0.0],
        ])
        loss = correlation_loss(t1_vecs, exec_dists)
        assert loss.item() > 1.0

    def test_gradient_flows(self):
        t1_vecs = torch.randn(8, 4, requires_grad=True)
        exec_dists = torch.rand(8, 8)
        exec_dists = (exec_dists + exec_dists.T) / 2
        exec_dists.fill_diagonal_(0)
        loss = correlation_loss(t1_vecs, exec_dists)
        loss.backward()
        assert t1_vecs.grad is not None

    def test_loss_bounded(self):
        """Loss should be in [0, 2]."""
        for seed in range(10):
            torch.manual_seed(seed)
            t1_vecs = torch.randn(16, 4)
            exec_dists = torch.rand(16, 16)
            exec_dists = (exec_dists + exec_dists.T) / 2
            exec_dists.fill_diagonal_(0)
            loss = correlation_loss(t1_vecs, exec_dists)
            assert 0 <= loss.item() <= 2.0


_SMOKE_PARAMS = dict(
    batch_size=16, n_steps=3, n_inputs=4, n_producers=2, prefetch=4,
    torch_threads=2, lr=1e-3, lr_min=1e-6, d_model=32, n_heads=2,
    n_layers=1, d_out=8, device='cpu', log_every=1, seed=42,
)


class TestTrainingSmoke:
    def test_few_steps_no_crash(self):
        model, losses = __import__('compressor').train(**_SMOKE_PARAMS)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)

    def test_loss_is_finite(self):
        _, losses = __import__('compressor').train(
            **{**_SMOKE_PARAMS, 'n_steps': 5, 'log_every': 100},
        )
        assert all(np.isfinite(l) for l in losses)
