"""Tests for the compressor model and training components."""

import numpy as np
import torch

from emulator import Instruction
from tokenizer import VOCAB_SIZE
from compressor import T1Compressor
from compressor.train import (
    tokenize_batch, _pearson, combined_loss, exec_distance,
)
from datagen import InlineProducer, random_instruction


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TestT1Compressor:
    def test_output_shapes(self):
        model = T1Compressor(vocab_size=VOCAB_SIZE, d_model=32, n_heads=2,
                             n_layers=1, d_out=16)
        ids, mask = tokenize_batch([Instruction('ADD', 5, 3, 7),
                                    Instruction('ADDI', 1, 0, 42)])
        t1, dt, dr = model(ids, mask)
        assert t1.shape == (2, 16)
        assert dt.shape == (2, 2)
        assert dr.shape == (2, 32)

    def test_deterministic(self):
        model = T1Compressor(vocab_size=VOCAB_SIZE, d_model=32, n_heads=2,
                             n_layers=1, d_out=16)
        model.eval()
        ids, mask = tokenize_batch([Instruction('ADD', 5, 3, 7)])
        with torch.no_grad():
            a, _, _ = model(ids, mask)
            b, _, _ = model(ids, mask)
        assert torch.allclose(a, b)

    def test_gradients_flow(self):
        model = T1Compressor(vocab_size=VOCAB_SIZE, d_model=32, n_heads=2,
                             n_layers=1, d_out=16)
        instrs = [random_instruction(np.random.default_rng(i)) for i in range(8)]
        ids, mask = tokenize_batch(instrs)
        t1, dt, dr = model(ids, mask)
        (t1.sum() + dt.sum() + dr.sum()).backward()
        for p in model.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# Loss / distance
# ---------------------------------------------------------------------------

class TestPearson:
    def test_perfect_correlation(self):
        t1_flat = torch.tensor([1.0, 3.0, 2.0])  # pairwise dists for 3 points
        exec_flat = torch.tensor([1.0, 3.0, 2.0])
        assert _pearson(t1_flat, exec_flat).item() < 0.01

    def test_gradient_flows(self):
        t1_flat = torch.randn(28, requires_grad=True)  # 8 choose 2 = 28 pairs
        exec_flat = torch.rand(28)
        _pearson(t1_flat, exec_flat).backward()
        assert t1_flat.grad is not None

    def test_weighted(self):
        t1_flat = torch.tensor([1.0, 3.0, 2.0])
        exec_flat = torch.tensor([1.0, 3.0, 2.0])
        weight = torch.tensor([0.5, 0.3, 0.2])
        assert _pearson(t1_flat, exec_flat, weight).item() < 0.01


class TestCombinedLoss:
    def test_gradient_flows_all_heads(self):
        model = T1Compressor(vocab_size=VOCAB_SIZE, d_model=32, n_heads=2,
                             n_layers=1, d_out=8)
        instrs = [random_instruction(np.random.default_rng(i)) for i in range(8)]
        ids, mask = tokenize_batch(instrs)
        t1, dt, dr = model(ids, mask)
        ed = torch.rand(8, 8); ed = (ed + ed.T) / 2; ed.fill_diagonal_(0)
        data_ranges = torch.rand(8) * 1e9
        combined_loss(t1, dt, dr, ed,
                      torch.zeros(8, dtype=torch.long),
                      torch.randint(0, 32, (8,)),
                      data_ranges).backward()
        for p in model.parameters():
            assert p.grad is not None


class TestExecDistance:
    def test_identical_zero(self):
        dv = np.array([[10, 20], [10, 20]], dtype=np.int64)
        pv = np.array([[4, 4], [4, 4]], dtype=np.int64)
        assert exec_distance(dv, pv, torch.device('cpu'))[0, 1].item() == 0.0

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        dv = rng.integers(-1000, 1000, size=(8, 4), dtype=np.int64)
        pv = rng.integers(0, 100, size=(8, 4), dtype=np.int64)
        d = exec_distance(dv, pv, torch.device('cpu'))
        assert torch.allclose(d, d.T)

    def test_pc_separates_branches(self):
        dv = np.array([[0, 0], [0, 0]], dtype=np.int64)
        pv = np.array([[4, 4], [100, 100]], dtype=np.int64)
        assert exec_distance(dv, pv, torch.device('cpu'))[0, 1].item() > 0


# ---------------------------------------------------------------------------
# Training smoke test (uses InlineProducer — no multiprocessing)
# ---------------------------------------------------------------------------

class TestTrainingSmoke:
    def _train_small_model(self, tmp_path):
        """Train a tiny model and save it. Returns the run directory."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        producer = InlineProducer(batch_size=8, n_inputs=4,
                                  n_batches=3, seed=42)
        model = T1Compressor(vocab_size=VOCAB_SIZE, d_model=32, n_heads=2,
                             n_layers=1, d_out=8).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for batch in producer:
            ids = torch.from_numpy(batch.token_ids).to(device)
            mask = torch.from_numpy(batch.padding_mask).to(device)
            ed = exec_distance(batch.data_vals, batch.pc_vals, device)
            dt = torch.from_numpy(batch.dest_types).to(device)
            dr = torch.from_numpy(batch.dest_regs).to(device)
            dv_range = batch.data_vals.max(axis=1) - batch.data_vals.min(axis=1)
            data_ranges = torch.tensor(dv_range, dtype=torch.float32, device=device)

            t1, dt_logits, dr_logits = model(ids, mask)
            loss = combined_loss(t1, dt_logits, dr_logits, ed, dt, dr, data_ranges)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        from compressor.train import save_run
        run_dir = save_run(model, losses, hparams={
            'vocab_size': VOCAB_SIZE, 'd_model': 32, 'n_heads': 2,
            'n_layers': 1, 'd_out': 8,
        }, out_dir=str(tmp_path))
        return losses, run_dir

    def test_inline_training(self, tmp_path):
        losses, _ = self._train_small_model(tmp_path)
        assert len(losses) == 3
        assert all(np.isfinite(l) for l in losses)

    def test_evaluate_after_training(self, tmp_path):
        from compressor.eval import evaluate
        _, run_dir = self._train_small_model(tmp_path)
        # Should complete without errors. Output goes to stdout.
        evaluate(str(run_dir), n_inputs=4)
