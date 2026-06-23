"""Unit tests for compressor.train loss math and helpers.

The trainers' loss functions are pure tensor math with hand-computable
results on tiny inputs. Each test here pins an expected value that was
either hand-derived from the formula or computed by reading the code —
never a "returns a float" smoke check. Where a head's learned output
would make the result opaque, we substitute a fixed (zero/constant)
head so the *target side* of the loss is exercised deterministically.
"""

import math

import numpy as np
import pytest
import torch

from datagen import AUX_CE_IGNORE, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS
from tokenizer import BOS, EOS, PAD, VOCAB_SIZE

from compressor.model import T1Compressor, T2Compressor
from compressor.train import (
    _compute_chunk_out_regs,
    _listmle_loss,
    _value_compress,
    binding_losses,
    build_optim_sched,
    prepare_decoder_targets,
    t2_pair_loss,
    t2_value_predict_loss,
    value_predict_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConstHead(torch.nn.Module):
    """Output a constant value, ignoring inputs — lets the target side
    of a value-prediction loss be hand-computed (pred is a known const)."""

    def __init__(self, out_dim, const=0.0):
        super().__init__()
        self.out_dim = out_dim
        self.const = const

    def forward(self, x):
        return torch.full((*x.shape[:-1], self.out_dim), self.const)


# ===========================================================================
# value_predict_loss
# ===========================================================================

def test_value_predict_loss_matches_hand_computed_mse():
    """Per-row MSE with a zero-prediction head equals mean over anchors
    of compressed-target squared; only (pair_valid & has_rd) rows count."""
    B, d, A = 3, 4, 2
    vecs = torch.zeros(B, d)
    anchor_states = torch.zeros(A, 32, dtype=torch.int32)
    src0 = torch.zeros(B, dtype=torch.int64)
    src1 = torch.zeros(B, dtype=torch.int64)
    row_outputs = torch.tensor([[3.0, 5.0], [7.0, 7.0], [100.0, 200.0]])
    pair_valid = torch.tensor([True, True, False])
    has_rd = torch.tensor([True, False, True])
    # mask = pair_valid & has_rd = [True, False, False] -> only row 0.

    loss = value_predict_loss(
        _ConstHead(1, 0.0), vecs, anchor_states, src0, src1,
        row_outputs, pair_valid, has_rd)

    tc = _value_compress(row_outputs)
    expected = (tc[0] ** 2).mean()   # row 0, pred=0
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-6)


def test_value_predict_loss_masked_rows_excluded():
    """Changing a masked-out row's target must NOT change the loss; only
    the single (valid & has_rd) row drives the mean."""
    B, d, A = 3, 4, 2
    vecs = torch.zeros(B, d)
    anchor_states = torch.zeros(A, 32, dtype=torch.int32)
    src0 = torch.zeros(B, dtype=torch.int64)
    src1 = torch.zeros(B, dtype=torch.int64)
    pair_valid = torch.tensor([True, True, False])
    has_rd = torch.tensor([True, False, True])

    base = torch.tensor([[3.0, 5.0], [7.0, 7.0], [100.0, 200.0]])
    perturbed = base.clone()
    perturbed[1] = 999.0       # masked (has_rd False)
    perturbed[2] = -42.0       # masked (pair_valid False)

    l_base = value_predict_loss(
        _ConstHead(1), vecs, anchor_states, src0, src1, base,
        pair_valid, has_rd)
    l_pert = value_predict_loss(
        _ConstHead(1), vecs, anchor_states, src0, src1, perturbed,
        pair_valid, has_rd)
    assert math.isclose(l_base.item(), l_pert.item(), rel_tol=1e-7)


def test_value_predict_loss_empty_mask_is_zero():
    """No (pair_valid & has_rd) rows -> exactly 0.0, no grad."""
    B, d, A = 2, 4, 2
    vecs = torch.zeros(B, d)
    anchor_states = torch.zeros(A, 32, dtype=torch.int32)
    src0 = torch.zeros(B, dtype=torch.int64)
    src1 = torch.zeros(B, dtype=torch.int64)
    row_outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pair_valid = torch.tensor([False, True])
    has_rd = torch.tensor([True, False])    # AND is all-False

    loss = value_predict_loss(
        _ConstHead(1), vecs, anchor_states, src0, src1, row_outputs,
        pair_valid, has_rd)
    assert loss.item() == 0.0


# ===========================================================================
# t2_value_predict_loss
# ===========================================================================

def test_t2_value_predict_loss_gather_and_mask():
    """out_slot_regs gather picks the right register column of out_regs;
    only out-valid (row, slot) pairs contribute, and return_count gives
    that pair count."""
    B, d, A = 2, 4, 2
    vecs = torch.zeros(B, d)
    anchor = torch.zeros(A, 32, dtype=torch.int32)
    in_slot = torch.full((B, MAX_INPUT_SLOTS), AUX_CE_IGNORE, dtype=torch.int64)
    out_slot = torch.full((B, MAX_OUTPUT_SLOTS), AUX_CE_IGNORE, dtype=torch.int64)
    # Row 0: slot 0 -> reg 5, slot 1 -> reg 9. Row 1: no out slots.
    out_slot[0, 0] = 5
    out_slot[0, 1] = 9
    out_regs = torch.zeros(B, A, 32, dtype=torch.int32)
    out_regs[0, 0, 5] = 10
    out_regs[0, 1, 5] = 20
    out_regs[0, 0, 9] = 30
    out_regs[0, 1, 9] = 40
    mask = torch.tensor([True, True])

    fake = type('T2', (), {'vp_head': _ConstHead(MAX_OUTPUT_SLOTS, 0.0)})()
    mse, count = t2_value_predict_loss(
        fake, vecs, anchor, in_slot, out_slot, out_regs, mask,
        return_count=True)

    # Two valid (row, slot) pairs: (0,0) gathers reg5 -> [10,20];
    # (0,1) gathers reg9 -> [30,40]. pred=0; per_slot = mean over anchors.
    ps0 = (_value_compress(torch.tensor([10.0, 20.0])) ** 2).mean()
    ps1 = (_value_compress(torch.tensor([30.0, 40.0])) ** 2).mean()
    expected = ((ps0 + ps1) / 2).item()
    assert count == 2
    assert math.isclose(mse.item(), expected, rel_tol=1e-6)


def test_t2_value_predict_loss_empty_masks_are_zero():
    """Empty row mask -> 0.0; all-IGNORE out_slots -> 0.0 (the two
    early-exit guards at the AUX_CE_IGNORE pair-mask)."""
    B, d, A = 2, 4, 2
    vecs = torch.zeros(B, d)
    anchor = torch.zeros(A, 32, dtype=torch.int32)
    in_slot = torch.full((B, MAX_INPUT_SLOTS), AUX_CE_IGNORE, dtype=torch.int64)
    out_slot = torch.full((B, MAX_OUTPUT_SLOTS), AUX_CE_IGNORE, dtype=torch.int64)
    out_slot[0, 0] = 5
    out_regs = torch.zeros(B, A, 32, dtype=torch.int32)
    fake = type('T2', (), {'vp_head': _ConstHead(MAX_OUTPUT_SLOTS, 0.0)})()

    # Empty row mask: not mask_t.any() -> first guard.
    z1 = t2_value_predict_loss(
        fake, vecs, anchor, in_slot, out_slot, out_regs,
        torch.tensor([False, False]))
    assert z1.item() == 0.0

    # All-IGNORE out slots with active rows: pair_mask empty -> second guard.
    all_ignore = torch.full((B, MAX_OUTPUT_SLOTS), AUX_CE_IGNORE,
                            dtype=torch.int64)
    z2 = t2_value_predict_loss(
        fake, vecs, anchor, in_slot, all_ignore, out_regs,
        torch.tensor([True, True]))
    assert z2.item() == 0.0


def test_t2_pair_loss_identical_behavior_targets_zero():
    B, A = 2, 2
    out_regs = torch.zeros(B, A, 32, dtype=torch.int32)
    live_out = torch.zeros(B, 32); live_out[:, 5] = 1.0
    valid = torch.tensor([True, True])
    loss0 = t2_pair_loss(torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]]),
                         out_regs, live_out, valid, scale=0.5, strides=(1,))
    assert loss0.item() == pytest.approx(0.0, abs=1e-6)
    loss1 = t2_pair_loss(torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0]]),
                         out_regs, live_out, valid, scale=0.5, strides=(1,))
    assert loss1.item() == pytest.approx(1.0, abs=1e-6)


def test_t2_pair_loss_different_behavior_pushes_apart():
    B, A = 2, 2
    out_regs = torch.zeros(B, A, 32, dtype=torch.int32); out_regs[1, :, 5] = 1000
    live_out = torch.zeros(B, 32); live_out[:, 5] = 1.0
    loss = t2_pair_loss(torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]]),
                        out_regs, live_out, torch.tensor([True, True]),
                        scale=0.5, strides=(1,))
    target = 0.5 * math.log1p(math.log1p(1000.0))
    assert loss.item() == pytest.approx(target ** 2, rel=1e-4)


def test_t2_pair_loss_invalid_rows_excluded_and_differentiable():
    z = t2_pair_loss(torch.eye(2, 4), torch.zeros(2, 2, 32, dtype=torch.int32),
                     torch.ones(2, 32), torch.tensor([False, False]),
                     scale=0.5, strides=(1,))
    assert z.item() == 0.0
    vecs = torch.randn(8, 16, requires_grad=True)
    loss = t2_pair_loss(vecs, torch.randint(-5000, 5000, (8, 4, 32), dtype=torch.int32),
                        (torch.rand(8, 32) > 0.5).float(), torch.ones(8, dtype=torch.bool),
                        scale=0.5)
    loss.backward()
    assert loss.dim() == 0 and torch.isfinite(vecs.grad).all()


def test_gen_shipped_out_regs_matches_recompute():
    """The out_regs the gen workers ship in the RVT batch must be bit-
    identical to what the trainer would recompute single-threaded (the
    gen-side shipping is a perf move, not a semantic change)."""
    from datagen.batch import collect_into_batches, generate_chunks
    from datagen.compare import make_anchor_states
    from datagen.generate import either, until_branch, length_cap

    anchors = make_anchor_states(8, 0)
    rng = np.random.default_rng(0)
    chunks = generate_chunks(either(until_branch(), length_cap(4)), rng,
                             eq_rate=0.05)
    batch = next(collect_into_batches(
        chunks, batch_size=64, twins=3, anchor_states=anchors, rng=rng,
        max_chunk_len=4))

    assert batch.out_regs.shape[0] == batch.tokens.shape[0]   # OB == B (T2)
    ref_out, ref_valid = _compute_chunk_out_regs(batch, anchors)
    assert np.array_equal(batch.out_regs, ref_out)
    assert np.array_equal(batch.out_regs_valid, ref_valid)


# ===========================================================================
# _listmle_loss
# ===========================================================================

def test_listmle_loss_matches_plackett_luce():
    """Uniform scores: per-step NLL is log(n_remaining) as previously
    chosen regs are masked to -inf and excluded from the next softmax."""
    n_regs = 3
    scores = torch.zeros(2, n_regs)
    slot_regs = torch.tensor(
        [[2, 0],
         [AUX_CE_IGNORE, AUX_CE_IGNORE]], dtype=torch.int64)
    active = torch.tensor([1.0, 1.0])

    loss = _listmle_loss(scores, slot_regs, 2, active)

    # Row 0: step0 picks reg2 from 3 uniform -> log 3; step1 picks reg0
    # from remaining 2 -> log 2. Per-chunk = (log3 + log2)/2 filled slots.
    # Row 1: 0 filled -> excluded from the chunk mean.
    expected = (math.log(3) + math.log(2)) / 2
    assert math.isclose(loss.item(), expected, rel_tol=1e-5)


def test_listmle_loss_zero_filled_rows_stay_finite():
    """The inf*0=nan guard (train.py:776-787): a row with fewer filled
    slots than K_max, whose register 0 was masked at an earlier step,
    must still yield a finite loss (torch.where, not multiply-by-mask)."""
    # Row has filled [reg0, reg1] but K_max=3. After reg0 is chosen and
    # masked, step 2 is unfilled -> safe_target collapses to 0 and
    # log_p[0] = -inf -> step_nll = +inf. The where() guard must drop it.
    scores = torch.tensor([[5.0, 1.0, 0.0]])
    slot_regs = torch.tensor([[0, 1, AUX_CE_IGNORE]], dtype=torch.int64)
    active = torch.tensor([1.0])

    loss = _listmle_loss(scores, slot_regs, 3, active)
    assert math.isfinite(loss.item())

    # And a batch where every row has zero filled slots -> finite 0.0
    # (weight sum clamps to 1, numerator is 0).
    all_ignore = torch.full((2, 2), AUX_CE_IGNORE, dtype=torch.int64)
    loss2 = _listmle_loss(
        torch.zeros(2, 3), all_ignore, 2, torch.tensor([1.0, 1.0]))
    assert math.isfinite(loss2.item())
    assert loss2.item() == 0.0


# ===========================================================================
# binding_losses (shared T1/T2 rename-equivariant register binding)
# ===========================================================================

class _StubBindingModel:
    """Binding heads return caller-fixed logits/scores, so binding_losses
    can be driven to a known value independent of learned weights."""

    def __init__(self, *, n_regs, li, lo, pc, in_s, out_s,
                 max_input_slots, max_output_slots):
        self.n_regs = n_regs
        self.max_input_slots = max_input_slots
        self.max_output_slots = max_output_slots
        self._li, self._lo, self._pc = li, lo, pc
        self._in, self._out = in_s, out_s

    def live_in_head(self, x):    return self._li
    def live_out_head(self, x):   return self._lo
    def pc_writes_head(self, x):  return self._pc
    def in_score_head(self, x):   return self._in
    def out_score_head(self, x):  return self._out


def test_binding_losses_jointly_satisfiable():
    """The whole point of mirroring T2: every binding target can be met at
    once. With heads predicting each target near-perfectly, all five losses
    go to ~0 simultaneously — there is no irreducible two-loss floor (which
    the old equivalence_loss vs src-register CE could not avoid)."""
    n_regs = 8
    # One row: reads regs {1,2} (slot order 1 then 2), writes {5}, no pc.
    live_in = torch.zeros(1, n_regs); live_in[0, [1, 2]] = 1.0
    live_out = torch.zeros(1, n_regs); live_out[0, 5] = 1.0
    pc = torch.zeros(1)
    in_slot = torch.full((1, 4), AUX_CE_IGNORE, dtype=torch.long)
    in_slot[0, 0], in_slot[0, 1] = 1, 2
    out_slot = torch.full((1, 4), AUX_CE_IGNORE, dtype=torch.long)
    out_slot[0, 0] = 5

    li = (live_in * 2 - 1) * 20.0
    lo = (live_out * 2 - 1) * 20.0
    pc_logit = ((pc * 2 - 1) * 20.0).unsqueeze(-1)
    in_s = torch.full((1, n_regs), -50.0); in_s[0, 1] = 100.0; in_s[0, 2] = 90.0
    out_s = torch.full((1, n_regs), -50.0); out_s[0, 5] = 100.0

    model = _StubBindingModel(
        n_regs=n_regs, li=li, lo=lo, pc=pc_logit, in_s=in_s, out_s=out_s,
        max_input_slots=4, max_output_slots=4)
    out = binding_losses(
        model, torch.randn(1, 16), live_in_t=live_in, live_out_t=live_out,
        pc_writes_t=pc, in_slot_t=in_slot, out_slot_t=out_slot)
    for k in ('live_in', 'live_out', 'pc_writes', 'in_slot', 'out_slot'):
        assert float(out[k]) < 1e-3, (k, float(out[k]))


def test_binding_losses_excludes_all_zero_mask_rows():
    """Rows with no live regs (invalid / mem-op) are excluded via active_f."""
    n_regs = 8
    live_in = torch.zeros(2, n_regs); live_in[0, 1] = 1.0   # row 1 all-zero
    live_out = torch.zeros(2, n_regs); live_out[0, 5] = 1.0
    pc = torch.zeros(2)
    in_slot = torch.full((2, 4), AUX_CE_IGNORE, dtype=torch.long); in_slot[0, 0] = 1
    out_slot = torch.full((2, 4), AUX_CE_IGNORE, dtype=torch.long); out_slot[0, 0] = 5
    z = torch.zeros(2, n_regs)
    model = _StubBindingModel(
        n_regs=n_regs, li=z, lo=z, pc=torch.zeros(2, 1), in_s=z, out_s=z,
        max_input_slots=4, max_output_slots=4)
    out = binding_losses(
        model, torch.randn(2, 16), live_in_t=live_in, live_out_t=live_out,
        pc_writes_t=pc, in_slot_t=in_slot, out_slot_t=out_slot)
    assert out['active_f'].tolist() == [1.0, 0.0]


def test_t1_compressor_uses_t2_binding_heads():
    """T1 carries T2's binding heads (and not the removed syntactic CE
    heads), so the shared binding_losses runs on a real T1 vector."""
    enc = T1Compressor(VOCAB_SIZE, d_model=32, n_heads=2, n_layers=1, d_out=16)
    for h in ('live_in_head', 'live_out_head', 'pc_writes_head',
              'in_score_head', 'out_score_head'):
        assert hasattr(enc, h), h
    for gone in ('dest_type_head', 'dest_reg_head', 'src_reg_head_0'):
        assert not hasattr(enc, gone), gone
    n = enc.n_regs
    live_in = torch.zeros(1, n); live_in[0, 1] = 1.0
    live_out = torch.zeros(1, n); live_out[0, 5] = 1.0
    in_slot = torch.full((1, enc.max_input_slots), AUX_CE_IGNORE, dtype=torch.long)
    in_slot[0, 0] = 1
    out_slot = torch.full((1, enc.max_output_slots), AUX_CE_IGNORE, dtype=torch.long)
    out_slot[0, 0] = 5
    out = binding_losses(
        enc, torch.randn(1, 16), live_in_t=live_in, live_out_t=live_out,
        pc_writes_t=torch.zeros(1), in_slot_t=in_slot, out_slot_t=out_slot)
    for k in ('live_in', 'live_out', 'pc_writes', 'in_slot', 'out_slot'):
        assert math.isfinite(float(out[k]))


# ===========================================================================
# build_optim_sched
# ===========================================================================

def test_build_optim_sched_no_scheduler_when_n_steps_falsy():
    p = [torch.nn.Parameter(torch.zeros(2))]
    _, s0 = build_optim_sched(p, 0.1, 0, device='cpu')
    _, sn = build_optim_sched(p, 0.1, None, device='cpu')
    assert s0 is None and sn is None


def test_build_optim_sched_warmup_then_cosine():
    """warmup_steps>0: LR ramps from ~lr*start_factor up to lr at the
    warmup boundary, then cosine-decays toward lr_min."""
    p = [torch.nn.Parameter(torch.zeros(2))]
    opt, sched = build_optim_sched(
        p, 0.1, 10, warmup_steps=4, lr_min=0.0, device='cpu')
    lrs = []
    for _ in range(10):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    # step 0 starts at lr * start_factor (1e-3) = 1e-4.
    assert math.isclose(lrs[0], 0.1 * 1e-3, rel_tol=1e-3)
    # rises monotonically through warmup to the peak lr at the boundary.
    assert lrs[0] < lrs[1] < lrs[2] < lrs[3] < lrs[4]
    assert math.isclose(lrs[4], 0.1, rel_tol=1e-6)
    # then cosine decays.
    assert lrs[4] > lrs[5] > lrs[9]
    assert lrs[9] < 0.05


def test_build_optim_sched_plain_cosine_when_no_warmup():
    """warmup_steps==0: plain cosine — LR starts at lr, decays toward
    lr_min by the last step."""
    p = [torch.nn.Parameter(torch.zeros(2))]
    opt, sched = build_optim_sched(
        p, 0.1, 10, warmup_steps=0, lr_min=0.0, device='cpu')
    lrs = []
    for _ in range(10):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert math.isclose(lrs[0], 0.1, rel_tol=1e-9)
    assert lrs[0] > lrs[5] > lrs[9]
    assert lrs[9] < 0.01


# ===========================================================================
# prepare_decoder_targets
# ===========================================================================

def test_prepare_decoder_targets_wraps_and_shifts():
    """dec_input = [BOS] + toks (drop last), dec_target = toks + [EOS]."""
    di, dt, dp = prepare_decoder_targets([[10, 11]], 'cpu')
    assert di.tolist() == [[BOS, 10, 11]]
    assert dt.tolist() == [[10, 11, EOS]]
    assert dp.tolist() == [[False, False, False]]


def test_prepare_decoder_targets_padding():
    """Unequal lengths pad to the longest; shorter row's tail is PAD and
    masked True."""
    di, dt, dp = prepare_decoder_targets([[10], [20, 21, 22]], 'cpu')
    # Row 0: [BOS,10,EOS] -> input [BOS,10] target [10,EOS], padded to len 4.
    assert di.tolist() == [[BOS, 10, PAD, PAD], [BOS, 20, 21, 22]]
    assert dt.tolist() == [[10, EOS, PAD, PAD], [20, 21, 22, EOS]]
    assert dp.tolist() == [[False, False, True, True],
                           [False, False, False, False]]


def test_prepare_decoder_targets_empty():
    assert prepare_decoder_targets([], 'cpu') == (None, None, None)


# ===========================================================================
# T2Compressor forward
# ===========================================================================

def test_t2compressor_forward_shape_and_padding_contract():
    """Output shape (B, d_out); return_pooled gives (B, d_model). With
    position 0 unmasked (the contract _assemble_chunk_seq guarantees) an
    otherwise-all-padded row does NOT NaN. Pinning the boundary: raw
    encode WITH a fully-padded row DOES NaN — protection lives in the
    caller (train.py:645), not in encode."""
    torch.manual_seed(0)
    t2 = T2Compressor(d_t1=8, d_model=16, n_heads=2, n_layers=1,
                      d_out=12, max_chunk_len=4)
    B, N = 3, 4
    seq = torch.randn(B, N, 8)

    # Realistic mask: row 2 fully padded EXCEPT position 0 (as
    # _assemble_chunk_seq forces). No NaN.
    pad = torch.zeros(B, N, dtype=torch.bool)
    pad[2, :] = True
    pad[:, 0] = False
    out, pooled = t2.encode(seq, pad, return_pooled=True)
    assert out.shape == (B, 12)
    assert pooled.shape == (B, 16)
    assert not torch.isnan(out).any()

    # Contract boundary: a truly all-padded row (position 0 masked too)
    # DOES NaN — the model relies on the caller unmasking position 0.
    pad_all = torch.zeros(B, N, dtype=torch.bool)
    pad_all[2, :] = True
    out_bad = t2.encode(seq, pad_all)
    assert torch.isnan(out_bad[2]).any()
