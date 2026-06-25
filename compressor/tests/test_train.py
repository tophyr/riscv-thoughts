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
    t2_value_predict_loss,
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
# t2_value_predict_loss (the one canonical vp loss, shared by T1 and T2)
# ===========================================================================

class _CaptureHead(torch.nn.Module):
    """Records the last feature tensor it was called with, returns zeros —
    lets a test read exactly what values the vp loss fed the head."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.last = None

    def forward(self, x):
        self.last = x.detach().clone()
        return torch.zeros((*x.shape[:-1], self.out_dim))


def test_t2_value_predict_loss_feeds_canonical_inputs_not_raw():
    """The head is fed the CANONICAL input value anchor[:, k+1] for
    filled input slot k — NOT the raw value at the slot's actual register
    (the old bug). Unfilled slots feed 0."""
    B, d, A = 1, 4, 2
    vecs = torch.zeros(B, d)
    # Distinct value per (anchor, position): pos p, anchor a -> 10*p + a.
    anchor = torch.zeros(A, 32, dtype=torch.int32)
    for p in range(32):
        for a in range(A):
            anchor[a, p] = 10 * p + a
    in_slot = torch.full((B, MAX_INPUT_SLOTS), AUX_CE_IGNORE, dtype=torch.int64)
    # Slot 0 filled, bound to actual reg 7 (canonical position is 1, not 7).
    in_slot[0, 0] = 7
    out_slot = torch.full((B, MAX_OUTPUT_SLOTS), AUX_CE_IGNORE, dtype=torch.int64)
    out_regs = torch.zeros(B, A, 32, dtype=torch.int32)

    head = _CaptureHead(MAX_OUTPUT_SLOTS)
    fake = type('T2', (), {'vp_head': head})()
    t2_value_predict_loss(
        fake, vecs, anchor, in_slot, out_slot, out_regs,
        torch.tensor([True]))

    in_vals = head.last[..., d:]                       # (B, A, MAX_IN)
    # Slot 0 fed canonical position 1 (= anchor[:, 1]), per anchor.
    canon1 = _value_compress(anchor[:, 1].float())     # (A,)
    raw7 = _value_compress(anchor[:, 7].float())
    assert torch.allclose(in_vals[0, :, 0], canon1, atol=1e-5)
    assert not torch.allclose(in_vals[0, :, 0], raw7)  # NOT the raw register
    # Unfilled slots feed 0.
    assert torch.allclose(in_vals[0, :, 1:], torch.zeros_like(in_vals[0, :, 1:]))


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
# _t1_predicted_wiring (tier-recursion routing)
# ===========================================================================

def test_t1_predicted_wiring_from_binding_scores():
    """T2 derives (in0,in1,out) from T1's predicted binding: out = top
    out_score; in0/in1 = top-2 in_score in read order; operand count gated by
    predicted live_in/live_out; absent operands route to x0."""
    from compressor.train import _t1_predicted_wiring
    n_regs, N, NEG = 8, 3, -50.0
    # instr0: reads 3 (first) then 5 (second), writes 7.
    # instr1: reads 2, writes 4.   instr2: reads nothing, writes 6 (LUI-like).
    in_s = torch.full((N, n_regs), NEG)
    in_s[0, 3] = 10.0; in_s[0, 5] = 5.0; in_s[1, 2] = 10.0
    out_s = torch.full((N, n_regs), NEG)
    out_s[0, 7] = 10.0; out_s[1, 4] = 10.0; out_s[2, 6] = 10.0
    live_in = torch.full((N, n_regs), -10.0)
    live_in[0, 3] = 10.0; live_in[0, 5] = 10.0; live_in[1, 2] = 10.0
    live_out = torch.full((N, n_regs), -10.0)
    live_out[0, 7] = 10.0; live_out[1, 4] = 10.0; live_out[2, 6] = 10.0

    class _Stub:
        n_regs = 8
        def in_score_head(self, T):   return in_s.unsqueeze(-1)
        def out_score_head(self, T):  return out_s.unsqueeze(-1)
        def live_in_head(self, T):    return live_in.unsqueeze(-1)
        def live_out_head(self, T):   return live_out.unsqueeze(-1)

    in0, in1, out = _t1_predicted_wiring(_Stub(), torch.zeros(N, n_regs, 4))
    assert in0.tolist() == [3, 2, 0]     # first operand / single / none
    assert in1.tolist() == [5, 0, 0]     # second operand / none / none
    assert out.tolist() == [7, 4, 6]
    # x0 never selected even if it scored highest.
    in_s[2, 0] = 100.0; live_in[2, 0] = 10.0
    i0b, _, _ = _t1_predicted_wiring(_Stub(), torch.zeros(N, n_regs, 4))
    assert i0b[2].item() == 0            # x0 masked out of contention -> absent


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
    can be driven to a known value independent of learned weights. The
    per-slot heads return (B, n_regs, 1) (binding_losses squeezes the last
    dim); pc_writes reads essence and returns (B, 1)."""

    def __init__(self, *, n_regs, li, lo, pc, in_s, out_s,
                 max_input_slots, max_output_slots):
        self.n_regs = n_regs
        self.max_input_slots = max_input_slots
        self.max_output_slots = max_output_slots
        self._li, self._lo, self._pc = li, lo, pc
        self._in, self._out = in_s, out_s

    def live_in_head(self, T):    return self._li.unsqueeze(-1)
    def live_out_head(self, T):   return self._lo.unsqueeze(-1)
    def pc_writes_head(self, e):  return self._pc
    def in_score_head(self, T):   return self._in.unsqueeze(-1)
    def out_score_head(self, T):  return self._out.unsqueeze(-1)


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
        model, torch.randn(1, n_regs, 4), torch.randn(1, 4),
        live_in_t=live_in, live_out_t=live_out,
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
        model, torch.randn(2, n_regs, 4), torch.randn(2, 4),
        live_in_t=live_in, live_out_t=live_out,
        pc_writes_t=pc, in_slot_t=in_slot, out_slot_t=out_slot)
    assert out['active_f'].tolist() == [1.0, 0.0]


def test_t1_compressor_uses_t2_binding_heads():
    """T1 carries the per-slot binding heads (and not the removed syntactic
    CE heads), so the shared binding_losses runs on a real (T, essence)."""
    from emulator import Instruction
    from tokenizer import encode_instruction
    from compressor.model import instruction_wiring

    enc = T1Compressor(vocab_size=VOCAB_SIZE, d_model=32, n_heads=2,
                       n_layers=1, max_window=32, d_out=16, d_event=16)
    for h in ('live_in_head', 'live_out_head', 'pc_writes_head',
              'in_score_head', 'out_score_head'):
        assert hasattr(enc, h), h
    for gone in ('dest_type_head', 'dest_reg_head', 'src_reg_head_0'):
        assert not hasattr(enc, gone), gone

    instr = Instruction('ADD', 5, 1, 2)               # reads 1,2 writes 5
    toks = encode_instruction(instr)
    tok = torch.tensor([toks]); pad = torch.zeros(1, len(toks), dtype=torch.bool)
    i0, i1, o = instruction_wiring(instr)
    tags = torch.randn(1, enc.n_regs); tags[:, 0] = 0.0
    T, ess = enc.encode_state(
        tok, pad, torch.tensor([i0]), torch.tensor([i1]), torch.tensor([o]), tags)

    n = enc.n_regs
    live_in = torch.zeros(1, n); live_in[0, [1, 2]] = 1.0
    live_out = torch.zeros(1, n); live_out[0, 5] = 1.0
    in_slot = torch.full((1, enc.max_input_slots), AUX_CE_IGNORE, dtype=torch.long)
    in_slot[0, 0], in_slot[0, 1] = 1, 2
    out_slot = torch.full((1, enc.max_output_slots), AUX_CE_IGNORE, dtype=torch.long)
    out_slot[0, 0] = 5
    out = binding_losses(
        enc, T, ess, live_in_t=live_in, live_out_t=live_out,
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
    """The equivariant T2 runs the register-state machine over a chunk of
    frozen-T1 essences threaded by per-instruction wiring. encode_state ->
    (T (B,n_regs,d_out+d_event), essence (B,d_out)); padded steps (active=0)
    are no-ops, so a chunk shorter than N does not NaN."""
    torch.manual_seed(0)
    t2 = T2Compressor(d_t1=8, d_out=12, d_event=16)
    B, N, R = 3, 4, t2.n_regs
    seq = torch.randn(B, N, 8)                       # per-instr T1 essences
    in0 = torch.randint(1, R, (B, N)); in1 = torch.randint(1, R, (B, N))
    out = torch.randint(1, R, (B, N))
    # row 2 is a 1-instruction chunk (rest padded -> active=0, no-op).
    active = torch.ones(B, N); active[2, 1:] = 0.0
    tags = torch.randn(B, R)

    T, essence = t2.encode_state(seq, in0, in1, out, active, tags)
    assert T.shape == (B, R, 12 + 16)
    assert essence.shape == (B, 12)
    assert not torch.isnan(T).any() and not torch.isnan(essence).any()
    # encode() returns just the essence.
    assert torch.allclose(t2.encode(seq, in0, in1, out, active, tags), essence)
