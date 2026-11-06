"""Unit tests for compressor.train loss math and helpers.

The trainers' loss functions are pure tensor math with hand-computable
results on tiny inputs. Each test here pins an expected value that was
either hand-derived from the formula or computed by reading the code —
never a "returns a float" smoke check. Where a head's learned output
would make the result opaque, we substitute a fixed (zero/constant)
head so the *target side* of the loss is exercised deterministically.
"""

import math

import pytest

import numpy as np
import torch

from datagen import AUX_CE_IGNORE, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS
from tokenizer import BOS, EOS, PAD, VOCAB_SIZE, decode_instruction

from compressor.model import T1Compressor, T2Compressor
from emulator import Instruction
from compressor.train import (
    _VP_VALUE_BITS,
    _listmle_loss,
    _value_bits,
    _value_smooth,
    binding_losses,
    build_optim_sched,
    vp_loss,
)

# vp_head output width per slot: 32 two's-complement bit-logits + 1 smooth scalar.
_VP_SLOT = _VP_VALUE_BITS + 1


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
# vp_loss (value prediction read from the essence, shared by T1 and T2)
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


def _vp_comp(head, d, de):
    from compressor.model import _COLOUR_DIM
    return type('C', (), {'vp_head': head, 'd_out': d, 'd_event': de,
                          'd_colour': _COLOUR_DIM, 'max_input_slots': MAX_INPUT_SLOTS})()


def test_vp_loss_feeds_the_essence_and_the_chunk_inputs_by_role():
    """The head is fed the unit-normalized essence and, for each of the
    chunk's inputs in canonical role order (in_roles), its shipped value at
    the anchor as [32 bits, 1 smooth] plus the input register's colour
    channels from T — no output is named to it. Unfilled positions feed
    zeros."""
    from compressor.model import _COLOUR_DIM as dc
    B, d, de, A, P = 1, 4, 2, 2, 4
    T = torch.zeros(B, 32, d + de + dc)
    T[0, 9, :d] = torch.tensor([1.0, 2.0, 3.0, 4.0])        # description of x9
    T[0, 7, d + de:d + de + dc] = 0.5                        # colour of leaf x7
    essence = torch.tensor([[3.0, 0.0, 4.0, 0.0]])
    in_values = torch.zeros(B, A, P, dtype=torch.int32)
    in_values[0, :, 0] = torch.tensor([70, 71], dtype=torch.int32)
    in_roles = torch.full((B, P), AUX_CE_IGNORE, dtype=torch.int64)
    in_roles[0, 0] = 7
    out_slot = torch.full((B, MAX_OUTPUT_SLOTS), AUX_CE_IGNORE, dtype=torch.int64)
    out_slot[0, 0] = 9
    out_values = torch.zeros(B, A, 2, dtype=torch.int32)
    head = _CaptureHead(MAX_OUTPUT_SLOTS * _VP_SLOT)
    vp_loss(_vp_comp(head, d, de), T, essence, in_values, out_slot, in_roles,
            out_values, torch.tensor([True]), n_out=2)
    x = head.last                                     # (B, A, d + I*(33+dc))
    assert x.shape == (B, A, d + MAX_INPUT_SLOTS * (_VP_SLOT + dc))
    assert torch.allclose(x[0, :, :d], torch.tensor([[0.6, 0.0, 0.8, 0.0]] * A))
    feats = x[0, :, d:].reshape(A, MAX_INPUT_SLOTS, _VP_SLOT + dc)
    assert torch.equal(feats[:, 0, :_VP_VALUE_BITS], _value_bits(in_values[0, :, 0].long()))
    assert torch.allclose(feats[:, 0, _VP_VALUE_BITS], _value_smooth(in_values[0, :, 0].float()))
    assert torch.all(feats[:, 0, _VP_SLOT:] == 0.5)
    assert torch.count_nonzero(feats[:, 1:]) == 0


class _FixedHead(torch.nn.Module):
    """Output one fixed prediction vector for every row and anchor."""

    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, x):
        return self.out.expand(*x.shape[:-1], -1)


def test_vp_loss_matches_predictions_to_outputs_by_the_cheapest_assignment():
    """The head's prediction slots are matched to the chunk's outputs by the
    assignment of least loss, so predicting the outputs in the other order
    costs nothing; only filled (row, output) pairs of unmasked rows count."""
    from compressor.model import _COLOUR_DIM
    B, d, de, A, P = 2, 4, 2, 1, 4
    T = torch.zeros(B, 32, d + de + _COLOUR_DIM)
    essence = torch.ones(B, d)
    in_values = torch.zeros(B, A, P, dtype=torch.int32)
    in_roles = torch.full((B, P), AUX_CE_IGNORE, dtype=torch.int64)
    out_slot = torch.full((B, MAX_OUTPUT_SLOTS), AUX_CE_IGNORE, dtype=torch.int64)
    out_slot[0, 0], out_slot[0, 1] = 5, 9
    out_values = torch.zeros(B, A, 2, dtype=torch.int32)
    out_values[0, 0, 0], out_values[0, 0, 1] = 10, 30

    def slot(v):
        bits = _value_bits(torch.tensor(v)).float() * 40 - 20     # confident logits
        return torch.cat([bits, _value_smooth(torch.tensor(float(v))).view(1)])
    fixed = torch.zeros(MAX_OUTPUT_SLOTS * _VP_SLOT)
    fixed[:_VP_SLOT], fixed[_VP_SLOT:2 * _VP_SLOT] = slot(30), slot(10)   # swapped
    comp = _vp_comp(_FixedHead(fixed), d, de)
    loss, count = vp_loss(comp, T, essence, in_values, out_slot, in_roles, out_values,
                          torch.tensor([True, False]), return_count=True, n_out=2)
    assert count == 2
    assert float(loss) < 1e-6
    masked = vp_loss(comp, T, essence, in_values, out_slot, in_roles, out_values,
                     torch.tensor([False, False]), n_out=2)
    assert float(masked) == 0.0


def test_gen_shipped_values_re_execute_from_their_inputs():
    """Every shipped output value re-executes from the chunk's shipped
    role-ordered input values alone, over register-dataflow chunks. (An output computed
    from the PC after a jump also depends on the jump's operands, which
    in_roles does not list; PC effects are out of scope.)"""
    from datagen.batch import collect_into_batches, generate_chunks
    from datagen.compare import draw_anchor_states
    from datagen.generate import build_opcode_table, length_cap

    rng = np.random.default_rng(0)
    alu = build_opcode_table({'R_ALU': 0.55, 'I_ALU': 0.45})
    chunks = generate_chunks(length_cap(4), rng, opcode_table=alu,
                             chain_rate=0.3)
    batch = next(collect_into_batches(
        chunks, batch_size=256,
        draw_anchors=lambda k: draw_anchor_states(rng, k * 8).reshape(k, 8, -1),
        max_chunk_len=4))

    from datagen.tests.test_batch import _reexecuted
    assert batch.out_values.shape[0] == batch.tokens.shape[0]   # OB == B (T2)
    for i in np.nonzero(batch.out_regs_valid)[0]:
        chunk = _decode_chunk_instructions(batch, i)
        got = _reexecuted(chunk, batch.out_slot_regs[i], batch.in_roles[i],
                          batch.in_role_values[i], 4)
        assert np.array_equal(got, batch.out_values[i]), chunk


def _decode_chunk_instructions(batch, row_idx):
    """Decode all instructions in chunk row_idx back to Instruction
    objects. Returns the list, or None if any decode fails or the
    chunk is empty/invalid. Caller masks via batch.valid."""
    if not batch.valid[row_idx]:
        return None
    instr_lens = batch.instr_lens[row_idx]
    tokens = batch.tokens[row_idx]
    instrs = []
    offset = 0
    for j in range(len(instr_lens)):
        L = int(instr_lens[j])
        if L == 0:
            continue
        toks = tokens[offset:offset + L].tolist()
        try:
            instr, _ = decode_instruction(toks, 0)
        except Exception:
            return None
        instrs.append(instr)
        offset += L
    return instrs if instrs else None


def _small_batches(n):
    import itertools
    from datagen.batch import collect_into_batches, generate_chunks
    from datagen.compare import draw_anchor_states
    from datagen.generate import length_cap
    rng = np.random.default_rng(0)
    chunks = generate_chunks(length_cap(2), rng)
    return list(itertools.islice(collect_into_batches(
        chunks, batch_size=16,
        draw_anchors=lambda k: draw_anchor_states(rng, k * 8).reshape(k, 8, -1),
        max_chunk_len=2), n))


def test_prefetcher_keeps_stream_order_across_workers():
    """The prefetcher spreads prep over several worker processes; the trainer
    must still see every batch, in stream order — including a stream length
    the worker count doesn't divide."""
    from compressor.train import (TrainBatchPrefetcher, _pack_step_arrays,
                                  _split_to_per_instruction)
    pack = dict(vp=True, valid=True, distance=True)
    batches = _small_batches(7)
    got = list(TrainBatchPrefetcher(iter(batches), pack, maxsize=6,
                                    n_workers=3))
    assert len(got) == len(batches)
    for want, item in zip(batches, got):
        ref = _pack_step_arrays(want, _split_to_per_instruction(want), **pack)
        assert np.array_equal(item[0], ref[0])
        assert item[1:] == ref[1:]


def test_packed_step_stages_to_the_arrays_it_packed():
    """A packed step must stage to exactly the tensors the trainer feeds the
    compiled step: each batch array cast to its dtype, in _prep's order."""
    import torch
    from compressor.train import (_H2DStager, _dense_chunk_grid,
                                  _pack_step_arrays, _split_to_per_instruction)
    for b in _small_batches(3):
        split = _split_to_per_instruction(b)
        packed, specs, _, shape = _pack_step_arrays(b, split, vp=True, valid=True,
                                                    distance=True)
        tok, pad, g0, g1, go, act = _dense_chunk_grid(split, *shape)
        L, F = torch.long, torch.float32
        want = [(tok, L), (pad, None), (g0, L), (g1, L), (go, L), (act, F),
                (b.live_in_mask, F), (b.live_out_mask, F), (b.pc_writes, F),
                (b.in_slot_regs, L), (b.out_slot_regs, L), (b.out_values, None),
                (b.out_regs_valid, None), (b.in_role_values, None), (b.in_roles, L),
                (b.valid, F)]
        got = _H2DStager('cpu')(packed, specs)
        assert len(got) == len(want) + 8     # + the register-file distance fields
        assert torch.equal(got[-8], torch.from_numpy(b.rf_pos))
        assert torch.equal(got[-7], torch.from_numpy(b.rf_pos_zero))
        assert torch.equal(got[-5], torch.from_numpy(b.rf_wo))
        assert torch.equal(got[-2], torch.from_numpy(b.rf_wo_free))
        assert torch.equal(got[-1], torch.from_numpy(b.n_wo).long())
        for g, (a, dt) in zip(got, want):
            w = torch.from_numpy(a) if dt is None else torch.from_numpy(a).to(dt)
            assert g.dtype == w.dtype and torch.equal(g, w)


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

    # Only a slot the instruction touched (a nonzero state row) is a candidate.
    T = torch.zeros(N, n_regs, 4)
    for b, slots in enumerate(((3, 5, 7), (2, 4), (6,))):
        T[b, list(slots)] = 1.0
    in0, in1, out = _t1_predicted_wiring(_Stub(), T)
    assert in0.tolist() == [3, 2, 0]     # first operand / single / none
    assert in1.tolist() == [5, 0, 0]     # second operand / none / none
    assert out.tolist() == [7, 4, 6]
    # x0 never selected even if it scored highest.
    in_s[2, 0] = 100.0; live_in[2, 0] = 10.0
    i0b, _, _ = _t1_predicted_wiring(_Stub(), T)
    assert i0b[2].item() == 0            # x0 masked out of contention -> absent
    # Nor an untouched slot: a live head that fires on it cannot name it, since
    # every untouched slot scores alike and a tie among them is not equivariant.
    in_s[1, 1] = 50.0; live_in[1, 1] = 10.0
    _, i1c, _ = _t1_predicted_wiring(_Stub(), T)
    assert i1c[1].item() == 0


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
    go to ~0 simultaneously — there is no irreducible two-loss floor."""
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
    """T1 carries the per-slot binding heads, so the shared binding_losses
    runs on a real (T, essence)."""
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
    T, ess = enc.encode_state(
        tok, pad, torch.tensor([i0]), torch.tensor([i1]), torch.tensor([o]))

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_lr_clock_reproduces_the_float_schedule_and_resumes():
    """On CUDA, Adam's lr is a device tensor driven by the float schedule
    running on a stand-in. Adam must see exactly the CPU schedule's values
    (to fp32), and a captured train state must resume onto the same sequence."""
    from compressor.train import capture_train_state, restore_train_state
    def run(device, n, state=None):
        p = [torch.nn.Parameter(torch.zeros(2, device=device))]
        opt, sched = build_optim_sched(p, 1e-3, 40, warmup_steps=5,
                                       lr_min=1e-6, device=device)
        if state is not None:
            restore_train_state(state, opt, sched)
        lrs = []
        for _ in range(n):
            lrs.append(float(opt.param_groups[0]['lr']))
            p[0].grad = torch.ones_like(p[0])
            opt.step()
            sched.step()
        return lrs, capture_train_state(n, opt, sched)
    want, _ = run('cpu', 40)
    got, _ = run('cuda', 40)
    assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(want, got))
    first, state = run('cuda', 17)
    rest, _ = run('cuda', 23, state)
    assert all(math.isclose(a, b, rel_tol=1e-6)
               for a, b in zip(want, first + rest))


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
# T2Compressor forward
# ===========================================================================

def test_t2compressor_forward_shape_and_padding_contract():
    """The equivariant T2 runs the register-state machine over a chunk of
    frozen-T1 essences threaded by per-instruction wiring. encode_state ->
    (T (B,n_regs,d_out+d_event+d_colour), essence (B,d_out)); padded steps (active=0)
    are no-ops, so a chunk shorter than N does not NaN."""
    torch.manual_seed(0)
    t2 = T2Compressor(d_t1=8, d_out=12, d_event=16)
    B, N, R = 3, 4, t2.n_regs
    seq = torch.randn(B, N, 8)                       # per-instr T1 essences
    in0 = torch.randint(1, R, (B, N)); in1 = torch.randint(1, R, (B, N))
    out = torch.randint(1, R, (B, N))
    # row 2 is a 1-instruction chunk (rest padded -> active=0, no-op).
    active = torch.ones(B, N); active[2, 1:] = 0.0

    T, essence = t2.encode_state(seq, in0, in1, out, active)
    assert T.shape == (B, R, 12 + 16 + t2.d_colour)
    assert essence.shape == (B, 12)
    assert not torch.isnan(T).any() and not torch.isnan(essence).any()
    # encode() returns just the essence.
    assert torch.allclose(t2.encode(seq, in0, in1, out, active), essence)


def test_out_slots_beyond_the_instruction_count_are_ignore():
    """`train_compressor` bounds its ListMLE loops at OUT_K = min(n_instrs, 16)
    because a K-instruction chunk writes at most K registers, so every later
    output slot is AUX_CE_IGNORE. Assert that on real oracle output rather than
    trusting the arithmetic."""
    import itertools
    import numpy as np
    from datagen import (AUX_CE_IGNORE, precompute_chunk, draw_anchor_states,
                         generate_chunks, length_cap)
    anchors = draw_anchor_states(np.random.default_rng(0), 8)
    rng = np.random.default_rng(0)
    n_checked = 0
    for chunk in itertools.islice(generate_chunks(length_cap(2), rng), 300):
        try:
            pre = precompute_chunk(chunk.instructions, anchors)
        except NotImplementedError:
            continue                      # memory ops carry no supervision
        assert len(chunk.instructions) <= 2
        assert (pre.out_slot_regs[2:] == AUX_CE_IGNORE).all(), chunk
        n_checked += 1
    assert n_checked > 50


def test_every_probe_and_entry_script_compiles():
    """A probe is never imported by the suite, so a syntax error in one passes
    every test and only surfaces when someone runs it hours later. Compiling
    them costs milliseconds."""
    import py_compile
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    files = sorted(root.glob('probes/*.py')) + sorted(root.glob('scripts/*.py'))
    assert len(files) > 5, files
    for f in files:
        py_compile.compile(str(f), doraise=True, cfile=str(f) + 'c')
        Path(str(f) + 'c').unlink(missing_ok=True)


_RF = ('rf_pos', 'rf_pos_zero', 'rf_pos_one', 'rf_wo', 'rf_wo_zero', 'rf_wo_one',
       'rf_wo_free', 'n_wo')


def _rf_batch(chunks, max_len):
    """A packed batch of `chunks` and its distance fields as tensors."""
    from datagen import draw_anchor_states
    from datagen.batch import build_targets, pack_batch, _make_valid_chunk
    rows = [_make_valid_chunk(c) for c in chunks]
    rng = np.random.default_rng(0)
    aux, ov, oiv, orv = build_targets(
        rows, lambda k: np.stack([draw_anchor_states(rng, 2) for _ in range(k)]), max_len)
    b = pack_batch(rows, target_B=len(rows), target_max_tokens=9 * max_len,
                   target_max_n_instrs=max_len, out_values=ov, in_role_values=oiv,
                   out_regs_valid=orv, aux_payload=aux)
    dt = {'rf_wo_free': torch.float32, 'rf_pos': torch.int32, 'rf_wo': torch.int32}
    return [torch.from_numpy(getattr(b, k)).to(dt.get(k, torch.int64)) for k in _RF]


_A = [Instruction('ADDI', 4, 3, 42)]
_CASES = [
    (_A, _A + [Instruction('ADDI', 7, 7, 0)]),          # a write that changes nothing
    (_A, _A + [Instruction('ADDI', 8, 7, 0)]),          # a copy
    (_A, _A + [Instruction('ADDI', 7, 7, 1)]),          # an in-place step
    (_A, [Instruction('ADDI', 4, 3, 43)]),              # an immediate moved by one
    (_A, [Instruction('ADDI', 9, 12, 42)]),             # a rename
    ([Instruction('SUB', 5, 6, 6)], [Instruction('ADDI', 5, 0, 1)]),   # zero against 1
    ([Instruction('ADDI', 5, 0, 0)], [Instruction('ADDI', 5, 0, 1)]),
    ([Instruction('XORI', 10, 20, -312), Instruction('ANDI', 19, 13, 1349)],
     [Instruction('XORI', 10, 20, -312), Instruction('ANDI', 19, 13, 1348)]),
    (_A, []),
]


def test_register_distance_targets_track_the_reference_distance():
    """Against tanh(d*) (`datagen.ideal`): exact zeros where d* is 0, within
    0.03 elsewhere (the two are averaged over different state families)."""
    from compressor.train import register_distance_targets
    from datagen.ideal import ideal_distance
    rows = [c for p in _CASES for c in p]
    D = register_distance_targets(*_rf_batch(rows, 2))
    for k, (a, b) in enumerate(_CASES):
        want = math.tanh(ideal_distance(a, b))
        got = float(D[2 * k, 2 * k + 1])
        if want == 0.0:
            assert got == 0.0, (a, b, got)
        else:
            assert abs(got - want) < 0.03, (a, b, got, want)
    assert torch.equal(D, D.t())


def test_every_partner_rewrite_is_at_zero():
    from compressor.train import register_distance_targets
    from datagen.generate import sample_chunk_partner_pair
    rng = np.random.default_rng(5)
    pairs = [sample_chunk_partner_pair(rng, 2) for _ in range(60)]
    D = register_distance_targets(*_rf_batch([c for p in pairs for c in p], 2))
    for k, (a, b) in enumerate(pairs):
        assert float(D[2 * k, 2 * k + 1]) == 0.0, (a, b)


def test_column_distance_matches_the_datagen_column_distance():
    from compressor.train import _column_distance, _inv_log_bound
    from datagen.knownbits import column_distance, observed_bits
    rng = np.random.default_rng(2)
    cols = [rng.integers(-2 ** 31, 2 ** 31, 64).astype(np.int32),
            (rng.integers(0, 2, 64)).astype(np.int32),
            np.full(64, 7, np.int32), np.full(64, 9, np.int32)]
    cols.append(cols[0] + 1)
    bits = [observed_bits(c) for c in cols]
    t = torch.from_numpy(np.stack(cols))
    z = torch.tensor([b[0] for b in bits]); o = torch.tensor([b[1] for b in bits])
    inv = _inv_log_bound(z[:, None], o[:, None], z[None], o[None])
    D = _column_distance(t[:, None], t[None], inv)
    for i in range(len(cols)):
        for j in range(len(cols)):
            want = column_distance(cols[i], bits[i], cols[j], bits[j])
            assert abs(float(D[i, j]) - want) < 1e-5, (i, j)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_graphed_distance_targets_equal_the_eager_ones():
    """The fused kernels, replayed as a graph against new contents at the
    captured addresses, give the eager targets."""
    from compressor.train import GraphedDistanceTargets, register_distance_targets
    from datagen.generate import sample_chunk_partner_pair
    dev = torch.device('cuda')
    g = GraphedDistanceTargets()
    bufs = None
    for seed in (3, 4, 5):
        rng = np.random.default_rng(seed)
        rows = [c for _ in range(24) for c in sample_chunk_partner_pair(rng, 2)]
        new = [t.to(dev) for t in _rf_batch(rows, 2)]
        if bufs is None:
            bufs = [torch.empty_like(t) for t in new]
        for b, t in zip(bufs, new):
            b.copy_(t)
        got = g(*bufs).clone()
        want = register_distance_targets(*bufs, fused=False)
        assert torch.allclose(got, want, atol=1e-5)


def test_distance_loss_targets_cosine_distance_and_weights_small_targets_up():
    """d is the target cosine distance — orthogonal rows at d = 1 and
    coincident rows at d = 0 cost nothing — and a pair's error weighs
    1 / (target chord + DISTANCE_EPS)."""
    from compressor.train import distance_loss, DISTANCE_EPS
    e = torch.eye(4)[:3]
    mask = torch.ones(3, dtype=torch.bool)
    assert float(distance_loss(e, torch.ones(3, 3), mask)) < 1e-6
    e2 = torch.stack([e[0], e[0], e[1]])               # rows 0 and 1 coincide
    t2 = torch.tensor([[0., 0., 1.], [0., 0., 1.], [1., 1., 0.]])
    assert float(distance_loss(e2, t2, mask)) < 1e-6
    t3 = t2.clone(); t3[0, 1] = t3[1, 0] = 1.0         # call the coincident pair unrelated
    c = math.sqrt(2.0)
    w = 1.0 / (c + DISTANCE_EPS)
    want = (2.0 * w) / (3 * w)                          # its chord misses by sqrt(2)
    assert abs(float(distance_loss(e2, t3, mask)) - want) < 1e-4
    t4 = t2.clone(); t4[0, 1] = t4[1, 0] = 0.5          # a nearer target weighs more
    c4 = 1.0
    w4 = 1.0 / (c4 + DISTANCE_EPS)
    want4 = (1.0 * w4) / (w4 + 2 * w)
    assert abs(float(distance_loss(e2, t4, mask)) - want4) < 1e-4


def test_vp_assignment_is_exact_by_enumeration_and_greedy_past_four():
    """Up to four outputs the least-cost assignment of every permutation; past
    four a greedy one, which recovers a planted assignment whose costs leave
    no tie."""
    from compressor.train import vp_assignment
    g = torch.Generator().manual_seed(0)
    for K in (2, 3, 4, 6):
        perm = torch.randperm(K, generator=g)
        cost = torch.rand(5, K, K, generator=g) + 1.0
        cost[:, perm, torch.arange(K)] = 0.01                 # slot perm[j] serves output j
        slot, total = vp_assignment(cost, torch.ones(5, K))
        assert torch.equal(slot, perm.expand(5, -1)), K
        assert torch.allclose(total, torch.full((5,), 0.01 * K))
