#!/usr/bin/env python3
"""Is the depth-2 vp deficit in the ESSENCE or in the vp HEAD?

An output slot is at dataflow depth 2 when the instruction that wrote it read
another instruction's result rather than a chunk input. Those slots carry an
order of magnitude more vp error than depth-1 slots. Three causes predict that
same symptom, and they call for different fixes:

  ESSENCE — the encoder never represented the composition. The op-cell's output
    geometry has to be legible to its own input on the next step, a demand the
    depth-1 case (every operand a freshly seeded leaf) never makes.
  HEAD BUDGET — the essence carries it, but depth-2 is ~2.5% of supervised
    slots and so contributes ~2.5% of the head's gradient.
  HEAD DEPTH — a depth-2 slot's value is f2(f1(a,b),c): the head must chain two
    32-bit arithmetic ops where a depth-1 slot needs one. The trained head has a
    single hidden layer.

All arms read FROZEN essences and are scored on held-out rows, so any gain is
attributable to the head alone; whatever gap survives every head-side arm is
the encoder's.

  (A) the model's OWN trained head — the reference, and the calibration for
      every other arm: a fit scoring below it is measuring its own budget.
  (B) that head fine-tuned on the natural slot mix — controls for "more fitting
      at all", so B-vs-C is the effect of the BUDGET SPLIT, not of fine-tuning.
  (C) that head fine-tuned on depth-2 slots only.
  (D) a fresh single-hidden-layer head, depth-2 slots only.
  (E) a fresh four-hidden-layer head, same data and step budget as D — D-vs-E
      is the effect of HEAD DEPTH with everything else held equal.

Depth-1 slots are reported split by whether their chunk also contains a depth-2
slot, which separates "composed slots are hard" from "chunks that compose are
hard".

Usage: depth_composition.py <t2_dir> <t1_path> [n_batches] [steps]
"""
import copy
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.model import _build_vp_head
from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              _value_bits, _value_smooth)
from datagen import (RVT_FORMAT, Batch, AUX_CE_IGNORE, MAX_INPUT_SLOTS,
                     MAX_OUTPUT_SLOTS, N_REGS, make_anchor_states)
from scripts._common import resolve_device, load_frozen_encoder, load_t2

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
LEN, N_ANCHORS, ANCHOR_SEED, NBITS = 2, 8, 0, 32
MB = 512

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)

CORPUS = f'/tmp/depth_comp_len{LEN}.rvt'
subprocess.run(
    f'python -m datagen --rule "cap={LEN}" --twins 0 --batch-size 128 '
    f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
    f'--config configs/instr_alu_writers.json --count {NB} > {CORPUS}',
    shell=True, check=True, stderr=subprocess.DEVNULL)

anchors = torch.from_numpy(
    np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)                  # (A, max_in)


def slot_depth(in0, in1, out, n):
    """Dataflow depth of every register slot after n instructions: 1 if written
    from chunk inputs only, 2 if an operand was itself written here. Slot 0 (x0)
    is never written."""
    d = np.zeros(N_REGS, dtype=np.int64)
    for k in range(n):
        i0, i1, o = int(in0[k]), int(in1[k]), int(out[k])
        nd = 1 + max(d[i0], d[i1])
        if o != 0:
            d[o] = nd
    return d


ess_l, fill_l, tgt_l, valid_l, depth_l = [], [], [], [], []
with open(CORPUS, 'rb') as f:
    for batch in RVT_FORMAT.reader(f, Batch):
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        (_, _, chunk_idx, slot_idx, n_per_chunk, w0, w1, wo) = split
        nch, mni = batch.instr_lens.shape
        with torch.no_grad():
            ct1, c0, c1, co, cact = _t2_assemble(
                t1, split, nch, mni, dev, route='binding')
            _, ess = t2.core(ct1, c0, c1, co, cact,
                             torch.randn(nch, t2.n_regs, device=dev))

        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        vp_mask = torch.from_numpy(batch.out_regs_valid).to(dev)
        valid = (out_slot != AUX_CE_IGNORE) & vp_mask.unsqueeze(-1)
        tgt = out_regs.gather(-1, out_slot.clamp(min=0).unsqueeze(1)
                              .expand(-1, N_ANCHORS, -1))       # (B,A,MAX_OUT)

        # depth of the slot each supervised output position refers to
        order = np.argsort(chunk_idx * 1000 + slot_idx)
        wo_s, w0_s, w1_s = wo[order], w0[order], w1[order]
        oreg = out_slot.clamp(min=0).cpu().numpy()
        vnp = valid.cpu().numpy()
        dep = np.zeros((nch, MAX_OUTPUT_SLOTS), dtype=np.int64)
        pos = 0
        for b in range(nch):
            n = int(n_per_chunk[b])
            if n:
                dv = slot_depth(w0_s[pos:pos + n], w1_s[pos:pos + n],
                                wo_s[pos:pos + n], n)
                dep[b] = dv[oreg[b]] * vnp[b]
            pos += n

        ess_l.append(ess.float())
        fill_l.append(in_slot != AUX_CE_IGNORE)
        tgt_l.append(tgt.to(torch.int32))
        valid_l.append(valid)
        depth_l.append(torch.from_numpy(dep).to(dev))

ESS = torch.cat(ess_l)                                   # (N, d_out)
FILL = torch.cat(fill_l)                                 # (N, max_in) bool
TGT = torch.cat(tgt_l)                                   # (N, A, MAX_OUT) int32
VALID = torch.cat(valid_l)                               # (N, MAX_OUT) bool
DEPTH = torch.cat(depth_l)                               # (N, MAX_OUT)
N = ESS.shape[0]
del ess_l, fill_l, tgt_l, valid_l, depth_l

# Reporting groups. A depth-1 slot in a chunk that also composes is scored apart
# from one in a chunk that does not: if both are degraded the chunk is hard, if
# only the composed slot is, composition is.
COMPOSES = ((DEPTH == 2) & VALID).any(-1, keepdim=True)
GROUPS = (('depth 1, plain chunk', (DEPTH == 1) & VALID & ~COMPOSES),
          ('depth 1, composing chunk', (DEPTH == 1) & VALID & COMPOSES),
          ('depth 2', (DEPTH == 2) & VALID))

cut = int(0.8 * N)
TR = torch.arange(cut, device=dev)
TE = torch.arange(cut, N, device=dev)
VP_IN = t2.d_out + MAX_INPUT_SLOTS * (NBITS + 1)
VP_OUT = MAX_OUTPUT_SLOTS * (NBITS + 1)


def head_input(idx):
    """(len(idx), A, VP_IN) — exactly the features t2_value_predict_loss feeds:
    the unit-normalized essence with the CANONICAL input values appended."""
    iv = ANCH_IN.unsqueeze(0) * FILL[idx].to(torch.int64).unsqueeze(1)
    ifeat = torch.cat(
        [_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
        dim=-1).reshape(idx.shape[0], N_ANCHORS, -1)
    vexp = F.normalize(ESS[idx], dim=-1).unsqueeze(1).expand(-1, N_ANCHORS, -1)
    return torch.cat([vexp, ifeat], dim=-1)


def score(head, idx, chunks=256):
    """Per-bit accuracy and exact 32-bit match over (row, slot, anchor) for each
    reporting group. Returns a list of (bit_hits, exact_hits, n)."""
    acc = [[0.0, 0.0, 0.0] for _ in GROUPS]
    with torch.no_grad():
        for s in range(0, idx.shape[0], chunks):
            i = idx[s:s + chunks]
            pred = head(head_input(i)).reshape(
                i.shape[0], N_ANCHORS, MAX_OUTPUT_SLOTS, NBITS + 1)
            hit = ((pred[..., :NBITS] > 0).float()
                   == _value_bits(TGT[i].to(torch.int64)))       # (b,A,S,32)
            bit, exact = hit.float().mean(-1), hit.all(-1).float()
            for g, (_, sel) in enumerate(GROUPS):
                m = sel[i].unsqueeze(1).float()
                acc[g][0] += float((bit * m).sum())
                acc[g][1] += float((exact * m).sum())
                acc[g][2] += float(m.expand_as(bit).sum())
    return acc


def report(name, acc):
    print(f'  {name}')
    for (label, _), (b, e, n) in zip(GROUPS, acc):
        if n < 1:
            continue
        print(f'    {label:26s} per-bit {100 * b / n:8.4f}%   '
              f'exact 32-bit {100 * e / n:8.4f}%   (n={int(n)})')


def fit(head, depth_only=None, seed=0, lr=3e-4):
    """Train `head` on frozen essences with the training loss, restricted to one
    dataflow depth when `depth_only` is set."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr)
    w = VALID.float()
    if depth_only is not None:
        w = w * (DEPTH == depth_only).float()
    rows = TR[w[TR].sum(-1) > 0]                 # rows with a supervised slot
    rng = np.random.default_rng(seed)
    for _ in range(STEPS):
        i = rows[torch.from_numpy(rng.integers(0, rows.numel(), MB)).to(dev)]
        pred = head(head_input(i)).reshape(
            MB, N_ANCHORS, MAX_OUTPUT_SLOTS, NBITS + 1)
        tb = _value_bits(TGT[i].to(torch.int64))
        bce = F.binary_cross_entropy_with_logits(
            pred[..., :NBITS], tb, reduction='none').mean(-1)
        sm = (pred[..., NBITS] - _value_smooth(TGT[i].float())) ** 2
        m = w[i].unsqueeze(1)
        loss = ((bce + 0.2 * sm) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    return head


print()
print(f'checkpoint = {T2_DIR}   len={LEN}   rows={N}  '
      f'(train {cut}, held-out {N - cut})   fit steps={STEPS}')
print(f'  supervised slots: depth-1 {int(((DEPTH == 1) & VALID).sum())}, '
      f'depth-2 {int(((DEPTH == 2) & VALID).sum())}, '
      f'depth-3+ {int(((DEPTH > 2) & VALID).sum())}')
print()
report("(A) the model's own trained head", score(t2.vp_head, TE))
print()
report('(B) that head fine-tuned, natural mix',
       score(fit(copy.deepcopy(t2.vp_head), lr=1e-4), TE))
print()
report('(C) that head fine-tuned, depth-2 only',
       score(fit(copy.deepcopy(t2.vp_head), depth_only=2, lr=1e-4), TE))
print()
report('(D) fresh 1-hidden-layer head, depth-2 only',
       score(fit(_build_vp_head(VP_IN, VP_OUT, 1).to(dev), depth_only=2), TE))
print()
report('(E) fresh 4-hidden-layer head, depth-2 only',
       score(fit(_build_vp_head(VP_IN, VP_OUT, 4).to(dev), depth_only=2), TE))
print()
