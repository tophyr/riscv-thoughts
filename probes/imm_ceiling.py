#!/usr/bin/env python3
"""Where is the I-type immediate lost — in T1's essence, or in T2's?

The value-prediction residual on two-instruction chunks is concentrated in
I-type instructions and in bit positions 0-11, the width of a RISC-V I-type
immediate. That residual is already present on SINGLE instructions, so it is
not a property of combining instructions. T2 ingests one frozen-T1 essence per
instruction, so a T2 essence can only carry an immediate that the T1 essence it
was built from carried first. This probe separates the two tiers.

Corpus: one-instruction chunks of register-writing ALU instructions, reported
split by whether the instruction is R-type (no immediate) or I-type (carrying a
12-bit immediate):

(A) The tiers' OWN trained value-prediction heads — exact 32-bit match of the
    predicted output register value at each anchor state. Whichever tier first
    falls below R-type accuracy is the tier that loses the immediate.

(B) A freshly fitted readout of the 12 immediate bits from each essence, scored
    on held-out rows as exact 12-bit match. This measures what the essence
    CONTAINS, independent of whether the trained head uses it: a readout that
    recovers the immediate from an essence whose own head cannot predict the
    output value locates the deficit in the head rather than the representation.

Usage: imm_ceiling.py <t2_dir> <t1_path> [n_batches]
"""
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              _value_bits, _value_smooth)
from datagen import (RVT_FORMAT, Batch, AUX_CE_IGNORE, MAX_INPUT_SLOTS,
                     N_REGS, make_anchor_states)
from emulator import I_TYPE
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 300
N_ANCHORS, ANCHOR_SEED, NBITS, IMM_BITS = 8, 0, 32, 12
HID, STEPS, MB = 1024, 8000, 4096

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)

CORPUS = '/tmp/imm_ceiling_len1.rvt'
subprocess.run(
    f'python -m datagen --rule "cap=1" --twins 0 --batch-size 128 '
    f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
    f'--config configs/instr_alu_writers.json --count {NB} > {CORPUS}',
    shell=True, check=True, stderr=subprocess.DEVNULL)

anchors = torch.from_numpy(
    np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)


def vp_exact(head, ess, in_slot, out_slot, out_regs):
    """Exact 32-bit match of `head`'s prediction for the chunk's single output
    slot, per (row, anchor). Returns (rows,) float in [0, 1]."""
    fill = (in_slot != AUX_CE_IGNORE).to(torch.int64)
    iv = ANCH_IN.unsqueeze(0) * fill.unsqueeze(1)
    ifeat = torch.cat([_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
                      dim=-1).reshape(ess.shape[0], N_ANCHORS, -1)
    vexp = F.normalize(ess, dim=-1).unsqueeze(1).expand(-1, N_ANCHORS, -1)
    pred = head(torch.cat([vexp, ifeat], dim=-1)).reshape(
        ess.shape[0], N_ANCHORS, -1, NBITS + 1)
    tgt = out_regs.gather(-1, out_slot.clamp(min=0).unsqueeze(1)
                          .expand(-1, N_ANCHORS, -1))
    hit = (pred[..., :NBITS] > 0).float() == _value_bits(tgt)
    return hit.all(-1)[:, :, 0].float().mean(1)          # slot 0 = the writer


ess1, ess2, slot2, pv2, imm, is_i, ok1, ok2 = [], [], [], [], [], [], [], []
with open(CORPUS, 'rb') as f:
    for batch in RVT_FORMAT.reader(f, Batch):
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        (itok, ipad, chunk_idx, _, n_per_chunk, w0, w1, wo) = split
        nch, mni = batch.instr_lens.shape
        with torch.no_grad():
            _, e1 = t1.encode_state(
                torch.from_numpy(itok).to(dev).long(),
                torch.from_numpy(ipad).to(dev),
                torch.from_numpy(w0).to(dev), torch.from_numpy(w1).to(dev),
                torch.from_numpy(wo).to(dev))
            ct1, c0, c1, co, cact = _t2_assemble(t1, split, nch, mni, dev,
                                                 route='binding')
            T2_state, e2 = t2.core(ct1, c0, c1, co, cact)
            # the one slot this chunk writes, before the pool sees it
            w2 = T2_state.gather(
                1, co[:, :1].unsqueeze(-1).expand(-1, -1, T2_state.shape[-1])
            ).squeeze(1)
            # The pool's ENTIRE view of that slot: its `pool_v` projection, and
            # its `pool_k` projection scored against the queries. A slot reaches
            # the essence only through those two vectors, so their concatenation
            # upper-bounds what any pooling of them can carry about it. A
            # checkpoint trained when all heads shared one d_value/n_pool_heads
            # projection loads with that projection repeated per head, so the
            # measurement counts the distinct dimensions it actually had.
            # One instruction means one write step, so the written slot's
            # write-time stamp is the whole ramp: (k+1)/K = 1.
            psrc = torch.cat(
                [w2[:, :t2.core.d_value],
                 torch.ones(w2.shape[0], 1, device=dev)], dim=-1)
            pview = torch.cat(
                [t2.core.pool_v(psrc),
                 t2.core.pool_k(psrc)
                 @ t2.core.pool_q.view(t2.core.n_pool_heads, -1).t()], dim=-1)

        # Chunks holding exactly one decodable instruction that writes a
        # register: the rows where an immediate is behaviorally live and a
        # single output slot is supervised.
        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        keep, im, ityp, e1_rows = [], [], [], []
        pos = 0
        for b in range(nch):
            n = int(n_per_chunk[b])
            if (n == 1 and batch.out_regs_valid[b]
                    and int(out_slot[b, 0]) != AUX_CE_IGNORE):
                try:
                    ins, _ = decode_instruction(batch.tokens[b].tolist(), 0)
                except Exception:
                    ins = None
                if ins is not None and int(wo[pos]) != 0:
                    keep.append(b); e1_rows.append(pos)
                    ityp.append(ins.opcode in I_TYPE)
                    im.append(ins.args[2] if ins.opcode in I_TYPE else 0)
            pos += n
        if not keep:
            continue
        kb = torch.tensor(keep, device=dev)
        kr = torch.tensor(e1_rows, device=dev)
        with torch.no_grad():
            ok1.append(vp_exact(t1.vp_head, e1[kr], in_slot[kb], out_slot[kb],
                                out_regs[kb]))
            ok2.append(vp_exact(t2.vp_head, e2[kb], in_slot[kb], out_slot[kb],
                                out_regs[kb]))
        ess1.append(e1[kr]); ess2.append(e2[kb]); slot2.append(w2[kb])
        pv2.append(pview[kb])
        imm.append(torch.tensor(im, device=dev))
        is_i.append(torch.tensor(ityp, device=dev))

ESS1, ESS2, SLOT2 = torch.cat(ess1), torch.cat(ess2), torch.cat(slot2)
PVIEW = torch.cat(pv2)
IMM, ISI = torch.cat(imm), torch.cat(is_i)
OK1, OK2 = torch.cat(ok1), torch.cat(ok2)
N = IMM.numel()

print()
print(f'  corpus = one-instruction chunks, n={N}  '
      f'(I-type {int(ISI.sum())}, R-type {int((~ISI).sum())})')
print()
print('  (A) each tier\'s own trained vp head — exact 32-bit match of the '
      'output value')
print(f'  {"":10s} {"R-type":>10s} {"I-type":>10s}')
for name, ok in (('T1', OK1), ('T2', OK2)):
    print(f'  {name:10s} {100 * ok[~ISI].mean():9.3f}% {100 * ok[ISI].mean():9.3f}%')


def fit_imm(ESS, tr, te):
    """Fit essence -> the 12 immediate bits; return (per-bit, exact) on `te`."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(ESS.shape[1], HID), nn.ReLU(),
                        nn.Linear(HID, HID), nn.ReLU(),
                        nn.Linear(HID, IMM_BITS)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    tgt = ((IMM.unsqueeze(-1) >> torch.arange(IMM_BITS, device=dev)) & 1).float()
    x = F.normalize(ESS, dim=-1)
    rng = np.random.default_rng(0)
    for _ in range(STEPS):
        i = tr[torch.from_numpy(rng.integers(0, tr.numel(), MB)).to(dev)]
        loss = F.binary_cross_entropy_with_logits(net(x[i]), tgt[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        hit = (net(x[te]) > 0).float() == tgt[te]
    return hit.float().mean(), hit.all(-1).float().mean()


# I-type rows only: an R-type instruction has no immediate to recover.
idx = torch.nonzero(ISI).squeeze(-1)
cut = int(0.8 * idx.numel())
TR, TE = idx[:cut], idx[cut:]
print()
print('  (B) fitted readout of the 12 immediate bits, held-out I-type rows '
      f'(n={TE.numel()})')
print(f'  {"":10s} {"per-bit":>10s} {"exact 12-bit":>14s}')
for name, E in (('T1 essence', ESS1), ('T2 slot', SLOT2),
                ('pool view', PVIEW), ('T2 essence', ESS2)):
    bit, ex = fit_imm(E, TR, TE)
    print(f'  {name:10s} {100 * bit:9.3f}% {100 * ex:13.3f}%')
print()
