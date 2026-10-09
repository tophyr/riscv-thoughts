#!/usr/bin/env python3
"""Isolate SUPERPOSITION: is all remaining vp loss caused by a second output
sharing the essence?

Corpus: len=2, R-type only (no immediates), filtered to DATA-INDEPENDENT pairs
(instr 2 neither reads instr 1's destination nor writes the same register).

Each chunk is encoded TWICE:
  both  - active=[1,1], so the essence must hold TWO outputs
  solo  - active=[1,0], instruction 2 masked off; ONE output in the essence

`value[out] = op_cell(content, r0, r1)` has no K dependence and the two writes
touch disjoint slots, so slot A's value is BIT-IDENTICAL between the runs (this
script asserts that rather than assuming it). Instruction A is written first, so
its canonical input positions (1,2) and therefore its vp target are unchanged.
The ONLY difference is whether B's output also occupies the essence.

Also splits pool-vs-head: the model's own vp_head must decode 16 output slots
from one vector through a single hidden layer, so a deficit could be the READOUT
rather than the pool. A dedicated probe that predicts ONLY slot A tests that — if
it recovers what the model's head cannot, the essence holds the information and
the head is the bottleneck.

Usage: vp_superposition.py <t2_dir> <t1_path> [n_batches]
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              _value_bits, _value_smooth)
from compressor.model import instruction_wiring
from datagen import (RVT_FORMAT, Batch, AUX_CE_IGNORE, MAX_INPUT_SLOTS,
                     N_REGS, make_anchor_states)
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 400
SLOT = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # which OUTPUT slot to
                            # analyse; = which instruction wrote it (write order)
N_ANCHORS, ANCHOR_SEED, NBITS = 8, 0, 32
HID, STEPS, MB = 1024, 20000, 4096

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
core = t2.core

CORPUS = f'/tmp/vp_super_r2_{os.getpid()}.rvt'
subprocess.run(
    f'python -m datagen --rule "cap=2" --twins 0 --batch-size 128 '
    f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
    f'--config configs/instr_r_only.json --count {NB} > {CORPUS}',
    shell=True, check=True, stderr=subprocess.DEVNULL)

anchors = torch.from_numpy(
    np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)


def head_acc(ess, in_slot, out_slot, out_regs):
    """Model's own vp_head -> per (row, out-slot) bit accuracy. Mirrors
    t2_value_predict_loss's feature construction exactly."""
    with torch.no_grad():
        fill_i = (in_slot != AUX_CE_IGNORE).to(torch.int64)
        iv = ANCH_IN.unsqueeze(0) * fill_i.unsqueeze(1)
        ifeat = torch.cat([_value_bits(iv),
                           _value_smooth(iv.float()).unsqueeze(-1)],
                          dim=-1).reshape(ess.shape[0], N_ANCHORS, -1)
        vexp = F.normalize(ess, dim=-1).unsqueeze(1).expand(-1, N_ANCHORS, -1)
        pr = t2.vp_head(torch.cat([vexp, ifeat], dim=-1)).reshape(
            ess.shape[0], N_ANCHORS, -1, NBITS + 1)
        tg = out_regs.gather(-1, out_slot.clamp(min=0).unsqueeze(1)
                             .expand(-1, N_ANCHORS, -1))
        return ((pr[..., :NBITS] > 0).float()
                == _value_bits(tg)).float().mean(-1).mean(1)     # (B, MAX_OUT)


both_acc, solo_acc, slot_delta = [], [], []
E_BOTH, E_SOLO, FILL, TGT = [], [], [], []
n_indep = n_seen = 0

with open(CORPUS, 'rb') as f:
    for batch in RVT_FORMAT.reader(f, Batch):
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        (_, _, chunk_idx, slot_idx, n_per_chunk, w0, w1, wo) = split
        nch, mni = batch.instr_lens.shape
        if mni < 2:
            continue

        # keep only DATA-INDEPENDENT 2-instruction R-type chunks
        order = np.argsort(chunk_idx * 1000 + slot_idx)
        w0s, w1s, wos = w0[order], w1[order], wo[order]
        keep = np.zeros(nch, dtype=bool)
        destA = np.zeros(nch, dtype=np.int64)
        pos = 0
        for b in range(nch):
            n = int(n_per_chunk[b])
            if n == 2:
                a0, a1, ao = int(w0s[pos]), int(w1s[pos]), int(wos[pos])
                b0, b1, bo = int(w0s[pos+1]), int(w1s[pos+1]), int(wos[pos+1])
                if ao != 0 and bo != 0 and ao != bo and b0 != ao and b1 != ao:
                    keep[b] = True
                    destA[b] = ao if SLOT == 0 else bo
            pos += n
        n_seen += nch
        if not keep.any():
            continue
        n_indep += int(keep.sum())
        ct1, c0, c1, co, cact = _t2_assemble(
            t1, split, nch, mni, dev, route='binding')
        # isolate ONLY the instruction that wrote the analysed slot
        solo_active = torch.zeros_like(cact)
        solo_active[:, SLOT] = cact[:, SLOT]
        with torch.no_grad():
            T_b, ess_b = core(ct1, c0, c1, co, cact)          # both
            T_s, ess_s = core(ct1, c0, c1, co, solo_active)   # solo

        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        vp_mask = torch.from_numpy(batch.out_regs_valid).to(dev)

        kb = torch.from_numpy(keep).to(dev)
        dA = torch.from_numpy(destA).to(dev)
        # slot A is the FIRST-written output, i.e. out-slot index 0
        okA = (out_slot[:, SLOT] == dA) & kb & vp_mask
        if not okA.any():
            continue
        idx = okA.nonzero(as_tuple=True)[0]

        # ASSERT slot A's value is untouched by masking instruction 2
        vb = T_b[idx, dA[idx]][:, :core.d_value]
        vs = T_s[idx, dA[idx]][:, :core.d_value]
        slot_delta.append(float((vb - vs).abs().max()))

        ab = head_acc(ess_b, in_slot, out_slot, out_regs)[idx, SLOT]
        as_ = head_acc(ess_s, in_slot, out_slot, out_regs)[idx, SLOT]
        both_acc.append(ab.cpu()); solo_acc.append(as_.cpu())

        E_BOTH.append(F.normalize(ess_b, dim=-1)[idx].cpu())
        E_SOLO.append(F.normalize(ess_s, dim=-1)[idx].cpu())
        FILL.append((in_slot != AUX_CE_IGNORE).to(torch.int64)[idx].cpu())
        TGT.append(out_regs[idx].gather(
            -1, out_slot[idx, SLOT].view(-1, 1, 1).expand(-1, N_ANCHORS, 1)
        ).squeeze(-1).cpu())

both_acc = torch.cat(both_acc); solo_acc = torch.cat(solo_acc)
print(f'checkpoint = {T2_DIR}')
print(f'  R-type len=2, data-independent pairs: {both_acc.numel()} rows, OUTPUT SLOT {SLOT} '
      f'(from {n_seen} chunks, {n_indep} independent)')
print()
print(f'  slot-A VALUE max |both - solo| = {max(slot_delta):.3e}   '
      f'<- 0 confirms the slot is untouched by the 2nd instruction')
print()
print('  MODEL vp_head on slot A (same target, same inputs, same slot value):')
print(f'    solo  (1 output in essence)   {100 * float(solo_acc.mean()):7.3f}%')
print(f'    both  (2 outputs in essence)  {100 * float(both_acc.mean()):7.3f}%')
print(f'    SUPERPOSITION COST            {100 * float(solo_acc.mean() - both_acc.mean()):7.3f} pts')

# pool vs head: a dedicated readout for slot A only
E_BOTH = torch.cat(E_BOTH).to(dev); E_SOLO = torch.cat(E_SOLO).to(dev)
FILL = torch.cat(FILL).to(dev); TGT = torch.cat(TGT).to(dev)
P = E_BOTH.shape[0]
perm = torch.randperm(P, generator=torch.Generator().manual_seed(0)).to(dev)
tr, te = perm[:int(.85 * P)], perm[int(.85 * P):]


def in_feat(idx, a):
    v = ANCH_IN[a].unsqueeze(0) * FILL[idx]
    return torch.cat([_value_bits(v),
                      _value_smooth(v.float()).unsqueeze(-1)],
                     dim=-1).reshape(idx.shape[0], -1)


print()
print(f'  DEDICATED probe (predicts ONLY slot A) — separates pool from readout')
print(f'    train={tr.numel()} test={te.numel()}')
for name, E in (('from BOTH essence', E_BOTH), ('from SOLO essence', E_SOLO)):
    torch.manual_seed(0)
    d_in = E.shape[1] + MAX_INPUT_SLOTS * (NBITS + 1)
    net = nn.Sequential(nn.Linear(d_in, HID), nn.ReLU(),
                        nn.Linear(HID, HID), nn.ReLU(),
                        nn.Linear(HID, NBITS)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    rng = np.random.default_rng(0)
    for _ in range(STEPS):
        i = tr[torch.from_numpy(rng.integers(0, tr.numel(), MB)).to(dev)]
        a = int(rng.integers(0, N_ANCHORS))
        loss = F.binary_cross_entropy_with_logits(
            net(torch.cat([E[i], in_feat(i, a)], dim=-1)), _value_bits(TGT[i, a]))
        opt.zero_grad(); loss.backward(); opt.step()
    acc = n = 0.0
    with torch.no_grad():
        for a in range(N_ANCHORS):
            for s in range(0, te.numel(), MB):
                i = te[s:s + MB]
                acc += float(((net(torch.cat([E[i], in_feat(i, a)], dim=-1)) > 0).float()
                              == _value_bits(TGT[i, a])).float().mean(-1).sum())
                n += i.numel()
    print(f'    {name:20s} {100 * acc / n:7.3f}%')
