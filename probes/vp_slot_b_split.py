#!/usr/bin/env python3
"""What costs the 2.25% on the SECOND-written output slot?

Established: slot values are bit-identical to the len=1 case; the essence
recovers output 0 at 100.000% and output 1 at only 97.75%; and the essence
(97.75%) already matches slot B's OWN full state (97.61%), so the pool is not
losing it — slot B never encoded it.

Two slices, both on output slot 1 (with slot 0 as control):

(A) OPERATION vs OPERANDS. From the essence, separately recover
      - the slot's OPCODE (10-way; "which operation")
      - the CANONICAL POSITION feeding each operand role ("which operands")
    If opcode is ~100% and positions are not, the deficit is operand BINDING,
    not operation identity.

(B) DISJOINT vs OVERLAPPING sources. Slot A's operands are always canonical
    positions (1,2) — it reads first. Slot B's are (3,4) only when its sources
    are disjoint from A's; a shared source shifts them to (2,3) etc. If slot B
    reaches ~100% on disjoint pairs and carries all the loss on overlapping
    ones, the cause is canonical-position ambiguity, not pooling or capacity.

Usage: vp_slot_b_split.py <t2_dir> <t1_path> [n_batches]
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
from datagen.generate import _ALU_R_OPS
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 400
N_ANCHORS, ANCHOR_SEED, NBITS = 8, 0, 32
HID, STEPS, MB = 1024, 15000, 4096
OPS = sorted(set(_ALU_R_OPS)); OP2I = {o: i for i, o in enumerate(OPS)}
NPOS = 6                      # canonical positions 1..5, plus 0 = "not found"

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
core = t2.core

CORPUS = '/tmp/vp_slotb_r2.rvt'
subprocess.run(
    f'python -m datagen --rule "cap=2" --twins 0 --batch-size 128 '
    f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
    f'--config configs/instr_r_only.json --count {NB} > {CORPUS}',
    shell=True, check=True, stderr=subprocess.DEVNULL)

anchors = torch.from_numpy(
    np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)

ESS, FILL, TGT, OPC, P0, P1, DISJ, SLOTID = [], [], [], [], [], [], [], []

with open(CORPUS, 'rb') as f:
    for batch in RVT_FORMAT.reader(f, Batch):
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        (_, _, chunk_idx, slot_idx, n_per_chunk, w0, w1, wo) = split
        nch, mni = batch.instr_lens.shape
        if mni < 2:
            continue
        tags = torch.randn(nch, t2.n_regs, device=dev)
        ct1, c0, c1, co, cact = _t2_assemble(
            t1, split, nch, mni, dev, route='binding')
        with torch.no_grad():
            _, ess = core(ct1, c0, c1, co, cact, tags)

        in_slot = batch.in_slot_regs.astype(np.int64)      # canonical read order
        out_slot_t = torch.from_numpy(
            batch.out_slot_regs.astype(np.int64)).to(dev)
        out_regs_t = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        vp_mask = batch.out_regs_valid
        in_slot_t = torch.from_numpy(in_slot).to(dev)
        essn = F.normalize(ess, dim=-1)

        order = np.argsort(chunk_idx * 1000 + slot_idx)
        w0s, w1s, wos = w0[order], w1[order], wo[order]
        pos = 0
        for b in range(nch):
            n = int(n_per_chunk[b])
            if n != 2 or not vp_mask[b]:
                pos += n; continue
            a0, a1, ao = int(w0s[pos]), int(w1s[pos]), int(wos[pos])
            b0, b1, bo = int(w0s[pos+1]), int(w1s[pos+1]), int(wos[pos+1])
            pos += n
            if not (ao and bo and ao != bo and b0 != ao and b1 != ao):
                continue
            toks = batch.tokens[b].tolist()
            try:
                iA, p = decode_instruction(toks, 0)
                iB, _ = decode_instruction(toks, p)
            except Exception:
                continue
            if iA.opcode not in OP2I or iB.opcode not in OP2I:
                continue
            canon = list(in_slot[b])          # register -> canonical position

            def cpos(r):
                try:
                    return canon.index(r) + 1
                except ValueError:
                    return 0
            disj = not ({a0, a1} & {b0, b1})
            for sl, (op, r0, r1, dst) in enumerate(
                    ((iA.opcode, a0, a1, ao), (iB.opcode, b0, b1, bo))):
                if int(batch.out_slot_regs[b, sl]) != dst:
                    continue
                ESS.append(essn[b].cpu()); SLOTID.append(sl)
                FILL.append((in_slot_t[b] != AUX_CE_IGNORE).to(torch.int64).cpu())
                TGT.append(out_regs_t[b, :, dst].cpu())
                OPC.append(OP2I[op]); P0.append(min(cpos(r0), NPOS - 1))
                P1.append(min(cpos(r1), NPOS - 1)); DISJ.append(disj)

ESS = torch.stack(ESS).to(dev); FILL = torch.stack(FILL).to(dev)
TGT = torch.stack(TGT).to(dev)
OPC = torch.tensor(OPC, device=dev); P0 = torch.tensor(P0, device=dev)
P1 = torch.tensor(P1, device=dev)
DISJ = torch.tensor(DISJ, device=dev); SLOTID = torch.tensor(SLOTID, device=dev)
P = ESS.shape[0]
perm = torch.randperm(P, generator=torch.Generator().manual_seed(0)).to(dev)
tr, te = perm[:int(.85 * P)], perm[int(.85 * P):]

print(f'checkpoint = {T2_DIR}')
print(f'  rows={P}  (slot0={int((SLOTID==0).sum())}, slot1={int((SLOTID==1).sum())})'
      f'  disjoint-source share={100*float(DISJ.float().mean()):.1f}%')


def in_feat(idx, a):
    v = ANCH_IN[a].unsqueeze(0) * FILL[idx]
    return torch.cat([_value_bits(v),
                      _value_smooth(v.float()).unsqueeze(-1)],
                     dim=-1).reshape(idx.shape[0], -1)


def fit(target, nclass, use_inputs, TR, TE):
    """Train a readout from the essence (optionally + canonical inputs) and
    return per-row correctness on the test split."""
    torch.manual_seed(0)
    d_in = ESS.shape[1] + (MAX_INPUT_SLOTS * (NBITS + 1) if use_inputs else 0)
    net = nn.Sequential(nn.Linear(d_in, HID), nn.ReLU(),
                        nn.Linear(HID, HID), nn.ReLU(),
                        nn.Linear(HID, nclass)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    rng = np.random.default_rng(0)
    for _ in range(STEPS):
        i = TR[torch.from_numpy(rng.integers(0, TR.numel(), MB)).to(dev)]
        a = int(rng.integers(0, N_ANCHORS))
        x = torch.cat([ESS[i], in_feat(i, a)], -1) if use_inputs else ESS[i]
        loss = F.cross_entropy(net(x), target[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = torch.cat([ESS[TE], in_feat(TE, 0)], -1) if use_inputs else ESS[TE]
        return (net(x).argmax(-1) == target[TE]).float()


def fit_vp(TR, TE):
    torch.manual_seed(0)
    d_in = ESS.shape[1] + MAX_INPUT_SLOTS * (NBITS + 1)
    net = nn.Sequential(nn.Linear(d_in, HID), nn.ReLU(),
                        nn.Linear(HID, HID), nn.ReLU(),
                        nn.Linear(HID, NBITS)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    rng = np.random.default_rng(0)
    for _ in range(STEPS):
        i = TR[torch.from_numpy(rng.integers(0, TR.numel(), MB)).to(dev)]
        a = int(rng.integers(0, N_ANCHORS))
        loss = F.binary_cross_entropy_with_logits(
            net(torch.cat([ESS[i], in_feat(i, a)], -1)), _value_bits(TGT[i, a]))
        opt.zero_grad(); loss.backward(); opt.step()
    acc = torch.zeros(TE.numel(), device=dev)
    with torch.no_grad():
        for a in range(N_ANCHORS):
            acc += ((net(torch.cat([ESS[TE], in_feat(TE, a)], -1)) > 0).float()
                    == _value_bits(TGT[TE, a])).float().mean(-1)
    return acc / N_ANCHORS


print()
print(f'  {"readout (own probe/slot)":26s} {"slot0":>9s} {"slot1":>9s}'
      f' {"s1 disjoint":>12s} {"s1 overlap":>11s}')
# CONTROL: also fit probes on the OVERLAP subset ALONE. With 88% of rows
# disjoint, a probe over the mixed set can score ~88% by always answering the
# disjoint default (position 4) and never learning the minority case at all --
# which is exactly what the mixed numbers look like. Training only on overlap
# removes that option: if accuracy jumps, the essence HAS the information and
# the earlier deficit was a class-imbalance artifact of the readout (and of the
# model's own vp_head, which trains on the same skewed distribution), NOT a
# missing signal.
out = {}
for sl in (0, 1):
    rows = (SLOTID == sl).nonzero(as_tuple=True)[0]
    rp = rows[torch.randperm(rows.numel(),
                             generator=torch.Generator().manual_seed(0)).to(dev)]
    TR, TE = rp[:int(.85 * rp.numel())], rp[int(.85 * rp.numel()):]
    out[sl] = (TE, {'vp (32-bit value)': fit_vp(TR, TE),
                    'OPCODE (10-way)': fit(OPC, len(OPS), False, TR, TE),
                    'operand0 canon POS': fit(P0, NPOS, False, TR, TE),
                    'operand1 canon POS': fit(P1, NPOS, False, TR, TE)})
print()
print('  === CONTROL: probes trained on the OVERLAP subset ONLY (slot 1) ===')
rows = ((SLOTID == 1) & ~DISJ).nonzero(as_tuple=True)[0]
rp = rows[torch.randperm(rows.numel(),
                         generator=torch.Generator().manual_seed(0)).to(dev)]
TRo, TEo = rp[:int(.8 * rp.numel())], rp[int(.8 * rp.numel()):]
print(f'    overlap rows={rows.numel()}  train={TRo.numel()}  test={TEo.numel()}')
for nm, tgt, nc in (('vp (32-bit value)', None, 0),
                    ('operand0 canon POS', P0, NPOS),
                    ('operand1 canon POS', P1, NPOS)):
    r = fit_vp(TRo, TEo) if tgt is None else fit(tgt, nc, False, TRo, TEo)
    print(f'    {nm:24s} {100 * float(r.mean()):7.3f}%')

print()
for name in ('vp (32-bit value)', 'OPCODE (10-way)',
             'operand0 canon POS', 'operand1 canon POS'):
    r0 = out[0][1][name]; TE1, r1 = out[1][0], out[1][1][name]
    d1 = DISJ[TE1]
    print(f'  {name:26s} {100*float(r0.mean()):8.3f}% '
          f'{100*float(r1.mean()):8.3f}% {100*float(r1[d1].mean()):11.3f}% '
          f'{100*float(r1[~d1].mean()):10.3f}%   (n disj={int(d1.sum())}, '
          f'ovl={int((~d1).sum())})')
