#!/usr/bin/env python3
"""Would giving the op-cell each operand's canonical rank be ENOUGH?

Feeding the canonical read rank to the op-cell would make operand binding a
local association instead of a cross-slot join the pool cannot do. This probe
decides whether that CAN work before any GPU time is spent.

The test is on slot B's OWN per-slot state (the essence tracks it at 97.75% vs
97.61%, so the pool is not the loser). Predict slot B's 32-bit output value from
that state plus the canonical input values, with and without the true canonical
ranks of B's two operands:

  WITHOUT rank  -- calibration. Must reproduce the known deficit (~100% on
                   disjoint-source pairs, ~83% on shared-source) or the probe is
                   measuring itself and neither arm is believable.
  WITH rank     -- the verdict. ~100% on overlap means the state holds
                   everything except the association, so supplying rank at the
                   op-cell is sufficient in principle. Still ~83% means slot B's
                   value has already symmetrized which operand was which, and
                   rank alone will not fix it -- the hypothesis dies here.

Note the ranks are given as SIDE INFO to a readout, not fed to the op-cell (the
frozen model was not trained that way). So this bounds what the change could
achieve; it cannot show the op-cell would learn to use it.

Usage: opcell_rank_sufficiency.py <t2_dir> <t1_path> [n_batches]
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
from datagen import (RVT_FORMAT, Batch, AUX_CE_IGNORE, MAX_INPUT_SLOTS,
                     N_REGS, make_anchor_states)
from datagen.generate import _ALU_R_OPS
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 400
N_ANCHORS, ANCHOR_SEED, NBITS = 8, 0, 32
HID, STEPS, MB = 1024, 15000, 4096
OPS = sorted(set(_ALU_R_OPS))
NPOS = 6                      # canonical positions 1..5, plus 0 = "not found"

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
core = t2.core

CORPUS = f'/tmp/opcell_rank_{os.getpid()}.rvt'
subprocess.run(
    f'python -m datagen --rule "cap=2" --twins 0 --batch-size 128 '
    f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
    f'--config configs/instr_r_only.json --count {NB} > {CORPUS}',
    shell=True, check=True, stderr=subprocess.DEVNULL)

anchors = torch.from_numpy(
    np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)

STATE, FILL, TGT, RANK0, RANK1, DISJ = [], [], [], [], [], []

with open(CORPUS, 'rb') as f:
    for batch in RVT_FORMAT.reader(f, Batch):
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        (_, _, chunk_idx, slot_idx, n_per_chunk, w0, w1, wo) = split
        nch, mni = batch.instr_lens.shape
        if mni < 2:
            continue
        ct1, c0, c1, co, cact = _t2_assemble(
            t1, split, nch, mni, dev, route='binding')
        with torch.no_grad():
            T, _ = core(ct1, c0, c1, co, cact)

        in_slot = batch.in_slot_regs.astype(np.int64)
        out_regs_t = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        in_slot_t = torch.from_numpy(in_slot).to(dev)
        vp_mask = batch.out_regs_valid

        order = np.argsort(chunk_idx * 1000 + slot_idx)
        w0s, w1s, wos = w0[order], w1[order], wo[order]
        pos = 0
        for b in range(nch):
            n = int(n_per_chunk[b])
            if n != 2 or not vp_mask[b]:
                pos += n
                continue
            a0, a1, ao = int(w0s[pos]), int(w1s[pos]), int(wos[pos])
            b0, b1, bo = int(w0s[pos + 1]), int(w1s[pos + 1]), int(wos[pos + 1])
            pos += n
            # data-INDEPENDENT pairs with distinct dests: the same subset the
            # ~83% deficit was measured on (vp_slot_b_split).
            if not (ao and bo and ao != bo and b0 != ao and b1 != ao):
                continue
            toks = batch.tokens[b].tolist()
            try:
                iA, p = decode_instruction(toks, 0)
                iB, _ = decode_instruction(toks, p)
            except Exception:
                continue
            if iA.opcode not in OPS or iB.opcode not in OPS:
                continue
            canon = list(in_slot[b])

            def cpos(r):
                try:
                    return canon.index(r) + 1
                except ValueError:
                    return 0

            if int(batch.out_slot_regs[b, 1]) != bo:
                continue
            STATE.append(T[b, bo].cpu())
            FILL.append((in_slot_t[b] != AUX_CE_IGNORE).to(torch.int64).cpu())
            TGT.append(out_regs_t[b, :, bo].cpu())
            RANK0.append(min(cpos(b0), NPOS - 1))
            RANK1.append(min(cpos(b1), NPOS - 1))
            DISJ.append(not ({a0, a1} & {b0, b1}))

STATE = torch.stack(STATE).to(dev); FILL = torch.stack(FILL).to(dev)
TGT = torch.stack(TGT).to(dev)
RANK0 = torch.tensor(RANK0, device=dev); RANK1 = torch.tensor(RANK1, device=dev)
DISJ = torch.tensor(DISJ, device=dev)
RANK_OH = torch.cat([F.one_hot(RANK0, NPOS), F.one_hot(RANK1, NPOS)],
                    dim=-1).float()
P = STATE.shape[0]

print(f'checkpoint = {T2_DIR}')
print(f'  rows={P}   disjoint-source share={100*float(DISJ.float().mean()):.1f}%'
      f'   (slot B only, data-independent pairs)')
# Base rate: the readout could answer "the modal position" and score this much.
mode0 = float((RANK0 == RANK0.mode().values).float().mean())
print(f'  base rate for operand0 position (modal answer) = {100*mode0:.1f}%')


def in_feat(idx, a):
    v = ANCH_IN[a].unsqueeze(0) * FILL[idx]
    return torch.cat([_value_bits(v),
                      _value_smooth(v.float()).unsqueeze(-1)],
                     dim=-1).reshape(idx.shape[0], -1)


def fit_vp(TR, TE, use_rank):
    """Predict slot B's 32-bit output from its state (+ canonical inputs,
    optionally + true operand ranks). Returns per-row (per-bit, exact)."""
    torch.manual_seed(0)
    d_in = (STATE.shape[1] + MAX_INPUT_SLOTS * (NBITS + 1)
            + (2 * NPOS if use_rank else 0))
    net = nn.Sequential(nn.Linear(d_in, HID), nn.ReLU(),
                        nn.Linear(HID, HID), nn.ReLU(),
                        nn.Linear(HID, NBITS)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    rng = np.random.default_rng(0)

    def feats(idx, a):
        parts = [STATE[idx], in_feat(idx, a)]
        if use_rank:
            parts.append(RANK_OH[idx])
        return torch.cat(parts, -1)

    for _ in range(STEPS):
        i = TR[torch.from_numpy(rng.integers(0, TR.numel(), MB)).to(dev)]
        a = int(rng.integers(0, N_ANCHORS))
        loss = F.binary_cross_entropy_with_logits(
            net(feats(i, a)), _value_bits(TGT[i, a]))
        opt.zero_grad(); loss.backward(); opt.step()
    bit = torch.zeros(TE.numel(), device=dev)
    exact = torch.zeros(TE.numel(), device=dev)
    with torch.no_grad():
        for a in range(N_ANCHORS):
            hit = (net(feats(TE, a)) > 0).float() == _value_bits(TGT[TE, a])
            bit += hit.float().mean(-1)
            exact += hit.all(-1).float()
    return bit / N_ANCHORS, exact / N_ANCHORS


perm = torch.randperm(P, generator=torch.Generator().manual_seed(0)).to(dev)
TR, TE = perm[:int(.85 * P)], perm[int(.85 * P):]
d = DISJ[TE]
print()
print(f'  {"slot B output value":22s} {"all":>9s} {"disjoint":>10s} {"OVERLAP":>10s}')
for use_rank in (False, True):
    bit, exact = fit_vp(TR, TE, use_rank)
    tag = 'state + rank' if use_rank else 'state only'
    for nm, r in (('per-bit', bit), ('EXACT 32-bit', exact)):
        print(f'  {tag + " / " + nm:22s} {100*float(r.mean()):8.3f}% '
              f'{100*float(r[d].mean()):9.3f}% {100*float(r[~d].mean()):9.3f}%')
print(f'\n  (n disjoint={int(d.sum())}, overlap={int((~d).sum())})')
