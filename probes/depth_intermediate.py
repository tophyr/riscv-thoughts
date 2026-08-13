#!/usr/bin/env python3
"""Is the depth-2 deficit LOST INFORMATION, or an unchained computation?

Every "depth-2 scores worse" measurement is confounded: a depth-2 slot's value
is f2(f1(a,b),c), so whatever reads it must chain two 32-bit arithmetic ops
where a depth-1 slot needs one. A readout can fail on that without any
information having been lost.

This removes the confound by handing the readout the INTERMEDIATE value —
f1(a,b) at each anchor — alongside the essence, which reduces the task to the
single op f2. In a two-instruction chunk that composes, the intermediate is
simply the other supervised output slot's target, so it needs no new oracle.

  (1) essence + canonical inputs                  — the baseline at this budget
  (2) essence + canonical inputs + intermediate   — the same, chain removed
  (3) canonical inputs + intermediate, NO essence — what is determined without
      the essence at all; f2's opcode and immediate live only there, so this is
      the floor a readout must beat to be using the essence

Read (2) against BOTH (1) and the trained head. If (2) reaches depth-1 quality,
the essence carries f2's operator and operands and the deficit is the chain,
not the representation — and since the intermediate is itself recoverable from
the essence (it is a depth-1 slot, ~99.96% exact), nothing was lost. If (2)
stays near (1), f2 itself was never represented.

Usage: depth_intermediate.py <t2_dir> <t1_path> [n_batches] [steps]
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
                     MAX_OUTPUT_SLOTS, N_REGS, make_anchor_states)
from scripts._common import resolve_device, load_frozen_encoder, load_t2

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
LEN, N_ANCHORS, ANCHOR_SEED, NBITS = 2, 8, 0, 32
HID, MB = 1024, 512

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)

CORPUS = f'/tmp/depth_inter_len{LEN}.rvt'
subprocess.run(
    f'python -m datagen --rule "cap={LEN}" --twins 0 --batch-size 128 '
    f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
    f'--config configs/instr_alu_writers.json --count {NB} > {CORPUS}',
    shell=True, check=True, stderr=subprocess.DEVNULL)

anchors = torch.from_numpy(
    np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)


def slot_depth(in0, in1, out, n):
    d = np.zeros(N_REGS, dtype=np.int64)
    for k in range(n):
        i0, i1, o = int(in0[k]), int(in1[k]), int(out[k])
        nd = 1 + max(d[i0], d[i1])
        if o != 0:
            d[o] = nd
    return d


cols = {k: [] for k in ('ess', 'fill', 'tgt', 'mid', 'ord')}
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
            ess = t2.encode(ct1, c0, c1, co, cact,
                            torch.randn(nch, t2.n_regs, device=dev)).float()

        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        vp_mask = torch.from_numpy(batch.out_regs_valid).to(dev)
        valid = ((out_slot != AUX_CE_IGNORE)
                 & vp_mask.unsqueeze(-1)).cpu().numpy()

        order = np.argsort(chunk_idx * 1000 + slot_idx)
        wo_s, w0_s, w1_s = wo[order], w0[order], w1[order]
        oreg = out_slot.clamp(min=0).cpu().numpy()
        rows, sd2, sd1, ords = [], [], [], []
        pos = 0
        for b in range(nch):
            n = int(n_per_chunk[b])
            if n:
                dv = slot_depth(w0_s[pos:pos + n], w1_s[pos:pos + n],
                                wo_s[pos:pos + n], n)
                sup = np.nonzero(valid[b])[0]
                d = dv[oreg[b, sup]]
                # Exactly one composed slot and one leaf slot: then the leaf
                # slot's register IS the operand the composed instruction read,
                # so its target is the intermediate value.
                if len(sup) == 2 and sorted(d.tolist()) == [1, 2]:
                    i2 = int(sup[d == 2][0])
                    rows.append(b)
                    sd2.append(int(oreg[b, i2]))
                    sd1.append(int(oreg[b, sup[d == 1][0]]))
                    ords.append(i2)
            pos += n
        if not rows:
            continue
        r = torch.tensor(rows, device=dev)
        cols['ess'].append(ess[r])
        cols['fill'].append(in_slot[r] != AUX_CE_IGNORE)
        cols['tgt'].append(out_regs[r].gather(
            -1, torch.tensor(sd2, device=dev).view(-1, 1, 1)
            .expand(-1, N_ANCHORS, 1)).squeeze(-1).to(torch.int32))
        cols['mid'].append(out_regs[r].gather(
            -1, torch.tensor(sd1, device=dev).view(-1, 1, 1)
            .expand(-1, N_ANCHORS, 1)).squeeze(-1).to(torch.int32))
        cols['ord'].append(torch.tensor(ords, device=dev))

D = {k: torch.cat(v) for k, v in cols.items()}
M = D['ess'].shape[0]
del cols
cut = int(0.8 * M)
TR, TE = torch.arange(cut, device=dev), torch.arange(cut, M, device=dev)


def parts(idx, use_ess, use_mid):
    """(len(idx), A, d) input for one arm."""
    iv = ANCH_IN.unsqueeze(0) * D['fill'][idx].to(torch.int64).unsqueeze(1)
    x = [torch.cat([_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
                   dim=-1).reshape(idx.shape[0], N_ANCHORS, -1)]
    if use_ess:
        x.append(F.normalize(D['ess'][idx], dim=-1).unsqueeze(1)
                 .expand(-1, N_ANCHORS, -1))
    if use_mid:
        m = D['mid'][idx].to(torch.int64)
        x.append(torch.cat([_value_bits(m),
                            _value_smooth(m.float()).unsqueeze(-1)], dim=-1))
    return torch.cat(x, dim=-1)


def run(name, use_ess, use_mid, seed=0):
    torch.manual_seed(seed)
    d_in = parts(TR[:1], use_ess, use_mid).shape[-1]
    net = nn.Sequential(nn.Linear(d_in, HID), nn.ReLU(),
                        nn.Linear(HID, NBITS + 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 3e-4)
    rng = np.random.default_rng(seed)
    for _ in range(STEPS):
        i = TR[torch.from_numpy(rng.integers(0, cut, MB)).to(dev)]
        pred = net(parts(i, use_ess, use_mid))
        tgt = D['tgt'][i].to(torch.int64)
        loss = (F.binary_cross_entropy_with_logits(
            pred[..., :NBITS], _value_bits(tgt)).mean()
            + 0.2 * ((pred[..., NBITS]
                      - _value_smooth(tgt.float())) ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(parts(TE, use_ess, use_mid))
        hit = ((pred[..., :NBITS] > 0).float()
               == _value_bits(D['tgt'][TE].to(torch.int64)))
    print(f'  {name:46s} per-bit {100 * hit.float().mean():8.4f}%   '
          f'exact 32-bit {100 * hit.all(-1).float().mean():8.4f}%')


with torch.no_grad():
    iv = ANCH_IN.unsqueeze(0) * D['fill'][TE].to(torch.int64).unsqueeze(1)
    ifeat = torch.cat([_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
                      dim=-1).reshape(TE.shape[0], N_ANCHORS, -1)
    v = F.normalize(D['ess'][TE], dim=-1).unsqueeze(1).expand(-1, N_ANCHORS, -1)
    pr = t2.vp_head(torch.cat([v, ifeat], dim=-1)).reshape(
        TE.shape[0], N_ANCHORS, MAX_OUTPUT_SLOTS, NBITS + 1)
    pr = pr.gather(2, D['ord'][TE].view(-1, 1, 1, 1)
                   .expand(-1, N_ANCHORS, 1, NBITS + 1)).squeeze(2)
    h = ((pr[..., :NBITS] > 0).float()
         == _value_bits(D['tgt'][TE].to(torch.int64)))

print()
print(f'checkpoint = {T2_DIR}   len={LEN}   composing chunks={M} '
      f'(train {cut}, held-out {M - cut})   fit steps={STEPS}')
print('  every row is one depth-2 output slot whose chunk also exposes its '
      'intermediate')
print()
print(f'  {"calibration: the model’s own vp head":46s} '
      f'per-bit {100 * h.float().mean():8.4f}%   '
      f'exact 32-bit {100 * h.all(-1).float().mean():8.4f}%')
print()
run('(1) essence + canonical inputs', True, False)
run('(2) essence + canonical inputs + intermediate', True, True)
run('(3) canonical inputs + intermediate, no essence', False, True)
print()
