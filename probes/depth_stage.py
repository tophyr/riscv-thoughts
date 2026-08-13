#!/usr/bin/env python3
"""WHERE does a depth-2 output value get lost — the slot, the pool, or neither?

Head-side arms (depth_composition.py) cannot beat the trained head on depth-2
slots, which puts the deficit in the encoder. This walks the encoder's stages
and recovers ONE quantity — the slot's output value at the 8 anchors — from
each, reported by dataflow depth. Depth-1 slots are the built-in control: they
run through the identical stages, so whichever stage first shows a depth-1 /
depth-2 gap is the stage that treats a composed value worse than a leaf one.

Stages, in pipeline order:
  T2 slot     — the register slot's own state, before the pool reads it.
  pool view   — the slot's `pool_v` image plus its `pool_k` scores against the
                queries. A slot reaches the essence only through those, so they
                bound what any pooling of them can carry about it.
  essence     — what the vp head and any future tier actually get.

Every stage is read by a freshly fitted readout of identical architecture and
step budget, so the stages are comparable to each other. They are NOT comparable
to the model's own head, which trained for the whole run; the trained head's
number is printed as the calibration ceiling, and a readout below it on depth-1
is measuring its own budget rather than the stage.

Usage: depth_stage.py <t2_dir> <t1_path> [n_batches] [steps]
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
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 600
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
LEN, N_ANCHORS, ANCHOR_SEED, NBITS = 2, 8, 0, 32
HID, MB = 1024, 512
N_ORD = 4          # one-hot of the slot's SSA write order (which output it is)

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
core = t2.core
assert core.pool_extra == 'write_time', core.pool_extra

CORPUS = f'/tmp/depth_stage_len{LEN}.rvt'
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
    """Dataflow depth of every register slot after n instructions."""
    d = np.zeros(N_REGS, dtype=np.int64)
    for k in range(n):
        i0, i1, o = int(in0[k]), int(in1[k]), int(out[k])
        nd = 1 + max(d[i0], d[i1])
        if o != 0:
            d[o] = nd
    return d


cols = {k: [] for k in ('slot', 'pool', 'ess', 'fill', 'tgt', 'dep', 'ord')}
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
            T, ess = core(ct1, c0, c1, co, cact,
                          torch.randn(nch, core.n_regs, device=dev))
            # The pool's whole view of every slot. write_time is REPLAYED from
            # the wiring rather than read off the state: event channel 1 holds
            # write_time plus the accumulated write/read pulses, so it is not
            # the clean ramp the pool is fed.
            wt = torch.zeros(nch, core.n_regs, device=dev)
            nx0 = torch.ones(core.n_regs, device=dev)
            nx0[0] = 0.0
            for k in range(mni):
                wr = (F.one_hot(co[:, k], core.n_regs).float() * nx0
                      * cact[:, k:k + 1])
                wt = wt * (1.0 - wr) + wr * (float(k + 1) / float(mni))
            psrc = torch.cat([T[..., :core.d_value], wt.unsqueeze(-1)], dim=-1)
            logits = core.pool_k(psrc) @ core.pool_q.view(
                core.n_pool_heads, -1).t()
            pview = torch.cat([core.pool_v(psrc), logits], dim=-1)  # (B,n_regs,·)
            if not cols['ess']:
                # The replayed pool_src must reproduce the model's own essence,
                # or every stage measured downstream of it is measuring a
                # different model.
                vv = core.pool_v(psrc).view(
                    nch, core.n_regs, core.n_pool_heads, -1)
                chk = torch.einsum('bsk,bskd->bkd',
                                   torch.softmax(logits, dim=1), vv)
                err = (chk.reshape(nch, -1) - ess).abs().max()
                assert err < 1e-4, f'pool replay mismatch: {err:.3e}'

        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        vp_mask = torch.from_numpy(batch.out_regs_valid).to(dev)
        valid = (out_slot != AUX_CE_IGNORE) & vp_mask.unsqueeze(-1)

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

        # Flatten to one row per SUPERVISED OUTPUT SLOT.
        r, s = torch.nonzero(valid, as_tuple=True)
        reg = out_slot[r, s].clamp(min=0)
        cols['slot'].append(T[r, reg])
        cols['pool'].append(pview[r, reg])
        cols['ess'].append(ess[r].float())
        cols['fill'].append(in_slot[r] != AUX_CE_IGNORE)
        cols['tgt'].append(out_regs[r][:, :, :].gather(
            -1, reg.view(-1, 1, 1).expand(-1, N_ANCHORS, 1)
        ).squeeze(-1).to(torch.int32))
        cols['dep'].append(torch.from_numpy(dep).to(dev)[r, s])
        cols['ord'].append(s.clamp(max=N_ORD - 1))

D = {k: torch.cat(v) for k, v in cols.items()}
M = D['ess'].shape[0]
del cols

cut = int(0.8 * M)
TR, TE = torch.arange(cut, device=dev), torch.arange(cut, M, device=dev)
GROUPS = (('depth 1', D['dep'] == 1), ('depth 2', D['dep'] == 2))


def feats(idx):
    """The canonical input values + which output slot this is — the context the
    vp head gets, minus the essence."""
    iv = ANCH_IN.unsqueeze(0) * D['fill'][idx].to(torch.int64).unsqueeze(1)
    ifeat = torch.cat(
        [_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
        dim=-1).reshape(idx.shape[0], N_ANCHORS, -1)
    ordh = F.one_hot(D['ord'][idx], N_ORD).float().unsqueeze(1).expand(
        -1, N_ANCHORS, -1)
    return torch.cat([ifeat, ordh], dim=-1)


def fit_stage(key, seed=0):
    """Fit `stage -> the slot's output value` with the training loss."""
    torch.manual_seed(seed)
    d_in = D[key].shape[1] + MAX_INPUT_SLOTS * (NBITS + 1) + N_ORD
    net = nn.Sequential(nn.Linear(d_in, HID), nn.ReLU(),
                        nn.Linear(HID, NBITS + 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 3e-4)
    rng = np.random.default_rng(seed)
    for _ in range(STEPS):
        i = TR[torch.from_numpy(rng.integers(0, cut, MB)).to(dev)]
        x = F.normalize(D[key][i], dim=-1).unsqueeze(1).expand(-1, N_ANCHORS, -1)
        pred = net(torch.cat([x, feats(i)], dim=-1))
        tgt = D['tgt'][i].to(torch.int64)
        loss = (F.binary_cross_entropy_with_logits(
            pred[..., :NBITS], _value_bits(tgt)).mean()
            + 0.2 * ((pred[..., NBITS]
                      - _value_smooth(tgt.float())) ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def score_stage(net, key, chunks=4096):
    acc = [[0.0, 0.0, 0.0] for _ in GROUPS]
    with torch.no_grad():
        for s in range(0, TE.shape[0], chunks):
            i = TE[s:s + chunks]
            x = F.normalize(D[key][i], dim=-1).unsqueeze(1).expand(
                -1, N_ANCHORS, -1)
            pred = net(torch.cat([x, feats(i)], dim=-1))
            hit = ((pred[..., :NBITS] > 0).float()
                   == _value_bits(D['tgt'][i].to(torch.int64)))
            bit, exact = hit.float().mean(-1), hit.all(-1).float()
            for g, (_, sel) in enumerate(GROUPS):
                m = sel[i].unsqueeze(1).float()
                acc[g][0] += float((bit * m).sum())
                acc[g][1] += float((exact * m).sum())
                acc[g][2] += float(m.expand_as(bit).sum())
    return acc


def trained_head_score(chunks=2048):
    """The model's own head, scored on the same held-out slots — the ceiling
    every fitted readout is calibrated against."""
    acc = [[0.0, 0.0, 0.0] for _ in GROUPS]
    with torch.no_grad():
        for s in range(0, TE.shape[0], chunks):
            i = TE[s:s + chunks]
            iv = ANCH_IN.unsqueeze(0) * D['fill'][i].to(torch.int64).unsqueeze(1)
            ifeat = torch.cat(
                [_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
                dim=-1).reshape(i.shape[0], N_ANCHORS, -1)
            v = F.normalize(D['ess'][i], dim=-1).unsqueeze(1).expand(
                -1, N_ANCHORS, -1)
            pred = t2.vp_head(torch.cat([v, ifeat], dim=-1)).reshape(
                i.shape[0], N_ANCHORS, MAX_OUTPUT_SLOTS, NBITS + 1)
            pred = pred.gather(
                2, D['ord'][i].view(-1, 1, 1, 1).expand(
                    -1, N_ANCHORS, 1, NBITS + 1)).squeeze(2)
            hit = ((pred[..., :NBITS] > 0).float()
                   == _value_bits(D['tgt'][i].to(torch.int64)))
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
        print(f'    {label:10s} per-bit {100 * b / n:8.4f}%   '
              f'exact 32-bit {100 * e / n:8.4f}%   (n={int(n)})')


print()
print(f'checkpoint = {T2_DIR}   len={LEN}   supervised slots={M} '
      f'(depth-1 {int((D["dep"] == 1).sum())}, depth-2 '
      f'{int((D["dep"] == 2).sum())})   fit steps={STEPS}')
print()
report("calibration ceiling: the model's own vp head", trained_head_score())
for key, label in (('slot', 'T2 slot'), ('pool', 'pool view'),
                   ('ess', 'T2 essence')):
    print()
    report(f'fitted readout from {label}', score_stage(fit_stage(key), key))
print()
