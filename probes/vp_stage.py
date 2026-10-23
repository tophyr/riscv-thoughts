#!/usr/bin/env python3
"""Where is a composed slot's value lost: in the op-cell, or in the pool?

On depth-2 output slots (the writer reads the other instruction's result), three
readouts of one architecture and budget each predict the slot's value at every
anchor from the canonical input values plus one feature set:

  gt       — ground-truth instruction features: each instruction's opcode and
             immediate, and which input (or the other instruction's result,
             or nothing the output depends on) feeds each operand. The ceiling for a readout of this shape.
  slot     — the composed register's T2 slot state, before pooling
  essence  — the pooled essence, plus which output slot is asked for

The trained vp head's own exact match on the same triples is the reference. If
`slot` reaches `gt` and `essence` does not, the pool loses the value; if `slot`
also falls short, the op-cell never formed it.

Trained on the live stream from stdin; scored on a held-out corpus.

Usage: datagen ... | vp_stage.py <t2_dir> <t1_path> <heldout_corpus> [steps]
"""
import math
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              _value_bits, _value_feat, vp_canonical_in_vals,
                              vp_head_predict)
from datagen import RVT_FORMAT, Batch, AUX_CE_IGNORE, N_REGS
from datagen.compare import _input_order, _written_regs
from emulator import R_TYPE
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction

T2_DIR, T1_PATH, HELDOUT = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
N_IN = 4                      # a two-instruction chunk has at most 4 inputs
OPS = ('ADD SUB XOR OR AND SLL SRL SRA SLT SLTU '
       'ADDI SLTI SLTIU XORI ORI ANDI SLLI SRLI SRAI').split()

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)


def slot_depth(in0, in1, out, n):
    d = np.zeros(N_REGS, dtype=np.int64)
    for k in range(n):
        nd = 1 + max(d[int(in0[k])], d[int(in1[k])])
        if int(out[k]):
            d[int(out[k])] = nd
    return d


def gt_feat(chunk, target_reg):
    """Per instruction: opcode one-hot, 12 immediate bits, and per operand which
    input position (or the other instruction's result) feeds it."""
    order = _input_order(chunk)
    f = []
    written = {}
    for k, ins in enumerate(chunk[:2]):
        oh = np.zeros(len(OPS)); oh[OPS.index(ins.opcode)] = 1
        if ins.opcode in R_TYPE:
            srcs, imm = list(ins.args[1:3]), 0
        else:
            srcs, imm = [ins.args[1], 0], ins.args[2]
        ib = np.array([(imm >> i) & 1 for i in range(12)], dtype=float)
        pos = np.zeros((2, N_IN + 3))
        for j, x in enumerate(srcs):
            if x == 0:
                pos[j, 0] = 1
            elif x in written:
                pos[j, N_IN + 1] = 1
            elif x in order:
                pos[j, 1 + order.index(x)] = 1
            else:                                  # a read the output ignores
                pos[j, N_IN + 2] = 1
        writes_target = float(target_reg in _written_regs(ins))
        f += [oh, ib, pos.ravel(), [writes_target]]
        for r in _written_regs(ins):
            written[r] = k
    return np.concatenate([np.asarray(x, dtype=float) for x in f])


def depth2_items(batch):
    """(row, output-slot index, register, chunk) for every depth-2 output slot,
    plus the T2 state and essence of the batch."""
    split = _split_to_per_instruction(batch)
    if split[0].shape[0] == 0:
        return None
    nch, mni = batch.instr_lens.shape
    n_per = split[4]
    with torch.no_grad():
        ct1, c0, c1, co, cact = _t2_assemble(t1, split, nch, mni, dev,
                                             route='binding')
        T, ess = t2.encode_state(ct1, c0, c1, co, cact)
    e0, e1, eo = c0.cpu().numpy(), c1.cpu().numpy(), co.cpu().numpy()
    items = []
    for b in np.nonzero(batch.out_regs_valid)[0]:
        n = int(n_per[b])
        if n < 2:
            continue
        dv = slot_depth(e0[b], e1[b], eo[b], n)
        toks, p, chunk = batch.tokens[b].tolist(), 0, []
        for _ in range(n):
            ins, p = decode_instruction(toks, p); chunk.append(ins)
        for si, r in enumerate(batch.out_slot_regs[b]):
            if r != AUX_CE_IGNORE and dv[int(r)] == 2:
                items.append((int(b), si, int(r), chunk))
    return items, T, ess


def features(items, T, ess, batch):
    rows = torch.tensor([i[0] for i in items], device=dev)
    regs = torch.tensor([i[2] for i in items], device=dev)
    si = torch.tensor([i[1] for i in items], device=dev)
    anchors = torch.from_numpy(batch.anchors).to(dev)
    in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
    iv = vp_canonical_in_vals(anchors, in_slot)[rows][..., :N_IN]   # (n, A, 4)
    A = iv.shape[1]
    inf = _value_feat(iv).reshape(len(items), A, -1)
    tgt = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)[rows]
    tgt = tgt.gather(-1, regs.view(-1, 1, 1).expand(-1, A, 1)).squeeze(-1)
    gt = torch.tensor(np.stack([gt_feat(i[3], i[2]) for i in items]),
                      dtype=torch.float32, device=dev)
    feats = {
        'gt': gt,
        'slot': T[rows, regs],
        'essence': torch.cat([F.normalize(ess[rows], dim=-1),
                              F.one_hot(si, 2).float()], -1),
    }
    return feats, inf, tgt, rows, si


def mlp(d_in):
    layers, d = [], d_in
    for _ in range(4):
        layers += [nn.Linear(d, 1024), nn.ReLU()]; d = 1024
    return nn.Sequential(*layers, nn.Linear(d, 32)).to(dev)


nets, opts = {}, {}
step = 0
for batch in RVT_FORMAT.reader(sys.stdin.buffer, Batch):
    if step >= STEPS:
        break
    got = depth2_items(batch)
    if not got or not got[0]:
        continue
    items, T, ess = got
    feats, inf, tgt, _, _ = features(items, T, ess, batch)
    A = inf.shape[1]
    lr = 3e-4 * (min(1, (step + 1) / 2000)
                 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
    for name, f in feats.items():
        if name not in nets:
            nets[name] = mlp(f.shape[-1] + inf.shape[-1])
            opts[name] = torch.optim.Adam(nets[name].parameters(), lr=lr)
        x = torch.cat([f.unsqueeze(1).expand(-1, A, -1), inf], -1)
        loss = F.binary_cross_entropy_with_logits(nets[name](x), _value_bits(tgt))
        for g in opts[name].param_groups:
            g['lr'] = lr
        opts[name].zero_grad(set_to_none=True)
        loss.backward()
        opts[name].step()
    step += 1
    if step % 5000 == 0:
        print(f'step {step}', flush=True)

acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
with torch.no_grad(), open(HELDOUT, 'rb') as fh:
    for batch in RVT_FORMAT.reader(fh, Batch):
        got = depth2_items(batch)
        if not got or not got[0]:
            continue
        items, T, ess = got
        feats, inf, tgt, rows, si = features(items, T, ess, batch)
        A = inf.shape[1]
        bits = _value_bits(tgt)
        anchors = torch.from_numpy(batch.anchors).to(dev)
        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        head = vp_head_predict(t2, ess, vp_canonical_in_vals(anchors, in_slot))
        head = head[rows, :, si, :32]
        res = {'trained vp head': ((head > 0).float() == bits).all(-1)}
        for name, f in feats.items():
            x = torch.cat([f.unsqueeze(1).expand(-1, A, -1), inf], -1)
            res[name] = ((nets[name](x) > 0).float() == bits).all(-1)
        for j, (_, _, r, chunk) in enumerate(items):
            w = chunk[1]
            kind = ('R-type writer, composed + fresh input'
                    if w.opcode in R_TYPE and all(w.args[1:3]) and
                    len({x for x in w.args[1:3]} - set(_written_regs(chunk[0]))) == 1
                    else 'other depth-2')
            for key in ('all depth-2', kind, f'writer {w.opcode}'):
                for name, ok in res.items():
                    a = acc[key][name]
                    a[0] += int(ok[j].sum()); a[1] += A

names = ['trained vp head', 'gt', 'slot', 'essence']
print(f'{"depth-2 slots":<44} {"triples":>8} ' + ' '.join(f'{n:>16}' for n in names))
keys = ['all depth-2', 'R-type writer, composed + fresh input', 'other depth-2'] + sorted(
    k for k in acc if k.startswith('writer') and acc[k]['gt'][1] >= 2000)
for k in keys:
    if k in acc:
        print(f'{k:<44} {acc[k]["gt"][1]:>8} ' + ' '.join(
            f'{100 * acc[k][n][0] / acc[k][n][1]:15.2f}%' for n in names))
