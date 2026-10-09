#!/usr/bin/env python3
"""Is immediate encoding the ENTIRE vp residual at len=2?

Bit-position data showed error concentrated in bits 0-11 (the RISC-V immediate
width) and ADDI far worse than ADD. That says immediates are *a* driver. It does
NOT establish they are the only one.

Decisive split: R-type instructions carry NO immediate at all (both operands are
register values the vp head is handed). If immediates were the whole residual,
every R-type output slot would be at ~100%. Whatever R-type error remains is a
SECOND mechanism.

Reports, for R-type and I-type separately: accuracy, per-bit-position curve,
share of total error, and the counterfactual "overall accuracy if I-type were
perfect". Also crossed with dataflow depth, since depth-2 carries its own
penalty independent of operand type.

Model's own trained head, forward passes only.
Usage: vp_imm_share.py <t2_dir> <t1_path> [chunk_len] [n_batches]
"""
import os
import subprocess
import sys

import numpy as np
import torch
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
LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 300
N_ANCHORS, ANCHOR_SEED, NBITS, MAX_OUT = 8, 0, 32, 16

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
core = t2.core

CORPUS = f'/tmp/vp_imm_len{LEN}_{os.getpid()}.rvt'
subprocess.run(
    f'python -m datagen --rule "cap={LEN}" --twins 0 --batch-size 128 '
    f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
    f'--config configs/instr_alu_writers.json --count {NB} > {CORPUS}',
    shell=True, check=True, stderr=subprocess.DEVNULL)

anchors = torch.from_numpy(
    np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)

bit_ok = {'R': torch.zeros(NBITS, device=dev), 'I': torch.zeros(NBITS, device=dev)}
bit_n = {'R': torch.zeros(NBITS, device=dev), 'I': torch.zeros(NBITS, device=dev)}
err = {'R': 0.0, 'I': 0.0}
cnt = {'R': 0.0, 'I': 0.0}
# crossed with dataflow depth
dep = {}


def slot_trace(in0, in1, out, n):
    d = np.zeros(N_REGS, dtype=np.int64)
    w = np.full(N_REGS, -1, dtype=np.int64)
    for k in range(n):
        i0, i1, o = int(in0[k]), int(in1[k]), int(out[k])
        nd = 1 + max(d[i0], d[i1])
        if o != 0:
            d[o], w[o] = nd, k
    return d, w


with open(CORPUS, 'rb') as f:
    for batch in RVT_FORMAT.reader(f, Batch):
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        (_, _, chunk_idx, slot_idx, n_per_chunk, w0, w1, wo) = split
        nch, mni = batch.instr_lens.shape
        ct1, c0, c1, co, cact = _t2_assemble(
            t1, split, nch, mni, dev, route='binding')
        with torch.no_grad():
            _, ess = core(ct1, c0, c1, co, cact)

        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        vp_mask = torch.from_numpy(batch.out_regs_valid).to(dev)

        with torch.no_grad():
            fill_i = (in_slot != AUX_CE_IGNORE).to(torch.int64)
            iv = ANCH_IN.unsqueeze(0) * fill_i.unsqueeze(1)
            ifeat = torch.cat(
                [_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
                dim=-1).reshape(ess.shape[0], N_ANCHORS, -1)
            vexp = F.normalize(ess, dim=-1).unsqueeze(1).expand(-1, N_ANCHORS, -1)
            pr = t2.vp_head(torch.cat([vexp, ifeat], dim=-1)).reshape(
                ess.shape[0], N_ANCHORS, -1, NBITS + 1)
            tg = out_regs.gather(-1, out_slot.clamp(min=0).unsqueeze(1)
                                 .expand(-1, N_ANCHORS, -1))
            hit = ((pr[..., :NBITS] > 0).float()
                   == _value_bits(tg)).float()              # (B,A,MAX_OUT,32)
            per_pair = hit.mean(-1).mean(1)                  # (B,MAX_OUT)

        # classify each output slot by the type of the instruction that wrote it
        order = np.argsort(chunk_idx * 1000 + slot_idx)
        wo_s, w0_s, w1_s = wo[order], w0[order], w1[order]
        is_i = np.zeros((nch, MAX_OUT), dtype=bool)
        known = np.zeros((nch, MAX_OUT), dtype=bool)
        depth_np = np.zeros((nch, MAX_OUT), dtype=np.int64)
        vnp = ((out_slot != AUX_CE_IGNORE)
               & vp_mask.unsqueeze(-1)).cpu().numpy()
        oreg = out_slot.clamp(min=0).cpu().numpy()
        pos = 0
        for b in range(nch):
            n = int(n_per_chunk[b])
            if n:
                dvec, _ = slot_trace(w0_s[pos:pos + n], w1_s[pos:pos + n],
                                     wo_s[pos:pos + n], n)
                toks = batch.tokens[b].tolist()
                ins, p = [], 0
                try:
                    for _ in range(n):
                        i, p = decode_instruction(toks, p)
                        ins.append(i)
                except Exception:
                    ins = []
                if len(ins) == n:
                    last = {}
                    for k in range(n):
                        if int(wo_s[pos + k]) != 0:
                            last[int(wo_s[pos + k])] = ins[k].opcode
                    for k in np.nonzero(vnp[b])[0]:
                        op = last.get(int(oreg[b, k]))
                        if op:
                            known[b, k] = True
                            is_i[b, k] = op in I_TYPE
                            depth_np[b, k] = dvec[oreg[b, k]]
            pos += n

        kn = torch.from_numpy(known).to(dev)
        ii = torch.from_numpy(is_i).to(dev)
        for tag, sel in (('R', kn & ~ii), ('I', kn & ii)):
            m = sel.float().unsqueeze(1).unsqueeze(-1)       # (B,1,MAX_OUT,1)
            bit_ok[tag] += (hit * m).sum(dim=(0, 1, 2))
            bit_n[tag] += m.expand_as(hit).sum(dim=(0, 1, 2))
            err[tag] += float(((1 - per_pair) * sel.float()).sum())
            cnt[tag] += float(sel.sum())
        dnp = depth_np
        for tag, sel in (('R', known & ~is_i), ('I', known & is_i)):
            for d in (1, 2):
                s2 = sel & (dnp == d)
                if s2.any():
                    k = (tag, d)
                    a, c = dep.get(k, (0.0, 0.0))
                    pp = per_pair.cpu().numpy()
                    dep[k] = (a + float(pp[s2].sum()), c + float(s2.sum()))

tot_err = err['R'] + err['I']
tot_n = cnt['R'] + cnt['I']
print(f'checkpoint = {T2_DIR}   len={LEN}')
print(f'  pairs={int(tot_n)}   overall bit acc = '
      f'{100 * (1 - tot_err / tot_n):.3f}%')
print()
for tag, label in (('R', 'R-type  (NO immediate)'), ('I', 'I-type  (12-bit imm)')):
    a = (100 * bit_ok[tag] / bit_n[tag].clamp(min=1)).cpu().numpy()
    acc = 100 * (1 - err[tag] / max(cnt[tag], 1))
    print(f'  {label:26s} acc={acc:7.3f}%   n={int(cnt[tag]):6d}   '
          f'share of total error = {100 * err[tag] / tot_err:5.1f}%')
    print(f'    {"":24s} bits 0-11 mean {a[:12].mean():6.3f}%   '
          f'bits 12-31 mean {a[12:].mean():6.3f}%')
print()
print('  crossed with dataflow depth')
for (tag, d), (a, c) in sorted(dep.items()):
    print(f'    {tag}-type depth {d}: {100 * a / c:7.3f}%   (n={int(c)})')
print()
cf = 100 * (1 - err['R'] / tot_n)
print(f'  COUNTERFACTUAL: if every I-type slot were perfect, overall = {cf:.3f}%')
print(f'    (i.e. immediates account for {100 * err["I"] / tot_err:.1f}% of the '
      f'residual; R-type accounts for {100 * err["R"] / tot_err:.1f}%)')
