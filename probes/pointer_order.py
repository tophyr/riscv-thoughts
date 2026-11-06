#!/usr/bin/env python3
"""Whether the essence depends on how a chunk's outputs would rank under a
fixed projection of their descriptions — an order an output pointer could be
taken from, and the order the encoder's pooling must not leak. An essence
that ignores order moves no further for a near-miss that swaps two outputs'
ranks than for one that keeps them (EXPERIMENT_LOG, findings 58 and 61).

Two-output len-2 chunks only (a one-output chunk has one rank):
  agreement     equivalent pairs, both chunks writing the same two registers —
                a commutative swap of one instruction in a natural chunk, and
                the partner rewrites; a flip is a register whose rank differs
                between the two.
  near-misses   a natural chunk against itself with one I-type immediate moved
                by 1 or by 2–64; split by whether the ranks swap, each pair's
                essence chord is read against its target chord √(2d), d the
                register-file target (`register_distance_targets`).

Usage: pointer_order.py <t2_run_dir> <t1_encoder.pt> [n]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from compressor.model import instruction_wiring
from compressor.train import _t2_assemble, register_distance_targets
from datagen.batch import build_targets, generate_chunks, pack_batch, _make_valid_chunk
from datagen.compare import draw_anchor_states
from datagen.generate import (build_opcode_table, length_cap, load_distribution,
                              sample_chunk_partner_pair)
from emulator import Instruction as I, run as emu_run
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import encode_instruction, PAD

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
core = t2.core
rng = np.random.default_rng(0)
COMM = {'ADD', 'AND', 'OR', 'XOR'}
I_IMM = {'ADDI', 'SLTI', 'SLTIU', 'XORI', 'ORI', 'ANDI'}


def build(chunks):
    toks, pads, ci, si, w0, w1, wo = [], [], [], [], [], [], []
    for c, instrs in enumerate(chunks):
        for s, ins in enumerate(instrs):
            t = encode_instruction(ins)
            toks.append(t + [PAD] * (9 - len(t)))
            pads.append([False] * len(t) + [True] * (9 - len(t)))
            a, b, o = instruction_wiring(ins)
            ci.append(c); si.append(s); w0.append(a); w1.append(b); wo.append(o)
    return (np.array(toks, dtype=np.int64).reshape(-1, 9),
            np.array(pads, dtype=bool).reshape(-1, 9),
            np.array(ci, dtype=np.int64), np.array(si, dtype=np.int64),
            np.array([len(x) for x in chunks], dtype=np.int64),
            np.array(w0, dtype=np.int64), np.array(w1, dtype=np.int64),
            np.array(wo, dtype=np.int64))


_U = None


def encode(chunks, bs=512):
    """(unit essences (n, d), each written register's rank (n, 32), -1
    unwritten): the rank under a fixed random projection (seed 0xD15C) of the
    value channels of the written registers' descriptions."""
    global _U
    es, ps = [], []
    d = core.d_value
    if _U is None:
        u = torch.randn(d, generator=torch.Generator().manual_seed(0xD15C))
        _U = (u / u.norm()).to(dev)
    for i in range(0, len(chunks), bs):
        sub = chunks[i:i + bs]
        with torch.no_grad():
            ct1, c0, c1, co, cact = _t2_assemble(t1, build(sub), len(sub), 2, dev, route='binding')
            T, ess = core(ct1, c0, c1, co, cact)
        desc = T[..., :d]
        wr = torch.zeros(len(sub), 32, dtype=torch.bool, device=dev)
        for k, c in enumerate(sub):
            wr[k, written(c)] = True
        proj = torch.where(wr, desc @ _U, torch.full_like(desc[..., 0], float('inf')))
        rank = (proj.unsqueeze(1) < proj.unsqueeze(2)).sum(-1)              # [r]: below r
        es.append(F.normalize(ess.float(), dim=-1).cpu())
        ps.append(torch.where(wr, rank, torch.full_like(rank, -1)).cpu())
    return torch.cat(es), torch.cat(ps)


def written(c):
    return sorted({i.args[0] for i in c if i.args[0] != 0})


def flips(pa, pb, regs):
    return any(int(pa[r]) != int(pb[r]) for r in regs)


def same(a, b):
    for _ in range(32):
        st = np.zeros(32, dtype=np.int32)
        st[1:] = rng.integers(-(2 ** 31), 2 ** 31 - 1, 31).astype(np.int32)
        if not np.array_equal(emu_run(a, st.copy())[0].regs, emu_run(b, st.copy())[0].regs):
            return False
    return True


def targets(pairs):
    rows = [_make_valid_chunk(c) for p in pairs for c in p]
    r = np.random.default_rng(1)
    aux, ov, oiv, orv = build_targets(rows, lambda k: np.stack([draw_anchor_states(r, 2) for _ in range(k)]), 2)
    b = pack_batch(rows, target_B=len(rows), target_max_tokens=18, target_max_n_instrs=2,
                   out_values=ov, in_role_values=oiv, out_regs_valid=orv, aux_payload=aux)
    dt = {'rf_wo_free': np.float32, 'rf_pos': np.int32, 'rf_wo': np.int32}
    f = [torch.from_numpy(getattr(b, k).astype(dt.get(k, np.int64))) for k in (
        'rf_pos', 'rf_pos_zero', 'rf_pos_one', 'rf_wo', 'rf_wo_zero', 'rf_wo_one', 'rf_wo_free', 'n_wo')]
    out = []
    for s in range(0, len(rows), 64):
        D = register_distance_targets(*[x[s:s + 64].to(dev) for x in f]).cpu()
        out += [float(D[k, k + 1]) for k in range(0, min(64, len(rows) - s), 2)]
    return np.sqrt(2.0 * np.array(out))


dist = load_distribution('configs/instr_alu_writers_chain010_rewrites025_minedlive025_aim050.json')
gen = generate_chunks(length_cap(2), rng, opcode_table=build_opcode_table(dist),
                      dest_regs=list(range(1, 32)), chain_rate=0.1)
two = []
while len(two) < 3 * N:
    c = next(gen).instructions
    if len(written(c)) == 2:
        two.append(c)

print(f'{T2_DIR.rstrip("/").split("/")[-1]}\n')
print('agreement on equivalent two-output pairs (fraction with a register whose rank differs):')
swaps = []
for c in two:
    k = [i for i, ins in enumerate(c) if ins.opcode in COMM and ins.args[1] != ins.args[2]]
    if k:
        d = list(c); a = d[k[0]].args; d[k[0]] = I(d[k[0]].opcode, a[0], a[2], a[1])
        swaps.append((c, d))
    if len(swaps) == N:
        break
rew = []
while len(rew) < N:
    x, y = sample_chunk_partner_pair(rng, 2)
    if len(x) == 2 and len(y) == 2 and len(written(x)) == 2 and written(x) == written(y):
        rew.append((x, y))
for name, pairs in (('commutative swap in a natural chunk', swaps), ('partner rewrites', rew)):
    ea, pa = encode([p[0] for p in pairs]); eb, pb = encode([p[1] for p in pairs])
    f = np.array([flips(pa[i], pb[i], written(pairs[i][0])) for i in range(len(pairs))])
    d = (ea - eb).norm(dim=-1).numpy()
    print(f'  {name:38s} n={len(pairs):5d}  flips {f.mean():6.1%}   chord: flipped {d[f].mean() if f.any() else float("nan"):.4f}'
          f'  agreeing {d[~f].mean():.4f}')

print('\nnear-misses (one immediate moved), two-output natural chunks:')
for lo, hi, label in ((1, 2, 'by 1'), (2, 65, 'by 2-64')):
    pairs = []
    for c in two:
        k = [i for i, ins in enumerate(c) if ins.opcode in I_IMM]
        if not k:
            continue
        i = k[int(rng.integers(len(k)))]; ins = c[i]
        imm = int(np.clip(int(ins.args[2]) + int(rng.choice([-1, 1])) * int(rng.integers(lo, hi)), -2048, 2047))
        if imm == ins.args[2]:
            continue
        d = list(c); d[i] = I(ins.opcode, ins.args[0], ins.args[1], imm)
        if same(c, d):
            continue
        pairs.append((c, d))
        if len(pairs) == N:
            break
    ea, pa = encode([p[0] for p in pairs]); eb, pb = encode([p[1] for p in pairs])
    f = np.array([flips(pa[i], pb[i], written(pairs[i][0])) for i in range(len(pairs))])
    chord = (ea - eb).norm(dim=-1).numpy()
    tgt = targets(pairs)
    for m, nm in ((f, 'ranks swap'), (~f, 'ranks kept')):
        if m.any():
            print(f'  {label:8s} {nm:15s} n={m.sum():5d} ({m.mean():5.1%})  chord {chord[m].mean():.4f}'
                  f'  target {tgt[m].mean():.4f}  chord - target {np.mean(chord[m] - tgt[m]):+.4f}')
