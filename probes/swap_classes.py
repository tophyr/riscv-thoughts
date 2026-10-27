#!/usr/bin/env python3
"""Is invariance to a commutative operand swap structural or learned, and where
does the difference live?

A commutative swap (`ADD rd,a,b` -> `ADD rd,b,a`) leaves the chunk's function
unchanged. Whether the essence is unchanged depends on which register slots the
swap moves between, so the pairs are split into four classes and every class is
encoded by the TRAINED pair and by a RANDOMLY INITIALISED pair of the same
architecture. A distance at float noise on the random pair is structural — no
gradient was involved; a distance the trained pair has reduced from the random
pair's is learned, and the ratio says how far.

  fresh operands, destination elsewhere   — operand 0 of a fresh pair always
        receives read rank 1, so the op-cell's inputs are identical and the two
        leaves merely trade slots, which a permutation-invariant pool cannot see.
  destination overwrites a swapped source — the surviving source keeps a
        different rank (2 against 1) in its read-only slot, which the pool sees.
  2nd instruction, operand already ranked — the swapped operand carries a rank
        the first instruction assigned, so the op-cell receives (e1,e3) in one
        chunk and (e3,e1) in the other; only training can make it commutative.
  2nd instruction, operand is a result     — the same, with the first
        instruction's result as the moved operand.

Direct distances between unit-normalized essences, no fitted readout, every
chunk encoded in a two-slot grid. The T1 rows encode the first instruction alone
on token wiring; the T2 rows route through T1's emitted wiring for the trained
pair and through token wiring for the random pair, whose binding heads emit
noise. The emulator checks every pair computes one function.

Usage: swap_classes.py <t2_dir> <t1_path> [n_pairs]
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.model import T1Compressor, T2Compressor, instruction_wiring
from compressor.train import _t2_assemble
from emulator import Instruction
from emulator import run as emu_run
from scripts._common import (VOCAB_SIZE, load_frozen_encoder, load_t2,
                             resolve_device)
from tokenizer import encode_instruction, PAD

MAX_INSTR_TOKENS = 9      # tokenizer.MAX_INSTR_TOKENS is not exported; see
                          # compressor.train._MAX_INSTR_TOKENS
COMM = ['ADD', 'AND', 'OR', 'XOR']

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, hp2 = load_t2(T2_DIR, t1, dev)
hp1 = json.load(open(T1_PATH.rsplit('/', 1)[0] + '/hparams.json'))
torch.manual_seed(0)
r1 = T1Compressor(
    vocab_size=VOCAB_SIZE, d_model=hp1['d_model'], n_heads=hp1['n_heads'],
    n_layers=hp1['n_layers'], max_window=hp1['max_window'], d_out=hp1['d_out'],
    d_event=hp1['d_event'], op_cell_depth=hp1['op_cell_depth'],
    vp_head_depth=hp1['vp_head_depth'],
    n_pool_heads=hp1['n_pool_heads']).to(dev).eval()
r2 = T2Compressor(
    d_t1=hp1['d_out'], d_out=hp2['d_out'], d_event=hp2['d_event'],
    op_cell_depth=hp2['op_cell_depth'], vp_head_depth=hp2['vp_head_depth'],
    n_pool_heads=hp2['n_pool_heads']).to(dev).eval()
rng = np.random.default_rng(0)


def build(chunks):
    toks, pads, ci, si, w0, w1, wo = [], [], [], [], [], [], []
    for c, instrs in enumerate(chunks):
        for s, ins in enumerate(instrs):
            t = encode_instruction(ins)
            toks.append(t + [PAD] * (MAX_INSTR_TOKENS - len(t)))
            pads.append([False] * len(t) + [True] * (MAX_INSTR_TOKENS - len(t)))
            a, b, o = instruction_wiring(ins)
            ci.append(c); si.append(s); w0.append(a); w1.append(b); wo.append(o)
    return (np.array(toks, dtype=np.int64), np.array(pads, dtype=bool),
            np.array(ci), np.array(si), np.array([len(x) for x in chunks]),
            np.array(w0), np.array(w1), np.array(wo))


def t2_enc(m1, m2, chunks, route, bs=512):
    out = []
    for i in range(0, len(chunks), bs):
        sub = chunks[i:i + bs]
        with torch.no_grad():
            ct1, c0, c1, co, ca = _t2_assemble(
                m1, build(sub), len(sub), 2, dev, route=route)
            out.append(F.normalize(m2.core(ct1, c0, c1, co, ca)[1].float(), dim=-1))
    return torch.cat(out)


def t1_enc(m1, instrs):
    sp = build([[i] for i in instrs])
    with torch.no_grad():
        _, ess = m1.encode_state(*(torch.from_numpy(x).to(dev)
                                   for x in (sp[0], sp[1], sp[5], sp[6], sp[7])))
    return F.normalize(ess.float(), dim=-1)


def fresh_pairs(alias):
    base, swap = [], []
    while len(base) < N:
        a, b = (int(x) for x in rng.integers(1, 32, 2))
        if a == b:
            continue
        rd = (int(rng.choice([a, b])) if alias
              else int(rng.choice([r for r in range(1, 32) if r not in (a, b)])))
        others = [r for r in range(1, 32) if r not in (a, b, rd)]
        rd2, c, d = (int(rng.choice(others)) for _ in range(3))
        op = COMM[rng.integers(4)]
        base.append([Instruction(op, rd, a, b), Instruction('XOR', rd2, c, d)])
        swap.append([Instruction(op, rd, b, a), Instruction('XOR', rd2, c, d)])
    return base, swap


def nonfresh_pairs(result):
    base, swap = [], []
    for _ in range(N):
        rd1, a, b, rd2, c = [int(x) for x in rng.permutation(31)[:5] + 1]
        s = rd1 if result else a
        base.append([Instruction('XOR', rd1, a, b), Instruction('ADD', rd2, s, c)])
        swap.append([Instruction('XOR', rd1, a, b), Instruction('ADD', rd2, c, s)])
    return base, swap


def check_equal(a, b, every=50):
    for k in range(0, len(a), max(1, len(a) // every)):
        st = np.zeros(32, dtype=np.int32)
        st[1:] = rng.integers(-(2 ** 31), 2 ** 31 - 1, 31).astype(np.int32)
        assert np.array_equal(emu_run(a[k], st.copy())[0].regs,
                              emu_run(b[k], st.copy())[0].regs), (a[k], b[k])


rows = [('fresh operands, destination elsewhere', fresh_pairs(False), True),
        ('destination overwrites a swapped source', fresh_pairs(True), True),
        ('2nd instr: operand already ranked by the 1st', nonfresh_pairs(False), False),
        ("2nd instr: operand is the 1st's result", nonfresh_pairs(True), False)]
for _, (base, swap), _ in rows:
    check_equal(base, swap)

print()
print(f'  commutative swap, {N} pairs per class, L2 between unit-normalized '
      f'essences; "random" = the same architecture at initialisation')
print(f'  {"swap class":46s} {"trained T1":>11s} {"random T1":>11s} '
      f'{"trained T2":>11s} {"random T2":>11s}')
for name, (base, swap), t1_row in rows:
    vals = []
    for m1, m2, route in ((t1, t2, 'binding'), (r1, r2, 'tokens')):
        d1 = ((t1_enc(m1, [x[0] for x in base]) - t1_enc(m1, [x[0] for x in swap]))
              .norm(dim=-1).mean().item() if t1_row else float('nan'))
        d2 = (t2_enc(m1, m2, base, route) - t2_enc(m1, m2, swap, route)) \
            .norm(dim=-1).mean().item()
        vals += [d1, d2]
    print(f'  {name:46s} {vals[0]:11.2e} {vals[2]:11.2e} {vals[1]:11.2e} {vals[3]:11.2e}')


def rand_chunk():
    return [Instruction(COMM[rng.integers(4)], *[int(v) for v in rng.integers(1, 32, 3)])
            for _ in range(2)]


X = [rand_chunk() for _ in range(N)]
Y = [rand_chunk() for _ in range(N)]
dx = [(t2_enc(m1, m2, X, route) - t2_enc(m1, m2, Y, route)).norm(dim=-1).mean().item()
      for m1, m2, route in ((t1, t2, 'binding'), (r1, r2, 'tokens'))]
print(f'  {"unrelated chunks (denominator)":46s} {"":11s} {"":11s} {dx[0]:11.2e} {dx[1]:11.2e}')

# Where the aliased swap's difference lives in the trained T2's final state:
# the written slot (fed by T1's content, which has the same leak one tier down)
# against the surviving source's read-only slot (rank 2 against rank 1).
base, swap, leaf, rd = [], [], [], []
for _ in range(N):
    a, b = (int(x) for x in rng.integers(1, 32, 2))
    while b == a:
        b = int(rng.integers(1, 32))
    others = [r for r in range(1, 32) if r not in (a, b)]
    rd2, c, e = (int(rng.choice(others)) for _ in range(3))
    base.append([Instruction('ADD', a, a, b), Instruction('XOR', rd2, c, e)])
    swap.append([Instruction('ADD', a, b, a), Instruction('XOR', rd2, c, e)])
    leaf.append(b); rd.append(a)
with torch.no_grad():
    Tb = t2.core(*_t2_assemble(t1, build(base), N, 2, dev, route='binding'))[0]
    Ts = t2.core(*_t2_assemble(t1, build(swap), N, 2, dev, route='binding'))[0]
idx = torch.arange(N, device=dev)
d = t2.d_out
print()
print('  trained T2, aliased swap `ADD a,a,b` vs `ADD a,b,a`: max |diff| of the value channels')
print(f'    written slot a:               {(Tb[idx, torch.tensor(rd, device=dev), :d] - Ts[idx, torch.tensor(rd, device=dev), :d]).abs().max().item():.2e}')
print(f'    surviving read-only slot b:   {(Tb[idx, torch.tensor(leaf, device=dev), :d] - Ts[idx, torch.tensor(leaf, device=dev), :d]).abs().max().item():.2e}')
print()
