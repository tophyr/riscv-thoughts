#!/usr/bin/env python3
"""Do two-instruction chunks computing one function land on one point, for
each chunk partner rewrite the generator draws, and for commutative swaps
in naturally drawn chunks?

Rows, 1,000 pairs each, essence chord distance between unit-normalized
essences:
  <rewrite> equivalent — the two spellings `datagen.generate.sample_chunk_partner_pair`
             draws for that rewrite (checked equal on the emulator). MUST be ~0.
  <rewrite> near-miss — the second spelling with one I-type immediate +1, or its
             first operation changed (checked different on the emulator): the
             denominator a small equivalent distance is read against.
  swap: <class>, 1 / 2 outputs — a naturally drawn len-2 chunk against itself
             with one commutative instruction's operands swapped, by how the
             swapped instruction meets the other one: `shared input` (both read
             a register from outside), `operand is a result` (it reads the
             other's result), `result is read` (the other reads its result);
             and by how many registers the chunk writes, since the distance
             term pairs one-output rows only.
  unrelated — naturally drawn len-2 chunks against each other.

Direct distances, no fitted readout.

Usage: partner_chunks.py <t2_dir> <t1_path> [n_pairs]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

import datagen.generate as G
from compressor.model import instruction_wiring
from compressor.train import _t2_assemble
from datagen.batch import generate_chunks
from datagen.compare import draw_anchor_states
from emulator import Instruction as I, run as emu_run
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import encode_instruction, PAD

MAX_INSTR_TOKENS = 9
T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
COMM = {'ADD', 'AND', 'OR', 'XOR'}

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
STATES = draw_anchor_states(np.random.default_rng(4), 64)
rng = np.random.default_rng(0)


def encode(chunks, bs=512):
    out = []
    for i in range(0, len(chunks), bs):
        sub = chunks[i:i + bs]
        toks, pads, ci, si, w0, w1, wo = [], [], [], [], [], [], []
        for c, instrs in enumerate(sub):
            for s, ins in enumerate(instrs):
                t = encode_instruction(ins)
                toks.append(t + [PAD] * (MAX_INSTR_TOKENS - len(t)))
                pads.append([False] * len(t) + [True] * (MAX_INSTR_TOKENS - len(t)))
                a, b, o = instruction_wiring(ins)
                ci.append(c); si.append(s); w0.append(a); w1.append(b); wo.append(o)
        split = (np.array(toks, dtype=np.int64), np.array(pads, dtype=bool),
                 np.array(ci), np.array(si), np.array([len(x) for x in sub]),
                 np.array(w0), np.array(w1), np.array(wo))
        with torch.no_grad():
            ct1, c0, c1, co, cact = _t2_assemble(
                t1, split, len(sub), max(len(x) for x in sub), dev, route='binding')
            _, ess = t2.core(ct1, c0, c1, co, cact)
        out.append(F.normalize(ess.float(), dim=-1))
    return torch.cat(out)


def dist(a, b):
    return float((encode(a) - encode(b)).norm(dim=-1).mean())


def same(x, y):
    for s in STATES:
        s = s.copy(); s[0] = 0
        if not np.array_equal(emu_run(x, s.copy())[0].regs, emu_run(y, s.copy())[0].regs):
            return False
    return True


_SWAP = {'ADD': 'SUB', 'SUB': 'ADD', 'XOR': 'OR', 'OR': 'XOR', 'AND': 'OR',
         'SLT': 'SLTU', 'SLTU': 'SLT', 'SLL': 'SRL', 'SRL': 'SLL', 'SRA': 'SRL',
         'SLLI': 'SRLI', 'SRLI': 'SLLI', 'SRAI': 'SRLI'}


def near_miss(y):
    for k, i in enumerate(y):
        if i.opcode in ('ADDI', 'XORI', 'ORI', 'ANDI', 'SLTI', 'SLTIU') and i.args[2] < 2047:
            z = list(y); z[k] = I(i.opcode, i.args[0], i.args[1], i.args[2] + 1)
            return z
    z = list(y); z[0] = I(_SWAP.get(y[0].opcode, 'ADD'), *y[0].args)
    return z


print(f'{"partner rewrite":20s} {"equivalent":>11s} {"near-miss":>10s}')
for rewrite in G._REWRITES:
    xs, ys, zs = [], [], []
    while len(xs) < N:
        x, y = G.sample_chunk_partner_pair(rng, 2, range(1, 32), (rewrite,))
        z = near_miss(y)
        assert same(x, y), (x, y)
        if same(x, z):
            continue
        xs.append(x); ys.append(y); zs.append(z)
    print(f'{rewrite:20s} {dist(xs, ys):11.4f} {dist(xs, zs):10.4f}', flush=True)

alu = G.build_opcode_table({'R_ALU': 0.55, 'I_ALU': 0.45})
natural = generate_chunks(G.length_cap(2), rng, opcode_table=alu,
                          dest_regs=list(range(1, 32)), chain_rate=0.5)
reads = lambda i: set(i.args[1:3]) if i.opcode in G._ALU_R_OPS else {i.args[1]}
CLASSES = ('shared input', 'operand is a result', 'result is read')
by = {(c, o): ([], []) for c in CLASSES for o in (1, 2)}
while min(len(v[0]) for v in by.values()) < N:
    c = next(natural).instructions
    k = [j for j, i in enumerate(c) if i.opcode in COMM and i.args[1] != i.args[2]]
    if not k:
        continue
    j = k[0]; other = c[1 - j]
    if j == 1 and other.args[0] in c[j].args[1:3]:
        cls = 'operand is a result'
    elif j == 0 and c[j].args[0] in reads(other):
        cls = 'result is read'
    elif set(c[j].args[1:3]) & reads(other):
        cls = 'shared input'
    else:
        continue
    sw = list(c); sw[j] = I(c[j].opcode, c[j].args[0], c[j].args[2], c[j].args[1])
    key = (cls, len({i.args[0] for i in c}))
    if len(by[key][0]) < N:
        by[key][0].append(c); by[key][1].append(sw)
for (cls, outs), (a, b) in by.items():
    print(f'swap: {cls:20s} {outs} output{"s" if outs > 1 else " "}: {dist(a, b):.4f}')
rand = [next(natural).instructions for _ in range(N)]
print(f'unrelated len-2 chunks: {dist(rand, rand[1:] + rand[:1]):.4f}')
