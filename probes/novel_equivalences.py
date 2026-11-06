#!/usr/bin/env python3
"""Whether a T2 collapses equivalences no named rewrite produces.

The partner rewrites (`datagen.generate._REWRITES`: commutative swap,
respelling within a T1 partner group, ADDI/XORI fold, ADDI/XORI cancellation,
reassociation, identity write) each make two chunks one function by one local
edit, and a collapse learned from them alone may be a collapse of the edits.
Every class here is an equivalence no rewrite produces — the mined pairs
(`datagen.mine`) do draw them unless the config excludes their opcodes — each
against a near-miss — the same second spelling with one immediate moved — so a class is
read as its equivalent distance beside its near-miss distance:

  andi_fold     ANDI t,a,k1 ; ANDI t,t,k2       vs  ANDI t,a,k1&k2
  ori_fold      ORI  t,a,k1 ; ORI  t,t,k2       vs  ORI  t,a,k1|k2
  slli_fold     SLLI t,a,s1 ; SLLI t,t,s2       vs  SLLI t,a,s1+s2
  srli_fold     SRLI t,a,s1 ; SRLI t,t,s2       vs  SRLI t,a,s1+s2
  neg_flip      SUB  t,a,b  ; SUB  t,x0,t       vs  SUB  t,b,a
  double_twice  ADD  t,a,a  ; ADD  t,t,t        vs  SLLI t,a,2
  add_sub       ADD  t,a,b  ; SUB  t,t,b        vs  ADDI t,a,0

Each pair is checked equivalent (and each near-miss inequivalent) on the
emulator. Distances are chords between unit-normalized essences, 1,000 pairs
per class; `unrelated` is two random two-instruction chunks.

Usage: novel_equivalences.py <t2_run_dir> <t1_encoder.pt>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from compressor.model import instruction_wiring
from compressor.train import _t2_assemble
from emulator import Instruction as I, run as emu_run
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import encode_instruction, PAD

MAX_INSTR_TOKENS = 9
N = 1000
T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
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
    return (np.array(toks, dtype=np.int64).reshape(-1, MAX_INSTR_TOKENS),
            np.array(pads, dtype=bool).reshape(-1, MAX_INSTR_TOKENS),
            np.array(ci, dtype=np.int64), np.array(si, dtype=np.int64),
            np.array([len(x) for x in chunks], dtype=np.int64),
            np.array(w0, dtype=np.int64), np.array(w1, dtype=np.int64),
            np.array(wo, dtype=np.int64))


def encode(chunks, bs=512):
    out = []
    for i in range(0, len(chunks), bs):
        sub = chunks[i:i + bs]
        with torch.no_grad():
            ct1, c0, c1, co, cact = _t2_assemble(t1, build(sub), len(sub), 2, dev, route='binding')
            _, ess = t2.core(ct1, c0, c1, co, cact)
        out.append(F.normalize(ess.float(), dim=-1))
    return torch.cat(out)


def same(a, b, n=64):
    for _ in range(n):
        st = np.zeros(32, dtype=np.int32)
        st[1:] = rng.integers(-(2 ** 31), 2 ** 31 - 1, 31).astype(np.int32)
        st[1:8] = rng.choice([0, 1, -1, 5, -5, 2 ** 31 - 1, -2 ** 31], 7)
        if not np.array_equal(emu_run(a, st.copy())[0].regs, emu_run(b, st.copy())[0].regs):
            return False
    return True


def imm12():
    return int(rng.integers(-2048, 2048))


def regs():
    t, a, b = (int(x) for x in rng.choice(np.arange(1, 32), 3, replace=False))
    return t, a, b


def sample(kind):
    """(spelling, equivalent spelling, near-miss of the second)."""
    t, a, b = regs()
    if kind == 'andi_fold':
        k1, k2 = imm12(), imm12()
        k = ((k1 & k2) & 0xFFF) - (((k1 & k2) & 0x800) << 1)
        return ([I('ANDI', t, a, k1), I('ANDI', t, t, k2)], [I('ANDI', t, a, k)],
                [I('ANDI', t, a, k ^ 1)])
    if kind == 'ori_fold':
        k1, k2 = imm12(), imm12()
        k = ((k1 | k2) & 0xFFF) - (((k1 | k2) & 0x800) << 1)
        return ([I('ORI', t, a, k1), I('ORI', t, t, k2)], [I('ORI', t, a, k)],
                [I('ORI', t, a, k ^ 1)])
    if kind in ('slli_fold', 'srli_fold'):
        op = kind[:4].upper()
        s1 = int(rng.integers(1, 15)); s2 = int(rng.integers(1, 31 - s1))
        return ([I(op, t, a, s1), I(op, t, t, s2)], [I(op, t, a, s1 + s2)],
                [I(op, t, a, s1 + s2 + 1)])
    if kind == 'neg_flip':
        return ([I('SUB', t, a, b), I('SUB', t, 0, t)], [I('SUB', t, b, a)],
                [I('ADD', t, b, a)])
    if kind == 'double_twice':
        return ([I('ADD', t, a, a), I('ADD', t, t, t)], [I('SLLI', t, a, 2)],
                [I('SLLI', t, a, 3)])
    if kind == 'add_sub':
        return ([I('ADD', t, a, b), I('SUB', t, t, b)], [I('ADDI', t, a, 0)],
                [I('ADDI', t, a, 1)])
    raise ValueError(kind)


KINDS = ['andi_fold', 'ori_fold', 'slli_fold', 'srli_fold', 'neg_flip',
         'double_twice', 'add_sub']
R_OPS = ['ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLT', 'SLTU', 'SLL', 'SRL', 'SRA']


def unrelated():
    return [I(str(rng.choice(R_OPS)), *(int(x) for x in rng.integers(1, 32, 3))) for _ in range(2)]


print(f'{"class":14s} {"equivalent":>11s} {"near-miss":>10s}  (chord, mean over {N} pairs)')
for kind in KINDS:
    trip = [sample(kind) for _ in range(N)]
    for x, y, z in trip[:50]:
        assert same(x, y), (kind, x, y)
        assert not same(y, z), (kind, y, z)
    ex, ey, ez = (encode([p[k] for p in trip]) for k in range(3))
    print(f'{kind:14s} {float((ex - ey).norm(dim=-1).mean()):11.4f} '
          f'{float((ey - ez).norm(dim=-1).mean()):10.4f}')
u = encode([unrelated() for _ in range(2 * N)])
print(f'{"unrelated":14s} {float((u[:N] - u[N:]).norm(dim=-1).mean()):11.4f}')
