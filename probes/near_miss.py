#!/usr/bin/env python3
"""Near-miss essence distances of a frozen T1: how far apart it puts pairs of
instructions that compute DIFFERENT functions, differing by a little or by a
lot, against what the behavioral distance asks for.

Rows, 2,000 pairs each, registers drawn at random and shared within a pair:
  ADDI k vs k+1, ADDI k vs k+512, ADD vs SUB, SLLI 3 vs SLLI 4.
Each row reports the mean chord distance between unit-normalized essences and
its target: twice the behavioral distance `distance_loss` trains toward
(`pair_distance_targets`, `compressor/train.py`), computed on the same pairs.
A graded space keeps k+1 well inside k+512.

Usage: near_miss.py <t1_run_dir> [<t1_run_dir> ...]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.model import instruction_wiring
from compressor.train import pair_distance_targets, witness_outputs
from datagen import AUX_CE_IGNORE, MAX_INPUT_SLOTS
from datagen.knownbits import output_known_bits
from datagen.signature import output_inputs
from emulator import Instruction as I
from scripts._common import load_frozen_encoder
from tokenizer import encode_instruction, PAD

N = 2000


def _grid(instrs):
    toks = [encode_instruction(i) for i in instrs]
    L = max(map(len, toks))
    tok = torch.full((len(instrs), L), PAD)
    pad = torch.ones(len(instrs), L, dtype=torch.bool)
    for j, t in enumerate(toks):
        tok[j, :len(t)] = torch.tensor(t)
        pad[j, :len(t)] = False
    return tok, pad


def encode(t1, instrs):
    tok, pad = _grid(instrs)
    w = torch.tensor([instruction_wiring(i) for i in instrs])
    with torch.no_grad():
        e = t1.encode_state(tok.cuda(), pad.cuda(), w[:, 0].cuda(), w[:, 1].cuda(), w[:, 2].cuda())[1]
    return F.normalize(e.float(), dim=-1)


def target(a, b):
    """Twice the behavioral distance of each pair (a[i], b[i])."""
    rows = a + b
    tok, pad = _grid(rows)
    oi = torch.full((len(rows), 1, MAX_INPUT_SLOTS), AUX_CE_IGNORE)
    for j, ins in enumerate(rows):
        regs = output_inputs([ins]).get(ins.args[0], [])
        oi[j, 0, :len(regs)] = torch.tensor(regs)
    kb = [output_known_bits([i])[i.args[0]] for i in rows]
    D = pair_distance_targets(witness_outputs(tok.unsqueeze(1), pad.unsqueeze(1), oi),
                              torch.tensor([k.zero for k in kb]), torch.tensor([k.one for k in kb]))
    n = len(a)
    return 2.0 * float(torch.stack([D[i, n + i] for i in range(n)]).mean())


def pairs():
    rng = np.random.default_rng(0)
    r = lambda: [int(x) for x in rng.integers(1, 32, 3)]
    k = rng.integers(-2000, 1400, N)
    out = {}
    for d in (1, 512):
        a, b = [], []
        for kk in k:
            rd, rs, _ = r()
            a.append(I('ADDI', rd, rs, int(kk)))
            b.append(I('ADDI', rd, rs, int(kk) + d))
        out[f'ADDI k vs k+{d}'] = (a, b)
    a, b = [], []
    for _ in range(N):
        rd, x, y = r()
        y = y if y != x else (x % 31) + 1
        a.append(I('ADD', rd, x, y))
        b.append(I('SUB', rd, x, y))
    out['ADD vs SUB'] = (a, b)
    a, b = [], []
    for _ in range(N):
        rd, x, _ = r()
        a.append(I('SLLI', rd, x, 3))
        b.append(I('SLLI', rd, x, 4))
    out['SLLI 3 vs SLLI 4'] = (a, b)
    return out


rows = pairs()
targets = {name: target(a, b) for name, (a, b) in rows.items()}
print('target  ' + '  '.join(f'{name} {t:.4f}' for name, t in targets.items()))
for path in sys.argv[1:]:
    t1 = load_frozen_encoder(path.rstrip('/') + '/encoder.pt', 'cuda')
    print(f'{path.rstrip("/").split("/")[-1]:24s}' + '  '.join(
        f'{name} {float((encode(t1, a) - encode(t1, b)).norm(dim=-1).mean()):.4f}'
        for name, (a, b) in rows.items()))
