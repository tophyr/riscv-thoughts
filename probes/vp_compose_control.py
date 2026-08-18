#!/usr/bin/env python3
"""At depth 2, is it the immediate that fails, or the intermediate it rides on?

The chunk shape is fixed throughout: two instructions, the second reading the
first's result, so the scored slot is at dataflow depth 2 in every arm. Only
WHERE the 12-bit literals sit changes:

    R->R    ADD  rB, rA, rX   ;  ADD  rC, rB, rY
    I->R    ADDI rB, rA, i1   ;  ADD  rC, rB, rY
    R->I    ADD  rB, rA, rX   ;  ADDI rC, rB, i2
    I->I    ADDI rB, rA, i1   ;  ADDI rC, rB, i2

Every arm computes a sum of the same shape and the same 32-bit width, so a
difference between them is not "composition is harder" and not "the target
value is bigger" — it is the literal specifically. On the corpus this cell has
only ~80 triples of `ADDI -> ADDI`, too few to rank; here the sample is chosen.

Two further arms separate the immediate from the intermediate it is applied to:

    cmp->I  SLT  rB, rA, rX   ;  ADDI rC, rB, i2
    I->cmp  ADDI rB, rA, i1   ;  SLT  rC, rB, rY

`SLT` produces 0 or 1, so `cmp->I` applies a literal to an intermediate carrying
one bit rather than 32. If that arm is healthy while `R->I` is not, what the
literal cannot survive is a wide intermediate, not composition.

Scoring is the model's own trained vp head on the slot the SECOND instruction
writes, against `precompute_chunk`'s oracle — no fitted readout.

Usage: vp_compose_control.py <t2_dir> <t1_path> [n_chunks]
"""
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.model import instruction_wiring
from compressor.train import _t2_assemble, _value_bits, _value_smooth
from datagen import AUX_CE_IGNORE, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS, N_REGS
from datagen.compare import make_anchor_states, precompute_chunk
from emulator import Instruction
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import encode_instruction, PAD

MAX_INSTR_TOKENS = 9
N_ANCHORS, ANCHOR_SEED, NBITS = 8, 0, 32

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)

anchors_np = np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))
anchors = torch.from_numpy(anchors_np).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)


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
            np.array(ci), np.array(si),
            np.array([len(x) for x in chunks]),
            np.array(w0), np.array(w1), np.array(wo))


def score(chunks, target_reg, bs=256):
    """Exact-32-bit vp rate on slot `target_reg[i]`, per chunk."""
    hits = trip = 0.0
    for i in range(0, len(chunks), bs):
        sub, tgt_regs = chunks[i:i + bs], target_reg[i:i + bs]
        split = build(sub)
        n = len(sub)
        with torch.no_grad():
            ct1, c0, c1, co, cact = _t2_assemble(
                t1, split, n, max(len(x) for x in sub), dev, route='binding')
            ess = t2.encode(ct1, c0, c1, co, cact,
                            torch.randn(n, t2.n_regs, device=dev))
        pre = [precompute_chunk(c, anchors_np) for c in sub]
        in_slot = torch.from_numpy(
            np.stack([p.in_slot_regs for p in pre]).astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(
            np.stack([p.out_regs for p in pre]).astype(np.int64)).to(dev)
        oslot = torch.from_numpy(
            np.stack([p.out_slot_regs for p in pre]).astype(np.int64)).to(dev)
        sel = torch.tensor([int(r) for r in tgt_regs], device=dev)
        tgt = out_regs.gather(
            -1, sel.view(-1, 1, 1).expand(-1, N_ANCHORS, 1)).squeeze(-1)
        with torch.no_grad():
            iv = (ANCH_IN.unsqueeze(0)
                  * (in_slot != AUX_CE_IGNORE).to(torch.int64).unsqueeze(1))
            ifeat = torch.cat(
                [_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
                dim=-1).reshape(n, N_ANCHORS, -1)
            v = F.normalize(ess, dim=-1).unsqueeze(1).expand(-1, N_ANCHORS, -1)
            pred = t2.vp_head(torch.cat([v, ifeat], dim=-1)).reshape(
                n, N_ANCHORS, MAX_OUTPUT_SLOTS, NBITS + 1)
        idx = (oslot == sel.unsqueeze(-1)).float().argmax(-1)
        ar = torch.arange(n, device=dev)
        p = pred[ar.unsqueeze(1), torch.arange(N_ANCHORS, device=dev).unsqueeze(0),
                 idx.unsqueeze(1)]
        hits += float(((p[..., :NBITS] > 0).float()
                       == _value_bits(tgt)).all(-1).float().sum())
        trip += n * N_ANCHORS
    return 100 * hits / trip, trip


rng = np.random.default_rng(0)
groups = defaultdict(lambda: ([], []))
for _ in range(N):
    rA, rB, rC, rX, rY = rng.choice(np.arange(1, 32), size=5, replace=False)
    i1 = int(rng.integers(-2048, 2048))
    i2 = int(rng.integers(-2048, 2048))
    rA, rB, rC, rX, rY = int(rA), int(rB), int(rC), int(rX), int(rY)
    ADD1 = Instruction('ADD', rB, rA, rX)
    ADDI1 = Instruction('ADDI', rB, rA, i1)
    SLT1 = Instruction('SLT', rB, rA, rX)
    for name, chunk in (
            ('R->R    ADD  ; ADD ', [ADD1, Instruction('ADD', rC, rB, rY)]),
            ('I->R    ADDI ; ADD ', [ADDI1, Instruction('ADD', rC, rB, rY)]),
            ('R->I    ADD  ; ADDI', [ADD1, Instruction('ADDI', rC, rB, i2)]),
            ('I->I    ADDI ; ADDI', [ADDI1, Instruction('ADDI', rC, rB, i2)]),
            ('cmp->I  SLT  ; ADDI', [SLT1, Instruction('ADDI', rC, rB, i2)]),
            ('I->cmp  ADDI ; SLT ', [ADDI1, Instruction('SLT', rC, rB, rY)])):
        groups[name][0].append(chunk)
        groups[name][1].append((rB, rC))

print()
print(f'checkpoint = {T2_DIR}')
print(f'  same depth-2 shape in every arm; only where the literals sit varies '
      f'  (n={N})')
print()
print('  Both of the chunk\'s output slots are scored. The INTERMEDIATE is the '
      'first')
print('  instruction\'s own slot, at depth 1; the COMPOSED slot is the second\'s,'
      ' at depth 2.')
print('  A literal intact in the intermediate and gone from the composed slot is'
      ' lost by')
print('  the second application, not by the first.')
print()
print(f'    {"arm":24s} {"triples":>9s} {"intermediate":>13s} '
      f'{"composed":>10s}')
for name in sorted(groups):
    chunks, regs = groups[name]
    mid, _ = score(chunks, [r[0] for r in regs])
    out, trip = score(chunks, [r[1] for r in regs])
    print(f'    {name:24s} {int(trip):9d} {mid:12.4f}% {out:9.4f}%')
print()
