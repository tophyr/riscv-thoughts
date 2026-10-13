#!/usr/bin/env python3
"""Which PART of a composed slot's description is recoverable, and at which
stage of the encoder?

The encoder never sees a register VALUE. Its inputs are each instruction's
opcode+immediate (through the frozen T1's essence) and the wiring; the anchor
values enter nowhere but the vp head's features. So a register slot holds a
symbolic description of the function that produced it, and a composed slot's
description has to compose two of them. The parts of such a description are
discrete, so they can be read back one at a time and located.

On `ADDI rB,rA,i1 ; ADDI rC,rB,i2`, three quantities at three stages:

  i1          the producer's literal, which reaches the composed slot only by
              being read back through the operand slot
  i2          the writer's own literal, which arrives in that slot's content
  i1+i2       the composite, which appears in NO instruction and exists only
              if the encoder computes it

  stages:  producer slot -> composed slot -> essence

How to read it. Each number is a fitted readout against a 1/4096 base rate for
a 12-bit literal, so a stage scoring near chance holds nothing of that
quantity. i2's column calibrates a stage — it says whether a literal is legible
there at all. A quantity high at one stage and low at the next locates where it
is lost. And the composite's column against its two summands says whether the
encoder folded them or merely carried both: summands high with the composite
low means it carried both separately.

An R-type arm (`ADD rB,rA,rX ; ADDI rC,rB,i2`) is scored alongside as the
control whose producer carries no literal at all.

Readout capacity is a confound: report the LINEAR and nonlinear columns
together, since a quantity only a nonlinear map recovers is present but
entangled, which is a different fact from absent.

Usage: compose_readback.py <t2_dir> <t1_path> [n_chunks] [fit_steps]
"""
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.model import instruction_wiring
from compressor.train import _t2_assemble, _value_bits
from datagen import MAX_INPUT_SLOTS, N_REGS
from datagen.compare import make_anchor_states
from emulator import Instruction
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import encode_instruction, PAD

MAX_INSTR_TOKENS = 9
N_ANCHORS, ANCHOR_SEED = 8, 0
NB_TARGET = 14          # bits of the recovered quantity (covers i1+i2 signed)
HID, MB = 1024, 512

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24000
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 12000

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
anchors_np = np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))


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


@torch.no_grad()
def encode(chunks, prod_regs, comp_regs, bs=256):
    """-> dict of stage name -> (n, d) features, all detached."""
    out = defaultdict(list)
    for i in range(0, len(chunks), bs):
        sub = chunks[i:i + bs]
        n = len(sub)
        split = build(sub)
        ct1, c0, c1, co, cact = _t2_assemble(
            t1, split, n, max(len(x) for x in sub), dev, route='binding')
        T, ess = t2.encode_state(ct1, c0, c1, co, cact)
        pr = torch.tensor(prod_regs[i:i + bs], device=dev)
        cr = torch.tensor(comp_regs[i:i + bs], device=dev)
        ar = torch.arange(n, device=dev)
        # value channels only: the dataflow state the op-cell threads.
        out['producer slot'].append(T[ar, pr, :t2.d_out].float().cpu())
        out['composed slot'].append(T[ar, cr, :t2.d_out].float().cpu())
        out['essence'].append(ess.float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def fit_recover(feats, target, steps=STEPS, seed=0):
    """Fit an MLP to recover `target` (int) as NB_TARGET bits; return exact-match
    % on a held-out fifth. One architecture and budget for every (stage,
    quantity) cell, so the cells are comparable to each other."""
    torch.manual_seed(seed)
    n = feats.shape[0]
    ntr = int(n * 0.8)
    X = feats.to(dev)
    Y = _value_bits(torch.as_tensor(target, device=dev))[..., :NB_TARGET]
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    net = nn.Sequential(nn.Linear(X.shape[1], HID), nn.ReLU(),
                        nn.Linear(HID, HID), nn.ReLU(),
                        nn.Linear(HID, NB_TARGET)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, ntr, (MB,), generator=g).to(dev)
        loss = F.binary_cross_entropy_with_logits(net(Xtr[idx]), Ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        pred = (net(Xte) > 0).float()
    return 100.0 * float((pred == Yte).all(-1).float().mean())


rng = np.random.default_rng(0)
ARMS = ('I->I  ADDI ; ADDI', 'R->I  ADD  ; ADDI')
chunks = {a: [] for a in ARMS}
prod = {a: [] for a in ARMS}
comp = {a: [] for a in ARMS}
i1s, i2s = [], []
for _ in range(N):
    rA, rB, rC, rX = (int(v) for v in
                      rng.choice(np.arange(1, 32), size=4, replace=False))
    i1 = int(rng.integers(-2048, 2048))
    i2 = int(rng.integers(-2048, 2048))
    i1s.append(i1); i2s.append(i2)
    chunks['I->I  ADDI ; ADDI'].append(
        [Instruction('ADDI', rB, rA, i1), Instruction('ADDI', rC, rB, i2)])
    chunks['R->I  ADD  ; ADDI'].append(
        [Instruction('ADD', rB, rA, rX), Instruction('ADDI', rC, rB, i2)])
    for a in ARMS:
        prod[a].append(rB); comp[a].append(rC)

i1s = np.array(i1s); i2s = np.array(i2s)
QUANTS = (('i1  (producer literal)', i1s),
          ('i2  (writer literal)', i2s),
          ('i1+i2  (the composite)', i1s + i2s))

print()
print(f'checkpoint = {T2_DIR}')
print(f'  n={N} chunks per arm, readout fit {STEPS} steps, held-out fifth, '
      f'exact match of {NB_TARGET} bits')
print(f'  base rate for a 12-bit literal = {100/4096:.4f}%')
for arm in ARMS:
    feats = encode(chunks[arm], prod[arm], comp[arm])
    print()
    print(f'  {arm}')
    print(f'    {"quantity":26s} ' +
          ' '.join(f'{s:>15s}' for s in feats))
    for qname, qval in QUANTS:
        if arm.startswith('R->I') and qname.startswith('i1'):
            continue                      # no literal in an R-type producer
        row = [fit_recover(feats[s], qval) for s in feats]
        print(f'    {qname:26s} ' + ' '.join(f'{v:14.4f}%' for v in row))
print()
