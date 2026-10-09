#!/usr/bin/env python3
"""Do two additive chains with the same total offset land on the same point?

`ADDI r1,r1,256; ADDI r1,r1,-16` and `ADDI r1,r1,32; ADDI r1,r1,208` compute
exactly the same function of r1 — the split between the two immediates is not a
property of the function, only their sum is. A value-numbering encoder must map
them to the same essence.

This is a direct measurement: build the pairs, encode them, take the distance.
No fitted readout, so nothing here depends on a probe's capacity or on how many
rows a group got — the failure mode that made every earlier attempt at this
question disagree with itself.

Both instructions write the SAME register, so the final state holds one slot
carrying `rs1 + (imm1+imm2)` and nothing else; the intermediate is overwritten
and the two chunks differ in no other way.

Reported as `scripts/eval.py` reports GVN collapse, and read the same way:

  fusion    — same total offset, different split. MUST be ~0.
  rename    — the same chunk with its registers relabeled. ~0 by construction;
              an assertion that the encoder is intact, not a result.
  different — chains whose totals differ by one. The separation denominator, and
              the number that says whether a small `fusion` distance means
              anything: an encoder that collapsed everything would score 0 on
              both.

Usage: additive_fusion.py <t2_dir> <t1_path> [n_pairs]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import _t2_assemble
from compressor.model import instruction_wiring
from emulator import Instruction
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from emulator import run as emu_run
from tokenizer import decode_instruction, encode_instruction, PAD

MAX_INSTR_TOKENS = 9      # tokenizer.MAX_INSTR_TOKENS is not exported; see
                          # compressor.train._MAX_INSTR_TOKENS

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
N_PAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
IMM_LO, IMM_HI = -2048, 2048          # the generator's 12-bit signed range

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)


def build(chunks):
    """[[Instruction, ...], ...] -> the split tuple `_t2_assemble` consumes."""
    toks, pads, ci, si, w0, w1, wo = [], [], [], [], [], [], []
    for c, instrs in enumerate(chunks):
        for s, ins in enumerate(instrs):
            t = encode_instruction(ins)
            assert len(t) <= MAX_INSTR_TOKENS, (ins, len(t))
            toks.append(t + [PAD] * (MAX_INSTR_TOKENS - len(t)))
            pads.append([False] * len(t) + [True] * (MAX_INSTR_TOKENS - len(t)))
            a, b, o = instruction_wiring(ins)
            ci.append(c); si.append(s); w0.append(a); w1.append(b); wo.append(o)
    return (np.array(toks, dtype=np.int64), np.array(pads, dtype=bool),
            np.array(ci), np.array(si),
            np.array([len(x) for x in chunks]),
            np.array(w0), np.array(w1), np.array(wo))


def encode(chunks, bs=512):
    """Chunk essences, unit-normalized (vp reads direction only).

    The encode is deterministic, so a distance between two sets reflects only
    the chunks, never a resampling artifact."""
    out = []
    for i in range(0, len(chunks), bs):
        sub = chunks[i:i + bs]
        split = build(sub)
        mni = max(len(x) for x in sub)
        with torch.no_grad():
            ct1, c0, c1, co, cact = _t2_assemble(
                t1, split, len(sub), mni, dev, route='binding')
            _, ess = t2.core(ct1, c0, c1, co, cact)
        out.append(F.normalize(ess.float(), dim=-1))
    return torch.cat(out)


def sext12(v):
    """Interpret the low 12 bits of `v` as a signed immediate."""
    v &= 0xFFF
    return v - 4096 if v & 0x800 else v


# Fusible families: an op that is associative in its immediate, so a two-step
# chain has ONE parameter and the split between the immediates is not a property
# of the function. `resplit(i1, i2, r)` returns a different pair with the same
# composite; `perturb(i1, i2)` returns a pair with a different one.
FAMILIES = {
    'ADDI': dict(
        combine=lambda a, b: a + b,
        resplit=lambda i1, i2, r: (lambda t, j: (j, t - j))(
            i1 + i2, int(r.integers(max(IMM_LO, i1 + i2 - IMM_HI + 1),
                                    min(IMM_HI - 1, i1 + i2 - IMM_LO) + 1))),
        perturb=lambda i1, i2: (i1, i2 + 1)),
    'XORI': dict(
        combine=lambda a, b: a ^ b,
        resplit=lambda i1, i2, r: (lambda j: (j, sext12(j ^ i1 ^ i2)))(
            int(r.integers(IMM_LO, IMM_HI))),
        perturb=lambda i1, i2: (i1, sext12(i2 ^ 1))),
}


def chain(op, rd, rs, i1, i2):
    """`op rd,rs,i1; op rd,rd,i2` — one slot, holding rs combined with both
    immediates. Writing the same register twice leaves the intermediate
    overwritten, so the essence describes only the composite."""
    return [Instruction(op, rd, rs, i1), Instruction(op, rd, rd, i2)]


rng = np.random.default_rng(0)
for OP, fam in FAMILIES.items():
    base, same, ren, diff = [], [], [], []
    for _ in range(N_PAIRS):
        rd, rs = int(rng.integers(1, 32)), int(rng.integers(1, 32))
        rd2, rs2 = int(rng.integers(1, 32)), int(rng.integers(1, 32))
        i1 = int(rng.integers(IMM_LO, IMM_HI))
        i2 = int(rng.integers(IMM_LO, IMM_HI))
        try:
            j1, j2 = fam['resplit'](i1, i2, rng)
        except ValueError:
            continue
        if not (IMM_LO <= j1 < IMM_HI and IMM_LO <= j2 < IMM_HI):
            continue
        k1, k2 = fam['perturb'](i1, i2)
        if not (IMM_LO <= k1 < IMM_HI and IMM_LO <= k2 < IMM_HI):
            continue
        base.append(chain(OP, rd, rs, i1, i2))
        same.append(chain(OP, rd, rs, j1, j2))
        ren.append(chain(OP, rd2, rs2, i1, i2))
        diff.append(chain(OP, rd, rs, k1, k2))

    # Hand-built chunks, so check both things that could quietly be false: that
    # the tokens round-trip, and that a "same composite" pair really is
    # behaviorally identical while a "different" one is not.
    for a, b, want_equal in ((base, same, True), (base, diff, False)):
        for k in range(0, len(a), max(1, len(a) // 50)):
            rows = build([a[k]])[0]
            got = [decode_instruction(rows[j].tolist(), 0)[0]
                   for j in range(rows.shape[0])]
            assert [str(x) for x in got] == [str(x) for x in a[k]], (got, a[k])
            st = np.zeros(32, dtype=np.int32)
            st[1:] = rng.integers(-(2 ** 20), 2 ** 20, 31).astype(np.int32)
            ra = emu_run(a[k], st.copy())[0].regs
            rb = emu_run(b[k], st.copy())[0].regs
            assert np.array_equal(ra, rb) == want_equal, (a[k], b[k], want_equal)

    B, S, R, D = encode(base), encode(same), encode(ren), encode(diff)
    d_same = (B - S).norm(dim=-1)
    d_ren = (B - R).norm(dim=-1)
    d_diff = (B - D).norm(dim=-1)

    print()
    print(f'  {OP}/{OP} chains writing one register — the essence describes '
          f'only the composite   (pairs={len(base)})')
    for name, d, note in (
            ('fusion   ', d_same, 'same composite, different split — MUST be ~0'),
            ('rename   ', d_ren, 'relabelled registers — ~0 by construction'),
            ('different', d_diff, 'a different composite — the denominator')):
        print(f'    {name} mean={float(d.mean()):.4f}  '
              f'median={float(d.median()):.4f}  '
              f'p95={float(d.quantile(0.95)):.4f}   ({note})')
    print(f'    separation different/fusion = '
          f'{float(d_diff.mean()) / max(float(d_same.mean()), 1e-9):.1f}x')
print()
