#!/usr/bin/env python3
"""Does the essence depend on WHERE a chunk's write happens and on how many
instructions the chunk holds, rather than on what it computes?

The pool reads each written slot's write STEP, `(k+1)/K` with `K` the padded
instruction count of the grid (`RegisterStateMachine.forward`), so a chunk
whose one write sits at step 1 of 2 and the same write at step 2 of 2 hand the
pool different numbers for the same function. The identity family asks the
same question through content: `ADDI r,r,0`, `ADD r,r,x0`, `ADDI r,r,k; ADDI
r,r,-k` and the empty chunk all leave the register file unchanged.

A direct measurement: build the chunks, encode them, take the distance between
unit-normalized essences. No fitted readout. Every chunk is encoded in a
two-slot grid (the cap=2 grid the model trained in) unless the line says
otherwise; a chunk shorter than the grid is padded with inactive slots exactly
as the trainer pads it.

The no-op used to shift a write is `ADD x0,x0,x0`: every read and write of x0
is masked in the core, so the step contributes nothing to the state but its
position. The frozen T1 has never seen an x0-only instruction, so the wiring it
emits for one is reported, and the position row is measured both through T1's
emitted wiring (the pipeline) and through token wiring (the core alone).

Reported as `scripts/eval.py` reports collapse, and read the same way: a pair
computing one function MUST be ~0; `different`/`distinct` rows are the
denominator without which a small distance means nothing; `rename` is ~0 by
construction and asserts the encoder is intact. The emulator checks every
equivalence and inequivalence claimed.

Usage: write_position.py <t2_dir> <t1_path> [n_pairs]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import _t2_assemble, _t1_predicted_wiring
from compressor.model import instruction_wiring
from emulator import Instruction
from emulator import run as emu_run
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import encode_instruction, PAD

MAX_INSTR_TOKENS = 9      # tokenizer.MAX_INSTR_TOKENS is not exported; see
                          # compressor.train._MAX_INSTR_TOKENS
IMM_LO, IMM_HI = -2048, 2048
NOP = Instruction('ADD', 0, 0, 0)
# The writers-ISA opcodes, for the `distinct` denominator.
R_OPS = ['ADD', 'SUB', 'SLL', 'SLT', 'SLTU', 'XOR', 'SRL', 'SRA', 'OR', 'AND']
I_OPS = ['ADDI', 'SLTI', 'SLTIU', 'XORI', 'ORI', 'ANDI']
SH_OPS = ['SLLI', 'SRLI', 'SRAI']

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
rng = np.random.default_rng(0)


def build(chunks):
    """[[Instruction, ...], ...] -> the split tuple `_t2_assemble` consumes."""
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


def encode(chunks, mni=2, route='binding', bs=512):
    """Unit-normalized essences of `chunks`, each encoded in an `mni`-slot grid."""
    out = []
    for i in range(0, len(chunks), bs):
        sub = chunks[i:i + bs]
        split = build(sub)
        with torch.no_grad():
            if split[0].shape[0] == 0:          # every chunk empty: no T1 work
                z = torch.zeros(len(sub), mni, dtype=torch.long, device=dev)
                ct1 = torch.zeros(len(sub), mni, t1.d_out, device=dev)
                _, ess = t2.core(ct1, z, z, z, z.float())
            else:
                ct1, c0, c1, co, cact = _t2_assemble(
                    t1, split, len(sub), mni, dev, route=route)
                _, ess = t2.core(ct1, c0, c1, co, cact)
        out.append(F.normalize(ess.float(), dim=-1))
    return torch.cat(out)


def dist(a, b):
    return (a - b).norm(dim=-1)


def check(a, b, want_equal, every=50):
    """Assert through the emulator that chunk pairs are (in)equivalent on the
    register file, on a sample of the pairs at random states."""
    for k in range(0, len(a), max(1, len(a) // every)):
        st = np.zeros(32, dtype=np.int32)
        st[1:] = rng.integers(-(2 ** 31), 2 ** 31 - 1, 31).astype(np.int32)
        ra = emu_run(a[k], st.copy())[0].regs if a[k] else st
        rb = emu_run(b[k], st.copy())[0].regs if b[k] else st
        assert np.array_equal(ra, rb) == want_equal, (a[k], b[k], want_equal)


def row(name, d, note):
    print(f'    {name:<28} mean={float(d.mean()):.4f}  '
          f'median={float(d.median()):.4f}  p95={float(d.quantile(0.95)):.4f}'
          f'   {note}')


def rand_instr():
    kind = rng.integers(3)
    rd, rs1, rs2 = (int(x) for x in rng.integers(1, 32, 3))
    if kind == 0:
        return Instruction(R_OPS[rng.integers(len(R_OPS))], rd, rs1, rs2)
    if kind == 1:
        return Instruction(I_OPS[rng.integers(len(I_OPS))], rd, rs1,
                           int(rng.integers(IMM_LO, IMM_HI)))
    return Instruction(SH_OPS[rng.integers(len(SH_OPS))], rd, rs1,
                       int(rng.integers(0, 32)))


# ---- what T1 emits for the x0 no-op -----------------------------------------
split = build([[NOP]])
with torch.no_grad():
    T1_state, _ = t1.encode_state(
        *(torch.from_numpy(x).to(dev) for x in
          (split[0], split[1], split[5], split[6], split[7])))
    w = tuple(int(x) for x in _t1_predicted_wiring(t1, T1_state))
print(f'\n  T1 emitted wiring for `{NOP}` (in0, in1, out): {w}'
      f'   {"(x0 everywhere: the step is inert)" if w == (0, 0, 0) else "(NOT inert — a phantom read or write)"}')

# ---- A. write position ------------------------------------------------------
first, second, single, diff, ren = [], [], [], [], []
for _ in range(N):
    rd, rs = int(rng.integers(1, 32)), int(rng.integers(1, 32))
    perm = np.concatenate([[0], rng.permutation(31) + 1])   # a rename keeps x0
    rd2, rs2 = int(perm[rd]), int(perm[rs])                  # and aliasing
    imm = int(rng.integers(IMM_LO, IMM_HI - 1))
    first.append([Instruction('ADDI', rd, rs, imm), NOP])
    second.append([NOP, Instruction('ADDI', rd, rs, imm)])
    single.append([Instruction('ADDI', rd, rs, imm)])
    diff.append([Instruction('ADDI', rd, rs, imm + 1), NOP])
    ren.append([Instruction('ADDI', rd2, rs2, imm), NOP])
check(first, second, True); check(first, single, True); check(first, diff, False)

print(f'\n  A. one `ADDI rd,rs,imm` and one x0 no-op; the write moves between '
      f'step 1 and step 2 of a two-slot grid   (pairs={N})')
for route in ('binding', 'tokens'):
    Fi, Se, Si, Di, Re = (encode(c, 2, route)
                          for c in (first, second, single, diff, ren))
    print(f'   route={route}:')
    row('position', dist(Fi, Se), 'write at step 1 vs step 2 — same function, MUST be ~0')
    row('nop vs pad', dist(Fi, Si), 'ADDI then no-op vs ADDI then inactive slot — the no-op step must be inert, so ~0 exactly')
    row('rename', dist(Fi, Re), '~0 by construction')
    row('different', dist(Fi, Di), 'imm+1, same position — the denominator')
    print(f'    separation different/position = '
          f'{float(dist(Fi, Di).mean()) / max(float(dist(Fi, Se).mean()), 1e-9):.1f}x')
S1 = encode(single, 1, 'binding')
S2 = encode(single, 2, 'binding')
row('grid 1 vs grid 2', dist(S1, S2),
    'the same single instruction, alone vs padded to two slots — padding dependence')

# ---- B. the identity family -------------------------------------------------
E, I0, I0b, I1, I2, N1, N2, X, Y = ([] for _ in range(9))
for _ in range(N):
    r = int(rng.integers(1, 32))
    k = int(rng.integers(1, IMM_HI - 1))           # k and -k and -k+1 all in range
    E.append([])
    I0.append([Instruction('ADDI', r, r, 0)])
    I0b.append([NOP, Instruction('ADDI', r, r, 0)])
    I1.append([Instruction('ADD', r, r, 0)])
    I2.append([Instruction('ADDI', r, r, k), Instruction('ADDI', r, r, -k)])
    N1.append([Instruction('ADDI', r, r, 1)])
    N2.append([Instruction('ADDI', r, r, k), Instruction('ADDI', r, r, -k + 1)])
    X.append([rand_instr(), rand_instr()])
    Y.append([rand_instr(), rand_instr()])
for a in (I0, I0b, I1, I2):
    check(E, a, True)
check(I0, N1, False); check(I2, N2, False)

e, i0, i0b, i1, i2, n1, n2, x, y = (encode(c) for c in (E, I0, I0b, I1, I2, N1, N2, X, Y))
print(f'\n  B. the identity on one register, five ways, in a two-slot grid   (pairs={N})')
row('ADDI r,r,0 vs ADD r,r,x0', dist(i0, i1), 'no-op exchange, both at step 1 — MUST be ~0')
row('ADDI r,r,0 vs +k;-k', dist(i0, i2), 'identity vs cancelling pair — write at step 1 vs 2')
row('nop;ADDI r,r,0 vs +k;-k', dist(i0b, i2), 'the same with the write at step 2 in both')
row('ADDI r,r,0 vs empty', dist(i0, e), 'identity vs no instruction at all')
row('+k;-k vs empty', dist(i2, e), '')
row('ADDI r,r,0 vs ADDI r,r,1', dist(i0, n1), 'NOT equivalent — denominator at one instruction')
row('+k;-k vs +k;-k+1', dist(i2, n2), 'NOT equivalent — denominator at two')
row('distinct', dist(x, y), 'unrelated two-instruction writers chunks')
print(f'    empty-chunk essence is one point: spread '
      f'{float((e - e.mean(0, keepdim=True)).norm(dim=-1).max()):.2e}')
print()
