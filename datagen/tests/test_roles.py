"""Tests for datagen.roles — canonical input roles.

Two equivalent chunks must ship identical witness outputs, whatever their
spelling or dataflow shape; each pair is checked equivalent on the emulator
first."""

import numpy as np
import pytest

from emulator import Instruction as I, run as emu_run
from datagen.batch import generate_chunks
from datagen.compare import draw_anchor_states, precompute_chunk
from datagen.generate import build_opcode_table, length_cap, random_perm, relabel

_COMM = {'ADD', 'AND', 'OR', 'XOR'}


def _equivalent(a, b, perm, rng):
    """Same register function at 200 structured states, b's registers being
    perm(a's)."""
    for s in draw_anchor_states(rng, 200):
        ra = emu_run(a, s.copy())[0].regs
        sb = s.copy() if perm is None else s[np.argsort(perm)].astype(np.int32)
        rb = emu_run(b, sb)[0].regs
        if perm is not None:
            rb = rb[perm]
        if not np.array_equal(ra, rb):
            return False
    return True


def _witness(chunk):
    pre = precompute_chunk(chunk, draw_anchor_states(np.random.default_rng(0), 2))
    n = int((pre.witness_slots >= 0).sum())
    return pre.witness[:n]


def _pairs(kind, rng, n):
    alu = build_opcode_table({'R_ALU': 0.55, 'I_ALU': 0.45})
    src = generate_chunks(length_cap(4), rng, opcode_table=alu, chain_rate=0.5)
    regs = lambda k: [int(x) for x in rng.choice(np.arange(1, 32), k, replace=False)]
    out = []
    while len(out) < n:
        t, a, b, c = regs(4)
        if kind == 'shape':
            out.append(([I('SUB', t, a, b), I('ADD', t, t, c)],
                        [I('SUB', t, c, b), I('ADD', t, t, a)], None))
        elif kind == 'difference':
            out.append(([I('SUB', t, a, b), I('SUB', t, t, c)],
                        [I('ADD', t, b, c), I('SUB', t, a, t)], None))
        elif kind == 'ignored':
            out.append(([I('ADD', t, a, b), I('SUB', t, t, a), I('ADD', c, c, b)],
                        [I('ADDI', t, b, 0), I('ADD', c, c, b)], None))
        elif kind == 'swap':
            ch = next(src).instructions
            if any(i.opcode in _COMM and i.args[1] != i.args[2] for i in ch):
                out.append((ch, [I(i.opcode, i.args[0], i.args[2], i.args[1])
                                 if i.opcode in _COMM else i for i in ch], None))
        elif kind == 'rename':
            ch = next(src).instructions
            p = random_perm(rng)
            out.append((ch, relabel(ch, p), p))
    return out


@pytest.mark.parametrize('kind', ['shape', 'difference', 'ignored', 'swap', 'rename'])
def test_equivalent_chunks_ship_identical_witness_outputs(kind):
    """Including the classes a colour-order placement gets wrong: a
    commutative swap whose operand another instruction also reads, and two
    dataflow shapes of one function."""
    rng = np.random.default_rng(3)
    for a, b, perm in _pairs(kind, rng, 60):
        assert _equivalent(a, b, perm, rng), (a, b)
        assert np.array_equal(_witness(a), _witness(b)), (a, b)
