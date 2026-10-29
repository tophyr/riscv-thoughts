"""The structural signature agrees with the emulator on which chunks compute
one register function.

Equivalence-preserving transformations must leave the signature unchanged on
every pair; on every control, two chunks with equal signatures must be equal
on the emulator (the signature never merges what the emulator separates — the
converse is not asserted, since identities, cancellations and immediate folds
are equal on the emulator and legitimately distinct in the signature). Equality
on the emulator is taken over the structured witness draw training uses, not
uniform values, which do not witness a comparison near its immediate.
"""
import itertools
import math

import numpy as np
import random

import pytest

from emulator import Instruction, run as emu_run
from datagen.compare import draw_anchor_states
from datagen.signature import COMMUTATIVE, dataflow, live_nodes, signature, output_inputs

R_OPS = ['ADD', 'SUB', 'SLL', 'SLT', 'SLTU', 'XOR', 'SRL', 'SRA', 'OR', 'AND']
I_OPS = ['ADDI', 'SLTI', 'SLTIU', 'XORI', 'ORI', 'ANDI']
SH_OPS = ['SLLI', 'SRLI', 'SRAI']
WITNESS = draw_anchor_states(np.random.default_rng(7), 64)
N = 300


def rand_instr(rng, rd=None):
    kind = rng.integers(3)
    rd = int(rng.integers(1, 32)) if rd is None else rd
    rs1, rs2 = (int(x) for x in rng.integers(1, 32, 2))
    if kind == 0:
        return Instruction(R_OPS[rng.integers(len(R_OPS))], rd, rs1, rs2)
    if kind == 1:
        return Instruction(I_OPS[rng.integers(len(I_OPS))], rd, rs1, int(rng.integers(-2048, 2048)))
    return Instruction(SH_OPS[rng.integers(len(SH_OPS))], rd, rs1, int(rng.integers(0, 32)))


def reads(ins):
    return [r for r in ([ins.args[1], ins.args[2]] if ins.opcode in R_OPS else [ins.args[1]]) if r]


def regs_equal(a, b, witnesses=None):
    for st in (WITNESS if witnesses is None else witnesses):
        st = st.astype(np.int32)
        if not np.array_equal(emu_run(a, st.copy())[0].regs, emu_run(b, st.copy())[0].regs):
            return False
    return True


def _relabel(chunk, m):
    out = []
    for ins in chunk:
        a = [m.get(x, x) for x in ins.args[:2]] + ([m.get(ins.args[2], ins.args[2])] if ins.opcode in R_OPS else [ins.args[2]])
        out.append(Instruction(ins.opcode, *a))
    return out


def rename_equivalent(a, b):
    """Is `b` a register rename of `a`? A rename changes the register file
    while leaving the essence, so the emulator alone cannot judge a
    signature-equal pair; this searches the renames that respect each
    register's role (written only / read only / both) and checks each on the
    emulator, cheap witnesses first."""
    def groups(chunk):
        w = {i.args[0] for i in chunk if i.args[0]}
        r = {x for i in chunk for x in reads(i)}
        return [sorted(w - r), sorted(r - w), sorted(w & r)]
    ga, gb = groups(a), groups(b)
    if [len(g) for g in ga] != [len(g) for g in gb]:
        return False
    if math.prod(math.factorial(len(g)) for g in gb) > 50_000:
        raise RuntimeError('too many renames to search')
    for perms in itertools.product(*(itertools.permutations(g) for g in gb)):
        m = {x: y for g_a, g_b in zip(ga, perms) for x, y in zip(g_a, g_b)}
        c = _relabel(a, m)
        if regs_equal(c, b, WITNESS[:4]) and regs_equal(c, b):
            return True
    return False


# ---- transformations that preserve the register function -------------------
def rename(chunk, rng):
    perm = np.concatenate([[0], rng.permutation(31) + 1])
    out = []
    for ins in chunk:
        a = list(ins.args)
        a[0], a[1] = int(perm[a[0]]), int(perm[a[1]])
        if ins.opcode in R_OPS:
            a[2] = int(perm[a[2]])
        out.append(Instruction(ins.opcode, *a))
    return out


def reorder(chunk, rng):
    for i in range(len(chunk) - 1):
        i1, i2 = chunk[i], chunk[i + 1]
        if (i1.args[0] != i2.args[0] and i1.args[0] not in reads(i2)
                and i2.args[0] not in reads(i1)):
            c = list(chunk); c[i], c[i + 1] = c[i + 1], c[i]
            return c
    return None


def comm_swap(chunk, rng):
    idx = [i for i, ins in enumerate(chunk) if ins.opcode in COMMUTATIVE and ins.args[1] != ins.args[2]]
    if not idx:
        return None
    i = int(rng.choice(idx)); c = list(chunk); a = c[i].args
    c[i] = Instruction(c[i].opcode, a[0], a[2], a[1])
    return c


def dead_insert(chunk, rng):
    """Prepend an instruction writing a register the chunk writes later and
    never reads before that write."""
    for j, ins in enumerate(chunk):
        rd = ins.args[0]
        if all(rd not in reads(k) for k in chunk[:j + 1]):
            return [rand_instr(rng, rd=rd)] + list(chunk)
    return None


# ---- controls that change the structure --------------------------------------
def ctrl_alias(chunk, rng):
    """Make an R-type op read one register twice, or stop it doing so."""
    idx = [i for i, ins in enumerate(chunk) if ins.opcode in R_OPS]
    if not idx:
        return None
    i = int(rng.choice(idx)); a = list(chunk[i].args)
    a[2] = int(rng.choice([r for r in range(1, 32) if r != a[1]])) if a[1] == a[2] else a[1]
    c = list(chunk); c[i] = Instruction(c[i].opcode, *a); return c


def ctrl_share(chunk, rng):
    """Make two LIVE instructions read a common register, or stop them doing
    so. Sharing with a dead instruction is a rename of a leaf, not a control."""
    edges, outputs = dataflow(chunk)
    live_ops = sorted(n[1] for n in live_nodes(edges, outputs) if n[0] == 'op')
    if len(live_ops) < 2:
        return None
    i, j = sorted(rng.choice(live_ops, 2, replace=False))
    ri = reads(chunk[i]); a = list(chunk[j].args)
    others = [r for r in range(1, 32) if r not in ri and r not in [k.args[0] for k in chunk]]
    a[1] = int(ri[0]) if a[1] not in ri else int(rng.choice(others))
    c = list(chunk); c[j] = Instruction(c[j].opcode, *a); return c


def ctrl_op(chunk, rng):
    i = int(rng.integers(len(chunk))); ins = chunk[i]
    pool = R_OPS if ins.opcode in R_OPS else (SH_OPS if ins.opcode in SH_OPS else I_OPS)
    c = list(chunk); c[i] = Instruction(str(rng.choice([o for o in pool if o != ins.opcode])), *ins.args)
    return c


def ctrl_imm(chunk, rng):
    idx = [i for i, ins in enumerate(chunk) if ins.opcode not in R_OPS]
    if not idx:
        return None
    i = int(rng.choice(idx)); a = list(chunk[i].args)
    a[2] = a[2] + 1 if a[2] < (31 if chunk[i].opcode in SH_OPS else 2047) else a[2] - 1
    c = list(chunk); c[i] = Instruction(c[i].opcode, *a); return c


def pairs(fn, length, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < N:
        base = [rand_instr(rng) for _ in range(length)]
        v = fn(base, rng)
        if v is not None:
            out.append((base, v))
    return out


@pytest.mark.parametrize('length', [2, 3])
@pytest.mark.parametrize('fn', [rename, reorder, comm_swap, dead_insert])
def test_equivalence_preserving_transformations_keep_the_signature(fn, length):
    for base, v in pairs(fn, length, seed=length):
        assert signature(base) == signature(v), (base, v)
        if fn is not rename:                     # a rename changes the register file, not the essence
            assert regs_equal(base, v), (base, v)


@pytest.mark.parametrize('length', [2, 3])
@pytest.mark.parametrize('fn', [ctrl_alias, ctrl_share, ctrl_op, ctrl_imm])
def test_the_signature_never_merges_what_the_emulator_separates(fn, length):
    """Signature-equal implies one register function — directly, or after a
    rename, since a control that moves a shared read from one live
    instruction to another can land on an isomorphic graph."""
    for base, v in pairs(fn, length, seed=10 + length):
        if signature(base) == signature(v):
            assert regs_equal(base, v) or rename_equivalent(base, v), (base, v)


def test_a_sharing_change_that_keeps_the_signature_is_a_rename():
    a = [Instruction('XOR', 31, 3, 2), Instruction('SLLI', 9, 25, 22), Instruction('XOR', 22, 28, 25)]
    b = [Instruction('XOR', 31, 3, 2), Instruction('SLLI', 9, 3, 22), Instruction('XOR', 22, 28, 25)]
    assert signature(a) == signature(b)
    assert not regs_equal(a, b)
    assert rename_equivalent(a, b)


def test_the_commutative_role_rule_is_load_bearing():
    """Colouring with every role ordered separates a swap whenever the swapped
    registers are read elsewhere; the encoder therefore needs the lower tier
    to say which operations are symmetric."""
    separated = 0
    for base, v in pairs(comm_swap, 2, seed=30):
        assert signature(base) == signature(v)
        separated += signature(base, symmetric_ops=frozenset()) != signature(v, symmetric_ops=frozenset())
    assert separated > 0


def test_dead_instructions_leave_no_trace():
    chunk = [Instruction('ADD', 5, 6, 7), Instruction('ADDI', 5, 8, 1)]
    assert signature(chunk) == signature([Instruction('ADDI', 5, 8, 1)])
    assert output_inputs(chunk) == {5: [8]}


def test_output_inputs_follow_the_graph_not_the_text():
    a = [Instruction('ADD', 5, 1, 2), Instruction('SUB', 6, 3, 4)]
    b = [Instruction('SUB', 6, 3, 4), Instruction('ADD', 5, 1, 2)]
    assert output_inputs(a) == output_inputs(b)
    assert {r: sorted(v) for r, v in output_inputs(a).items()} == {5: [1, 2], 6: [3, 4]}
    # `SUB x6,x4,x3` is `SUB x6,x3,x4` renamed: same signature, and the
    # minuend lands in the same gather position either way, because the order
    # is by role colour rather than by register number.
    p, q = [Instruction('SUB', 6, 4, 3)], [Instruction('SUB', 6, 3, 4)]
    assert signature(p) == signature(q)
    assert output_inputs(p)[6].index(4) == output_inputs(q)[6].index(3)


def test_a_rename_never_moves_an_input_between_gather_positions():
    """out_inputs of a renamed chunk is the original's under the rename, for
    every output and position — commutative operands included, whose order
    would otherwise follow register numbers."""
    rng = random.Random(41)
    ops = ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'SLT', 'SLL']
    for _ in range(500):
        chunk = [Instruction(rng.choice(ops), rng.randrange(1, 32), rng.randrange(1, 32),
                             rng.randrange(1, 32)) for _ in range(2)]
        perm = [0] + rng.sample(range(1, 32), 31)
        renamed = [Instruction(i.opcode, *(perm[r] for r in i.args)) for i in chunk]
        want = {perm[r]: [perm[x] for x in v] for r, v in output_inputs(chunk).items()}
        assert output_inputs(renamed) == want, (chunk, renamed)


def test_memory_and_control_flow_are_out_of_scope():
    with pytest.raises(NotImplementedError):
        signature([Instruction('LW', 5, 0, 6)])
