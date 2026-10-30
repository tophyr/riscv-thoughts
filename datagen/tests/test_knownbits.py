"""Known-bits analysis against the emulator: every output it describes obeys
its known bits, every observed distance is within `max_distance`, and the
cases it must be exact on are."""
import random

import numpy as np

from datagen.compare import draw_anchor_states
from datagen.knownbits import (MASK, KnownBits, max_distance,
                               output_distance, output_known_bits,
                               wrap_distance)
from emulator import Instruction, run as emu_run

R_OPS = ['ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU']
I_OPS = ['ADDI', 'XORI', 'ORI', 'ANDI', 'SLTI', 'SLTIU']
SH_OPS = ['SLLI', 'SRLI', 'SRAI']


def _states(rng, n):
    s = draw_anchor_states(np.random.default_rng(rng.randrange(1 << 30)), n).astype(np.int64)
    edge = np.array([0, 1, -1, 2 ** 31 - 1, -2 ** 31, 0x55555555, -0x55555556, 7],
                    dtype=np.int64)
    s = np.concatenate([s, np.tile(edge[:, None], (1, 32))])
    s[:, 0] = 0
    return s


def _instr(rng, regs):
    k = rng.randrange(3)
    rd, a, b = (rng.choice(regs) for _ in range(3))
    if k == 0:
        return Instruction(rng.choice(R_OPS), rd, a, b)
    if k == 1:
        return Instruction(rng.choice(I_OPS), rd, a, rng.choice([0, 1, -1, 15, 255, -2048, 2047,
                                                                   rng.randrange(-2048, 2048)]))
    return Instruction(rng.choice(SH_OPS), rd, a, rng.randrange(32))


def _outputs(chunk, states):
    return {s_i: emu_run(chunk, st.astype(np.int32).copy())[0].regs for s_i, st in enumerate(states)}


def test_every_output_obeys_its_known_bits():
    rng = random.Random(3)
    for _ in range(400):
        regs = [0, 1, 2, 3, 4]                 # few registers, so aliasing and chains happen
        chunk = [_instr(rng, regs) for _ in range(rng.randint(1, 3))]
        kb = output_known_bits(chunk)
        for st in _states(rng, 24):
            regs_out = emu_run(chunk, st.astype(np.int32).copy())[0].regs
            for r, k in kb.items():
                v = int(regs_out[r]) & MASK
                assert v & k.zero == 0 and v & k.one == k.one, (chunk, r, hex(v), k)


def test_every_observed_distance_is_within_max_distance():
    rng = random.Random(4)
    for _ in range(300):
        a = [_instr(rng, [0, 1, 2, 3]) for _ in range(rng.randint(1, 2))]
        b = [_instr(rng, [0, 1, 2, 3]) for _ in range(rng.randint(1, 2))]
        ka, kb = output_known_bits(a), output_known_bits(b)
        if not ka or not kb:
            continue
        ra, rb = next(iter(ka)), next(iter(kb))
        m = max_distance(ka[ra], kb[rb])
        for st in _states(rng, 16):
            va = emu_run(a, st.astype(np.int32).copy())[0].regs[ra]
            vb = emu_run(b, st.astype(np.int32).copy())[0].regs[rb]
            assert wrap_distance(va, vb) <= m, (a, b)
            assert 0.0 <= output_distance(va, vb, m) <= 1.0


def test_the_cases_it_must_be_exact_on():
    I = Instruction
    kb = lambda chunk, r: output_known_bits(chunk)[r]
    # A comparison can only set bit 0.
    assert kb([I('SLT', 3, 1, 2)], 3) == KnownBits(MASK & ~1, 0)
    # A logical right shift by 4 clears the top four bits.
    assert kb([I('SRLI', 3, 1, 4)], 3).zero == 0xF0000000
    # Known bits propagate through a chain: only bits 4..11 can be set.
    assert kb([I('ANDI', 3, 1, 0xFF), I('SLLI', 4, 3, 4)], 4).zero == MASK & ~0xFF0
    assert kb([I('SLT', 3, 1, 2), I('SLLI', 4, 3, 3)], 4).zero == MASK & ~0x8
    # x0 is known 0; an instruction of constants is fully known.
    assert kb([I('ADDI', 3, 0, -5)], 3) == KnownBits.constant(-5)
    assert kb([I('ADD', 3, 0, 0)], 3) == KnownBits.constant(0)
    # Two comparisons are at most 1 apart; a comparison against a full-range
    # value, and two full-range values, at most 2^31.
    slt = kb([I('SLT', 3, 1, 2)], 3)
    full = kb([I('ADD', 3, 1, 2)], 3)
    assert max_distance(slt, slt) == 1
    assert max_distance(slt, KnownBits.constant(0)) == 1
    assert max_distance(slt, full) == 2 ** 31
    assert max_distance(full, full) == 2 ** 31


def test_output_distance_is_zero_when_equal_and_one_at_the_maximum():
    assert output_distance(5, 5, 2 ** 31) == 0.0
    assert output_distance(0, 2 ** 31, 2 ** 31) == 1.0
    assert output_distance(1, 0, 1) == 1.0
    # The wrap-around reading: -1 and 0 are 1 apart, and so are INT_MAX and INT_MIN.
    assert wrap_distance(-1, 0) == 1
    assert wrap_distance(2 ** 31 - 1, -2 ** 31) == 1
    assert wrap_distance(2 ** 31, 2 ** 31 - 1) == 1
