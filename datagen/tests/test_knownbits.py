"""Known bits read from behaviour: two spellings of one function get one
bound, the bound holds the cases a range must be exact on, and the distance
it scales is 0 when equal and 1 at the maximum."""
import numpy as np

from datagen.compare import _eval_ssa_numpy, bound_states, live_nodes, to_ssa
from datagen.knownbits import (MASK, KnownBits, column_distance, max_distance,
                               observed_bits)
from emulator import Instruction as I


def _bits(chunk, r):
    ssa = to_ssa(chunk)
    out = _eval_ssa_numpy(ssa, len(chunk), bound_states(), live_nodes(ssa))[0]
    return KnownBits(*observed_bits(out[:, r]))


def test_spellings_of_one_function_get_one_bound():
    zero = [_bits([I('ADDI', 5, 0, 0)], 5), _bits([I('SUB', 5, 6, 6)], 5),
            _bits([I('XOR', 5, 6, 6)], 5)]
    assert zero[0] == zero[1] == zero[2] == KnownBits.constant(0)
    sign = [_bits([I('SLTI', 5, 6, 0)], 5), _bits([I('SRLI', 5, 6, 31)], 5)]
    assert sign[0] == sign[1]


def test_the_cases_a_range_must_be_exact_on():
    # A comparison can only set bit 0, and does set it somewhere.
    assert _bits([I('SLT', 3, 1, 2)], 3) == KnownBits(MASK & ~1, 0)
    # A logical right shift by 4 clears the top four bits.
    assert _bits([I('SRLI', 3, 1, 4)], 3).zero == 0xF0000000
    # Through a chain only bits 4..11 can be set.
    assert _bits([I('ANDI', 3, 1, 0xFF), I('SLLI', 4, 3, 4)], 4).zero == MASK & ~0xFF0
    # Two comparisons are at most 1 apart; a comparison against a full-range
    # value, and two full-range values, at most 2^31.
    slt, full = _bits([I('SLT', 3, 1, 2)], 3), _bits([I('ADD', 3, 1, 2)], 3)
    assert max_distance(slt, slt) == 1
    assert max_distance(slt, KnownBits.constant(0)) == 1
    assert max_distance(slt, full) == 2 ** 31
    assert max_distance(full, full) == 2 ** 31


def test_column_distance_is_zero_when_equal_and_one_at_the_maximum():
    def d(a, b):
        a, b = np.array(a, np.int32), np.array(b, np.int32)
        return column_distance(a, observed_bits(a), b, observed_bits(b))
    assert d([5, 6], [5, 6]) == 0.0
    assert d([0, 0], [-2 ** 31, -2 ** 31]) == 1.0
    assert d([0, 1], [1, 0]) == 1.0
    # The wrap-around reading: -1 and 0 are 1 apart, and so are INT_MAX and INT_MIN.
    assert d([-1], [0]) == d([2 ** 31 - 1], [-2 ** 31]) == d([1], [0])
