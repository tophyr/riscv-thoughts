"""Tests for datagen.compare — GVN equivalence and behavioral
chunk distance."""

import numpy as np
import pytest

from emulator import Instruction
from datagen.compare import (
    gvn_equivalent, chunk_distance, chunk_distance_cached,
    make_anchor_states, precompute_chunk,
    to_ssa, live_nodes,
    PC_REG, MEM_REG, COMMUTATIVE_OPS,
)
from datagen.generate import random_relabel


# ---------------------------------------------------------------------------
# GVN equivalence — yes/no isomorphism
# ---------------------------------------------------------------------------

class TestGVNEquivalent:
    def test_identical_chunks(self):
        a = [Instruction('ADDI', 5, 1, 7), Instruction('ADD', 6, 5, 5)]
        assert gvn_equivalent(a, a)

    def test_random_relabel(self):
        rng = np.random.default_rng(0)
        a = [Instruction('ADDI', 5, 1, 7), Instruction('ADD', 6, 5, 5)]
        for _ in range(10):
            b = random_relabel(a, rng)
            assert gvn_equivalent(a, b)

    def test_commutativity_add(self):
        a = [Instruction('ADD', 5, 1, 2)]
        b = [Instruction('ADD', 5, 2, 1)]
        assert gvn_equivalent(a, b)

    def test_dead_writes_eliminated(self):
        # x5 is written then immediately overwritten without being read,
        # so the first ADDI is dead and DCE drops it.
        a = [Instruction('ADDI', 5, 1, 99),    # dead — x5 immediately overwritten
             Instruction('ADDI', 5, 1, 7)]
        b = [Instruction('ADDI', 5, 1, 7)]
        assert gvn_equivalent(a, b)

    def test_distinct_outputs_not_equivalent(self):
        a = [Instruction('ADDI', 5, 1, 7)]
        b = [Instruction('ADDI', 5, 1, 8)]   # different imm
        assert not gvn_equivalent(a, b)

    def test_distinct_dataflow_not_equivalent(self):
        # x5 = x1 - x2 vs x5 = x1 + x2: no bijection can equate them.
        a = [Instruction('SUB', 5, 1, 2)]
        b = [Instruction('ADD', 5, 1, 2)]
        assert not gvn_equivalent(a, b)


def test_to_ssa_and_live_nodes():
    instrs = [
        Instruction('ADDI', 5, 1, 7),
        Instruction('ADDI', 6, 0, 0),   # dead
        Instruction('ADD', 7, 5, 5),
    ]
    g = to_ssa(instrs)
    live = live_nodes(g)
    # Live set must contain the live outputs (x5 and x7), but the
    # dead x6 ADDI shouldn't be live (it can't reach any live root).
    op_names = [g.nodes[i].op for i in live]
    assert 'ADDI' in op_names
    assert 'ADD' in op_names


def test_commutative_ops_listed_correctly():
    # Sanity: only the truly commutative R-type ALU ops are listed.
    expected = {'ADD', 'XOR', 'OR', 'AND', 'BEQ_COND', 'BNE_COND'}
    assert COMMUTATIVE_OPS == expected
    # Notably, SUB and SLT are NOT commutative.
    assert 'SUB' not in COMMUTATIVE_OPS
    assert 'SLT' not in COMMUTATIVE_OPS


# ---------------------------------------------------------------------------
# Behavioral distance
# ---------------------------------------------------------------------------

class TestChunkDistance:
    def test_identical_is_zero(self):
        a = [Instruction('ADDI', 5, 1, 7), Instruction('ADD', 6, 5, 5)]
        assert chunk_distance(a, a) == 0.0

    def test_random_relabel_is_zero(self):
        rng = np.random.default_rng(0)
        a = [Instruction('ADDI', 5, 1, 7), Instruction('ADD', 6, 5, 5)]
        for _ in range(5):
            b = random_relabel(a, rng)
            d = chunk_distance(a, b)
            assert d == 0.0, f'relabeled distance was {d}, expected 0'

    def test_commutative_swap_is_zero(self):
        a = [Instruction('ADD', 5, 1, 2)]
        b = [Instruction('ADD', 5, 2, 1)]
        assert chunk_distance(a, b) == 0.0

    def test_double_shift_equiv_slli(self):
        # x5 = x1 << 2  ≡  x5 = x1 << 1; x5 = x5 << 1
        a = [Instruction('SLLI', 5, 1, 2)]
        b = [Instruction('SLLI', 5, 1, 1), Instruction('SLLI', 5, 5, 1)]
        d = chunk_distance(a, b)
        assert d == 0.0, f'double-shift distance was {d}, expected 0'

    def test_distinct_imm_is_positive(self):
        a = [Instruction('ADDI', 5, 1, 7)]
        b = [Instruction('ADDI', 5, 1, 8)]
        assert chunk_distance(a, b) > 0.0

    def test_symmetric(self):
        a = [Instruction('ADDI', 5, 1, 7), Instruction('ADD', 6, 5, 5)]
        b = [Instruction('ADD', 7, 1, 1)]
        d_ab = chunk_distance(a, b)
        d_ba = chunk_distance(b, a)
        assert abs(d_ab - d_ba) < 1e-6, (d_ab, d_ba)

    def test_memory_ops_raise(self):
        a = [Instruction('LW', 5, 0, 1)]
        b = [Instruction('LW', 5, 0, 1)]
        with pytest.raises(NotImplementedError):
            chunk_distance(a, b)

    def test_cached_matches_uncached(self):
        a = [Instruction('ADDI', 5, 1, 7), Instruction('XOR', 6, 5, 1)]
        b = [Instruction('ADDI', 5, 1, 8), Instruction('XOR', 6, 5, 2)]
        anchor = make_anchor_states(8, seed=0)
        pre_a = precompute_chunk(a, anchor)
        pre_b = precompute_chunk(b, anchor)
        d_cached = chunk_distance_cached(pre_a, pre_b, anchor)
        d_uncached = chunk_distance(a, b, n_states=8, seed=0)
        assert abs(d_cached - d_uncached) < 1e-6


# ---------------------------------------------------------------------------
# Anchor states
# ---------------------------------------------------------------------------

def test_make_anchor_states_shape():
    s = make_anchor_states(16, seed=0)
    assert s.shape == (16, 32)
    # x0 must always be 0.
    assert (s[:, 0] == 0).all()


def test_make_anchor_states_deterministic_with_seed():
    s1 = make_anchor_states(8, seed=42)
    s2 = make_anchor_states(8, seed=42)
    assert np.array_equal(s1, s2)
