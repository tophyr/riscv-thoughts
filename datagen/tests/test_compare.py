"""Tests for datagen.compare — behavioral chunk distance.

The behavioral distance scalar (the GVN bijection oracle) is the
authoritative equivalence test: two chunks behave identically iff their
distance is exactly 0. Each property below — relabeling, commutativity,
DCE of dead writes, etc. — is asserted against that production code path.
"""

import numpy as np
import pytest

from emulator import Instruction, run, make_ctx
from datagen.behavioral_oracle import (
    behavioral_distance, behavioral_distance_cached,
)
from datagen.compare import (
    make_anchor_states, precompute_chunk, precompute_row_outputs,
    AUX_CE_IGNORE,
)
from datagen.generate import random_relabel


# ---------------------------------------------------------------------------
# Behavioral distance — the dormant GVN bijection oracle
# ---------------------------------------------------------------------------

class TestChunkDistance:
    def test_identical_is_zero(self):
        a = [Instruction('ADDI', 5, 1, 7), Instruction('ADD', 6, 5, 5)]
        assert behavioral_distance(a, a) == 0.0

    def test_random_relabel_is_zero(self):
        rng = np.random.default_rng(0)
        a = [Instruction('ADDI', 5, 1, 7), Instruction('ADD', 6, 5, 5)]
        for _ in range(5):
            b = random_relabel(a, rng)
            d = behavioral_distance(a, b)
            assert d == 0.0, f'relabeled distance was {d}, expected 0'

    def test_commutative_swap_is_zero(self):
        a = [Instruction('ADD', 5, 1, 2)]
        b = [Instruction('ADD', 5, 2, 1)]
        assert behavioral_distance(a, b) == 0.0

    def test_dead_writes_collapse_to_zero(self):
        # The first ADDI is dead — x5 is overwritten before being read.
        # DCE inside analyze_chunk should drop it; behaviorally these
        # two chunks produce identical observable state.
        a = [Instruction('ADDI', 5, 1, 99), Instruction('ADDI', 5, 1, 7)]
        b = [Instruction('ADDI', 5, 1, 7)]
        d = behavioral_distance(a, b)
        assert d == 0.0, f'dead-write distance was {d}, expected 0'

    def test_double_shift_equiv_slli(self):
        # x5 = x1 << 2  ≡  x5 = x1 << 1; x5 = x5 << 1
        a = [Instruction('SLLI', 5, 1, 2)]
        b = [Instruction('SLLI', 5, 1, 1), Instruction('SLLI', 5, 5, 1)]
        d = behavioral_distance(a, b)
        assert d == 0.0, f'double-shift distance was {d}, expected 0'

    def test_distinct_imm_is_positive(self):
        a = [Instruction('ADDI', 5, 1, 7)]
        b = [Instruction('ADDI', 5, 1, 8)]
        assert behavioral_distance(a, b) > 0.0

    def test_distinct_dataflow_is_positive(self):
        # SUB and ADD differ behaviorally on essentially every input pair;
        # no bijection over input names can collapse them.
        a = [Instruction('SUB', 5, 1, 2)]
        b = [Instruction('ADD', 5, 1, 2)]
        assert behavioral_distance(a, b) > 0.0

    def test_symmetric(self):
        a = [Instruction('ADDI', 5, 1, 7), Instruction('ADD', 6, 5, 5)]
        b = [Instruction('ADD', 7, 1, 1)]
        d_ab = behavioral_distance(a, b)
        d_ba = behavioral_distance(b, a)
        assert abs(d_ab - d_ba) < 1e-6, (d_ab, d_ba)

    def test_memory_ops_raise(self):
        a = [Instruction('LW', 5, 0, 1)]
        b = [Instruction('LW', 5, 0, 1)]
        with pytest.raises(NotImplementedError):
            behavioral_distance(a, b)

    def test_cached_matches_uncached(self):
        a = [Instruction('ADDI', 5, 1, 7), Instruction('XOR', 6, 5, 1)]
        b = [Instruction('ADDI', 5, 1, 8), Instruction('XOR', 6, 5, 2)]
        anchor = make_anchor_states(8, seed=0)
        pre_a = precompute_chunk(a, anchor)
        pre_b = precompute_chunk(b, anchor)
        d_cached = behavioral_distance_cached(pre_a, pre_b, anchor)
        d_uncached = behavioral_distance(a, b, n_states=8, seed=0)
        assert abs(d_cached - d_uncached) < 1e-6


# ---------------------------------------------------------------------------
# Anchor states
# ---------------------------------------------------------------------------

def test_make_anchor_states_shape():
    s = make_anchor_states(16, seed=0)
    assert s.shape == (16, 32)
    # x0 must always be 0.
    assert (s[:, 0] == 0).all()


def test_make_anchor_states_x0_zeroed_across_seeds():
    # x0 is hardwired zero; make_anchor_states must zero column 0 regardless
    # of seed, while the rest of the row is seed-dependent random state.
    for seed in (0, 7, 42):
        s = make_anchor_states(8, seed=seed)
        assert (s[:, 0] == 0).all()
        assert (s[:, 1:] != 0).any()   # not a degenerate all-zero file


# ---------------------------------------------------------------------------
# precompute_row_outputs — the LIVE T1 value-prediction label. rd_values must
# equal the destination register's value from actually running the instruction
# on each anchor state (compared against the CPU emulator).
# ---------------------------------------------------------------------------

def _emu_rd(instr, anchors, rd):
    ctx = make_ctx()
    out = []
    for i in range(anchors.shape[0]):
        st, _, _ = run([instr], regs=anchors[i].copy(), pc=0,
                       _ctx=ctx, max_steps=1)
        out.append(int(st.regs[rd]))
    return out


class TestRowOutputs:
    def test_rd_values_shape_and_has_rd(self):
        anchors = make_anchor_states(8, seed=0)
        ro = precompute_row_outputs(Instruction('ADD', 5, 1, 2), anchors)
        assert ro.has_rd
        assert ro.rd_values.shape == (8,)

    def test_x0_destination_has_no_rd(self):
        anchors = make_anchor_states(4, seed=0)
        # Writing x0 is a NOP — no destination register is updated.
        ro = precompute_row_outputs(Instruction('ADD', 0, 1, 2), anchors)
        assert ro.has_rd is False

    # precompute_row_outputs must execute on the RAW anchor state (the
    # instruction's actual source registers), NOT the canonical baseline
    # out_regs (precompute_chunk relocates inputs to CANON_POSITIONS). For
    # ADD x7,x5,x9 the label must be anchor[x5]+anchor[x9]; ADD x5,x1,x2 would
    # pass even with the old canonical-baseline bug since its inputs already
    # sit at the canonical positions, so these use non-canonical sources.
    def test_add_rd_values_match_emulator(self):
        anchors = make_anchor_states(8, seed=3)
        instr = Instruction('ADD', 7, 5, 9)
        ro = precompute_row_outputs(instr, anchors)
        expected = _emu_rd(instr, anchors, rd=7)
        for i in range(8):
            assert int(ro.rd_values[i]) == expected[i], i

    def test_addi_rd_values_match_emulator(self):
        anchors = make_anchor_states(8, seed=3)
        instr = Instruction('ADDI', 7, 3, -42)
        ro = precompute_row_outputs(instr, anchors)
        expected = _emu_rd(instr, anchors, rd=7)
        for i in range(8):
            assert int(ro.rd_values[i]) == expected[i], i


# ---------------------------------------------------------------------------
# precompute_chunk aux targets — the T2 register-identity supervision schema.
# ---------------------------------------------------------------------------

class TestAuxTargets:
    def test_known_instruction_slots_and_masks(self):
        anchors = make_anchor_states(4, seed=0)
        # ADD x5, x1, x2 — behavioral inputs {x1,x2}, output {x5}.
        pre = precompute_chunk([Instruction('ADD', 5, 1, 2)], anchors)
        assert pre.behavioral_inputs == [1, 2]
        assert pre.reg_outs == [5]
        assert np.nonzero(pre.live_in_mask)[0].tolist() == [1, 2]
        assert np.nonzero(pre.live_out_mask)[0].tolist() == [5]
        # in_slot i = i-th behavioral input (syntactic first-read order).
        assert pre.in_slot_regs[0] == 1
        assert pre.in_slot_regs[1] == 2
        assert (pre.in_slot_regs[2:] == AUX_CE_IGNORE).all()
        # out_slot i = i-th SSA-write-order output.
        assert pre.out_slot_regs[0] == 5
        assert (pre.out_slot_regs[1:] == AUX_CE_IGNORE).all()
        assert pre.pc_explicit is False

    def test_memory_op_raises(self):
        # Mem ops are out of V1 distance scope; precompute_chunk raises so
        # batch.py routes them to the None/empty-aux path.
        anchors = make_anchor_states(4, seed=0)
        with pytest.raises(NotImplementedError):
            precompute_chunk([Instruction('LW', 5, 0, 1)], anchors)
