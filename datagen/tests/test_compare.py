"""Tests for datagen.compare — the canonical SSA/anchor-execution schema
(`precompute_chunk`): anchor states + the per-chunk binding/aux targets that
training consumes.
"""

import numpy as np
import pytest

from emulator import Instruction
from datagen.compare import (
    make_anchor_states, precompute_chunk,
    AUX_CE_IGNORE,
)


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
