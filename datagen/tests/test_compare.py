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

    def test_canonical_positions_go_to_behavioral_inputs(self):
        """Behavioral input i must execute at canonical position i+1.

        The value-prediction head is fed `anchor[:, k+1]` for input slot k, and
        `in_slot_regs[k]` is the k-th behavioral input, so the baseline has to
        put that input at that position. Spending a position on a live-but-
        non-behavioral input shifts every later input off its slot: the target
        then depends on a read that never reaches the encoder, and two chunks
        the encoder cannot tell apart get different targets.

        `SLTIU rd,rs,-300` compares UNSIGNED against a sign-extended
        0xFFFFFED4, so it is the constant 1 and its source is read but not
        behavioral.
        """
        anchors = make_anchor_states(8, seed=0)
        second = Instruction('ADDI', 9, 5, 100)
        # x11 is read, live (its instruction writes an output) and irrelevant.
        far = precompute_chunk([Instruction('SLTIU', 7, 11, -300), second],
                               anchors)
        # the same chunk with no earlier read at all
        near = precompute_chunk([Instruction('SLTIU', 7, 5, -300), second],
                                anchors)
        assert far.inputs == [11, 5] and far.behavioral_inputs == [5]
        for pre in (far, near):
            assert pre.in_slot_regs[0] == 5
            # x9 = x5 + 100 with x5 sourced from canonical position 1
            assert (pre.out_regs[:, 9] == anchors[:, 1] + 100).all()

    def test_supervision_is_rename_equivariant(self):
        """Everything the oracle emits must permute with a register rename.

        The model is exactly equivariant by construction, so a chunk and its
        rename produce the same representation; if the oracle labels them
        differently, one of the two is being graded against a target it cannot
        distinguish. `state_max_err` checks this for the model and nothing
        checked it for the targets.

        `out_regs` is asserted EQUAL rather than permuted: canonicalization
        re-sources inputs by read position, so the VALUES are register-name
        free — that is the whole point of the canonical frame.
        """
        from datagen.generate import random_perm, relabel

        anchors = make_anchor_states(8, seed=0)
        rng = np.random.default_rng(0)
        chunks = [
            # regression anchor: canonical position 1 IS register x1, so this
            # chunk writes x1 exactly the value x1 already held. Deciding
            # "unmodified" from that made the emitted supervision depend on
            # register names.
            [Instruction('ADD', 1, 7, 0)],
            [Instruction('ADD', 5, 3, 4), Instruction('ADDI', 6, 5, 100)],
            [Instruction('SLTIU', 4, 9, -300), Instruction('ADDI', 8, 9, 12)],
            [Instruction('ADD', 5, 5, 0)],            # true no-op
            [Instruction('XORI', 2, 2, 0)],           # true no-op
        ]
        for chunk in chunks:
            for _ in range(4):
                perm = random_perm(rng)
                a = precompute_chunk(chunk, anchors)
                b = precompute_chunk(relabel(chunk, perm), anchors)

                def permute(regs):
                    return [int(perm[r]) for r in regs]

                # behavioral_inputs is in first-read order, which a rename
                # preserves. reg_outs is sorted by register NUMBER, so only
                # its membership is meaningful — out_slot_regs below is the
                # ordered one (SSA write order).
                assert permute(a.behavioral_inputs) == b.behavioral_inputs, chunk
                assert set(permute(a.reg_outs)) == set(b.reg_outs), chunk
                for name in ('in_slot_regs', 'out_slot_regs'):
                    fa, fb = getattr(a, name), getattr(b, name)
                    filled = fa != AUX_CE_IGNORE
                    assert ((fb != AUX_CE_IGNORE) == filled).all(), (chunk, name)
                    assert (perm[fa[filled]] == fb[filled]).all(), (chunk, name)
                for name in ('live_in_mask', 'live_out_mask'):
                    ma, mb = getattr(a, name), getattr(b, name)
                    assert (mb[perm] == ma).all(), (chunk, name)
                # the canonical target VALUES carry no register identity
                for ra in a.reg_outs:
                    assert np.array_equal(
                        a.out_regs[:, ra], b.out_regs[:, int(perm[ra])]), chunk

    def test_memory_op_raises(self):
        # Mem ops are out of V1 distance scope; precompute_chunk raises so
        # batch.py routes them to the None/empty-aux path.
        anchors = make_anchor_states(4, seed=0)
        with pytest.raises(NotImplementedError):
            precompute_chunk([Instruction('LW', 5, 0, 1)], anchors)
