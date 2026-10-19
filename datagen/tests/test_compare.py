"""Tests for datagen.compare — the canonical SSA/anchor-execution schema
(`precompute_chunk`): anchor states + the per-chunk binding/aux targets that
training consumes.
"""

import numpy as np
import pytest

from emulator import Instruction
from datagen.compare import (
    draw_anchor_states, heldout_anchor_states, instruction_reads,
    precompute_chunk, AUX_CE_IGNORE,
)


def _anchors(n, seed=0):
    return draw_anchor_states(np.random.default_rng(seed), n)


# ---------------------------------------------------------------------------
# Anchor states
# ---------------------------------------------------------------------------

def test_draw_anchor_states_shape_and_x0():
    for seed in (0, 7, 42):
        s = _anchors(16, seed)
        assert s.shape == (16, 32) and s.dtype == np.int32
        assert (s[:, 0] == 0).all()        # x0 is hardwired zero
        assert (s[:, 1:] != 0).any()


def test_draw_anchor_states_is_structured():
    """Uniform int32 never produces zeros, small values or equal registers,
    which makes comparisons against an immediate constant."""
    s = _anchors(512)[:, 1:]
    assert (s == 0).mean() > 0.05
    assert (np.abs(s.astype(np.int64)) < 64).mean() > 0.2
    assert (s[:, 0] == s[:, 1]).mean() > 0.01


def test_heldout_anchor_states_is_fixed():
    assert np.array_equal(heldout_anchor_states(), heldout_anchor_states())
    assert heldout_anchor_states().shape == (64, 32)


# ---------------------------------------------------------------------------
# Structural labels — decided on fixed witness states, never on anchors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('instr, reads', [
    (Instruction('SLTIU', 4, 5, 40), [5]),    # varies only below 40
    (Instruction('SLTIU', 4, 5, -300), [5]),  # unsigned vs 0xFFFFFED4
    (Instruction('SLTIU', 4, 5, 0), []),      # x <u 0 is always false
    (Instruction('SLTU', 4, 0, 5), [5]),      # x5 != 0
    (Instruction('XOR', 3, 5, 5), []),
    (Instruction('SUB', 3, 5, 5), []),
    (Instruction('ANDI', 3, 5, 0), []),
    (Instruction('SUB', 3, 0, 5), [5]),
    (Instruction('SLL', 3, 0, 7), []),        # 0 << anything
    (Instruction('ADD', 5, 6, 7), [6, 7]),
    (Instruction('ADDI', 0, 5, 3), []),       # writes x0: no output
    (Instruction('BEQ', 5, 6, 8), [5, 6]),
])
def test_instruction_reads(instr, reads):
    assert instruction_reads(instr) == reads


@pytest.mark.parametrize('chunk', [
    [Instruction('ADD', 5, 5, 0)],
    [Instruction('ADDI', 5, 5, 1), Instruction('ADDI', 5, 5, -1)],
])
def test_a_same_value_write_is_a_read_and_a_write(chunk):
    """Writing a register the value it already held is still a write: calling it
    a no-op would make the targets impose identity == empty chunk, an
    equivalence for the essence to learn, not for the binding to be told."""
    pre = precompute_chunk(chunk, _anchors(8))
    assert pre.behavioral_inputs == [5] and pre.reg_outs == [5]
    assert (pre.out_regs[:, 5] == _anchors(8)[:, 1]).all()


def test_a_dead_instructions_reads_are_inputs():
    """The input list is the order T2's read ranks number registers in, and a
    dead instruction's reads are ranked: listing only inputs that reach an
    output would put x8 at canonical position 1 while T2 numbers it 3."""
    anchors = _anchors(8)
    pre = precompute_chunk([Instruction('ADD', 5, 6, 7),
                            Instruction('ADDI', 5, 8, 1)], anchors)
    assert pre.behavioral_inputs == [6, 7, 8] and pre.reg_outs == [5]
    assert (pre.out_regs[:, 5] == anchors[:, 3] + 1).all()


def _derived_reads(op, rd, rs1, rs2_or_imm):
    """Read relevance derived by hand from each operation's semantics, with no
    sampled values: a source is irrelevant only where the result is the same
    for every 32-bit value it can hold."""
    if op in ('ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI', 'SLTI', 'SLTIU'):
        imm = rs2_or_imm
        constant = ((op == 'ANDI' and imm == 0) or (op == 'ORI' and imm == -1)
                    or (op == 'SLTIU' and imm == 0))   # x <u 0 is always false
        return [] if rs1 == 0 or constant else [rs1]
    rs2 = rs2_or_imm
    if rs1 == rs2:            # one register read twice, or x0 twice
        if rs1 == 0 or op in ('SUB', 'XOR', 'SLT', 'SLTU'):
            return []         # r-r = r^r = 0; r<r is false
        return [rs1]
    if rs1 == 0:              # 0 op r
        return [] if op in ('AND', 'SLL', 'SRL', 'SRA') else [rs2]
    if rs2 == 0:              # r op 0
        return [] if op in ('AND', 'SLTU') else [rs1]
    return [rs1, rs2]


def test_witness_reads_match_the_derivation_on_every_alu_form():
    """The witness states must find every input an instruction depends on, at
    every immediate, operand pattern and destination alias — checked against an
    independent derivation rather than against more sampled states."""
    R = ('ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU')
    I = ('ADDI', 'XORI', 'ORI', 'ANDI', 'SLTI', 'SLTIU')
    S = ('SLLI', 'SRLI', 'SRAI')
    forms = []
    for op in R:
        for rs1, rs2 in ((5, 6), (5, 5), (0, 5), (5, 0), (0, 0)):
            for rd in {9, rs1 or 9, rs2 or 9}:
                forms.append((op, rd, rs1, rs2))
    for op in I + S:
        for imm in (range(32) if op in S else range(-2048, 2048)):
            for rs1, rd in ((5, 9), (5, 5), (0, 9)):
                forms.append((op, rd, rs1, imm))
    wrong = [f for f in forms
             if instruction_reads(Instruction(*f)) != _derived_reads(*f)]
    assert not wrong, (len(wrong), wrong[:10])
    assert len(forms) > 70000


def test_labels_do_not_depend_on_the_anchor_draw():
    """Labels are a property of the chunk: per-batch anchors may change the
    values, never which registers are inputs and outputs."""
    chunks = [
        [Instruction('SLTIU', 4, 5, 40)],
        [Instruction('SLTU', 9, 0, 18), Instruction('SRL', 12, 19, 2)],
        [Instruction('ANDI', 6, 5, 0x0F0), Instruction('SLTIU', 7, 6, 1)],
        [Instruction('OR', 14, 24, 14), Instruction('SRA', 25, 11, 14)],
    ]
    for chunk in chunks:
        labels = {(tuple(p.behavioral_inputs), tuple(p.reg_outs))
                  for p in (precompute_chunk(chunk, _anchors(8, seed))
                            for seed in range(20))}
        assert len(labels) == 1, (chunk, labels)


# ---------------------------------------------------------------------------
# precompute_chunk aux targets — the T2 register-identity supervision schema.
# ---------------------------------------------------------------------------

class TestAuxTargets:
    def test_known_instruction_slots_and_masks(self):
        anchors = _anchors(4)
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

        `XOR rd,rs,rs` is the constant 0, so its source is read but not
        behavioral.
        """
        anchors = _anchors(8)
        second = Instruction('ADDI', 9, 5, 100)
        # x11 is read, live (its instruction writes an output) and irrelevant.
        far = precompute_chunk([Instruction('XOR', 7, 11, 11), second], anchors)
        # the same chunk with no earlier read at all
        near = precompute_chunk([Instruction('XOR', 7, 5, 5), second], anchors)
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

        anchors = _anchors(8)
        rng = np.random.default_rng(0)
        chunks = [
            # regression anchor: canonical position 1 IS register x1, so this
            # chunk writes x1 exactly the value x1 already held. Deciding
            # "unmodified" from that made the emitted supervision depend on
            # register names.
            [Instruction('ADD', 1, 7, 0)],
            [Instruction('ADD', 5, 3, 4), Instruction('ADDI', 6, 5, 100)],
            [Instruction('SLTIU', 4, 9, -300), Instruction('ADDI', 8, 9, 12)],
            [Instruction('XOR', 4, 9, 9), Instruction('ADDI', 8, 9, 12)],
            [Instruction('ADD', 5, 5, 0)],            # same-value self-write
            [Instruction('XORI', 2, 2, 0)],           # same-value self-write
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
        anchors = _anchors(4)
        with pytest.raises(NotImplementedError):
            precompute_chunk([Instruction('LW', 5, 0, 1)], anchors)


# ---------------------------------------------------------------------------
# chain_group: raising the density of dataflow-composed chunks
# ---------------------------------------------------------------------------

def test_chain_group_makes_each_instruction_read_its_predecessor():
    import numpy as np
    from datagen.generate import chain_group
    from emulator import Instruction
    rng = np.random.default_rng(0)
    g = [Instruction('ADDI', 6, 3, 100), Instruction('ADDI', 9, 4, 25)]
    out = chain_group(g, rng)
    assert out[0] == g[0]                      # the producer is untouched
    assert out[1].opcode == 'ADDI'             # the opcode is untouched
    assert out[1].args[0] == 9                 # so is the destination
    assert out[1].args[1] == 6                 # the source now reads x6
    assert out[1].args[2] == 25                # so is the immediate


def test_chain_group_rewrites_one_source_of_an_r_type():
    import numpy as np
    from datagen.generate import chain_group
    from emulator import Instruction
    for seed in range(8):
        rng = np.random.default_rng(seed)
        g = [Instruction('ADD', 6, 3, 4), Instruction('SUB', 9, 7, 8)]
        out = chain_group(g, rng)
        assert out[1].opcode == 'SUB' and out[1].args[0] == 9
        assert 6 in (out[1].args[1], out[1].args[2]), out[1]
        # exactly ONE source is rewritten, the other keeps its draw
        assert {out[1].args[1], out[1].args[2]} & {7, 8}


def test_chain_group_skips_a_producer_that_writes_x0():
    """An x0 write discards its result, so reading it composes nothing."""
    import numpy as np
    from datagen.generate import chain_group
    from emulator import Instruction
    rng = np.random.default_rng(0)
    g = [Instruction('ADDI', 0, 3, 100), Instruction('ADDI', 9, 4, 25)]
    out = chain_group(g, rng)
    assert out[1] == g[1]


def test_chain_group_leaves_a_single_instruction_alone():
    import numpy as np
    from datagen.generate import chain_group
    from emulator import Instruction
    rng = np.random.default_rng(0)
    g = [Instruction('ADDI', 6, 3, 100)]
    assert chain_group(g, rng) == g


def test_chain_rate_raises_composed_density_without_moving_opcodes():
    """The point of the knob: more dataflow composition, same opcode mix."""
    import itertools
    from collections import Counter
    import numpy as np
    from datagen import length_cap
    from datagen.batch import generate_chunks
    from datagen.generate import build_opcode_table, load_distribution

    dist = load_distribution('configs/instr_alu_writers.json')
    table = build_opcode_table(dist)

    def measure(chain_rate, n=3000):
        rng = np.random.default_rng(0)
        it = generate_chunks(length_cap(2), rng, opcode_table=table,
                             dest_regs=list(range(1, 32)),
                             chain_rate=chain_rate)
        composed = total = 0
        ops = Counter()
        for ch in itertools.islice(it, n):
            ins = ch.instructions
            ops.update(i.opcode for i in ins)
            total += 1
            if len(ins) != 2:
                continue
            rd = ins[0].args[0]
            reads = ((ins[1].args[1],) if ins[1].opcode.endswith('I')
                     else (ins[1].args[1], ins[1].args[2]))
            composed += int(rd != 0 and rd in reads)
        return composed / total, ops

    base_rate, base_ops = measure(0.0)
    hi_rate, hi_ops = measure(1.0)
    assert base_rate < 0.15, base_rate
    assert hi_rate > 0.85, hi_rate
    # the opcode distribution must be unchanged: same support, similar mass
    assert set(base_ops) == set(hi_ops)
    nb, nh = sum(base_ops.values()), sum(hi_ops.values())
    for op in base_ops:
        assert abs(base_ops[op] / nb - hi_ops[op] / nh) < 0.02, op


def test_every_augmentation_rate_comes_from_the_config_alone():
    """A corpus is ONE reviewable file. `chain.rate`, `equivalences.rate` and
    `invalidity.rate` are read from the config and nothing else can set them,
    which is what makes the config a run copies into its directory
    (scripts/record_corpus.py) a complete description of what it trained on."""
    from datagen.generate import load_distribution, validate_distribution
    d = load_distribution('configs/instr_alu_writers_chain010.json')
    assert d['chain']['rate'] == 0.10
    validate_distribution(dict(d))
    # a corpus with no chain block chains nothing
    plain = load_distribution('configs/instr_alu_writers.json')
    assert plain.get('chain', {}).get('rate', 0.0) == 0.0
    assert plain.get('equivalences', {}).get('rate', 0.0) == 0.0
    assert plain.get('invalidity', {}).get('rate', 0.0) == 0.0


def test_chain_config_block_is_validated():
    from datagen.generate import validate_distribution
    import pytest
    for bad in ({'R_ALU': 1.0, 'chain': {'rate': 1.5}},
                {'R_ALU': 1.0, 'chain': {'rate': -0.1}},
                {'R_ALU': 1.0, 'chain': {'rat': 0.1}},
                {'R_ALU': 1.0, 'chain': 0.1}):
        with pytest.raises(ValueError):
            validate_distribution(dict(bad))
    validate_distribution({'R_ALU': 1.0, 'chain': {'rate': 0.25}})
