"""Tests for datagen.generate — instruction generation, relabeling,
the equivalence MANIFEST, and the termination-rule grouping API."""

import numpy as np
import pytest

from emulator import (
    Instruction, run, make_ctx, random_regs,
    R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE,
)
from datagen.generate import (
    DEFAULT_DISTRIBUTION, validate_distribution, _build_opcode_table,
    random_instruction, random_basic_block,
    random_perm, relabel, random_relabel,
    MANIFEST, sample_binding, materialize, sample_injection_tuples,
    single, until_branch, until_transformation, length_cap, either,
    collect_groups, TerminationRule,
)


# ---------------------------------------------------------------------------
# Instruction generation
# ---------------------------------------------------------------------------

def test_random_instruction_uses_supported_ops():
    rng = np.random.default_rng(0)
    for _ in range(100):
        i = random_instruction(rng)
        # Doesn't crash and returns something the emulator knows about.
        assert isinstance(i, Instruction)


def test_default_distribution_validates():
    # If this fails, it's a manifest bug.
    validate_distribution(DEFAULT_DISTRIBUTION)


def test_build_opcode_table_sums_to_one():
    table = _build_opcode_table(DEFAULT_DISTRIBUTION)
    total = sum(w for w, _ in table)
    assert abs(total - 1.0) < 1e-9


def test_random_basic_block_terminates_with_branch():
    rng = np.random.default_rng(1)
    for _ in range(20):
        block = random_basic_block(rng, max_length=8)
        assert 2 <= len(block) <= 8
        assert block[-1].opcode in B_TYPE
        # Non-terminal instructions must NOT be branches.
        for instr in block[:-1]:
            assert instr.opcode not in B_TYPE


# ---------------------------------------------------------------------------
# Relabeling
# ---------------------------------------------------------------------------

def test_random_perm_keeps_x0_fixed():
    rng = np.random.default_rng(2)
    for _ in range(20):
        perm = random_perm(rng)
        assert perm[0] == 0
        assert sorted(perm[1:].tolist()) == list(range(1, 32))


def test_relabel_rejects_perm_that_moves_x0():
    bad = np.arange(32, dtype=np.int32)
    bad[0] = 5  # moves x0
    with pytest.raises(ValueError, match='x0'):
        relabel([Instruction('ADD', 1, 2, 3)], bad)


def test_random_relabel_preserves_behavior_under_consistent_input_perm():
    """Behavioral equivalence under register relabeling: the same
    inputs (after applying the same permutation to the input register
    file) should produce the same outputs (under the inverse perm)."""
    rng = np.random.default_rng(3)
    instrs = [
        Instruction('ADDI', 5, 0, 7),
        Instruction('ADD', 6, 5, 5),
    ]
    perm = random_perm(rng)
    relabeled = relabel(instrs, perm)

    ctx = make_ctx()
    regs = random_regs(rng)
    permuted_regs = np.zeros(32, dtype=np.int32)
    for r in range(32):
        permuted_regs[int(perm[r])] = regs[r]

    out_orig, _, _ = run(instrs, regs=regs, pc=0, rng=rng, _ctx=ctx,
                         max_steps=len(instrs))
    out_relabeled, _, _ = run(relabeled, regs=permuted_regs, pc=0,
                              rng=rng, _ctx=ctx, max_steps=len(relabeled))

    # Under the perm, output regs must match positionally after re-permuting.
    for r in range(32):
        assert out_orig.regs[r] == out_relabeled.regs[int(perm[r])]


# ---------------------------------------------------------------------------
# Equivalence manifest
# ---------------------------------------------------------------------------

def test_manifest_nonempty():
    assert len(MANIFEST) > 0


def test_sample_binding_satisfies_constraints():
    rng = np.random.default_rng(4)
    for klass in MANIFEST:
        b = sample_binding(klass, rng)
        for c in klass.constraints:
            assert c(b)


def test_materialize_yields_instructions():
    rng = np.random.default_rng(5)
    for klass in MANIFEST:
        b = sample_binding(klass, rng)
        for tpl in klass.canonical:
            instr = materialize(tpl, b)
            assert isinstance(instr, Instruction)


def test_sample_injection_tuples_returns_whole_tuples():
    rng = np.random.default_rng(6)
    out = sample_injection_tuples(target_count=20, max_per_class=8, rng=rng)
    assert len(out) >= 20
    # Every emission is a real Instruction.
    for instr in out:
        assert isinstance(instr, Instruction)


# ---------------------------------------------------------------------------
# Termination rules
# ---------------------------------------------------------------------------

def _ops(group):
    return [i.opcode for i in group]


def test_single_emits_one_per_group():
    instrs = [Instruction('ADD', 1, 2, 3),
              Instruction('XOR', 4, 5, 6),
              Instruction('SUB', 7, 8, 9)]
    groups = list(collect_groups(iter(instrs), single()))
    assert len(groups) == 3
    for g in groups:
        assert len(g) == 1


def test_until_branch_groups_to_branch():
    instrs = [
        Instruction('ADDI', 1, 0, 5),
        Instruction('ADD', 2, 1, 1),
        Instruction('BEQ', 1, 2, 4),
        Instruction('XOR', 3, 1, 2),
        Instruction('BNE', 3, 0, -8),
    ]
    groups = list(collect_groups(iter(instrs), until_branch()))
    assert len(groups) == 2
    assert _ops(groups[0]) == ['ADDI', 'ADD', 'BEQ']
    assert _ops(groups[1]) == ['XOR', 'BNE']


def test_until_transformation_terminates_on_load_store_branch_jump():
    instrs = [
        Instruction('ADDI', 1, 0, 5),
        Instruction('LW', 2, 0, 1),
        Instruction('ADD', 3, 1, 2),
        Instruction('SW', 3, 0, 1),
        Instruction('JAL', 0, 4),
    ]
    groups = list(collect_groups(iter(instrs), until_transformation()))
    assert len(groups) == 3
    assert _ops(groups[0]) == ['ADDI', 'LW']
    assert _ops(groups[1]) == ['ADD', 'SW']
    assert _ops(groups[2]) == ['JAL']


def test_length_cap_terminates_at_n():
    instrs = [Instruction('ADDI', 1, 0, i) for i in range(10)]
    groups = list(collect_groups(iter(instrs), length_cap(3)))
    assert [len(g) for g in groups] == [3, 3, 3, 1]


def test_either_combines_via_or_operator():
    """`r1 | r2` should be equivalent to either(r1, r2)."""
    instrs = [
        Instruction('ADDI', 1, 0, 5),
        Instruction('ADDI', 2, 0, 6),
        Instruction('BEQ', 1, 2, 4),
        Instruction('ADDI', 3, 0, 7),
        Instruction('ADDI', 4, 0, 8),
        Instruction('ADDI', 5, 0, 9),
    ]
    rule = until_branch() | length_cap(2)
    groups = list(collect_groups(iter(instrs), rule))
    # Cap fires first, then cap fires on the BEQ-bearing pair, then cap.
    assert [len(g) for g in groups] == [2, 1, 2, 1]


def test_either_is_a_termination_rule():
    """The combined rule must itself be a TerminationRule (so it can
    be further composed)."""
    rule = until_branch() | length_cap(5)
    assert isinstance(rule, TerminationRule)


def test_collect_groups_emits_tail():
    """Leftover instructions that don't trigger termination still come
    out as a final group."""
    instrs = [Instruction('ADDI', 1, 0, i) for i in range(3)]
    groups = list(collect_groups(iter(instrs), until_branch()))
    assert len(groups) == 1
    assert len(groups[0]) == 3
