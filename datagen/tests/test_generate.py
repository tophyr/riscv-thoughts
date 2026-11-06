"""Tests for datagen.generate — instruction generation, relabeling,
the equivalence MANIFEST, and the termination-rule grouping API."""

import numpy as np
import pytest

from emulator import (
    Instruction, run, make_ctx, random_regs, ALL_OPCODES,
)
from datagen.generate import (
    DEFAULT_DISTRIBUTION, build_opcode_table,
    random_instruction,
    random_perm, relabel,
    MANIFEST, sample_binding,
    single, until_branch, until_transformation, length_cap, either,
    collect_groups,
)


# ---------------------------------------------------------------------------
# Instruction generation
# ---------------------------------------------------------------------------

def test_random_instruction_uses_supported_ops():
    rng = np.random.default_rng(0)
    for _ in range(200):
        i = random_instruction(rng)
        # Every emitted opcode must be one the emulator can execute.
        assert i.opcode in ALL_OPCODES, i.opcode


def test_build_opcode_table_sums_to_one():
    table = build_opcode_table(DEFAULT_DISTRIBUTION)
    total = sum(w for w, _ in table)
    assert abs(total - 1.0) < 1e-9


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
    # collect_groups requires a bounded rule; the cap is large enough not
    # to fire, so this still tests until_branch's termination behavior.
    groups = list(collect_groups(iter(instrs), until_branch() | length_cap(100)))
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
    groups = list(collect_groups(
        iter(instrs), until_transformation() | length_cap(100)))
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


def test_collect_groups_emits_tail():
    """Leftover instructions that don't trigger termination still come
    out as a final group."""
    instrs = [Instruction('ADDI', 1, 0, i) for i in range(3)]
    groups = list(collect_groups(iter(instrs), until_branch() | length_cap(100)))
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_collect_groups_rejects_unbounded_rule():
    """A rule with no length cap (bare until_branch / until_transformation)
    can build an infinite group from the stream, so collect_groups refuses
    it rather than risk looping forever."""
    instrs = [Instruction('ADDI', 1, 0, 5)]
    for rule in (until_branch(), until_transformation()):
        with pytest.raises(ValueError, match='unbounded'):
            list(collect_groups(iter(instrs), rule))


def test_partner_groups_are_exact_on_the_emulator():
    """Every canonical spelling in a partner group writes the same value to rd
    on every register file tried, so any two are a true equivalent pair; every
    contrast spelling differs from them on some register file."""
    from datagen.compare import draw_anchor_states
    from datagen.generate import PARTNER_GROUPS, materialize
    from emulator import run as emu_run
    rng = np.random.default_rng(7)
    # Drawn states plus all-zero and all-ones register files: a contrast such
    # as `SLTI rd,rs,1` against `SLTI rd,rs,0` differs only where rs == 0.
    states = np.concatenate([draw_anchor_states(rng, 64),
                             np.zeros((1, 32), np.int64), np.full((1, 32), -1)])
    states[:, 0] = 0
    for group in PARTNER_GROUPS:
        for _ in range(50):
            b = sample_binding(group, rng)
            outs = []
            for tpl in group.canonical:
                ins = materialize(tpl, b)
                outs.append([int(emu_run([ins], st.astype(np.int32).copy())[0].regs[b['rd']])
                             for st in states])
            assert all(o == outs[0] for o in outs), group.name
            for tpl in group.contrast:
                if tpl.args[0] != 'rd':
                    continue
                ins = materialize(tpl, b)
                c = [int(emu_run([ins], st.astype(np.int32).copy())[0].regs[b['rd']]) for st in states]
                assert c != outs[0], (group.name, ins)


def test_partner_pairs_compute_one_function():
    """A sampled partner pair writes the same value to rd on every register
    file: two different spellings of one function."""
    from datagen.compare import draw_anchor_states
    from datagen.generate import sample_partner_pair
    from emulator import run as emu_run
    rng = np.random.default_rng(11)
    states = draw_anchor_states(rng, 32)
    states[:, 0] = 0
    for _ in range(400):
        a, b = sample_partner_pair(rng)
        assert (a.opcode, a.args) != (b.opcode, b.args)
        for st in states:
            assert (emu_run([a], st.astype(np.int32).copy())[0].regs[a.args[0]]
                    == emu_run([b], st.astype(np.int32).copy())[0].regs[b.args[0]]), (a, b)


def test_every_rewrite_keeps_its_chunk_equivalent():
    """Each chunk partner rewrite, embedded in random context
    at lengths 2, 4 and 8, computes the same register function on the
    emulator; a rewrite left out of the list is never drawn."""
    from emulator import run as emu_run
    import datagen.generate as G
    from datagen.compare import draw_anchor_states
    rng = np.random.default_rng(21)
    states = draw_anchor_states(np.random.default_rng(3), 64)
    for name in G._REWRITES:
        for L in (2, 4, 8):
            for _ in range(80):
                x, y = G.sample_chunk_partner_pair(rng, L, range(1, 32), (name,))
                assert len(x) <= L and len(y) <= L and x != y, (name, x, y)
                for s in states:
                    s = s.copy(); s[0] = 0
                    assert np.array_equal(emu_run(x, s.copy())[0].regs,
                                          emu_run(y, s.copy())[0].regs), (name, x, y)


def test_the_proof_separates_a_sampled_match_from_an_equivalence():
    """`single_equivalents` filters by sampled states only. `(x & -996) <u 203`
    holds exactly when x <u 1024, so SLTIU x,1024 is equivalent, while 512, a
    match at every search state, is not; the proof keeps only the first."""
    from datagen.mine import single_equivalents
    from datagen.prove import equivalent
    chain = [Instruction('ANDI', 27, 13, -996), Instruction('SLTIU', 27, 27, 203)]
    cands = {c.args[2]: c for c in single_equivalents(chain, 27) if c.opcode == 'SLTIU'}
    assert 512 in cands and 1024 in cands
    assert equivalent(chain, [cands[1024]]) is True
    assert equivalent(chain, [cands[512]]) is False


def test_searched_pairs_are_fresh_proven_and_respect_the_exclusion():
    """Each draw is a new chain and a single instruction proved equal on every
    register file; an excluded opcode appears in neither."""
    from datagen.mine import sample_mined_pair
    from datagen.prove import equivalent
    rng = np.random.default_rng(4)
    shifts = {'SLL', 'SRL', 'SRA', 'SLLI', 'SRLI', 'SRAI'}
    seen = set()
    for _ in range(30):
        a, b = sample_mined_pair(rng, exclude=frozenset(shifts))
        assert len(a) == 2 and len(b) == 1
        assert equivalent(a, b) is True, (a, b)
        assert not {i.opcode for i in a + b} & shifts, (a, b)
        seen.add(str((a, b)))
    assert len(seen) == 30


def test_a_pooled_pair_is_drawn_at_most_reuse_times_and_stays_proven(monkeypatch):
    """Each mined pair leaves the pool after `reuse` draws, so the search runs
    once per `reuse` draws once the pool is full; any POOL_SIZE consecutive
    draws are distinct; every draw, renamed afresh, is still proved
    equivalent."""
    import collections
    import datagen.mine as mine
    from datagen.prove import equivalent
    rng = np.random.default_rng(9)
    mine._POOLS.clear()
    for a, b in (mine.sample_pooled_pair(rng, reuse=3) for _ in range(40)):
        assert equivalent(a, b) is True, (a, b)
    mine._POOLS.clear()
    made = iter(range(10 ** 6))
    monkeypatch.setattr(mine, 'sample_mined_pair', lambda rng, exclude: (
        [Instruction('ADDI', 5, 6, next(made) % 2000)], [Instruction('ADDI', 5, 6, 0)]))
    draws = [mine.sample_pooled_pair(rng, reuse=3) for _ in range(3000)]
    ids = [a[0].args[2] for a, _ in draws]
    assert max(collections.Counter(ids).values()) <= 3
    # Any POOL_SIZE consecutive draws are distinct pairs, so no batch repeats one.
    assert all(len(set(ids[i:i + mine.POOL_SIZE])) == mine.POOL_SIZE
               for i in range(len(ids) - mine.POOL_SIZE))
    mined = next(made)
    assert abs(mined - (mine.POOL_SIZE + 3000 / 3)) < 60, mined
