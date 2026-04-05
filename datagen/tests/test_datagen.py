"""Unit tests for the RV32I data generator."""

import numpy as np
import pytest
from emulator import Instruction, RV32IState, execute, random_regs
from datagen import (
    generate_sequence,
    rename_registers,
    permute_regs,
    reorder_independent,
    insert_nops,
    remove_nops,
    strength_reduce,
    decompose_immediate,
    mutate_immediate,
    mutate_opcode,
    generate_equivalence_pair,
    generate_near_equivalence_pair,
    generate_training_sample,
)


class TestGenerateSequence:
    def test_correct_length(self):
        seq = generate_sequence(10, np.random.default_rng(42))
        assert len(seq) == 10

    def test_all_instructions_valid(self):
        seq = generate_sequence(20, np.random.default_rng(42))
        for instr in seq:
            assert isinstance(instr, Instruction)

    def test_deterministic_with_seed(self):
        s1 = generate_sequence(10, np.random.default_rng(42))
        s2 = generate_sequence(10, np.random.default_rng(42))
        for a, b in zip(s1, s2):
            assert a.opcode == b.opcode
            assert a.args == b.args

    def test_different_seeds_differ(self):
        s1 = generate_sequence(10, np.random.default_rng(0))
        s2 = generate_sequence(10, np.random.default_rng(1))
        # Extremely unlikely to be identical
        assert any(a.opcode != b.opcode or a.args != b.args
                    for a, b in zip(s1, s2))

    def test_r_type_only(self):
        seq = generate_sequence(20, np.random.default_rng(42), r_type_prob=1.0)
        for instr in seq:
            assert instr.opcode in ('ADD', 'SUB', 'XOR', 'OR', 'AND',
                                     'SLL', 'SRL', 'SRA', 'SLT', 'SLTU')

    def test_i_type_only(self):
        seq = generate_sequence(20, np.random.default_rng(42), r_type_prob=0.0)
        for instr in seq:
            assert instr.opcode in ('ADDI', 'XORI', 'ORI', 'ANDI',
                                     'SLLI', 'SRLI', 'SRAI', 'SLTI', 'SLTIU')

    def test_executes_without_error(self):
        seq = generate_sequence(20, np.random.default_rng(42))
        regs = random_regs(np.random.default_rng(42))
        execute(seq, initial_regs=regs)  # should not raise


class TestRenameRegisters:
    def test_preserves_semantics(self):
        """Renamed sequence with permuted initial state produces permuted output."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            seq = generate_sequence(10, rng)
            initial_regs = random_regs(rng)
            renamed, perm = rename_registers(seq, np.random.default_rng(seed + 100))
            permuted_regs = permute_regs(initial_regs, perm)
            state_orig = execute(seq, initial_regs=initial_regs)
            state_renamed = execute(renamed, initial_regs=permuted_regs)
            # Inverse-permute the renamed state back to original namespace
            inv_perm = [0] * len(perm)
            for old, new in enumerate(perm):
                inv_perm[new] = old
            state_back = RV32IState(permute_regs(state_renamed.regs, inv_perm))
            assert state_orig == state_back, (
                f'Seed {seed}: register renaming changed semantics'
            )

    def test_x0_unchanged(self):
        rng = np.random.default_rng(42)
        seq = [Instruction('ADDI', 1, 0, 5)]  # reads x0
        renamed, perm = rename_registers(seq, rng)
        # x0 maps to x0
        assert perm[0] == 0
        # x0 should still appear as a source in the renamed instruction
        assert renamed[0].args[1] == 0

    def test_produces_different_sequence(self):
        rng = np.random.default_rng(42)
        seq = generate_sequence(10, np.random.default_rng(0))
        renamed, _perm = rename_registers(seq, rng)
        any_different = any(
            a.args != b.args for a, b in zip(seq, renamed)
        )
        assert any_different

    def test_returns_permutation(self):
        rng = np.random.default_rng(42)
        seq = [Instruction('ADDI', 1, 0, 5)]
        _renamed, perm = rename_registers(seq, rng)
        assert len(perm) == 16
        assert sorted(perm) == list(range(16))  # valid permutation


class TestReorderIndependent:
    def test_preserves_semantics(self):
        """Reordered sequence must produce identical execution state."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            seq = generate_sequence(10, rng)
            initial_regs = random_regs(rng)
            reordered = reorder_independent(seq, np.random.default_rng(seed + 100))
            state_orig = execute(seq, initial_regs=initial_regs)
            state_reordered = execute(reordered, initial_regs=initial_regs)
            assert state_orig == state_reordered, (
                f'Seed {seed}: reordering changed semantics'
            )

    def test_preserves_length(self):
        rng = np.random.default_rng(42)
        seq = generate_sequence(10, rng)
        reordered = reorder_independent(seq, np.random.default_rng(0))
        assert len(reordered) == len(seq)

    def test_single_instruction(self):
        seq = [Instruction('ADDI', 1, 0, 5)]
        result = reorder_independent(seq, np.random.default_rng(0))
        assert len(result) == 1


class TestInsertRemoveNops:
    def test_insert_preserves_semantics(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            seq = generate_sequence(10, rng)
            initial_regs = random_regs(rng)
            with_nops = insert_nops(seq, np.random.default_rng(seed + 100), count=3)
            state_orig = execute(seq, initial_regs=initial_regs)
            state_nops = execute(with_nops, initial_regs=initial_regs)
            assert state_orig == state_nops, f'Seed {seed}: nop insertion changed semantics'

    def test_insert_increases_length(self):
        rng = np.random.default_rng(42)
        seq = generate_sequence(10, rng)
        with_nops = insert_nops(seq, rng, count=3)
        assert len(with_nops) == 13

    def test_remove_nops(self):
        seq = [
            Instruction('ADD', 0, 0, 0),
            Instruction('ADDI', 1, 0, 5),
            Instruction('ADD', 0, 0, 0),
        ]
        cleaned = remove_nops(seq)
        assert len(cleaned) == 1
        assert cleaned[0].opcode == 'ADDI'

    def test_insert_then_remove_preserves_original(self):
        rng = np.random.default_rng(42)
        seq = generate_sequence(10, rng)
        # Make sure original has no nops
        seq = remove_nops(seq)
        with_nops = insert_nops(seq, rng, count=5)
        recovered = remove_nops(with_nops)
        assert len(recovered) == len(seq)
        for a, b in zip(seq, recovered):
            assert a.opcode == b.opcode
            assert a.args == b.args


class TestStrengthReduce:
    def test_add_to_shift(self):
        seq = [Instruction('ADD', 5, 3, 3)]  # x5 = x3 + x3
        rng = np.random.default_rng(42)
        result = strength_reduce(seq, rng)
        assert result is not None
        assert result[0].opcode == 'SLLI'
        assert result[0].args == (5, 3, 1)

    def test_shift_to_add(self):
        seq = [Instruction('SLLI', 5, 3, 1)]  # x5 = x3 << 1
        rng = np.random.default_rng(42)
        result = strength_reduce(seq, rng)
        assert result is not None
        assert result[0].opcode == 'ADD'
        assert result[0].args == (5, 3, 3)

    def test_preserves_semantics(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            seq = [Instruction('ADD', 5, 3, 3)]
            initial_regs = random_regs(rng)
            reduced = strength_reduce(seq, np.random.default_rng(0))
            if reduced is not None:
                state_orig = execute(seq, initial_regs=initial_regs)
                state_reduced = execute(reduced, initial_regs=initial_regs)
                assert state_orig == state_reduced

    def test_no_applicable_reduction(self):
        seq = [Instruction('ADD', 5, 3, 7)]  # rs1 != rs2
        result = strength_reduce(seq, np.random.default_rng(42))
        assert result is None

    def test_slli_not_1_no_reduction(self):
        seq = [Instruction('SLLI', 5, 3, 2)]  # shift by 2, not 1
        result = strength_reduce(seq, np.random.default_rng(42))
        assert result is None


class TestDecomposeImmediate:
    def test_splits_addi(self):
        seq = [Instruction('ADDI', 5, 0, 1000)]
        rng = np.random.default_rng(42)
        result = decompose_immediate(seq, rng)
        assert result is not None
        assert len(result) == 2
        assert result[0].opcode == 'ADDI'
        assert result[1].opcode == 'ADDI'
        # Second instruction reads from same dest as first writes
        assert result[1].args[1] == result[0].args[0]

    def test_preserves_semantics(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            imm = int(rng.integers(-2000, 2001))
            if abs(imm) < 2:
                continue
            seq = [Instruction('ADDI', 5, 0, imm)]
            decomposed = decompose_immediate(seq, np.random.default_rng(seed + 100))
            if decomposed is not None:
                state_orig = execute(seq)
                state_decomposed = execute(decomposed)
                assert state_orig == state_decomposed, (
                    f'Seed {seed}, imm={imm}: decomposition changed semantics'
                )

    def test_small_immediate_no_decomposition(self):
        seq = [Instruction('ADDI', 5, 0, 1)]
        result = decompose_immediate(seq, np.random.default_rng(42))
        assert result is None


class TestMutateImmediate:
    def test_changes_immediate(self):
        seq = [Instruction('ADDI', 5, 0, 100)]
        rng = np.random.default_rng(42)
        result = mutate_immediate(seq, rng)
        assert result is not None
        assert result[0].args[2] != 100

    def test_produces_different_state(self):
        seq = [Instruction('ADDI', 5, 0, 100)]
        rng = np.random.default_rng(42)
        result = mutate_immediate(seq, rng)
        if result is not None:
            s1 = execute(seq)
            s2 = execute(result)
            assert s1 != s2

    def test_no_i_type_returns_none(self):
        seq = [Instruction('ADD', 5, 3, 7)]
        result = mutate_immediate(seq, np.random.default_rng(42))
        assert result is None

    def test_shift_stays_in_range(self):
        seq = [Instruction('SLLI', 5, 3, 0)]  # minimum shift
        for seed in range(50):
            result = mutate_immediate(seq, np.random.default_rng(seed))
            if result is not None:
                assert 0 <= result[0].args[2] <= 31


class TestMutateOpcode:
    def test_changes_opcode(self):
        seq = [Instruction('ADD', 5, 3, 7)]
        rng = np.random.default_rng(42)
        result = mutate_opcode(seq, rng)
        assert result is not None
        assert result[0].opcode != 'ADD'
        assert result[0].opcode == 'SUB'  # ADD's mutation partner

    def test_preserves_args(self):
        seq = [Instruction('ADD', 5, 3, 7)]
        result = mutate_opcode(seq, np.random.default_rng(42))
        if result is not None:
            assert result[0].args == (5, 3, 7)


class TestGenerateEquivalencePair:
    def test_states_are_equal(self):
        for seed in range(20):
            pair = generate_equivalence_pair(10, np.random.default_rng(seed))
            assert pair['state_a'] == pair['state_b'], (
                f"Seed {seed}, transform={pair['transformation']}: "
                f"states differ"
            )

    def test_has_required_keys(self):
        pair = generate_equivalence_pair(10, np.random.default_rng(42))
        assert 'seq_a' in pair
        assert 'seq_b' in pair
        assert 'initial_regs' in pair
        assert 'state_a' in pair
        assert 'state_b' in pair
        assert 'transformation' in pair

    def test_sequences_are_different(self):
        # At least some pairs should have different sequences
        any_different = False
        for seed in range(20):
            pair = generate_equivalence_pair(10, np.random.default_rng(seed))
            if len(pair['seq_a']) != len(pair['seq_b']):
                any_different = True
                break
            if any(a.opcode != b.opcode or a.args != b.args
                    for a, b in zip(pair['seq_a'], pair['seq_b'])):
                any_different = True
                break
        assert any_different


class TestGenerateNearEquivalencePair:
    def test_states_differ(self):
        any_differ = False
        for seed in range(20):
            pair = generate_near_equivalence_pair(10, np.random.default_rng(seed))
            if pair['state_a'] != pair['state_b']:
                any_differ = True
                break
        assert any_differ

    def test_has_distance(self):
        pair = generate_near_equivalence_pair(10, np.random.default_rng(42))
        assert 'distance' in pair
        assert 'regs_differ' in pair['distance']

    def test_has_required_keys(self):
        pair = generate_near_equivalence_pair(10, np.random.default_rng(42))
        assert 'seq_a' in pair
        assert 'seq_b' in pair
        assert 'mutation' in pair


class TestGenerateTrainingSample:
    def test_equivalent_label(self):
        # With equiv_prob=1.0, always generates equivalent pairs
        sample = generate_training_sample(10, np.random.default_rng(42),
                                          equiv_prob=1.0)
        assert sample['label'] == 'equivalent'

    def test_near_equivalent_label(self):
        sample = generate_training_sample(10, np.random.default_rng(42),
                                          equiv_prob=0.0)
        assert sample['label'] == 'near_equivalent'

    def test_mixed(self):
        labels = set()
        for seed in range(50):
            sample = generate_training_sample(10, np.random.default_rng(seed))
            labels.add(sample['label'])
        assert 'equivalent' in labels
        assert 'near_equivalent' in labels
