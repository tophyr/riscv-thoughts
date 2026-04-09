"""Unit tests for the RV32I emulator wrapper."""

import numpy as np
import pytest
from emulator import Instruction, RV32IState, run, random_regs


def execute(instructions, initial_regs=None, max_steps=None):
    """Test helper: run instructions, return only register state."""
    state, _, _ = run(instructions, regs=initial_regs, max_steps=max_steps)
    return state


class TestInstruction:
    def test_valid_opcode(self):
        instr = Instruction('ADD', 5, 3, 7)
        assert instr.opcode == 'ADD'
        assert instr.args == (5, 3, 7)

    def test_invalid_opcode(self):
        with pytest.raises(AssertionError, match='unknown opcode'):
            Instruction('FAKE', 1, 2, 3)

    def test_repr_r_type(self):
        instr = Instruction('ADD', 5, 3, 7)
        assert repr(instr) == 'ADD x5, x3, x7'

    def test_repr_i_type(self):
        instr = Instruction('ADDI', 5, 3, 42)
        assert repr(instr) == 'ADDI x5, x3, 42'


class TestRV32IState:
    def test_x0_forced_zero(self):
        regs = np.ones(32, dtype=np.int32)
        state = RV32IState(regs)
        assert state.regs[0] == 0
        assert state.regs[1] == 1

    def test_immutable_copy(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[1] = 42
        state = RV32IState(regs)
        regs[1] = 99  # mutate original
        assert state.regs[1] == 42  # state unaffected

    def test_equality(self):
        r1 = np.zeros(32, dtype=np.int32)
        r1[5] = 100
        r2 = np.zeros(32, dtype=np.int32)
        r2[5] = 100
        assert RV32IState(r1) == RV32IState(r2)

    def test_inequality(self):
        r1 = np.zeros(32, dtype=np.int32)
        r2 = np.zeros(32, dtype=np.int32)
        r2[5] = 1
        assert RV32IState(r1) != RV32IState(r2)

    def test_equality_type_mismatch(self):
        state = RV32IState(np.zeros(32, dtype=np.int32))
        assert state != "not a state"

    def test_distance_identical(self):
        regs = np.zeros(32, dtype=np.int32)
        s = RV32IState(regs)
        d = s.distance(s)
        assert d['regs_differ'] == 0
        assert d['l1'] == 0
        assert d['l2'] == 0.0
        assert d['hamming'] == 0
        assert d['which_differ'] == []

    def test_distance_one_reg(self):
        r1 = np.zeros(32, dtype=np.int32)
        r2 = np.zeros(32, dtype=np.int32)
        r2[5] = 1
        d = RV32IState(r1).distance(RV32IState(r2))
        assert d['regs_differ'] == 1
        assert d['l1'] == 1
        assert d['which_differ'] == [5]

    def test_distance_multiple_regs(self):
        r1 = np.zeros(32, dtype=np.int32)
        r2 = np.zeros(32, dtype=np.int32)
        r2[3] = 10
        r2[7] = -20
        d = RV32IState(r1).distance(RV32IState(r2))
        assert d['regs_differ'] == 2
        assert d['l1'] == 30
        assert set(d['which_differ']) == {3, 7}

    def test_repr(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[5] = 42
        s = RV32IState(regs)
        assert 'x5' in repr(s)
        assert '42' in repr(s)


class TestExecuteArithmetic:
    def test_addi(self):
        state = execute([Instruction('ADDI', 5, 0, 42)])
        assert state.regs[5] == 42

    def test_addi_negative(self):
        state = execute([Instruction('ADDI', 5, 0, -10)])
        assert state.regs[5] == -10

    def test_add(self):
        prog = [
            Instruction('ADDI', 1, 0, 10),
            Instruction('ADDI', 2, 0, 20),
            Instruction('ADD', 3, 1, 2),
        ]
        state = execute(prog)
        assert state.regs[3] == 30

    def test_sub(self):
        prog = [
            Instruction('ADDI', 1, 0, 30),
            Instruction('ADDI', 2, 0, 12),
            Instruction('SUB', 3, 1, 2),
        ]
        state = execute(prog)
        assert state.regs[3] == 18


class TestExecuteBitwise:
    def test_xor(self):
        prog = [
            Instruction('ADDI', 1, 0, 0xFF),
            Instruction('ADDI', 2, 0, 0x0F),
            Instruction('XOR', 3, 1, 2),
        ]
        state = execute(prog)
        assert state.regs[3] == 0xF0

    def test_or(self):
        prog = [
            Instruction('ADDI', 1, 0, 0xA0),
            Instruction('ADDI', 2, 0, 0x0B),
            Instruction('OR', 3, 1, 2),
        ]
        state = execute(prog)
        assert state.regs[3] == 0xAB

    def test_and(self):
        prog = [
            Instruction('ADDI', 1, 0, 0xFF),
            Instruction('ANDI', 2, 1, 0x0F),
        ]
        state = execute(prog)
        assert state.regs[2] == 0x0F


class TestExecuteShift:
    def test_slli(self):
        prog = [
            Instruction('ADDI', 1, 0, 1),
            Instruction('SLLI', 2, 1, 4),
        ]
        state = execute(prog)
        assert state.regs[2] == 16

    def test_srli(self):
        prog = [
            Instruction('ADDI', 1, 0, 16),
            Instruction('SRLI', 2, 1, 4),
        ]
        state = execute(prog)
        assert state.regs[2] == 1

    def test_srai_negative(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[1] = np.int32(-16)
        prog = [Instruction('SRAI', 2, 1, 2)]
        state = execute(prog, initial_regs=regs)
        assert state.regs[2] == -4  # arithmetic shift preserves sign


class TestExecuteCompare:
    def test_slt_true(self):
        prog = [
            Instruction('ADDI', 1, 0, 5),
            Instruction('ADDI', 2, 0, 10),
            Instruction('SLT', 3, 1, 2),
        ]
        state = execute(prog)
        assert state.regs[3] == 1

    def test_slt_false(self):
        prog = [
            Instruction('ADDI', 1, 0, 10),
            Instruction('ADDI', 2, 0, 5),
            Instruction('SLT', 3, 1, 2),
        ]
        state = execute(prog)
        assert state.regs[3] == 0

    def test_sltu(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[1] = np.int32(-1)  # 0xFFFFFFFF unsigned
        regs[2] = np.int32(1)
        prog = [Instruction('SLTU', 3, 1, 2)]
        state = execute(prog, initial_regs=regs)
        assert state.regs[3] == 0  # 0xFFFFFFFF > 1 unsigned


class TestExecuteBranch:
    def test_beq_taken(self):
        prog = [
            Instruction('ADDI', 1, 0, 5),
            Instruction('ADDI', 2, 0, 5),
            Instruction('BEQ', 1, 2, 8),   # skip next instruction
            Instruction('ADDI', 3, 0, 99),  # skipped
            Instruction('ADDI', 4, 0, 42),  # lands here
        ]
        state = execute(prog)
        assert state.regs[3] == 0   # skipped
        assert state.regs[4] == 42  # executed

    def test_beq_not_taken(self):
        prog = [
            Instruction('ADDI', 1, 0, 5),
            Instruction('ADDI', 2, 0, 6),
            Instruction('BEQ', 1, 2, 8),
            Instruction('ADDI', 3, 0, 99),  # not skipped
            Instruction('ADDI', 4, 0, 42),
        ]
        state = execute(prog)
        assert state.regs[3] == 99
        assert state.regs[4] == 42

    def test_bne(self):
        prog = [
            Instruction('ADDI', 1, 0, 5),
            Instruction('ADDI', 2, 0, 6),
            Instruction('BNE', 1, 2, 8),
            Instruction('ADDI', 3, 0, 99),  # skipped
            Instruction('ADDI', 4, 0, 42),
        ]
        state = execute(prog)
        assert state.regs[3] == 0
        assert state.regs[4] == 42


class TestExecuteUpperImmediate:
    def test_lui(self):
        prog = [Instruction('LUI', 5, 0x12345)]
        state = execute(prog)
        assert state.regs[5] == 0x12345000


class TestExecuteJump:
    def test_jal(self):
        prog = [
            Instruction('JAL', 1, 8),       # x1 = PC+4, jump +8
            Instruction('ADDI', 2, 0, 99),  # skipped
            Instruction('ADDI', 3, 0, 42),  # lands here
        ]
        state = execute(prog)
        assert state.regs[1] == 4   # return address
        assert state.regs[2] == 0   # skipped
        assert state.regs[3] == 42


class TestExecuteInitialState:
    def test_custom_initial_regs(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[1] = 100
        prog = [Instruction('ADDI', 2, 1, 5)]
        state = execute(prog, initial_regs=regs)
        assert state.regs[2] == 105

    def test_x0_stays_zero(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[0] = 999
        prog = [Instruction('ADDI', 1, 0, 1)]
        state = execute(prog, initial_regs=regs)
        assert state.regs[0] == 0
        assert state.regs[1] == 1


class TestExecuteMaxSteps:
    def test_infinite_loop_terminates(self):
        prog = [Instruction('BEQ', 0, 0, 0)]  # branch to self
        state = execute(prog, max_steps=100)
        assert state.regs[0] == 0


class TestEquivalences:
    """Semantic-preserving transformations from the design doc."""

    def test_strength_reduction_shift_vs_add(self):
        regs = random_regs(np.random.default_rng(42))
        prog_a = [Instruction('ADD', 2, 1, 1)]
        prog_b = [Instruction('SLLI', 2, 1, 1)]
        assert execute(prog_a, initial_regs=regs) == execute(prog_b, initial_regs=regs)

    def test_identity_insertion(self):
        regs = random_regs(np.random.default_rng(42))
        prog_a = [Instruction('ADDI', 1, 1, 5)]
        prog_b = [
            Instruction('ADD', 0, 0, 0),  # nop
            Instruction('ADDI', 1, 1, 5),
        ]
        assert execute(prog_a, initial_regs=regs) == execute(prog_b, initial_regs=regs)

    def test_immediate_decomposition(self):
        prog_a = [Instruction('ADDI', 5, 0, 1000)]
        prog_b = [
            Instruction('ADDI', 5, 0, 500),
            Instruction('ADDI', 5, 5, 500),
        ]
        assert execute(prog_a) == execute(prog_b)

    def test_instruction_reordering(self):
        regs = random_regs(np.random.default_rng(42))
        prog_a = [
            Instruction('ADDI', 1, 0, 10),
            Instruction('ADDI', 2, 0, 20),
        ]
        prog_b = [
            Instruction('ADDI', 2, 0, 20),
            Instruction('ADDI', 1, 0, 10),
        ]
        assert execute(prog_a, initial_regs=regs) == execute(prog_b, initial_regs=regs)


class TestRandomRegs:
    def test_x0_always_zero(self):
        for seed in range(10):
            regs = random_regs(np.random.default_rng(seed))
            assert regs[0] == 0

    def test_deterministic_with_seed(self):
        r1 = random_regs(np.random.default_rng(42))
        r2 = random_regs(np.random.default_rng(42))
        assert np.array_equal(r1, r2)

    def test_different_seeds_differ(self):
        r1 = random_regs(np.random.default_rng(0))
        r2 = random_regs(np.random.default_rng(1))
        assert not np.array_equal(r1, r2)

    def test_shape_and_dtype(self):
        regs = random_regs()
        assert regs.shape == (32,)
        assert regs.dtype == np.int32

    def test_structured_equal_groups(self):
        rng = np.random.default_rng(0)
        n = 10000
        count = 0
        for _ in range(n):
            regs = random_regs(rng)
            # Count how many non-x0 registers share a value with
            # at least one other non-x0 register.
            from collections import Counter
            vals = Counter(int(regs[k]) for k in range(1, 32))
            grouped = sum(c for c in vals.values() if c > 1)
            if grouped >= 8:
                count += 1
        rate = count / n
        assert 0.10 < rate < 0.25, f'grouped-equal rate {rate:.3f} outside [0.10, 0.25]'

    def test_structured_small_values(self):
        rng = np.random.default_rng(0)
        n = 10000
        count = 0
        for _ in range(n):
            regs = random_regs(rng)
            if any(0 <= regs[k] <= 31 for k in range(1, 32)):
                count += 1
        rate = count / n
        assert 0.05 < rate < 0.20, f'small value rate {rate:.3f} outside [0.05, 0.20]'

    def test_structured_extra_zeros(self):
        rng = np.random.default_rng(0)
        n = 10000
        count = 0
        for _ in range(n):
            regs = random_regs(rng)
            if sum(regs[k] == 0 for k in range(1, 32)) > 0:
                count += 1
        rate = count / n
        assert 0.03 < rate < 0.15, f'extra zero rate {rate:.3f} outside [0.03, 0.15]'


class TestRunAllInstructionTypes:
    """Tests for run() with all RV32I instruction types, including
    SparseMemory, arbitrary addresses, and random PC."""

    def test_alu(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[3] = 10; regs[7] = 20
        state, pc, _ = run([Instruction('ADD', 5, 3, 7)], regs=regs)
        assert state.regs[5] == 30 and pc == 4

    def test_load_arbitrary_address(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[3] = 1000000
        s1, _, _ = run([Instruction('LW', 5, 4, 3)], regs=regs,
                       rng=np.random.default_rng(42))
        s2, _, _ = run([Instruction('LW', 5, 4, 3)], regs=regs,
                       rng=np.random.default_rng(42))
        assert s1.regs[5] == s2.regs[5]

    def test_store_arbitrary_address(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[3] = -500000
        regs[5] = np.int32(-559038737)
        _, pc, mem = run([Instruction('SW', 5, 8, 3)], regs=regs)
        assert pc == 4
        addr = -500000 + 8
        val = (int(mem[addr]) | (int(mem[addr+1]) << 8)
               | (int(mem[addr+2]) << 16) | (int(mem[addr+3]) << 24))
        assert val == 0xDEADBEEF

    def test_branch(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[1] = 5; regs[2] = 5
        _, pc, _ = run([Instruction('BEQ', 1, 2, 16)], regs=regs, pc=100)
        assert pc == 116

    def test_jal(self):
        state, pc, _ = run([Instruction('JAL', 1, 100)], pc=500)
        assert state.regs[1] == 504 and pc == 600

    def test_auipc_vs_lui(self):
        s1, _, _ = run([Instruction('AUIPC', 5, 1)], pc=1000)
        s2, _, _ = run([Instruction('LUI', 5, 1)], pc=1000)
        assert s1.regs[5] != s2.regs[5]

    def test_mem_returned(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[3] = 42
        _, _, mem = run([Instruction('SB', 3, 0, 0)], regs=regs)
        assert int(mem[0]) == 42
