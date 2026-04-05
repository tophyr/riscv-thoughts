"""Tests for the training data producer."""

import numpy as np
from emulator import Instruction, SparseMemory
from datagen import (
    Batch, InlineProducer, ParallelProducer,
    random_instruction, produce_batch,
    dest_type, dest_reg, extract_data_val,
)
from emulator import R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE


class TestRandomInstruction:
    def test_all_types_generated(self):
        rng = np.random.default_rng(42)
        seen = set()
        for _ in range(1000):
            instr = random_instruction(rng)
            if instr.opcode in R_TYPE: seen.add('R')
            elif instr.opcode in I_TYPE: seen.add('I')
            elif instr.opcode in LOAD_TYPE: seen.add('Load')
            elif instr.opcode in STORE_TYPE: seen.add('Store')
            elif instr.opcode in B_TYPE: seen.add('Branch')
            elif instr.opcode in ('LUI', 'AUIPC'): seen.add('U')
            elif instr.opcode == 'JAL': seen.add('JAL')
            elif instr.opcode == 'JALR': seen.add('JALR')
        assert seen == {'R', 'I', 'Load', 'Store', 'Branch', 'U', 'JAL', 'JALR'}


class TestDestExtraction:
    def test_alu(self):
        assert dest_type(Instruction('ADD', 5, 3, 7)) == 0
        assert dest_reg(Instruction('ADD', 5, 3, 7)) == 5

    def test_store(self):
        assert dest_type(Instruction('SW', 5, 0, 3)) == 1
        assert dest_reg(Instruction('SW', 5, 0, 3)) == 0

    def test_branch(self):
        assert dest_type(Instruction('BEQ', 1, 2, 8)) == 0
        assert dest_reg(Instruction('BEQ', 1, 2, 8)) == 0

    def test_load(self):
        assert dest_type(Instruction('LW', 5, 0, 3)) == 0
        assert dest_reg(Instruction('LW', 5, 0, 3)) == 5

    def test_jal(self):
        assert dest_type(Instruction('JAL', 1, 8)) == 0
        assert dest_reg(Instruction('JAL', 1, 8)) == 1


class TestDataValExtraction:
    def test_alu(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[5] = 42
        assert extract_data_val(Instruction('ADD', 5, 3, 7), regs, None) == 42

    def test_branch(self):
        regs = np.zeros(32, dtype=np.int32)
        assert extract_data_val(Instruction('BEQ', 1, 2, 8), regs, None) == 0

    def test_store(self):
        smem = SparseMemory()
        smem[12345] = 0xAB
        smem[12346] = 0xCD
        regs = np.zeros(32, dtype=np.int32)
        regs[3] = 12345
        assert extract_data_val(
            Instruction('SW', 5, 0, 3), regs, smem) == 0xCDAB


class TestProduceBatch:
    def test_shape(self):
        batch = produce_batch(16, 4, np.random.default_rng(42))
        assert isinstance(batch, Batch)
        assert batch.token_ids.shape[0] == 16
        assert batch.data_vals.shape == (16, 4)
        assert batch.pc_vals.shape == (16, 4)
        assert batch.dest_types.shape == (16,)
        assert batch.dest_regs.shape == (16,)

    def test_deterministic(self):
        b1 = produce_batch(16, 4, np.random.default_rng(42))
        b2 = produce_batch(16, 4, np.random.default_rng(42))
        assert np.array_equal(b1.data_vals, b2.data_vals)
        assert np.array_equal(b1.pc_vals, b2.pc_vals)


class TestInlineProducer:
    def test_yields_correct_count(self):
        batches = list(InlineProducer(batch_size=8, n_inputs=4,
                                      n_batches=5, seed=42))
        assert len(batches) == 5

    def test_batches_are_valid(self):
        for batch in InlineProducer(batch_size=8, n_inputs=4,
                                    n_batches=3, seed=42):
            assert isinstance(batch, Batch)
            assert batch.token_ids.shape[0] == 8


class TestParallelProducer:
    def test_yields_correct_count(self):
        with ParallelProducer(batch_size=8, n_inputs=4, n_batches=5,
                              seed=42, n_workers=2, prefetch=3) as p:
            assert len(list(p)) == 5

    def test_batches_are_valid(self):
        with ParallelProducer(batch_size=8, n_inputs=4, n_batches=3,
                              seed=42, n_workers=2, prefetch=3) as p:
            for batch in p:
                assert isinstance(batch, Batch)
                assert batch.token_ids.shape[0] == 8


