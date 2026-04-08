"""Tests for the training data producer."""

import io
import numpy as np
import pytest
from emulator import Instruction, SparseMemory
from datagen import (
    Batch,
    random_instruction, produce_batch,
    dest_type, dest_reg, extract_data_val,
    write_batch, read_batch,
    write_stream_header, read_stream_header,
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


class TestBatchIO:
    def test_roundtrip(self):
        batch = produce_batch(16, 4, np.random.default_rng(42))
        buf = io.BytesIO()
        write_stream_header(buf)
        write_batch(buf, batch)
        buf.seek(0)
        read_stream_header(buf)
        batch2 = read_batch(buf)
        assert np.array_equal(batch.token_ids, batch2.token_ids)
        assert np.array_equal(batch.padding_mask, batch2.padding_mask)
        assert np.array_equal(batch.data_vals, batch2.data_vals)
        assert np.array_equal(batch.pc_vals, batch2.pc_vals)
        assert np.array_equal(batch.dest_types, batch2.dest_types)
        assert np.array_equal(batch.dest_regs, batch2.dest_regs)
        assert batch2.instructions is None

    def test_multiple_batches(self):
        rng = np.random.default_rng(42)
        buf = io.BytesIO()
        write_stream_header(buf)
        for _ in range(3):
            write_batch(buf, produce_batch(8, 4, rng))
        buf.seek(0)
        read_stream_header(buf)
        batches = []
        while True:
            b = read_batch(buf)
            if b is None:
                break
            batches.append(b)
        assert len(batches) == 3

    def test_eof(self):
        assert read_batch(io.BytesIO()) is None

    def test_bad_magic(self):
        buf = io.BytesIO(b'XXXX\x01' + b'\x00' * 6)
        with pytest.raises(ValueError, match='Bad magic'):
            read_stream_header(buf)

    def test_truncated_batch(self):
        buf = io.BytesIO()
        write_stream_header(buf)
        buf.write(b'\x02\x00\x00\x00')  # partial header
        buf.seek(len(buf.getvalue()) - 4)  # seek to after stream header
        # Actually, seek to right after stream header to read the partial batch header
        buf.seek(0)
        read_stream_header(buf)
        with pytest.raises(EOFError):
            read_batch(buf)

    def test_empty_after_header(self):
        buf = io.BytesIO()
        write_stream_header(buf)
        buf.seek(0)
        read_stream_header(buf)
        assert read_batch(buf) is None
