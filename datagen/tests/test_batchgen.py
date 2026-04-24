"""Tests for the single-instruction batch generator."""

import io
import numpy as np
import pytest

from datagen import (
    InstructionBatch, produce_instruction_batch,
    extract_data_val, dest_type, dest_reg,
)
from datagen.batchgen import (
    write_stream_header, read_stream_header,
    write_batch, read_batch,
)
from emulator import Instruction, SparseMemory


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

    def test_data_val_alu(self):
        regs = np.zeros(32, dtype=np.int32)
        regs[5] = 42
        assert extract_data_val(Instruction('ADD', 5, 3, 7), regs, None) == 42

    def test_data_val_branch(self):
        regs = np.zeros(32, dtype=np.int32)
        assert extract_data_val(Instruction('BEQ', 1, 2, 8), regs, None) == 0

    def test_data_val_store_readback(self):
        """extract_data_val reads back the stored value from memory."""
        smem = SparseMemory()
        smem[100] = 0xAB
        smem[101] = 0xCD
        regs = np.zeros(32, dtype=np.int32)
        regs[3] = 100
        val = extract_data_val(Instruction('SH', 5, 0, 3), regs, smem)
        assert val == 0xCDAB


class TestProduceInstructionBatch:
    def test_shapes(self):
        rng = np.random.default_rng(0)
        batch = produce_instruction_batch(16, 4, rng)
        assert isinstance(batch, InstructionBatch)
        assert batch.token_ids.shape[0] == 16
        assert batch.data_vals.shape == (16, 4)
        assert batch.pc_vals.shape == (16, 4)
        assert batch.dest_types.shape == (16,)
        assert batch.dest_regs.shape == (16,)

    def test_deterministic(self):
        b1 = produce_instruction_batch(8, 4, np.random.default_rng(42))
        b2 = produce_instruction_batch(8, 4, np.random.default_rng(42))
        assert np.array_equal(b1.data_vals, b2.data_vals)
        assert np.array_equal(b1.pc_vals, b2.pc_vals)


class TestInstructionBatchIO:
    def test_roundtrip(self):
        rng = np.random.default_rng(42)
        batch = produce_instruction_batch(8, 4, rng)
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
            write_batch(buf, produce_instruction_batch(4, 2, rng))
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
        buf = io.BytesIO(b'XXXX\x02' + b'\x00' * 7)
        with pytest.raises(ValueError, match='Bad magic'):
            read_stream_header(buf)
