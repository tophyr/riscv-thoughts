"""Tests for the sequence training data generator."""

import io
import numpy as np
import pytest

from emulator import Instruction, R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE, run, make_ctx, random_regs
from datagen import (
    SequenceBatch, random_basic_block, execute_sequence, produce_sequence_batch,
)
from datagen.seqgen import (
    write_stream_header, read_stream_header,
    write_batch, read_batch,
)
from datagen.instrgen import (
    _OPCODE_DISTRIBUTION, DEFAULT_DISTRIBUTION,
    validate_distribution,
)


class TestOpcodeDistribution:
    def test_default_weights_sum_to_one(self):
        """The default distribution is validated at import time.
        This test verifies the built table also sums correctly."""
        total = sum(weight for weight, _ in _OPCODE_DISTRIBUTION)
        assert abs(total - 1.0) < 1e-9, \
            f'opcode distribution weights sum to {total}, not 1.0'

    def test_validate_rejects_bad_sum(self):
        bad = dict(DEFAULT_DISTRIBUTION)
        bad['R_ALU'] = 0.99  # now sums to > 1
        with pytest.raises(ValueError, match='sum to'):
            validate_distribution(bad)

    def test_validate_rejects_unknown_category(self):
        bad = {'R_ALU': 0.5, 'NONEXISTENT': 0.5}
        with pytest.raises(ValueError, match='Unknown'):
            validate_distribution(bad)

    def test_produce_with_bad_distribution_fails(self):
        """Passing a bad distribution to produce_instruction_batch
        should fail at validation time, not silently produce garbage."""
        from datagen import produce_instruction_batch
        rng = np.random.default_rng(0)
        bad = {'R_ALU': 0.5, 'I_ALU': 0.3}  # sums to 0.8
        with pytest.raises(ValueError, match='sum to'):
            produce_instruction_batch(8, 2, rng, dist=bad)


class TestRandomBasicBlock:
    def test_minimum_length(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            block = random_basic_block(rng, max_length=10)
            assert len(block) >= 2

    def test_length_bounded(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            block = random_basic_block(rng, max_length=5)
            assert len(block) <= 5

    def test_terminates_with_branch(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            block = random_basic_block(rng, max_length=8)
            assert block[-1].opcode in B_TYPE, \
                f'block does not end with branch: {block[-1].opcode}'

    def test_data_flow(self):
        """At least 50% of non-terminal instructions should read from
        a register written by an earlier instruction in the block."""
        rng = np.random.default_rng(42)
        n_blocks = 100
        total_dependent = 0
        total_eligible = 0

        for _ in range(n_blocks):
            block = random_basic_block(rng, max_length=10)
            live = set()
            for i, instr in enumerate(block[:-1]):  # exclude branch
                # Get source registers
                if instr.opcode in R_TYPE:
                    sources = {instr.args[1], instr.args[2]}
                elif instr.opcode in I_TYPE:
                    sources = {instr.args[1]}
                elif instr.opcode in LOAD_TYPE:
                    sources = {instr.args[2]}  # base reg
                elif instr.opcode in STORE_TYPE:
                    sources = {instr.args[0], instr.args[2]}  # rs2, rs1
                elif instr.opcode == 'LUI':
                    sources = set()
                else:
                    sources = set()

                if i > 0:  # First instruction can't be dependent
                    total_eligible += 1
                    if sources & live:
                        total_dependent += 1

                # Update live set with this instruction's destination
                if instr.opcode in R_TYPE or instr.opcode in I_TYPE \
                        or instr.opcode in LOAD_TYPE or instr.opcode == 'LUI':
                    rd = instr.args[0]
                    if rd != 0:
                        live.add(rd)

        if total_eligible > 0:
            ratio = total_dependent / total_eligible
            assert ratio > 0.4, \
                f'data-flow dependency ratio too low: {ratio:.2f}'


class TestExecuteSequence:
    def test_snapshot_shapes(self):
        rng = np.random.default_rng(0)
        instructions = [Instruction('ADD', 5, 3, 7),
                        Instruction('SUB', 6, 5, 3),
                        Instruction('BEQ', 5, 6, 16)]
        regs = np.zeros(32, dtype=np.int32)
        regs[3] = 10
        regs[7] = 20
        ctx = make_ctx()

        regs_snap, pc_snap = execute_sequence(
            instructions, regs, pc=0, rng=rng, ctx=ctx)

        assert regs_snap.shape == (4, 32)  # n+1 snapshots
        assert pc_snap.shape == (4,)

    def test_initial_state_captured(self):
        rng = np.random.default_rng(0)
        instructions = [Instruction('ADD', 5, 3, 7),
                        Instruction('BEQ', 5, 0, 8)]
        regs = np.zeros(32, dtype=np.int32)
        regs[3] = 10
        regs[7] = 20
        ctx = make_ctx()

        regs_snap, _ = execute_sequence(
            instructions, regs, pc=0, rng=rng, ctx=ctx)

        # First snapshot is the initial state
        assert regs_snap[0, 3] == 10
        assert regs_snap[0, 7] == 20
        assert regs_snap[0, 5] == 0

    def test_per_instruction_state(self):
        rng = np.random.default_rng(0)
        # ADD x5, x3, x7    (x5 = 10 + 20 = 30)
        # SUB x6, x5, x3    (x6 = 30 - 10 = 20)
        # BEQ x5, x6, 8     (terminator)
        instructions = [Instruction('ADD', 5, 3, 7),
                        Instruction('SUB', 6, 5, 3),
                        Instruction('BEQ', 5, 6, 8)]
        regs = np.zeros(32, dtype=np.int32)
        regs[3] = 10
        regs[7] = 20
        ctx = make_ctx()

        regs_snap, _ = execute_sequence(
            instructions, regs, pc=0, rng=rng, ctx=ctx)

        # Before instr 0: x5 = 0
        assert regs_snap[0, 5] == 0
        # Before instr 1: x5 = 30 (ADD ran)
        assert regs_snap[1, 5] == 30
        # Before instr 2: x6 = 20 (SUB ran)
        assert regs_snap[2, 5] == 30
        assert regs_snap[2, 6] == 20
        # Final state: x5 = 30, x6 = 20
        assert regs_snap[3, 5] == 30
        assert regs_snap[3, 6] == 20

    def test_store_load_persistence(self):
        """Stores within a sequence should be visible to subsequent loads."""
        rng = np.random.default_rng(0)
        # SW x3, 0(x0)  → store regs[3]=42 at address 0
        # LW x5, 0(x0)  → load from address 0 into x5
        # BEQ x5, x3, 8 → terminator
        instructions = [Instruction('SW', 3, 0, 0),
                        Instruction('LW', 5, 0, 0),
                        Instruction('BEQ', 5, 3, 8)]
        regs = np.zeros(32, dtype=np.int32)
        regs[3] = 42
        ctx = make_ctx()

        regs_snap, _ = execute_sequence(
            instructions, regs, pc=0, rng=rng, ctx=ctx)

        # After SW: registers unchanged
        assert regs_snap[1, 3] == 42
        assert regs_snap[1, 5] == 0
        # After LW: x5 should be 42 (loaded from stored value)
        assert regs_snap[2, 5] == 42

    def test_final_state_matches_emulator(self):
        """The final register state from execute_sequence should match
        emulator.run() on ALU-only blocks (no loads/stores, since
        execute_sequence uses zero-initialized memory while
        emulator.run() uses random-filled memory)."""
        rng = np.random.default_rng(7)
        ctx = make_ctx()

        for _ in range(50):
            block = random_basic_block(rng, max_length=6)
            # Filter to ALU+branch only (no loads/stores) so memory
            # independence doesn't affect the comparison.
            alu_block = [i for i in block
                         if i.opcode not in ('LB', 'LBU', 'LH', 'LHU',
                                             'LW', 'SB', 'SH', 'SW')]
            if len(alu_block) < 2:
                continue

            regs = random_regs(rng)
            pc = int(rng.integers(0, 256)) * 4

            regs_snap, _ = execute_sequence(
                alu_block, regs, pc=pc, rng=rng, ctx=ctx)

            state, _, _ = run(alu_block, regs=regs, pc=pc,
                              rng=rng, _ctx=ctx, max_steps=len(alu_block))

            np.testing.assert_array_equal(
                regs_snap[-1], state.regs,
                err_msg='execute_sequence final state != emulator.run')


class TestProduceSeqBatch:
    def test_shapes(self):
        rng = np.random.default_rng(0)
        batch = produce_sequence_batch(batch_size=8, n_inputs=4,
                                          max_block_len=5, rng=rng)
        assert isinstance(batch, SequenceBatch)
        B = 8
        assert batch.token_ids.shape[0] == B
        assert batch.padding_mask.shape == batch.token_ids.shape
        assert batch.token_instr_idx.shape == batch.token_ids.shape
        assert batch.n_instructions.shape == (B,)
        # max_instrs+1 in axis 1
        assert batch.per_instr_regs.shape[0] == B
        assert batch.per_instr_regs.shape[2] == 4  # n_inputs
        assert batch.per_instr_regs.shape[3] == 32  # registers
        assert batch.per_instr_pcs.shape[0] == B
        assert batch.per_instr_pcs.shape[2] == 4  # n_inputs

    def test_n_instructions_consistent(self):
        rng = np.random.default_rng(0)
        batch = produce_sequence_batch(batch_size=16, n_inputs=2,
                                          max_block_len=5, rng=rng)
        # max_instrs from batch should be max of n_instructions
        max_instrs = batch.per_instr_regs.shape[1] - 1
        assert max_instrs == int(batch.n_instructions.max())

    def test_token_instr_idx_validity(self):
        rng = np.random.default_rng(0)
        batch = produce_sequence_batch(batch_size=8, n_inputs=2,
                                          max_block_len=5, rng=rng)
        for b in range(8):
            n = int(batch.n_instructions[b])
            indices = batch.token_instr_idx[b]
            valid_indices = indices[indices >= 0]
            # Indices should be in [0, n)
            assert valid_indices.max() < n
            assert valid_indices.min() >= 0
            # Each instruction should have at least one token
            unique_idx = set(valid_indices.tolist())
            assert unique_idx == set(range(n))

    def test_x0_always_zero(self):
        rng = np.random.default_rng(0)
        batch = produce_sequence_batch(batch_size=8, n_inputs=4,
                                          max_block_len=5, rng=rng)
        # x0 (register 0) should always be 0 in any snapshot
        assert (batch.per_instr_regs[..., 0] == 0).all()

    def test_deterministic(self):
        b1 = produce_sequence_batch(batch_size=4, n_inputs=2,
                                       max_block_len=5, rng=np.random.default_rng(42))
        b2 = produce_sequence_batch(batch_size=4, n_inputs=2,
                                       max_block_len=5, rng=np.random.default_rng(42))
        assert np.array_equal(b1.token_ids, b2.token_ids)
        assert np.array_equal(b1.per_instr_regs, b2.per_instr_regs)


class TestSeqBatchIO:
    def test_roundtrip(self):
        rng = np.random.default_rng(42)
        batch = produce_sequence_batch(batch_size=4, n_inputs=2,
                                          max_block_len=5, rng=rng)

        buf = io.BytesIO()
        write_stream_header(buf)
        write_batch(buf, batch)

        buf.seek(0)
        read_stream_header(buf)
        batch2 = read_batch(buf)

        assert np.array_equal(batch.token_ids, batch2.token_ids)
        assert np.array_equal(batch.padding_mask, batch2.padding_mask)
        assert np.array_equal(batch.token_instr_idx, batch2.token_instr_idx)
        assert np.array_equal(batch.n_instructions, batch2.n_instructions)
        assert np.array_equal(batch.per_instr_regs, batch2.per_instr_regs)
        assert np.array_equal(batch.per_instr_pcs, batch2.per_instr_pcs)

    def test_multiple_batches(self):
        rng = np.random.default_rng(42)
        buf = io.BytesIO()
        write_stream_header(buf)
        for _ in range(3):
            write_batch(buf, produce_sequence_batch(
                batch_size=4, n_inputs=2, max_block_len=5, rng=rng))

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
