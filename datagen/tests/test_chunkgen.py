"""Tests for the T2 chunker (datagen/chunkgen.py).

Verifies boundary detection, terminator classification, length cap
behavior, register-delta correctness, invalidity augmentation, and
RVC binary I/O round-trips.
"""

import io
import numpy as np
import pytest

from datagen.seqgen import (
    SequenceBatch, produce_sequence_batch, _encode_with_instr_idx,
    execute_sequence,
)
from datagen.chunkgen import (
    chunk_rvs, augment_chunkbatch_with_invalid,
    write_stream_header, read_stream_header, write_batch, read_batch,
    classify_opcode_token,
    TYPE_NON_TERMINATOR, TYPE_LOAD, TYPE_STORE, TYPE_BRANCH,
    TYPE_JUMP, TYPE_CAPPED, TYPE_TAIL,
    INVALID_SPANNING, INVALID_MULTI, INVALID_OVERLONG,
)
from emulator import Instruction, make_ctx, random_regs
from tokenizer import PAD


def _hand_rvs(sequences, n_inputs=2):
    """Build an RVS batch from a list of instruction lists."""
    rng = np.random.default_rng(0)
    encoded = [_encode_with_instr_idx(instrs) for instrs in sequences]
    max_tokens = max(len(toks) for toks, _ in encoded)
    max_instrs = max(len(s) for s in sequences) if sequences else 0
    B = len(sequences)

    token_ids = np.full((B, max_tokens), PAD, dtype=np.int64)
    padding_mask = np.ones((B, max_tokens), dtype=np.bool_)
    token_instr_idx = np.full((B, max_tokens), -1, dtype=np.int32)
    n_instructions = np.array([len(s) for s in sequences], dtype=np.int32)

    for b, (toks, idx) in enumerate(encoded):
        n = len(toks)
        token_ids[b, :n] = toks
        padding_mask[b, :n] = False
        token_instr_idx[b, :n] = idx

    per_instr_regs = np.zeros((B, max_instrs + 1, n_inputs, 32),
                              dtype=np.int32)
    per_instr_pcs = np.zeros((B, max_instrs + 1, n_inputs),
                             dtype=np.int64)
    ctx = make_ctx()
    for s in range(n_inputs):
        regs = random_regs(rng)
        pc = int(rng.integers(0, 1024)) * 4
        for b, instrs in enumerate(sequences):
            regs_snap, pc_snap = execute_sequence(instrs, regs, pc, rng, ctx)
            n = len(instrs)
            per_instr_regs[b, :n + 1, s, :] = regs_snap
            per_instr_pcs[b, :n + 1, s] = pc_snap

    return SequenceBatch(
        token_ids=token_ids,
        padding_mask=padding_mask,
        token_instr_idx=token_instr_idx,
        n_instructions=n_instructions,
        per_instr_regs=per_instr_regs,
        per_instr_pcs=per_instr_pcs,
    )


# ---------------------------------------------------------------------------
# Boundary detection and basic chunk properties
# ---------------------------------------------------------------------------

def test_total_instructions_preserved():
    """Total instructions across chunks equals total in source sequences."""
    rng = np.random.default_rng(42)
    rvs = produce_sequence_batch(8, n_inputs=2, max_block_len=10, rng=rng)
    cb = chunk_rvs(rvs, max_chunk_len=16)
    expected_total = int(rvs.n_instructions.sum())
    got_total = int(cb.chunk_lens.sum())
    assert got_total == expected_total, (
        f'expected {expected_total} total instrs across chunks, got {got_total}')


def test_chunks_within_cap():
    """All chunk lengths are within [1, max_chunk_len]."""
    rng = np.random.default_rng(44)
    rvs = produce_sequence_batch(8, n_inputs=2, max_block_len=10, rng=rng)
    for cap in (4, 8, 16):
        cb = chunk_rvs(rvs, max_chunk_len=cap)
        for c in range(cb.token_ids.shape[0]):
            L = int(cb.chunk_lens[c])
            assert 1 <= L <= cap, f'cap={cap}, chunk {c} len={L} out of range'


def test_terminator_classification_per_kind():
    """Each terminator opcode kind produces the right chunk_type."""
    seqs = [
        [Instruction('ADD', 1, 1, 1), Instruction('LW',  2, 0, 1)],   # LOAD
        [Instruction('ADD', 1, 1, 1), Instruction('SW',  2, 0, 1)],   # STORE
        [Instruction('ADD', 1, 1, 1), Instruction('BEQ', 1, 2, 8)],   # BRANCH
        [Instruction('ADD', 1, 1, 1), Instruction('JAL', 2, 8)],      # JUMP
        [Instruction('ADD', 1, 1, 1), Instruction('JALR', 2, 1, 0)],  # JUMP
    ]
    expected = [TYPE_LOAD, TYPE_STORE, TYPE_BRANCH, TYPE_JUMP, TYPE_JUMP]
    rvs = _hand_rvs(seqs)
    cb = chunk_rvs(rvs, max_chunk_len=16)
    # One chunk per sequence (all terminate at instruction 1).
    assert cb.token_ids.shape[0] == len(seqs)
    # Chunks for sequences are emitted in source order.
    for b in range(len(seqs)):
        assert int(cb.chunk_type[b]) == expected[b], (
            f'seq {b}: chunk_type={int(cb.chunk_type[b])}, expected {expected[b]}')


def test_capped_chunk_long_alu_run():
    """A long pure-ALU sequence with cap=4 produces a capped + a tail."""
    instrs = [Instruction('ADD', i + 1, i + 1, i + 1) for i in range(7)]
    rvs = _hand_rvs([instrs])
    cb = chunk_rvs(rvs, max_chunk_len=4)
    # Should be 2 chunks: [0:4] capped, [4:7] tail.
    assert cb.token_ids.shape[0] == 2
    assert int(cb.chunk_lens[0]) == 4
    assert int(cb.chunk_lens[1]) == 3
    assert int(cb.chunk_type[0]) == TYPE_CAPPED
    assert int(cb.chunk_type[1]) == TYPE_TAIL


def test_tail_chunk():
    """Sequence ending in non-terminator gets a TAIL-typed final chunk."""
    instrs = [Instruction('ADD', 1, 1, 1), Instruction('ADD', 2, 2, 2),
              Instruction('LW', 3, 0, 1), Instruction('ADD', 4, 4, 4)]
    rvs = _hand_rvs([instrs])
    cb = chunk_rvs(rvs, max_chunk_len=16)
    # First chunk: [0:3] LOAD (ADD, ADD, LW). Second: [3:4] TAIL (ADD).
    assert cb.token_ids.shape[0] == 2
    assert int(cb.chunk_type[0]) == TYPE_LOAD
    assert int(cb.chunk_type[1]) == TYPE_TAIL
    assert int(cb.chunk_lens[0]) == 3
    assert int(cb.chunk_lens[1]) == 1


def test_single_terminator_only():
    """A sequence of just one branch: one chunk of length 1."""
    instrs = [Instruction('BEQ', 1, 2, 8)]
    rvs = _hand_rvs([instrs])
    cb = chunk_rvs(rvs, max_chunk_len=16)
    assert cb.token_ids.shape[0] == 1
    assert int(cb.chunk_lens[0]) == 1
    assert int(cb.chunk_type[0]) == TYPE_BRANCH


def test_reg_delta_matches_for_full_sequence():
    """For a 1-chunk sequence, reg_delta equals regs_after - regs_before."""
    instrs = [Instruction('ADD', 1, 2, 3), Instruction('SUB', 4, 1, 2),
              Instruction('BEQ', 1, 4, 8)]  # 3 instructions terminating in branch
    rvs = _hand_rvs([instrs])
    cb = chunk_rvs(rvs, max_chunk_len=16)
    # Single chunk covering all 3 instructions.
    assert cb.token_ids.shape[0] == 1
    expected = (rvs.per_instr_regs[0, 3, :, :]
                - rvs.per_instr_regs[0, 0, :, :])
    assert np.array_equal(cb.reg_delta[0], expected)


def test_per_instruction_tokens_populated():
    """Each instruction's token slot is populated correctly."""
    instrs = [Instruction('ADD', 1, 2, 3), Instruction('LW', 4, 0, 1)]
    rvs = _hand_rvs([instrs])
    cb = chunk_rvs(rvs, max_chunk_len=16)
    assert cb.token_ids.shape[0] == 1
    # First instruction (ADD): 4 tokens. Second (LW): 6 tokens.
    instr_lens = (~cb.instr_pad[0]).sum(axis=1)
    assert int(instr_lens[0]) == 4
    assert int(instr_lens[1]) == 6


def test_classify_opcode_helper():
    """classify_opcode_token maps each opcode type correctly."""
    from tokenizer.tokenizer import _OP_TO_TOKEN
    assert classify_opcode_token(_OP_TO_TOKEN['ADD']) == TYPE_NON_TERMINATOR
    assert classify_opcode_token(_OP_TO_TOKEN['LW']) == TYPE_LOAD
    assert classify_opcode_token(_OP_TO_TOKEN['SW']) == TYPE_STORE
    assert classify_opcode_token(_OP_TO_TOKEN['BEQ']) == TYPE_BRANCH
    assert classify_opcode_token(_OP_TO_TOKEN['JAL']) == TYPE_JUMP
    assert classify_opcode_token(_OP_TO_TOKEN['JALR']) == TYPE_JUMP


def test_valid_mask_all_true_pre_augment():
    """chunk_rvs produces only valid chunks; valid_mask is all True."""
    rng = np.random.default_rng(48)
    rvs = produce_sequence_batch(8, n_inputs=2, max_block_len=10, rng=rng)
    cb = chunk_rvs(rvs, max_chunk_len=16)
    assert cb.valid_mask.all()


# ---------------------------------------------------------------------------
# Invalid-chunk augmentation
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_cb():
    rng = np.random.default_rng(100)
    rvs = produce_sequence_batch(16, n_inputs=2, max_block_len=10, rng=rng)
    return chunk_rvs(rvs, max_chunk_len=16)


def test_augment_invalidity_rate(valid_cb):
    rng = np.random.default_rng(200)
    n_valid = valid_cb.token_ids.shape[0]
    aug = augment_chunkbatch_with_invalid(
        valid_cb, invalidity_rate=0.2, rng=rng)
    n_total = aug.token_ids.shape[0]
    n_invalid = int((~aug.valid_mask).sum())
    assert abs(n_invalid / n_total - 0.2) < 0.05
    assert int(aug.valid_mask.sum()) == n_valid


def test_augment_zero_rate_is_passthrough(valid_cb):
    aug = augment_chunkbatch_with_invalid(
        valid_cb, invalidity_rate=0.0,
        storage_max_chunk_len=valid_cb.token_ids.shape[1])
    assert aug.token_ids.shape[0] == valid_cb.token_ids.shape[0]
    assert aug.valid_mask.all()


def test_augment_storage_padding(valid_cb):
    rng = np.random.default_rng(201)
    aug = augment_chunkbatch_with_invalid(
        valid_cb, invalidity_rate=0.2, storage_max_chunk_len=24, rng=rng)
    assert aug.token_ids.shape[1] == 24
    assert aug.instr_pad.shape[1] == 24


def test_augment_storage_too_small_raises(valid_cb):
    with pytest.raises(ValueError, match='input padding'):
        augment_chunkbatch_with_invalid(
            valid_cb, invalidity_rate=0.2, storage_max_chunk_len=8)


def test_augment_invalid_type_codes(valid_cb):
    rng = np.random.default_rng(202)
    aug = augment_chunkbatch_with_invalid(
        valid_cb, invalidity_rate=0.5, rng=rng)
    invalid_indices = np.flatnonzero(~aug.valid_mask)
    valid_codes = {INVALID_SPANNING, INVALID_MULTI, INVALID_OVERLONG}
    for i in invalid_indices:
        assert int(aug.chunk_type[i]) in valid_codes


def test_augment_invalid_lens_in_range(valid_cb):
    rng = np.random.default_rng(203)
    storage = 24
    aug = augment_chunkbatch_with_invalid(
        valid_cb, invalidity_rate=0.3,
        storage_max_chunk_len=storage, rng=rng)
    invalid_indices = np.flatnonzero(~aug.valid_mask)
    for i in invalid_indices:
        L = int(aug.chunk_lens[i])
        assert 1 <= L <= storage


def test_augment_overlong_actually_overlong(valid_cb):
    rng = np.random.default_rng(204)
    aug = augment_chunkbatch_with_invalid(
        valid_cb, invalidity_rate=0.3,
        type_weights={'overlong': 1.0},
        storage_max_chunk_len=24, rng=rng)
    invalid_indices = np.flatnonzero(~aug.valid_mask)
    overlong_count = 0
    for i in invalid_indices:
        if int(aug.chunk_type[i]) == INVALID_OVERLONG:
            assert int(aug.chunk_lens[i]) > 16
            overlong_count += 1
    assert overlong_count > 0


def test_augment_invalid_reg_delta_zero(valid_cb):
    rng = np.random.default_rng(206)
    aug = augment_chunkbatch_with_invalid(
        valid_cb, invalidity_rate=0.3, rng=rng)
    invalid_indices = np.flatnonzero(~aug.valid_mask)
    for i in invalid_indices:
        assert (aug.reg_delta[i] == 0).all()


def test_augment_valid_unchanged(valid_cb):
    """Valid chunks at the front of the augmented batch are unchanged."""
    rng = np.random.default_rng(208)
    aug = augment_chunkbatch_with_invalid(
        valid_cb, invalidity_rate=0.3,
        storage_max_chunk_len=valid_cb.token_ids.shape[1], rng=rng)
    n_valid = valid_cb.token_ids.shape[0]
    assert np.array_equal(aug.token_ids[:n_valid], valid_cb.token_ids)
    assert np.array_equal(aug.chunk_lens[:n_valid], valid_cb.chunk_lens)
    assert aug.valid_mask[:n_valid].all()


# ---------------------------------------------------------------------------
# Binary I/O round-trip
# ---------------------------------------------------------------------------

def test_binary_round_trip(valid_cb):
    rng = np.random.default_rng(300)
    aug = augment_chunkbatch_with_invalid(
        valid_cb, invalidity_rate=0.2, storage_max_chunk_len=24, rng=rng)
    buf = io.BytesIO()
    write_stream_header(buf)
    write_batch(buf, aug)
    buf.seek(0)
    read_stream_header(buf)
    got = read_batch(buf)
    assert np.array_equal(aug.token_ids, got.token_ids)
    assert np.array_equal(aug.instr_pad, got.instr_pad)
    assert np.array_equal(aug.chunk_lens, got.chunk_lens)
    assert np.array_equal(aug.valid_mask, got.valid_mask)
    assert np.array_equal(aug.chunk_type, got.chunk_type)
    assert np.array_equal(aug.reg_delta, got.reg_delta)


def test_binary_eof():
    """Reading past end returns None cleanly."""
    assert read_batch(io.BytesIO()) is None


def test_binary_bad_magic():
    buf = io.BytesIO(b'XXXX\x01' + b'\x00' * 6)
    with pytest.raises(ValueError, match='Bad magic'):
        read_stream_header(buf)
