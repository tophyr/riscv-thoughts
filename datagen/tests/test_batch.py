"""Tests for datagen.batch — Chunk + Batch containers, the
BinaryFormat machinery (RVT), pack helpers, and twin construction."""

import io

import numpy as np
import pytest

from emulator import Instruction
from datagen.batch import (
    Chunk, Batch, RVT_FORMAT, BinaryFormat, BodyField,
    pack_batch, padding_mask,
    generate_chunks, collect_into_batches, build_twins,
    build_row_outputs, RowOutputsPayload, AuxPayload,
    _make_valid_chunk, _make_invalid_chunk,
)
from datagen.compare import (
    make_anchor_states, precompute_chunk, precompute_row_outputs,
    AUX_CE_IGNORE, N_REGS,
)
from datagen.generate import length_cap, single, until_branch
from tokenizer.tokenizer import MAX_INSTR_TOKENS


def _instrs_to_chunk(instrs):
    return _make_valid_chunk(instrs)


def _pack(chunks, max_chunk_len=4):
    """Pack to a fixed shape the way production does — targets derived
    from a chunking-rule length cap (collect_into_batches' formula)."""
    return pack_batch(
        list(chunks),
        target_B=len(chunks),
        target_max_tokens=max_chunk_len * MAX_INSTR_TOKENS,
        target_max_n_instrs=max_chunk_len)


# ---------------------------------------------------------------------------
# pack_batch + padding_mask
# ---------------------------------------------------------------------------

def test_pack_basic():
    chunks = [
        _instrs_to_chunk([Instruction('ADDI', 1, 0, 5),
                          Instruction('ADD', 2, 1, 1)]),
        _instrs_to_chunk([Instruction('XOR', 3, 0, 0)]),
    ]
    batch = _pack(chunks)
    assert batch.tokens.shape[0] == 2
    assert batch.token_lens[0] > batch.token_lens[1]
    assert batch.valid.tolist() == [True, True]
    # No row-outputs payload supplied → row-* fields empty.
    assert batch.row_outputs.shape[0] == 0


def test_pack_rejects_empty():
    with pytest.raises(ValueError):
        _pack([])


def test_padding_mask():
    chunks = [
        _instrs_to_chunk([Instruction('ADDI', 1, 0, 5)]),
        _instrs_to_chunk([Instruction('ADD', 2, 3, 4),
                          Instruction('SUB', 5, 6, 7)]),
    ]
    batch = _pack(chunks)
    pm = padding_mask(batch)
    assert pm.shape == batch.tokens.shape
    # Row 0 is shorter; should have True (padded) past its token_len.
    n0 = int(batch.token_lens[0])
    assert not pm[0, :n0].any()
    assert pm[0, n0:].all()


def test_invalid_chunk_carries_no_instructions():
    ch = _make_invalid_chunk([5, 6, 7, 8])
    assert ch.valid is False
    assert ch.instructions is None
    assert ch.tokens == [5, 6, 7, 8]


def test_instr_lens_populated_for_valid_chunks():
    """Each row's instr_lens reflects per-instruction token counts;
    invalid rows are all-zero."""
    chunks = [
        _instrs_to_chunk([Instruction('ADDI', 1, 0, 5)]),               # 1 instr
        _instrs_to_chunk([Instruction('ADD', 2, 1, 1),
                          Instruction('SUB', 3, 1, 2)]),                # 2 instrs
        _make_invalid_chunk([5, 6, 7]),
    ]
    batch = _pack(chunks)
    # Row 0: one instruction, length matches its token count.
    assert batch.instr_lens[0, 0] == int(batch.token_lens[0])
    # Row 1: two instructions, sum matches token_lens.
    assert batch.instr_lens[1, 0] + batch.instr_lens[1, 1] == int(batch.token_lens[1])
    # Row 1: only first two slots populated.
    assert (batch.instr_lens[1, 2:] == 0).all()
    # Row 2: invalid, all-zero instr_lens.
    assert (batch.instr_lens[2] == 0).all()


# ---------------------------------------------------------------------------
# RVT binary I/O
# ---------------------------------------------------------------------------

def _sample_batch():
    rng = np.random.default_rng(0)
    anchors = make_anchor_states(4, seed=0)
    gen = collect_into_batches(
        generate_chunks(length_cap(2), rng),
        batch_size=8, twins=1,
        anchor_states=anchors, rng=rng,
        max_chunk_len=2,
    )
    return next(gen)


def test_round_trip():
    b = _sample_batch()
    buf = io.BytesIO()
    RVT_FORMAT.write_batch(buf, b)
    buf.seek(0)
    got = RVT_FORMAT.read_batch(buf, Batch)
    assert np.array_equal(b.tokens, got.tokens)
    assert np.array_equal(b.token_lens, got.token_lens)
    assert np.array_equal(b.valid, got.valid)
    assert np.array_equal(b.instr_lens, got.instr_lens)
    assert np.array_equal(b.in_slot_regs, got.in_slot_regs)
    assert np.array_equal(b.out_slot_regs, got.out_slot_regs)


def test_eof_clean():
    assert RVT_FORMAT.read_batch(io.BytesIO(b''), Batch) is None


def test_pass_through_bytes():
    b = _sample_batch()
    buf = io.BytesIO()
    RVT_FORMAT.write_batch(buf, b)
    written = buf.getvalue()
    buf.seek(0)
    assert RVT_FORMAT.read_batch_bytes(buf) == written


def test_reader_concatenates():
    # No stream header: batches are self-describing, so writing them
    # back-to-back == cat-ing files; the reader yields all of them.
    rng = np.random.default_rng(0)
    anchors = make_anchor_states(4, seed=0)
    gen = collect_into_batches(
        generate_chunks(length_cap(2), rng),
        batch_size=8, twins=1,
        anchor_states=anchors, rng=rng,
        max_chunk_len=2,
    )
    b1, b2 = next(gen), next(gen)
    buf = io.BytesIO()
    RVT_FORMAT.write_batch(buf, b1)
    RVT_FORMAT.write_batch(buf, b2)
    buf.seek(0)
    assert len(list(RVT_FORMAT.reader(buf, Batch))) == 2


def test_no_row_outputs_round_trip():
    """RB=0 case: pack a batch with no row-outputs payload and verify it
    serializes (the row-* fields are empty)."""
    chunks = [_instrs_to_chunk([Instruction('ADD', 1, 2, 3)])]
    b = _pack(chunks)
    buf = io.BytesIO()
    RVT_FORMAT.write_batch(buf, b)
    buf.seek(0)
    got = RVT_FORMAT.read_batch(buf, Batch)
    assert got.row_outputs.shape == (0, 0)
    assert got.row_has_rd.shape == (0,)
    assert got.pair_valid.shape == (0,)


def test_bad_magic_raises():
    b = _sample_batch()
    buf = io.BytesIO()
    RVT_FORMAT.write_batch(buf, b)
    bad = io.BytesIO(b'XXXX' + buf.getvalue()[4:])
    with pytest.raises(ValueError, match='Bad magic'):
        RVT_FORMAT.read_batch(bad, Batch)


# ---------------------------------------------------------------------------
# Wire-format failure modes — truncation, version, and dtype corruption.
# Each batch is self-describing: prefix (4 magic + 1 version + N dtype chars)
# then a uint32 dimension header then the raw body.
# ---------------------------------------------------------------------------

def _serialized_sample():
    b = _sample_batch()
    buf = io.BytesIO()
    RVT_FORMAT.write_batch(buf, b)
    return buf.getvalue()


def test_truncated_after_partial_prefix_raises():
    data = _serialized_sample()
    # A nonzero-but-short read of the prefix is a torn batch, not clean EOF.
    truncated = data[:RVT_FORMAT.batch_prefix_size - 1]
    with pytest.raises(EOFError, match='prefix'):
        RVT_FORMAT.read_batch(io.BytesIO(truncated), Batch)


def test_truncated_after_prefix_raises():
    data = _serialized_sample()
    # Full prefix present, header missing entirely → header read comes up short.
    truncated = data[:RVT_FORMAT.batch_prefix_size]
    with pytest.raises(EOFError, match='header'):
        RVT_FORMAT.read_batch(io.BytesIO(truncated), Batch)


def test_truncated_after_partial_header_raises():
    data = _serialized_sample()
    prefix_n = RVT_FORMAT.batch_prefix_size
    header_n = RVT_FORMAT.batch_header.size
    truncated = data[:prefix_n + header_n - 1]
    with pytest.raises(EOFError, match='header'):
        RVT_FORMAT.read_batch(io.BytesIO(truncated), Batch)


def test_truncated_after_partial_body_raises():
    data = _serialized_sample()
    prefix_n = RVT_FORMAT.batch_prefix_size
    header_n = RVT_FORMAT.batch_header.size
    # Prefix + header intact, body cut one byte short.
    truncated = data[:-1]
    assert len(truncated) > prefix_n + header_n
    with pytest.raises(EOFError, match='body'):
        RVT_FORMAT.read_batch(io.BytesIO(truncated), Batch)


def test_truncated_body_passthrough_raises():
    """read_batch_bytes (the cat/pass-through path) also rejects a torn body."""
    data = _serialized_sample()
    with pytest.raises(EOFError, match='body'):
        RVT_FORMAT.read_batch_bytes(io.BytesIO(data[:-1]))


def test_bad_version_raises():
    data = bytearray(_serialized_sample())
    # Version byte sits immediately after the 4-byte magic.
    assert data[4] == RVT_FORMAT.version
    data[4] = RVT_FORMAT.version + 1
    with pytest.raises(ValueError, match='version'):
        RVT_FORMAT.read_batch(io.BytesIO(bytes(data)), Batch)


def test_corrupt_dtype_signature_raises():
    data = bytearray(_serialized_sample())
    # Dtype-signature chars follow magic(4) + version(1).
    sig_start = 5
    data[sig_start] ^= 0xFF
    with pytest.raises(ValueError, match='[Dd]type'):
        RVT_FORMAT.read_batch(io.BytesIO(bytes(data)), Batch)


# ---------------------------------------------------------------------------
# pack_batch overflow + payload-shape checks
# ---------------------------------------------------------------------------

def test_pack_rejects_too_many_chunks():
    chunks = [_instrs_to_chunk([Instruction('ADD', 1, 2, 3)]) for _ in range(3)]
    with pytest.raises(ValueError, match='exceeds target_B'):
        pack_batch(chunks, target_B=2,
                   target_max_tokens=4 * MAX_INSTR_TOKENS,
                   target_max_n_instrs=4)


def test_pack_rejects_token_overflow():
    chunks = [_instrs_to_chunk([Instruction('ADDI', 1, 0, 5),
                                Instruction('ADD', 2, 1, 1)])]
    # target_max_tokens=1 is smaller than this chunk's token count.
    with pytest.raises(ValueError, match='max_tokens'):
        pack_batch(chunks, target_B=1, target_max_tokens=1,
                   target_max_n_instrs=4)


def test_pack_rejects_instr_count_overflow():
    chunks = [_instrs_to_chunk([Instruction('ADD', 1, 2, 3),
                                Instruction('SUB', 4, 5, 6),
                                Instruction('XOR', 7, 8, 9)])]
    with pytest.raises(ValueError, match='max_n_instrs'):
        pack_batch(chunks, target_B=1,
                   target_max_tokens=8 * MAX_INSTR_TOKENS,
                   target_max_n_instrs=2)


def test_pack_rejects_aux_first_dim_mismatch():
    chunks = [_instrs_to_chunk([Instruction('ADD', 1, 2, 3)])]
    # AuxPayload sized for 2 rows but only 1 chunk.
    bad_aux = AuxPayload(
        live_in_mask=np.zeros((2, N_REGS), dtype=bool),
        live_out_mask=np.zeros((2, N_REGS), dtype=bool),
        pc_writes=np.zeros((2,), dtype=bool),
        in_slot_regs=np.full((2, 32), AUX_CE_IGNORE, dtype=np.int8),
        out_slot_regs=np.full((2, 16), AUX_CE_IGNORE, dtype=np.int8),
    )
    with pytest.raises(ValueError, match='aux_payload first dim'):
        pack_batch(chunks, target_B=1,
                   target_max_tokens=4 * MAX_INSTR_TOKENS,
                   target_max_n_instrs=4, aux_payload=bad_aux)


def test_pack_rejects_row_outputs_first_dim_mismatch():
    chunks = [_instrs_to_chunk([Instruction('ADD', 1, 2, 3)])]
    bad_payload = RowOutputsPayload(
        row_outputs=np.zeros((2, 4), dtype=np.float64),  # 2 rows, 1 chunk
        has_rd=np.zeros((2,), dtype=bool),
        pair_valid=np.zeros((2,), dtype=bool),
    )
    with pytest.raises(ValueError, match='row_outputs'):
        pack_batch(chunks, target_B=2,
                   target_max_tokens=4 * MAX_INSTR_TOKENS,
                   target_max_n_instrs=4,
                   row_outputs_payload=bad_payload)


# ---------------------------------------------------------------------------
# pack_batch row-outputs branch — real RowOutputsPayload via build_row_outputs
# ---------------------------------------------------------------------------

def test_pack_row_outputs_branch_populates_fields():
    anchors = make_anchor_states(4, seed=0)
    rng = np.random.default_rng(0)
    # One ALU source (writes x5) + one x0-write NOP (no rd).
    chunks = [
        _instrs_to_chunk([Instruction('ADD', 5, 1, 2)]),
        _instrs_to_chunk([Instruction('ADD', 0, 1, 2)]),
    ]
    chunks_out, payload, aux = build_row_outputs(
        chunks, twins=0, anchor_states=anchors, rng=rng)
    target_B = len(chunks_out) + 2  # leave padding rows
    batch = pack_batch(
        chunks_out, target_B=target_B,
        target_max_tokens=MAX_INSTR_TOKENS,
        target_max_n_instrs=1,
        row_outputs_payload=payload, aux_payload=aux)

    # RB == B (row-outputs mode pads to target_B).
    assert batch.row_outputs.shape == (target_B, 4)
    assert batch.row_has_rd.shape == (target_B,)
    assert batch.pair_valid.shape == (target_B,)
    # First two rows are real single-instr ALU → pair_valid True; the
    # ADD x5 writes a real reg (has_rd), the ADD x0 does not.
    assert batch.pair_valid[0] and batch.pair_valid[1]
    assert batch.row_has_rd[0] and not batch.row_has_rd[1]
    # Padding rows past actual_B: pair_valid False.
    assert not batch.pair_valid[len(chunks_out):].any()


def test_pack_row_outputs_masks_invalid_rows():
    anchors = make_anchor_states(4, seed=0)
    rng = np.random.default_rng(0)
    chunks = [
        _instrs_to_chunk([Instruction('ADD', 5, 1, 2)]),
        _make_invalid_chunk([5, 6, 7]),         # invalid window → not meaningful
        _instrs_to_chunk([Instruction('LW', 5, 0, 1)]),  # mem-op → not meaningful
    ]
    chunks_out, payload, aux = build_row_outputs(
        chunks, twins=0, anchor_states=anchors, rng=rng)
    batch = pack_batch(
        chunks_out, target_B=len(chunks_out),
        target_max_tokens=MAX_INSTR_TOKENS,
        target_max_n_instrs=1,
        row_outputs_payload=payload, aux_payload=aux)
    # Only the first row is a behaviorally-meaningful single ALU instr.
    assert batch.pair_valid.tolist() == [True, False, False]
    # The masked rows carry a zero row-outputs vector.
    assert (batch.row_outputs[1] == 0).all()
    assert (batch.row_outputs[2] == 0).all()


def test_recognized_by_streamfmt():
    from scripts._streamfmt import detect_format, RVT
    b = _sample_batch()
    buf = io.BytesIO()
    RVT_FORMAT.write_batch(buf, b)
    # detect_format peeks, so wrap in a buffered (peekable) reader, as the
    # real tools' open(...,'rb')/stdin.buffer streams already are.
    f = io.BufferedReader(io.BytesIO(buf.getvalue()))
    assert detect_format(f) is RVT


# ---------------------------------------------------------------------------
# build_twins
# ---------------------------------------------------------------------------

def test_build_twins_builds_twin_clusters():
    rng = np.random.default_rng(0)
    anchors = make_anchor_states(4, seed=0)
    chunks = [
        _instrs_to_chunk([Instruction('ADDI', 1, 0, 5),
                          Instruction('ADD', 2, 1, 1)]),
        _instrs_to_chunk([Instruction('XOR', 3, 1, 1)]),
        _instrs_to_chunk([Instruction('OR', 4, 1, 2)]),
    ]
    out, _aux, _or, _orv = build_twins(
        chunks, twins=2, anchor_states=anchors, rng=rng)
    # Each source plus its 2 twins forms a cluster of 3 → 3*3 rows.
    assert len(out) == 3 * 3


def test_build_twins_aux_content_and_none_rows():
    """build_twins / _aux_from_precomputeds produce correct live masks +
    slot regs for a known instruction, and zero-mask / AUX_CE_IGNORE rows
    for chunks without a usable Precomputed (mem-op, invalid)."""
    rng = np.random.default_rng(0)
    anchors = make_anchor_states(4, seed=0)
    chunks = [
        _instrs_to_chunk([Instruction('ADD', 5, 1, 2)]),   # row 0: real aux
        _make_invalid_chunk([5, 6, 7]),                    # invalid → None row
        _instrs_to_chunk([Instruction('LW', 9, 0, 3)]),    # mem-op → None row
    ]
    out, aux, _or, _orv = build_twins(chunks, twins=0, anchor_states=anchors, rng=rng)
    assert len(out) == 3  # twins=0 → no growth

    # Row 0: ADD x5,x1,x2.
    assert np.nonzero(aux.live_in_mask[0])[0].tolist() == [1, 2]
    assert np.nonzero(aux.live_out_mask[0])[0].tolist() == [5]
    assert aux.in_slot_regs[0, 0] == 1
    assert aux.in_slot_regs[0, 1] == 2
    assert (aux.in_slot_regs[0, 2:] == AUX_CE_IGNORE).all()
    assert aux.out_slot_regs[0, 0] == 5
    assert (aux.out_slot_regs[0, 1:] == AUX_CE_IGNORE).all()
    assert not aux.pc_writes[0]

    # Rows 1 (invalid) and 2 (mem-op): zero masks, all-ignore slots.
    for r in (1, 2):
        assert not aux.live_in_mask[r].any()
        assert not aux.live_out_mask[r].any()
        assert (aux.in_slot_regs[r] == AUX_CE_IGNORE).all()
        assert (aux.out_slot_regs[r] == AUX_CE_IGNORE).all()


def test_build_row_outputs_pair_valid_and_zero_rows():
    """build_row_outputs marks single-ALU rows pair_valid and zero-fills /
    masks mem-op and multi-instruction rows (which get no value target)."""
    anchors = make_anchor_states(4, seed=0)
    rng = np.random.default_rng(0)
    chunks = [
        _instrs_to_chunk([Instruction('ADD', 5, 1, 2)]),         # single ALU
        _instrs_to_chunk([Instruction('LW', 6, 0, 3)]),          # mem-op
        _instrs_to_chunk([Instruction('ADD', 7, 1, 2),
                          Instruction('SUB', 8, 7, 1)]),         # multi-instr
    ]
    out, payload, aux = build_row_outputs(
        chunks, twins=0, anchor_states=anchors, rng=rng)
    assert len(out) == 3
    assert payload.pair_valid[0] and payload.has_rd[0]
    # Rows 1 (mem-op) and 2 (multi-instr): zero row, pair_valid False.
    assert not payload.pair_valid[1]
    assert not payload.pair_valid[2]
    assert (payload.row_outputs[1] == 0).all()
    assert (payload.row_outputs[2] == 0).all()


def test_build_row_outputs_live_label_matches_emulator():
    """build_row_outputs ships, per row, the destination register's value
    under each anchor — the LIVE T1 value-prediction target. Use an
    instruction whose inputs are NOT already at the canonical slots (x5,x9)
    so a regression to the canonical-baseline bug would be visible (ADD
    x5,x1,x2 would pass by coincidence)."""
    from emulator import run, make_ctx
    anchors = make_anchor_states(4, seed=3)
    rng = np.random.default_rng(0)
    instr = Instruction('ADD', 7, 5, 9)
    out, payload, aux = build_row_outputs(
        [_instrs_to_chunk([instr])], twins=0,
        anchor_states=anchors, rng=rng)
    ctx = make_ctx()
    for i in range(4):
        st, _, _ = run([instr], regs=anchors[i].copy(), pc=0,
                       _ctx=ctx, max_steps=1)
        assert int(payload.row_outputs[0, i]) == int(st.regs[7])


def test_build_twins_drops_memory_op_chunks():
    rng = np.random.default_rng(0)
    anchors = make_anchor_states(4, seed=0)
    chunks = [
        _instrs_to_chunk([Instruction('LW', 1, 0, 0)]),
        _instrs_to_chunk([Instruction('ADD', 2, 3, 4)]),
        _instrs_to_chunk([Instruction('SUB', 5, 6, 7)]),
    ]
    out, _aux, _or, _orv = build_twins(
        chunks, twins=1, anchor_states=anchors, rng=rng)
    # The mem-op chunk gets no twins.
    # 1 mem-op + 2 sources × 2 cluster_size = 1 + 4 = 5.
    assert len(out) == 5


# ---------------------------------------------------------------------------
# Fixed-shape (allocator-fragmentation) invariant on collect_into_batches.
# Every batch from one configured generator MUST share an identical shape so
# PyTorch's caching allocator doesn't fragment over a long run.
# ---------------------------------------------------------------------------

def _drain(gen, n):
    return [next(gen) for _ in range(n)]


def test_fixed_shape_with_twins_and_invalids():
    rng = np.random.default_rng(0)
    anchors = make_anchor_states(4, seed=0)
    invalid_rng = np.random.default_rng(99)

    def invalid_provider():
        # Variable-length bogus windows — would yield ragged shapes if the
        # padding-to-target guarantee weren't enforced.
        n = int(invalid_rng.integers(2, 6))
        return invalid_rng.integers(4, 20, size=n).tolist()

    gen = collect_into_batches(
        generate_chunks(length_cap(3), rng),
        batch_size=8, twins=2,
        anchor_states=anchors, rng=rng,
        invalid_rate=0.25, invalid_provider=invalid_provider,
        max_invalid_window=20, max_chunk_len=3)
    batches = _drain(gen, 5)

    shape0 = batches[0].tokens.shape
    ilen0 = batches[0].instr_lens.shape
    for b in batches:
        assert b.tokens.shape == shape0
        assert b.instr_lens.shape == ilen0

    # invalid_rate>0 must actually produce invalid (valid==False) rows that
    # are real invalid windows (token_len>0), not just padding rows.
    saw_invalid_window = False
    for b in batches:
        for i in range(b.tokens.shape[0]):
            if not b.valid[i] and b.token_lens[i] > 0:
                saw_invalid_window = True
    assert saw_invalid_window, 'invalid_rate>0 yielded no invalid windows'


def test_fixed_shape_row_outputs_mode():
    rng = np.random.default_rng(1)
    anchors = make_anchor_states(4, seed=0)
    gen = collect_into_batches(
        generate_chunks(single(), rng),
        batch_size=8, twins=1,
        anchor_states=anchors, rng=rng,
        max_chunk_len=1, row_outputs_mode=True)
    batches = _drain(gen, 5)

    shape0 = batches[0].tokens.shape
    ilen0 = batches[0].instr_lens.shape
    ro0 = batches[0].row_outputs.shape
    for b in batches:
        assert b.tokens.shape == shape0
        assert b.instr_lens.shape == ilen0
        assert b.row_outputs.shape == ro0
        # row_outputs_mode populates the row-* payload (RB == B, not 0).
        assert b.row_outputs.shape[0] == b.tokens.shape[0]

    # row_outputs_mode actually populated meaningful rows.
    assert any(b.pair_valid.any() for b in batches)
    assert any(b.row_has_rd.any() for b in batches)
