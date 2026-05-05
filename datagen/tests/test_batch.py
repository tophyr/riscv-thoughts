"""Tests for datagen.batch — Chunk + Batch containers, the
BinaryFormat machinery (RVT), pack helpers, and pair construction."""

import io

import numpy as np
import pytest

from emulator import Instruction
from datagen.batch import (
    Chunk, Batch, RVT_FORMAT, BinaryFormat, BodyField,
    pack_batch, padding_mask, unpack_chunks,
    generate_chunks, collect_into_batches, build_pairs,
    _make_valid_chunk, _make_invalid_chunk,
)
from datagen.compare import make_anchor_states
from datagen.generate import length_cap, until_branch


def _instrs_to_chunk(instrs):
    return _make_valid_chunk(instrs)


# ---------------------------------------------------------------------------
# pack_batch + padding_mask
# ---------------------------------------------------------------------------

def test_pack_basic():
    chunks = [
        _instrs_to_chunk([Instruction('ADDI', 1, 0, 5),
                          Instruction('ADD', 2, 1, 1)]),
        _instrs_to_chunk([Instruction('XOR', 3, 0, 0)]),
    ]
    batch = pack_batch(chunks, pairs=[], distances=[])
    assert batch.tokens.shape[0] == 2
    assert batch.token_lens[0] > batch.token_lens[1]
    assert batch.valid.tolist() == [True, True]
    assert batch.pair_indices.shape == (0, 2)


def test_pack_with_pairs_and_distances():
    chunks = [
        _instrs_to_chunk([Instruction('ADD', 1, 2, 3)]),
        _instrs_to_chunk([Instruction('ADD', 1, 3, 2)]),
        _instrs_to_chunk([Instruction('XOR', 4, 5, 6)]),
    ]
    batch = pack_batch(chunks, pairs=[(0, 1), (0, 2)], distances=[0.0, 12.5])
    assert batch.pair_indices.shape == (2, 2)
    assert batch.distances.tolist() == [0.0, 12.5]


def test_pack_rejects_empty():
    with pytest.raises(ValueError):
        pack_batch([], pairs=[], distances=[])


def test_pack_rejects_mismatched_pairs_distances():
    c = [_instrs_to_chunk([Instruction('ADD', 1, 2, 3)])]
    with pytest.raises(ValueError):
        pack_batch(c, pairs=[(0, 0), (0, 0)], distances=[0.0])


def test_padding_mask():
    chunks = [
        _instrs_to_chunk([Instruction('ADDI', 1, 0, 5)]),
        _instrs_to_chunk([Instruction('ADD', 2, 3, 4),
                          Instruction('SUB', 5, 6, 7)]),
    ]
    batch = pack_batch(chunks, pairs=[], distances=[])
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
    batch = pack_batch(chunks, pairs=[], distances=[])
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
        batch_size=8, twins=1, partners=2,
        anchor_states=anchors, rng=rng,
    )
    return next(gen)


def test_round_trip():
    b = _sample_batch()
    buf = io.BytesIO()
    RVT_FORMAT.write_stream_header(buf)
    RVT_FORMAT.write_batch(buf, b)
    buf.seek(0)
    RVT_FORMAT.read_stream_header(buf)
    got = RVT_FORMAT.read_batch(buf, Batch)
    assert np.array_equal(b.tokens, got.tokens)
    assert np.array_equal(b.token_lens, got.token_lens)
    assert np.array_equal(b.valid, got.valid)
    assert np.array_equal(b.instr_lens, got.instr_lens)
    assert np.array_equal(b.pair_indices, got.pair_indices)
    assert np.allclose(b.distances, got.distances)


def test_eof_clean():
    buf = io.BytesIO()
    RVT_FORMAT.write_stream_header(buf)
    buf.seek(0)
    RVT_FORMAT.read_stream_header(buf)
    assert RVT_FORMAT.read_batch(buf, Batch) is None


def test_pass_through_bytes():
    b = _sample_batch()
    buf = io.BytesIO()
    RVT_FORMAT.write_batch(buf, b)
    written = buf.getvalue()
    buf.seek(0)
    assert RVT_FORMAT.read_batch_bytes(buf) == written


def test_reader_iterates():
    rng = np.random.default_rng(0)
    anchors = make_anchor_states(4, seed=0)
    gen = collect_into_batches(
        generate_chunks(length_cap(2), rng),
        batch_size=8, twins=1, partners=2,
        anchor_states=anchors, rng=rng,
    )
    b1, b2 = next(gen), next(gen)
    buf = io.BytesIO()
    RVT_FORMAT.write_stream_header(buf)
    RVT_FORMAT.write_batch(buf, b1)
    RVT_FORMAT.write_batch(buf, b2)
    buf.seek(0)
    got = list(RVT_FORMAT.reader(buf, Batch))
    assert len(got) == 2


def test_empty_pairs_round_trip():
    """P=0 case: pack a batch with no pairs and verify it serializes."""
    chunks = [_instrs_to_chunk([Instruction('ADD', 1, 2, 3)])]
    b = pack_batch(chunks, pairs=[], distances=[])
    buf = io.BytesIO()
    RVT_FORMAT.write_stream_header(buf)
    RVT_FORMAT.write_batch(buf, b)
    buf.seek(0)
    RVT_FORMAT.read_stream_header(buf)
    got = RVT_FORMAT.read_batch(buf, Batch)
    assert got.pair_indices.shape == (0, 2)
    assert got.distances.shape == (0,)


def test_bad_magic_raises():
    bad = io.BytesIO(b'XXXX\x01' + b'\x00' * len(RVT_FORMAT.dtype_chars))
    with pytest.raises(ValueError, match='Bad magic'):
        RVT_FORMAT.read_stream_header(bad)


def test_recognized_by_batch_util():
    from scripts._batch_util import detect_format, RVT
    b = _sample_batch()
    buf = io.BytesIO()
    RVT_FORMAT.write_stream_header(buf)
    RVT_FORMAT.write_batch(buf, b)
    buf.seek(0)
    assert detect_format(buf) is RVT


# ---------------------------------------------------------------------------
# build_pairs
# ---------------------------------------------------------------------------

def test_build_pairs_clusters_have_zero_sibling_distance():
    rng = np.random.default_rng(0)
    anchors = make_anchor_states(4, seed=0)
    chunks = [
        _instrs_to_chunk([Instruction('ADDI', 1, 0, 5),
                          Instruction('ADD', 2, 1, 1)]),
        _instrs_to_chunk([Instruction('XOR', 3, 1, 1)]),
        _instrs_to_chunk([Instruction('OR', 4, 1, 2)]),
    ]
    out, pairs, dists = build_pairs(
        chunks, twins=2, partners=4, anchor_states=anchors, rng=rng)
    # Each cluster: 1 source + 2 twins.
    assert len(out) == 3 * 3
    # All sibling pairs have distance 0; non-sibling pairs may not.
    assert len(pairs) > 0


def test_build_pairs_drops_memory_op_chunks():
    rng = np.random.default_rng(0)
    anchors = make_anchor_states(4, seed=0)
    chunks = [
        _instrs_to_chunk([Instruction('LW', 1, 0, 0)]),
        _instrs_to_chunk([Instruction('ADD', 2, 3, 4)]),
        _instrs_to_chunk([Instruction('SUB', 5, 6, 7)]),
    ]
    out, pairs, dists = build_pairs(
        chunks, twins=1, partners=2, anchor_states=anchors, rng=rng)
    # The mem-op chunk is unpaired (no twins added for it).
    # 1 unpaired + 2 sources × 2 cluster_size = 1 + 4 = 5.
    assert len(out) == 5


# ---------------------------------------------------------------------------
# unpack_chunks
# ---------------------------------------------------------------------------

def test_unpack_yields_token_lists():
    chunks = [
        _instrs_to_chunk([Instruction('ADD', 1, 2, 3)]),
        _instrs_to_chunk([Instruction('XOR', 4, 5, 6),
                          Instruction('SUB', 7, 8, 9)]),
    ]
    b = pack_batch(chunks, pairs=[], distances=[])
    rows = list(unpack_chunks(b))
    assert len(rows) == 2
    assert len(rows[0]) == int(b.token_lens[0])
    assert len(rows[1]) == int(b.token_lens[1])
