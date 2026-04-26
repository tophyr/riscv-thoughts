"""Tests for the T2 chunker.

Verifies that:
- Chunks cover every instruction in every source sequence exactly once.
- Terminator classification matches the opcode's category.
- Chunk lengths respect the max_chunk_len cap.
- Register-state deltas are computed correctly from RVS snapshots.
- Edge cases: capped chunks (long pure-ALU runs), tail chunks (no
  terminator at end of sequence).
"""

import numpy as np
import pytest
import torch

from datagen.seqgen import (
    SequenceBatch, produce_sequence_batch, _encode_with_instr_idx,
    execute_sequence,
)
from emulator import Instruction, make_ctx, random_regs
from tokenizer import PAD
from compressor.model import T1Compressor
from compressor.chunker import (
    chunk_rvs_to_t2_batch, augment_t2_with_invalid,
    TYPE_NON_TERMINATOR, TYPE_LOAD, TYPE_STORE, TYPE_BRANCH,
    TYPE_JUMP, TYPE_CAPPED, TYPE_TAIL,
    INVALID_SPANNING, INVALID_MULTI, INVALID_OVERLONG,
)
from tokenizer import VOCAB_SIZE


@pytest.fixture(scope='module')
def t1_encoder():
    """A small randomly-initialized T1Compressor — we don't care
    about training quality for chunker tests; we just need an
    encoder that produces vectors of the right shape."""
    torch.manual_seed(0)
    m = T1Compressor(VOCAB_SIZE, d_model=32, n_heads=2, n_layers=1,
                     d_out=8, max_window=32)
    m.eval()
    return m


def _hand_rvs(sequences, n_inputs=2):
    """Build an RVS batch from a list of instruction lists.

    Mirrors the produce_sequence_batch internals but takes
    explicit sequences so we can construct edge cases.
    """
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


def test_full_coverage(t1_encoder):
    """Every instruction in every sequence appears in exactly one chunk."""
    rng = np.random.default_rng(42)
    rvs = produce_sequence_batch(8, n_inputs=2, max_block_len=10, rng=rng)
    t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=16)

    for b in range(rvs.token_ids.shape[0]):
        n = int(rvs.n_instructions[b])
        instrs_covered = []
        for c in range(t2.chunk_emissions.shape[0]):
            if int(t2.sequence_idx[c]) != b:
                continue
            s = int(t2.instr_start[c])
            e = int(t2.instr_end[c])
            instrs_covered.extend(range(s, e))
        assert sorted(instrs_covered) == list(range(n)), (
            f'seq {b}: covered {sorted(instrs_covered)}, expected '
            f'{list(range(n))}')


def test_chunk_lens_match_ranges(t1_encoder):
    """chunk_lens[c] == instr_end[c] - instr_start[c]."""
    rng = np.random.default_rng(43)
    rvs = produce_sequence_batch(4, n_inputs=2, max_block_len=8, rng=rng)
    t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=16)
    for c in range(t2.chunk_emissions.shape[0]):
        L = int(t2.chunk_lens[c])
        s = int(t2.instr_start[c])
        e = int(t2.instr_end[c])
        assert L == e - s, f'chunk {c}: len {L} != range {e}-{s}'


def test_chunks_within_cap(t1_encoder):
    """All chunk lengths are within [1, max_chunk_len]."""
    rng = np.random.default_rng(44)
    rvs = produce_sequence_batch(8, n_inputs=2, max_block_len=10, rng=rng)
    for cap in (4, 8, 16):
        t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=cap)
        for c in range(t2.chunk_emissions.shape[0]):
            L = int(t2.chunk_lens[c])
            assert 1 <= L <= cap, f'cap={cap}, chunk {c} len={L} out of range'


def test_terminator_classification(t1_encoder):
    """The terminator-typed chunks really end at the right opcode type."""
    rng = np.random.default_rng(45)
    rvs = produce_sequence_batch(8, n_inputs=2, max_block_len=10, rng=rng)
    t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=16)

    # Reconstruct first-token-of-each-instruction map to verify types.
    from compressor.chunker import _classify_opcode_token
    for c in range(t2.chunk_emissions.shape[0]):
        b = int(t2.sequence_idx[c])
        e = int(t2.instr_end[c])
        chunk_type = int(t2.chunk_type[c])
        # Find opcode token of last instruction in chunk.
        positions = np.flatnonzero(rvs.token_instr_idx[b] == (e - 1))
        last_op = int(rvs.token_ids[b, positions[0]])
        derived = _classify_opcode_token(last_op)
        # CAPPED and TAIL chunks have a non-terminator last instruction.
        if chunk_type in (TYPE_CAPPED, TYPE_TAIL):
            assert derived == TYPE_NON_TERMINATOR, (
                f'chunk {c} marked {chunk_type} but last instr is a terminator')
        else:
            assert derived == chunk_type, (
                f'chunk {c}: classified {chunk_type}, '
                f'last instr derives {derived}')


def test_reg_delta_matches_per_instr_regs(t1_encoder):
    """reg_delta[c] = per_instr_regs[b, end] - per_instr_regs[b, start]."""
    rng = np.random.default_rng(46)
    rvs = produce_sequence_batch(4, n_inputs=2, max_block_len=8, rng=rng)
    t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=16)
    for c in range(t2.chunk_emissions.shape[0]):
        b = int(t2.sequence_idx[c])
        s = int(t2.instr_start[c])
        e = int(t2.instr_end[c])
        expected = (rvs.per_instr_regs[b, e, :, :]
                    - rvs.per_instr_regs[b, s, :, :])
        got = t2.reg_delta[c].cpu().numpy()
        assert np.array_equal(got, expected), (
            f'chunk {c}: reg_delta mismatch')


def test_capped_chunk_long_alu_run(t1_encoder):
    """A long pure-ALU sequence with cap=4 produces multiple capped chunks."""
    # 7 ADDs in a row, no terminator anywhere.
    instrs = [Instruction('ADD', i + 1, i + 1, i + 1) for i in range(7)]
    rvs = _hand_rvs([instrs])
    t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=4)

    # Should be 2 chunks: [0:4] capped, [4:7] tail.
    assert t2.chunk_emissions.shape[0] == 2
    assert int(t2.chunk_lens[0]) == 4
    assert int(t2.chunk_lens[1]) == 3
    assert int(t2.chunk_type[0]) == TYPE_CAPPED
    assert int(t2.chunk_type[1]) == TYPE_TAIL


def test_tail_chunk(t1_encoder):
    """Sequence ending in non-terminator gets a TAIL-typed final chunk."""
    instrs = [Instruction('ADD', 1, 1, 1), Instruction('ADD', 2, 2, 2),
              Instruction('LW', 3, 0, 1), Instruction('ADD', 4, 4, 4)]
    rvs = _hand_rvs([instrs])
    t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=16)
    # First chunk: [0:3] LOAD (ADD, ADD, LW). Second: [3:4] TAIL (ADD).
    assert t2.chunk_emissions.shape[0] == 2
    assert int(t2.chunk_type[0]) == TYPE_LOAD
    assert int(t2.chunk_type[1]) == TYPE_TAIL
    assert int(t2.chunk_lens[0]) == 3
    assert int(t2.chunk_lens[1]) == 1


def test_single_terminator_only(t1_encoder):
    """A sequence that's just one branch: one chunk of length 1."""
    instrs = [Instruction('BEQ', 1, 2, 8)]
    rvs = _hand_rvs([instrs])
    t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=16)
    assert t2.chunk_emissions.shape[0] == 1
    assert int(t2.chunk_lens[0]) == 1
    assert int(t2.chunk_type[0]) == TYPE_BRANCH


def test_each_terminator_kind(t1_encoder):
    """Verify each terminator opcode type produces the right chunk_type."""
    seqs = [
        [Instruction('ADD', 1, 1, 1), Instruction('LW',  2, 0, 1)],   # LOAD
        [Instruction('ADD', 1, 1, 1), Instruction('SW',  2, 0, 1)],   # STORE
        [Instruction('ADD', 1, 1, 1), Instruction('BEQ', 1, 2, 8)],   # BRANCH
        [Instruction('ADD', 1, 1, 1), Instruction('JAL', 2, 8)],      # JUMP
        [Instruction('ADD', 1, 1, 1), Instruction('JALR', 2, 1, 0)],  # JUMP
    ]
    expected = [TYPE_LOAD, TYPE_STORE, TYPE_BRANCH, TYPE_JUMP, TYPE_JUMP]
    rvs = _hand_rvs(seqs)
    t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=16)
    # One chunk per sequence (all terminate at the second instruction).
    assert t2.chunk_emissions.shape[0] == len(seqs)
    for b in range(len(seqs)):
        c = next(i for i in range(t2.chunk_emissions.shape[0])
                 if int(t2.sequence_idx[i]) == b)
        assert int(t2.chunk_type[c]) == expected[b], (
            f'seq {b}: chunk_type={int(t2.chunk_type[c])}, '
            f'expected {expected[b]}')


def test_emissions_zero_outside_chunk_len(t1_encoder):
    """Positions beyond chunk_lens[c] in chunk_emissions[c] are zero."""
    rng = np.random.default_rng(47)
    rvs = produce_sequence_batch(4, n_inputs=2, max_block_len=8, rng=rng)
    t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=16)
    for c in range(t2.chunk_emissions.shape[0]):
        L = int(t2.chunk_lens[c])
        beyond = t2.chunk_emissions[c, L:]
        assert (beyond == 0).all(), f'chunk {c}: nonzero emissions beyond len'


def test_valid_mask_all_true(t1_encoder):
    """The basic chunker produces only valid chunks; valid_mask is all True."""
    rng = np.random.default_rng(48)
    rvs = produce_sequence_batch(8, n_inputs=2, max_block_len=10, rng=rng)
    t2 = chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=16)
    assert t2.valid_mask.all()


# ---------------------------------------------------------------------------
# Invalid-chunk augmentation tests
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_t2(t1_encoder):
    """A valid-only T2Batch generated from a small RVS batch."""
    rng = np.random.default_rng(100)
    rvs = produce_sequence_batch(16, n_inputs=2, max_block_len=10, rng=rng)
    return chunk_rvs_to_t2_batch(rvs, t1_encoder, max_chunk_len=16)


def test_augment_invalidity_rate(valid_t2):
    """The invalidity rate matches what was requested."""
    rng = np.random.default_rng(200)
    n_valid = valid_t2.chunk_emissions.shape[0]
    augmented = augment_t2_with_invalid(
        valid_t2, invalidity_rate=0.2, rng=rng)
    n_total = augmented.chunk_emissions.shape[0]
    n_invalid = (~augmented.valid_mask).sum().item()
    # n_invalid / n_total should be close to 0.2 (allow off-by-one for rounding).
    assert abs(n_invalid / n_total - 0.2) < 0.05, (
        f'rate {n_invalid/n_total:.3f} far from 0.2')
    # Valids preserved.
    assert augmented.valid_mask.sum().item() == n_valid


def test_augment_zero_rate_is_noop(valid_t2):
    """invalidity_rate=0 returns the same batch (modulo possible repad)."""
    n_valid = valid_t2.chunk_emissions.shape[0]
    augmented = augment_t2_with_invalid(
        valid_t2, invalidity_rate=0.0, storage_max_chunk_len=16)
    assert augmented.chunk_emissions.shape[0] == n_valid
    assert augmented.valid_mask.all()


def test_augment_storage_padding(valid_t2):
    """Output chunk_emissions has the requested storage padding."""
    rng = np.random.default_rng(201)
    augmented = augment_t2_with_invalid(
        valid_t2, invalidity_rate=0.2, storage_max_chunk_len=24, rng=rng)
    assert augmented.chunk_emissions.shape[1] == 24


def test_augment_storage_too_small_raises(valid_t2):
    """Asking for storage smaller than input padding errors."""
    with pytest.raises(ValueError, match='smaller'):
        augment_t2_with_invalid(
            valid_t2, invalidity_rate=0.2, storage_max_chunk_len=8)


def test_augment_invalid_type_codes(valid_t2):
    """Invalid chunks carry one of the INVALID_* type codes."""
    rng = np.random.default_rng(202)
    augmented = augment_t2_with_invalid(
        valid_t2, invalidity_rate=0.5, rng=rng)
    invalid_indices = (~augmented.valid_mask).nonzero(as_tuple=True)[0]
    valid_codes = {INVALID_SPANNING, INVALID_MULTI, INVALID_OVERLONG}
    for i in invalid_indices.tolist():
        code = int(augmented.chunk_type[i])
        assert code in valid_codes, f'unexpected invalid code {code}'


def test_augment_invalid_lens_in_range(valid_t2):
    """Invalid chunk lengths fit in storage."""
    rng = np.random.default_rng(203)
    storage = 24
    augmented = augment_t2_with_invalid(
        valid_t2, invalidity_rate=0.3,
        storage_max_chunk_len=storage, rng=rng)
    invalid_indices = (~augmented.valid_mask).nonzero(as_tuple=True)[0]
    for i in invalid_indices.tolist():
        L = int(augmented.chunk_lens[i])
        assert 1 <= L <= storage


def test_augment_overlong_actually_overlong(valid_t2):
    """Overlong-typed chunks have len > 16."""
    rng = np.random.default_rng(204)
    # Force overlong-only.
    augmented = augment_t2_with_invalid(
        valid_t2, invalidity_rate=0.3,
        type_weights={'overlong': 1.0},
        storage_max_chunk_len=24, rng=rng)
    invalid_indices = (~augmented.valid_mask).nonzero(as_tuple=True)[0]
    overlong_count = 0
    for i in invalid_indices.tolist():
        if int(augmented.chunk_type[i]) == INVALID_OVERLONG:
            assert int(augmented.chunk_lens[i]) > 16, (
                f'overlong-typed chunk has len {int(augmented.chunk_lens[i])}')
            overlong_count += 1
    # At least some of the invalids should have actually been overlong
    # (the rest may have hit the fallback path).
    assert overlong_count > 0


def test_augment_multi_only_correct_length_distribution(valid_t2):
    """Multi-only augmentation produces lengths that sum two valid chunks."""
    rng = np.random.default_rng(205)
    augmented = augment_t2_with_invalid(
        valid_t2, invalidity_rate=0.5,
        type_weights={'multi': 1.0},
        storage_max_chunk_len=24, rng=rng)
    invalid_indices = (~augmented.valid_mask).nonzero(as_tuple=True)[0]
    # Most multi-typed chunks should have len ≥ 2 (two chunks combined).
    for i in invalid_indices.tolist():
        if int(augmented.chunk_type[i]) == INVALID_MULTI:
            assert int(augmented.chunk_lens[i]) >= 2


def test_augment_invalid_reg_delta_zero(valid_t2):
    """Invalid chunks have zero reg_delta (they're filtered from direction loss)."""
    rng = np.random.default_rng(206)
    augmented = augment_t2_with_invalid(
        valid_t2, invalidity_rate=0.3, rng=rng)
    invalid_indices = (~augmented.valid_mask).nonzero(as_tuple=True)[0]
    for i in invalid_indices.tolist():
        assert (augmented.reg_delta[i] == 0).all()


def test_augment_invalid_provenance_negative(valid_t2):
    """Invalid chunks have provenance fields set to -1 sentinel."""
    rng = np.random.default_rng(207)
    augmented = augment_t2_with_invalid(
        valid_t2, invalidity_rate=0.3, rng=rng)
    invalid_indices = (~augmented.valid_mask).nonzero(as_tuple=True)[0]
    for i in invalid_indices.tolist():
        assert int(augmented.sequence_idx[i]) == -1
        assert int(augmented.instr_start[i]) == -1
        assert int(augmented.instr_end[i]) == -1


def test_augment_valid_unchanged(valid_t2):
    """Valid chunks are passed through unchanged."""
    rng = np.random.default_rng(208)
    augmented = augment_t2_with_invalid(
        valid_t2, invalidity_rate=0.3,
        storage_max_chunk_len=valid_t2.chunk_emissions.shape[1], rng=rng)
    n_valid = valid_t2.chunk_emissions.shape[0]
    # First n_valid rows of augmented match valid_t2.
    torch.testing.assert_close(
        augmented.chunk_emissions[:n_valid], valid_t2.chunk_emissions)
    torch.testing.assert_close(
        augmented.chunk_lens[:n_valid], valid_t2.chunk_lens)
    assert augmented.valid_mask[:n_valid].all()
