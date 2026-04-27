"""T2 chunk generation: split RVS sequences at register-state-transformation
boundaries, optionally augment with invalid chunks.

A T2 thought is a register-state transformation block: a maximal contiguous
subsequence of instructions where only the LAST may be a memory access
(load/store) or control-flow change (branch/jump). See
WHAT_IS_A_THOUGHT.md / STREAMING_COMPRESSOR.md.

This module is pure-Python/numpy. The trainer is responsible for running
T1 encoder on the per-instruction tokens to produce T2 input embeddings;
this module only handles boundary detection, augmentation, and binary I/O.

Outputs are RVC-format batches (RV Chunked):
- per-chunk per-instruction token spans, padded
- chunk metadata: length, validity, type, register-state delta

Mux-friendly: this module is invoked by `scripts/chunk_t2.py` as a
CPU-side pipe stage between sequence generation and training. Multiple
parallel instances can run via `mux_batches.py`.

Why no "partial" at T2: strict prefixes of valid chunks are themselves
valid by the structural rules.

Why no "bogus" at T2: every T1 emission is by construction a valid T1
output, so concatenating random emissions produces sequences that are
structurally indistinguishable from spanning/multi (just noisier). T2
should learn its response to such inputs from the spanning/multi/
overlong signal — emergence rather than enumeration.
"""

import struct
from dataclasses import dataclass

import numpy as np

from datagen.seqgen import SequenceBatch
from datagen.instrgen import (
    _LOAD_OPS, _STORE_OPS, _BRANCH_OPS,
)
from tokenizer.tokenizer import _OP_TO_TOKEN
from tokenizer import PAD


# Maximum tokens per instruction. Longest is neg-imm JAL: op + rd + NEG +
# 6 hex = 9 tokens.
MAX_INSTR_TOKENS = 9

# Default cap on T2 chunk length (instructions per chunk).
DEFAULT_MAX_CHUNK_LEN = 16


# Terminator type codes per chunk.
TYPE_NON_TERMINATOR = 0  # not used as a chunk_type, only per-instr classification
TYPE_LOAD           = 1
TYPE_STORE          = 2
TYPE_BRANCH         = 3
TYPE_JUMP           = 4   # JAL or JALR
TYPE_CAPPED         = 5   # chunk hit max_chunk_len without a terminator
TYPE_TAIL           = 6   # chunk runs off the end of the sequence

# Invalidity type codes (continue numbering).
INVALID_SPANNING = 7
INVALID_MULTI    = 8
INVALID_OVERLONG = 9


DEFAULT_INVALIDITY_WEIGHTS = {
    'spanning': 0.45,
    'multi':    0.35,
    'overlong': 0.20,
}


# Build opcode-token sets once at import.
_LOAD_TOKS   = frozenset(_OP_TO_TOKEN[op] for op in _LOAD_OPS)
_STORE_TOKS  = frozenset(_OP_TO_TOKEN[op] for op in _STORE_OPS)
_BRANCH_TOKS = frozenset(_OP_TO_TOKEN[op] for op in _BRANCH_OPS)
_JAL_TOK     = _OP_TO_TOKEN['JAL']
_JALR_TOK    = _OP_TO_TOKEN['JALR']


def classify_opcode_token(opcode_token: int) -> int:
    """Return the terminator type code for an opcode token id.

    Returns one of TYPE_NON_TERMINATOR / TYPE_LOAD / TYPE_STORE /
    TYPE_BRANCH / TYPE_JUMP.
    """
    if opcode_token in _LOAD_TOKS:
        return TYPE_LOAD
    if opcode_token in _STORE_TOKS:
        return TYPE_STORE
    if opcode_token in _BRANCH_TOKS:
        return TYPE_BRANCH
    if opcode_token == _JAL_TOK or opcode_token == _JALR_TOK:
        return TYPE_JUMP
    return TYPE_NON_TERMINATOR


# ---------------------------------------------------------------------------
# Batch container
# ---------------------------------------------------------------------------

@dataclass
class ChunkBatch:
    """A batch of T2 chunks expressed as token spans (RVC format).

    Each chunk is a sequence of up to max_n_instrs instructions, each
    of up to MAX_INSTR_TOKENS tokens. Positions beyond chunk_lens[c]
    in the instruction axis, and beyond the actual instruction length
    in the token axis, are padded.

    The trainer is responsible for running the T1 encoder on the
    per-instruction tokens to produce T2 input embeddings. This batch
    type carries no T1 outputs.
    """
    token_ids:     np.ndarray   # (B, max_n_instrs, MAX_INSTR_TOKENS) int64
    instr_pad:     np.ndarray   # (B, max_n_instrs, MAX_INSTR_TOKENS) bool — True where padded
    chunk_lens:    np.ndarray   # (B,) int32 — instructions per chunk (1..max_n_instrs)
    valid_mask:    np.ndarray   # (B,) bool
    chunk_type:    np.ndarray   # (B,) int8 — TYPE_* (valid) or INVALID_* (augmented)
    reg_delta:     np.ndarray   # (B, n_inputs, 32) int32 — zeros for invalid


# ---------------------------------------------------------------------------
# Boundary detection — RVS sequences -> per-chunk records
# ---------------------------------------------------------------------------

def _build_per_instruction_tokens(rvs: SequenceBatch):
    """Extract per-instruction token spans from the flat RVS layout.

    Returns:
      tokens:      (B, max_n_instr, MAX_INSTR_TOKENS) int64
      instr_pad:   (B, max_n_instr, MAX_INSTR_TOKENS) bool — True where padded
      first_op:    (B, max_n_instr) int64 — opcode token at position 0,
                   or PAD where the (b, i) slot is unused.

    Vectorized: computes per-token (b, i, k) coordinates via cumulative
    counting on the token_instr_idx field. Avoids the per-(b,i)
    `np.flatnonzero` loop the original implementation used.
    """
    B = rvs.token_ids.shape[0]
    if rvs.n_instructions.size == 0:
        max_n_instr = 0
    else:
        max_n_instr = int(rvs.n_instructions.max())

    tokens = np.full((B, max_n_instr, MAX_INSTR_TOKENS), PAD, dtype=np.int64)
    instr_pad = np.ones((B, max_n_instr, MAX_INSTR_TOKENS), dtype=bool)
    first_op = np.full((B, max_n_instr), PAD, dtype=np.int64)

    if max_n_instr == 0:
        return tokens, instr_pad, first_op

    idx = rvs.token_instr_idx        # (B, max_tokens) int32
    toks = rvs.token_ids             # (B, max_tokens) int64

    # For each (b, t), compute the position-within-instruction k:
    # walking left to right, k restarts at 0 each time idx changes.
    # Vectorized: k = t - first_t_for(b, idx[b, t]), where first_t_for
    # is the first column index where idx equals i.
    # We compute that via cumulative argmin trick: for each row, the
    # first occurrence of each value can be precomputed. But variable-
    # length is awkward. Use a per-row Python loop (faster than the
    # original because the inner work is O(max_tokens) numpy).
    for b in range(B):
        n = int(rvs.n_instructions[b])
        if n == 0:
            continue
        idx_row = idx[b]                        # (max_tokens,)
        tok_row = toks[b]                       # (max_tokens,)
        # Mask of token positions assigned to a real instruction.
        valid_pos = (idx_row >= 0) & (idx_row < n)
        if not valid_pos.any():
            continue
        # Position-within-instruction: count occurrences of each idx
        # up to and including the current position.
        # cumcount-by-group via argsort.
        # Simpler approach: for each i in 0..n-1, find positions and
        # assign in one shot. We avoid Python-side loops by using
        # idx_row as a scatter dimension.
        instr_ids = idx_row[valid_pos].astype(np.int64)         # (T_b,)
        token_vals = tok_row[valid_pos]                          # (T_b,)
        # For each (i, k), find the position-within-instruction.
        # k_for_token[t] = how many earlier tokens in this row had the
        # same instruction id.
        # Computed via cumulative counts: walk idx_row sequentially.
        # Numpy doesn't have group-by-cumcount built-in, but since
        # tokens for the same instr appear contiguously in RVS layout
        # (instructions are emitted in order, tokens within instr are
        # contiguous), we can compute k as t - start[i] where start[i]
        # is the first column with idx==i.
        # That contiguity holds for the seqgen output (verified by
        # inspection of _encode_with_instr_idx).
        positions = np.flatnonzero(valid_pos)                    # (T_b,)
        # For each token, find the first position with the same instr_id.
        # Since tokens are contiguous per-instr, positions[0] is the start
        # of instr_ids[0]; runs of equal instr_ids share a start.
        # Detect run starts:
        run_start = np.empty_like(instr_ids, dtype=np.int64)
        run_start[0] = positions[0]
        if instr_ids.size > 1:
            new_run = instr_ids[1:] != instr_ids[:-1]
            run_starts = np.where(new_run, positions[1:], 0)
            # Forward-fill the run start:
            cum = np.maximum.accumulate(
                np.concatenate([[positions[0]], run_starts]))
            run_start = cum
        k = positions - run_start                                # (T_b,)

        # Bounds: an instruction shouldn't exceed MAX_INSTR_TOKENS.
        if k.max(initial=-1) >= MAX_INSTR_TOKENS:
            raise ValueError(
                f'Instruction in batch row {b} has > {MAX_INSTR_TOKENS} tokens')

        tokens[b, instr_ids, k] = token_vals
        instr_pad[b, instr_ids, k] = False

    # First-op token is the (b, i, 0) slot wherever instr_pad[b, i, 0]
    # is False (the slot is populated).
    populated = ~instr_pad[:, :, 0]
    if populated.any():
        first_op[populated] = tokens[:, :, 0][populated]

    return tokens, instr_pad, first_op


def _walk_chunks(per_instr_type, n_instructions, max_chunk_len):
    """Walk each sequence and emit chunks at terminators or the cap.

    Yields dicts with: sequence_idx, instr_start, instr_end (exclusive),
    chunk_type. chunk_type is the terminator's TYPE_*, or TYPE_CAPPED
    if the cap fired without a terminator, or TYPE_TAIL if the chunk
    runs off the end of the sequence without terminating.
    """
    B = per_instr_type.shape[0]
    for b in range(B):
        n = int(n_instructions[b])
        cur_start = 0
        i = 0
        while i < n:
            t = int(per_instr_type[b, i])
            chunk_len_so_far = i - cur_start + 1
            if t != TYPE_NON_TERMINATOR:
                yield {
                    'sequence_idx': b,
                    'instr_start':  cur_start,
                    'instr_end':    i + 1,
                    'chunk_type':   t,
                }
                cur_start = i + 1
            elif chunk_len_so_far >= max_chunk_len:
                yield {
                    'sequence_idx': b,
                    'instr_start':  cur_start,
                    'instr_end':    i + 1,
                    'chunk_type':   TYPE_CAPPED,
                }
                cur_start = i + 1
            i += 1
        if cur_start < n:
            yield {
                'sequence_idx': b,
                'instr_start':  cur_start,
                'instr_end':    n,
                'chunk_type':   TYPE_TAIL,
            }


def chunk_rvs(rvs: SequenceBatch,
              max_chunk_len: int = DEFAULT_MAX_CHUNK_LEN,
              storage_max_chunk_len: int = None) -> ChunkBatch:
    """Convert an RVS sequence batch into a valid-only ChunkBatch.

    Walks each sequence, classifies instructions by opcode terminator
    type, and groups them into chunks at terminators or at the
    max_chunk_len cap. Each chunk's per-instruction tokens are stored
    padded to MAX_INSTR_TOKENS in the token axis and storage_max_chunk_len
    in the instruction axis (defaults to max_chunk_len; bump higher if
    the caller will augment with overlong chunks downstream).

    All chunks produced here have valid_mask=True. Augmentation lives
    in `augment_chunkbatch_with_invalid`.
    """
    if storage_max_chunk_len is None:
        storage_max_chunk_len = max_chunk_len

    tokens_3d, instr_pad_3d, first_op = _build_per_instruction_tokens(rvs)
    B, max_n_instr, _ = tokens_3d.shape

    per_instr_type = np.zeros((B, max_n_instr), dtype=np.int8)
    for b in range(B):
        for i in range(int(rvs.n_instructions[b])):
            per_instr_type[b, i] = classify_opcode_token(int(first_op[b, i]))

    chunks = list(_walk_chunks(
        per_instr_type, rvs.n_instructions, max_chunk_len))

    n_chunks = len(chunks)
    n_inputs = rvs.per_instr_regs.shape[2] if n_chunks > 0 else 4

    out_tokens = np.full(
        (n_chunks, storage_max_chunk_len, MAX_INSTR_TOKENS),
        PAD, dtype=np.int64)
    out_pad = np.ones(
        (n_chunks, storage_max_chunk_len, MAX_INSTR_TOKENS), dtype=bool)
    out_lens = np.zeros(n_chunks, dtype=np.int32)
    out_type = np.zeros(n_chunks, dtype=np.int8)
    out_reg_delta = np.zeros((n_chunks, n_inputs, 32), dtype=np.int32)

    for c, info in enumerate(chunks):
        b = info['sequence_idx']
        s = info['instr_start']
        e = info['instr_end']
        L = e - s
        if L > storage_max_chunk_len:
            raise ValueError(
                f'Chunk length {L} exceeds storage_max_chunk_len '
                f'{storage_max_chunk_len}')

        out_tokens[c, :L] = tokens_3d[b, s:e]
        out_pad[c, :L] = instr_pad_3d[b, s:e]
        out_lens[c] = L
        out_type[c] = info['chunk_type']

        regs_before = rvs.per_instr_regs[b, s, :, :]
        regs_after = rvs.per_instr_regs[b, e, :, :]
        out_reg_delta[c] = regs_after - regs_before

    return ChunkBatch(
        token_ids=out_tokens,
        instr_pad=out_pad,
        chunk_lens=out_lens,
        valid_mask=np.ones(n_chunks, dtype=bool),
        chunk_type=out_type,
        reg_delta=out_reg_delta,
    )


# ---------------------------------------------------------------------------
# Invalidity augmentation (token-level)
# ---------------------------------------------------------------------------

def _build_invalidity_table(weights):
    positive = {k: v for k, v in weights.items() if v > 0}
    if not positive:
        raise ValueError(
            'invalidity weights must have at least one positive entry')
    total = sum(positive.values())
    return [(v / total, k) for k, v in positive.items()]


def _sample_invalidity_type(rng, table):
    roll = rng.random()
    c = 0.0
    for w, name in table:
        c += w
        if roll < c:
            return name
    return table[-1][1]


def _gen_invalid_multi(cb: ChunkBatch, rng, max_total_len):
    """Concatenate two complete valid chunks. Total instr count ≤ max_total_len."""
    n = cb.token_ids.shape[0]
    for _ in range(20):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        len_a = int(cb.chunk_lens[a])
        len_b = int(cb.chunk_lens[b])
        L = len_a + len_b
        if 2 <= L <= max_total_len:
            tokens = np.concatenate(
                [cb.token_ids[a, :len_a], cb.token_ids[b, :len_b]], axis=0)
            pad = np.concatenate(
                [cb.instr_pad[a, :len_a], cb.instr_pad[b, :len_b]], axis=0)
            return tokens, pad, L, INVALID_MULTI
    a = int(rng.integers(0, n))
    len_a = int(cb.chunk_lens[a])
    return (cb.token_ids[a, :len_a].copy(),
            cb.instr_pad[a, :len_a].copy(), len_a, INVALID_MULTI)


def _gen_invalid_spanning(cb: ChunkBatch, rng, max_total_len):
    """Tail of valid chunk A (incl terminator) + head of valid chunk B."""
    n = cb.token_ids.shape[0]
    for _ in range(20):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        len_a = int(cb.chunk_lens[a])
        len_b = int(cb.chunk_lens[b])
        if len_a < 1 or len_b < 1:
            continue
        j = int(rng.integers(1, len_a + 1))
        k = int(rng.integers(1, len_b + 1))
        if j == len_a and k == len_b:
            continue  # full + full = multi
        L = j + k
        if not (2 <= L <= max_total_len):
            continue
        tokens = np.concatenate([
            cb.token_ids[a, len_a - j : len_a],
            cb.token_ids[b, :k],
        ], axis=0)
        pad = np.concatenate([
            cb.instr_pad[a, len_a - j : len_a],
            cb.instr_pad[b, :k],
        ], axis=0)
        return tokens, pad, L, INVALID_SPANNING
    return _gen_invalid_multi(cb, rng, max_total_len)


def _gen_invalid_overlong(cb: ChunkBatch, rng, max_total_len, min_overlong=17):
    """Concatenate enough valid chunks to exceed the validity length cap."""
    if max_total_len < min_overlong:
        return _gen_invalid_multi(cb, rng, max_total_len)
    n = cb.token_ids.shape[0]
    target_L = int(rng.integers(min_overlong, max_total_len + 1))
    parts_tok, parts_pad = [], []
    cur_L = 0
    for _ in range(20):
        if cur_L >= target_L:
            break
        ci = int(rng.integers(0, n))
        len_c = int(cb.chunk_lens[ci])
        if len_c == 0:
            continue
        take = min(len_c, target_L - cur_L)
        parts_tok.append(cb.token_ids[ci, :take])
        parts_pad.append(cb.instr_pad[ci, :take])
        cur_L += take
    if not parts_tok or cur_L < min_overlong:
        return _gen_invalid_multi(cb, rng, max_total_len)
    tokens = np.concatenate(parts_tok, axis=0)
    pad = np.concatenate(parts_pad, axis=0)
    return tokens, pad, cur_L, INVALID_OVERLONG


_INVALIDITY_GENERATORS = {
    'spanning': _gen_invalid_spanning,
    'multi':    _gen_invalid_multi,
    'overlong': _gen_invalid_overlong,
}


def augment_chunkbatch_with_invalid(
    cb: ChunkBatch,
    invalidity_rate: float = 0.2,
    type_weights=None,
    storage_max_chunk_len: int = 24,
    rng=None,
) -> ChunkBatch:
    """Add invalid chunks (spanning / multi / overlong) to a valid-only ChunkBatch.

    invalidity_rate: target fraction of invalid chunks in the returned
        batch. n_invalid = round(n_valid * r / (1 - r)).
    type_weights: dict over {spanning, multi, overlong}. Defaults to
        DEFAULT_INVALIDITY_WEIGHTS.
    storage_max_chunk_len: padding cap for the instruction axis. Must
        accommodate the longest invalid chunk; defaults to 24 to fit
        overlong chunks.
    """
    if type_weights is None:
        type_weights = DEFAULT_INVALIDITY_WEIGHTS
    if rng is None:
        rng = np.random.default_rng()

    n_valid = cb.token_ids.shape[0]
    if storage_max_chunk_len < cb.token_ids.shape[1]:
        raise ValueError(
            f'storage_max_chunk_len={storage_max_chunk_len} < '
            f'input padding {cb.token_ids.shape[1]}')
    if n_valid == 0 or invalidity_rate <= 0:
        return _repad_chunkbatch(cb, storage_max_chunk_len)

    n_invalid = int(round(n_valid * invalidity_rate / (1 - invalidity_rate)))
    if n_invalid <= 0:
        return _repad_chunkbatch(cb, storage_max_chunk_len)

    type_table = _build_invalidity_table(type_weights)
    n_inputs = cb.reg_delta.shape[1]

    inv_tokens = np.full(
        (n_invalid, storage_max_chunk_len, MAX_INSTR_TOKENS),
        PAD, dtype=np.int64)
    inv_pad = np.ones(
        (n_invalid, storage_max_chunk_len, MAX_INSTR_TOKENS), dtype=bool)
    inv_lens = np.zeros(n_invalid, dtype=np.int32)
    inv_type = np.zeros(n_invalid, dtype=np.int8)

    for k in range(n_invalid):
        type_name = _sample_invalidity_type(rng, type_table)
        gen = _INVALIDITY_GENERATORS[type_name]
        tokens, pad, L, type_code = gen(cb, rng, storage_max_chunk_len)
        inv_tokens[k, :L] = tokens
        inv_pad[k, :L] = pad
        inv_lens[k] = L
        inv_type[k] = type_code

    valid_padded = _repad_chunkbatch(cb, storage_max_chunk_len)

    out = ChunkBatch(
        token_ids=np.concatenate(
            [valid_padded.token_ids, inv_tokens], axis=0),
        instr_pad=np.concatenate(
            [valid_padded.instr_pad, inv_pad], axis=0),
        chunk_lens=np.concatenate(
            [valid_padded.chunk_lens, inv_lens], axis=0),
        valid_mask=np.concatenate(
            [valid_padded.valid_mask, np.zeros(n_invalid, dtype=bool)], axis=0),
        chunk_type=np.concatenate(
            [valid_padded.chunk_type, inv_type], axis=0),
        reg_delta=np.concatenate([
            valid_padded.reg_delta,
            np.zeros((n_invalid, n_inputs, 32), dtype=np.int32),
        ], axis=0),
    )
    return out


def _repad_chunkbatch(cb: ChunkBatch, storage_max_chunk_len: int) -> ChunkBatch:
    """Re-pad the instruction axis to a possibly-larger size."""
    cur = cb.token_ids.shape[1]
    if cur == storage_max_chunk_len:
        return cb
    if cur > storage_max_chunk_len:
        raise ValueError(
            f'cannot shrink padding from {cur} to {storage_max_chunk_len}')
    n = cb.token_ids.shape[0]
    new_tokens = np.full(
        (n, storage_max_chunk_len, MAX_INSTR_TOKENS), PAD, dtype=np.int64)
    new_pad = np.ones(
        (n, storage_max_chunk_len, MAX_INSTR_TOKENS), dtype=bool)
    new_tokens[:, :cur] = cb.token_ids
    new_pad[:, :cur] = cb.instr_pad
    return ChunkBatch(
        token_ids=new_tokens,
        instr_pad=new_pad,
        chunk_lens=cb.chunk_lens,
        valid_mask=cb.valid_mask,
        chunk_type=cb.chunk_type,
        reg_delta=cb.reg_delta,
    )


# ---------------------------------------------------------------------------
# Binary I/O — RVC format
#
# Stream format:
#   Stream header (once): 4-byte magic "RVC\x00" + 1-byte version (1)
#                         + 6 dtype chars
#   Per batch:            16-byte header (B, max_n_instrs, max_instr_tokens,
#                         n_inputs as uint32) + raw array data in field order:
#                         token_ids, instr_pad, chunk_lens, valid_mask,
#                         chunk_type, reg_delta
# ---------------------------------------------------------------------------

_MAGIC = b'RVC\x00'
_VERSION = 1
_STREAM_HEADER = struct.Struct('<4sB6s')
_BATCH_HEADER = struct.Struct('<IIII')

_FIELD_DTYPES = (
    np.dtype(np.int64),   # token_ids
    np.dtype(np.bool_),   # instr_pad
    np.dtype(np.int32),   # chunk_lens
    np.dtype(np.bool_),   # valid_mask
    np.dtype(np.int8),    # chunk_type
    np.dtype(np.int32),   # reg_delta
)
_DTYPE_CHARS = b''.join(dt.char.encode() for dt in _FIELD_DTYPES)


def _batch_body_size(B: int, max_n_instrs: int, max_instr_tokens: int,
                     n_inputs: int) -> int:
    return (
        B * max_n_instrs * max_instr_tokens * _FIELD_DTYPES[0].itemsize
        + B * max_n_instrs * max_instr_tokens * _FIELD_DTYPES[1].itemsize
        + B * _FIELD_DTYPES[2].itemsize
        + B * _FIELD_DTYPES[3].itemsize
        + B * _FIELD_DTYPES[4].itemsize
        + B * n_inputs * 32 * _FIELD_DTYPES[5].itemsize
    )


def write_stream_header(f):
    f.write(_STREAM_HEADER.pack(_MAGIC, _VERSION, _DTYPE_CHARS))


def read_stream_header(f):
    buf = f.read(_STREAM_HEADER.size)
    if len(buf) < _STREAM_HEADER.size:
        raise ValueError('Missing stream header')
    magic, version, dtype_chars = _STREAM_HEADER.unpack(buf)
    if magic != _MAGIC:
        raise ValueError(f'Bad magic: {magic!r} (expected {_MAGIC!r})')
    if version != _VERSION:
        raise ValueError(f'Unsupported RVC version: {version}')
    if dtype_chars != _DTYPE_CHARS:
        raise ValueError(f'Dtype mismatch: {dtype_chars!r}')


def write_batch(f, batch: ChunkBatch):
    B, max_n_instrs, max_instr_tokens = batch.token_ids.shape
    n_inputs = batch.reg_delta.shape[1]
    f.write(_BATCH_HEADER.pack(B, max_n_instrs, max_instr_tokens, n_inputs))
    f.write(batch.token_ids.tobytes())
    f.write(batch.instr_pad.tobytes())
    f.write(batch.chunk_lens.tobytes())
    f.write(batch.valid_mask.tobytes())
    f.write(batch.chunk_type.tobytes())
    f.write(batch.reg_delta.tobytes())


def read_batch(f) -> ChunkBatch | None:
    header = f.read(_BATCH_HEADER.size)
    if len(header) == 0:
        return None
    if len(header) < _BATCH_HEADER.size:
        raise EOFError(f'Truncated batch header ({len(header)} bytes)')
    B, max_n_instrs, max_instr_tokens, n_inputs = _BATCH_HEADER.unpack(header)

    def _read_array(dtype, shape):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        buf = f.read(nbytes)
        if len(buf) < nbytes:
            raise EOFError(
                f'Truncated batch data (got {len(buf)}, expected {nbytes})')
        return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()

    return ChunkBatch(
        token_ids=_read_array(_FIELD_DTYPES[0],
                              (B, max_n_instrs, max_instr_tokens)),
        instr_pad=_read_array(_FIELD_DTYPES[1],
                              (B, max_n_instrs, max_instr_tokens)),
        chunk_lens=_read_array(_FIELD_DTYPES[2], (B,)),
        valid_mask=_read_array(_FIELD_DTYPES[3], (B,)),
        chunk_type=_read_array(_FIELD_DTYPES[4], (B,)),
        reg_delta=_read_array(_FIELD_DTYPES[5], (B, n_inputs, 32)),
    )


def read_batch_bytes(f) -> bytes | None:
    """Read one complete batch as raw bytes. For pass-through use."""
    header = f.read(_BATCH_HEADER.size)
    if len(header) == 0:
        return None
    if len(header) < _BATCH_HEADER.size:
        raise EOFError(f'Truncated batch header ({len(header)} bytes)')
    B, max_n_instrs, max_instr_tokens, n_inputs = _BATCH_HEADER.unpack(header)
    body_size = _batch_body_size(B, max_n_instrs, max_instr_tokens, n_inputs)
    body = f.read(body_size)
    if len(body) < body_size:
        raise EOFError(
            f'Truncated batch body (got {len(body)}, expected {body_size})')
    return header + body


class ChunkBatchReader:
    """Iterable wrapper for reading RVC batches from a binary stream."""

    def __init__(self, f):
        read_stream_header(f)
        self._f = f

    def __iter__(self):
        return self

    def __next__(self):
        batch = read_batch(self._f)
        if batch is None:
            raise StopIteration
        return batch
