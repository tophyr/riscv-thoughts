"""T2 input chunking from RVS sequences via a frozen T1 encoder.

Walks each instruction sequence, runs the frozen T1 encoder on each
instruction, and groups the resulting T1 emission vectors into T2
chunks. Boundaries are placed at terminator instructions (loads,
stores, branches, JAL, JALR) or at a max-length cap.

Why terminator-based chunking: a T2 thought is a register-state
transformation block (see WHAT_IS_A_THOUGHT.md / STREAMING_COMPRESSOR.md).
Inside a T2 thought only register state evolves — purely bounded.
A memory access or control-flow change is the chunk's exit; it
commits the chunk's effect to memory, signals a control transfer,
or otherwise hands control back to the world. The 16-instruction
cap is a safety bound for ALU runs that don't naturally terminate.

The chunker is the structural-rule emulation of what gates would
do at T1: it splits the T1 emission stream at the same boundaries
the gates would (eventually) learn. This lets us train T2 without
first solving T1 gates.

`chunk_rvs_to_t2_batch` produces only valid chunks. Invalid-chunk
augmentation (spanning / multi / overlong) is added by
`augment_t2_with_invalid`.

Why no "partial" at T2: strict prefixes of valid chunks are
themselves valid by the structural rules (≤1 non-pure instr at
the end, length in range), so partial doesn't produce a
meaningful invalid window the way it does at T1.

Why no "bogus" at T2: every T1 emission is by construction a
valid T1 output, so concatenating random emissions produces
sequences that are structurally indistinguishable from spanning/
multi (just noisier). The only ways to construct "actually bogus"
T2 inputs would be (a) low-magnitude T1 vectors, which require a
threshold we deliberately avoid in the smooth-magnitude framing,
or (b) off-distribution random vectors, which doesn't generalize
to a real failure mode. T2 should learn its response to such
inputs from the spanning/multi/overlong signal — emergence
rather than enumeration.
"""

from dataclasses import dataclass

import numpy as np
import torch

from datagen.seqgen import SequenceBatch
from datagen.instrgen import (
    _ALU_R_OPS, _ALU_I_OPS, _LOAD_OPS, _STORE_OPS, _BRANCH_OPS,
)
from tokenizer.tokenizer import _OP_TO_TOKEN
from tokenizer import PAD


# Maximum tokens per instruction. Longest is neg-imm JAL:
# op + rd + NEG + 6 hex = 9 tokens.
MAX_INSTR_TOKENS = 9

# Default cap on T2 chunk length (instructions per chunk).
DEFAULT_MAX_CHUNK_LEN = 16


# Terminator type codes. Used for both per-instruction classification
# during walk and per-chunk type for debugging / eval.
TYPE_NON_TERMINATOR = 0
TYPE_LOAD           = 1
TYPE_STORE          = 2
TYPE_BRANCH         = 3
TYPE_JUMP           = 4   # JAL or JALR
TYPE_CAPPED         = 5   # chunk hit max_chunk_len without a terminator
TYPE_TAIL           = 6   # chunk runs off the end of the sequence


# Build opcode-token sets once at import.
_LOAD_TOKS   = frozenset(_OP_TO_TOKEN[op] for op in _LOAD_OPS)
_STORE_TOKS  = frozenset(_OP_TO_TOKEN[op] for op in _STORE_OPS)
_BRANCH_TOKS = frozenset(_OP_TO_TOKEN[op] for op in _BRANCH_OPS)
_JAL_TOK     = _OP_TO_TOKEN['JAL']
_JALR_TOK    = _OP_TO_TOKEN['JALR']


def _classify_opcode_token(opcode_token: int) -> int:
    """Return the terminator type code for an opcode token id."""
    if opcode_token in _LOAD_TOKS:
        return TYPE_LOAD
    if opcode_token in _STORE_TOKS:
        return TYPE_STORE
    if opcode_token in _BRANCH_TOKS:
        return TYPE_BRANCH
    if opcode_token == _JAL_TOK or opcode_token == _JALR_TOK:
        return TYPE_JUMP
    return TYPE_NON_TERMINATOR


@dataclass
class T2Batch:
    """A batch of T2 input chunks.

    Each chunk is a sequence of T1 emission vectors corresponding to
    a contiguous run of instructions from one source RVS sequence.
    Positions beyond chunk_lens[c] in chunk_emissions[c] are zeroed.
    """
    chunk_emissions: torch.Tensor   # (B_t2, max_chunk_len, d_out) float32
    chunk_lens:      torch.Tensor   # (B_t2,)                      int64
    valid_mask:      torch.Tensor   # (B_t2,)                      bool
    reg_delta:       torch.Tensor   # (B_t2, n_inputs, 32)         int32
    chunk_type:      torch.Tensor   # (B_t2,)                      int8 — TYPE_*
    sequence_idx:    torch.Tensor   # (B_t2,)                      int32 — source seq idx in RVS batch
    instr_start:     torch.Tensor   # (B_t2,)                      int32 — first instr in source seq
    instr_end:       torch.Tensor   # (B_t2,)                      int32 — exclusive last instr


def _build_per_instruction_tokens(rvs: SequenceBatch):
    """Extract per-instruction token spans from the flat RVS layout.

    Returns:
      tokens:      (B, max_n_instr, MAX_INSTR_TOKENS) int64
      pad_mask:    (B, max_n_instr, MAX_INSTR_TOKENS) bool — True where padded
      first_op:    (B, max_n_instr) int64 — opcode token at position 0,
                   or PAD where the (b, i) slot is unused.
    """
    B = rvs.token_ids.shape[0]
    max_n_instr = int(rvs.n_instructions.max())

    tokens = np.full((B, max_n_instr, MAX_INSTR_TOKENS), PAD, dtype=np.int64)
    pad_mask = np.ones((B, max_n_instr, MAX_INSTR_TOKENS), dtype=bool)
    first_op = np.full((B, max_n_instr), PAD, dtype=np.int64)

    for b in range(B):
        n = int(rvs.n_instructions[b])
        idx_row = rvs.token_instr_idx[b]
        tok_row = rvs.token_ids[b]
        for i in range(n):
            positions = np.flatnonzero(idx_row == i)
            if positions.size == 0:
                continue
            if positions.size > MAX_INSTR_TOKENS:
                # Should not happen for valid RV32I tokenizations.
                raise ValueError(
                    f'Instruction at (b={b}, i={i}) has '
                    f'{positions.size} tokens > MAX_INSTR_TOKENS={MAX_INSTR_TOKENS}')
            span = tok_row[positions]
            L = span.size
            tokens[b, i, :L] = span
            pad_mask[b, i, :L] = False
            first_op[b, i] = span[0]

    return tokens, pad_mask, first_op


@torch.no_grad()
def _encode_per_instruction(t1_encoder, tokens, pad_mask, device):
    """Run T1 encoder on a (B, max_n_instr, MAX_INSTR_TOKENS) tensor.

    Reshapes into (B*max_n_instr, MAX_INSTR_TOKENS), forwards through
    the encoder, reshapes back to (B, max_n_instr, d_out).
    """
    B, M, T = tokens.shape
    flat_tokens = torch.from_numpy(tokens).to(device).view(B * M, T)
    flat_pad = torch.from_numpy(pad_mask).to(device).view(B * M, T)
    # Force position 0 non-pad on empty rows so the transformer's
    # attention doesn't see an all-padding sequence (it complains
    # and the result for those rows is discarded anyway).
    all_pad = flat_pad.all(dim=1)
    flat_pad[all_pad, 0] = False

    vecs = t1_encoder.encode(flat_tokens, flat_pad)  # (B*M, d_out)
    return vecs.view(B, M, -1)


def _walk_chunks(per_instr_type, n_instructions, max_chunk_len):
    """Walk each sequence and emit chunks at terminators or the cap.

    per_instr_type: (B, max_n_instr) int — TYPE_* per instruction
    n_instructions: (B,) int — actual instruction count per sequence
    max_chunk_len: int

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
        # Tail: any unterminated suffix at end of sequence.
        if cur_start < n:
            yield {
                'sequence_idx': b,
                'instr_start':  cur_start,
                'instr_end':    n,
                'chunk_type':   TYPE_TAIL,
            }


def chunk_rvs_to_t2_batch(rvs: SequenceBatch, t1_encoder,
                          max_chunk_len: int = DEFAULT_MAX_CHUNK_LEN,
                          device='cpu') -> T2Batch:
    """Chunk an RVS batch into a T2Batch for training.

    Steps:
    1. Extract per-instruction token spans from the flat RVS layout.
    2. Run frozen T1 encoder on every instruction (batched).
    3. Classify each instruction's terminator type from its opcode.
    4. Walk each sequence, group instructions into chunks at terminators
       or at the max-length cap.
    5. Pack the resulting chunks into a T2Batch with T1 emissions,
       lengths, validity flags, register-state deltas, and metadata.

    All chunks produced here are valid by construction; valid_mask is
    all True. Invalid-chunk augmentation lives in a separate module
    and is composed by the caller.
    """
    if not torch.is_tensor(rvs.token_ids):
        # rvs is plain numpy as deserialized from disk.
        pass

    # Step 1+3: per-instruction tokens and opcode-based classification.
    tokens, pad_mask, first_op = _build_per_instruction_tokens(rvs)
    B, max_n_instr, _ = tokens.shape
    per_instr_type = np.zeros((B, max_n_instr), dtype=np.int8)
    for b in range(B):
        for i in range(int(rvs.n_instructions[b])):
            per_instr_type[b, i] = _classify_opcode_token(int(first_op[b, i]))

    # Step 2: encode every instruction through frozen T1.
    per_instr_emissions = _encode_per_instruction(
        t1_encoder, tokens, pad_mask, device)  # (B, max_n_instr, d_out)
    d_out = per_instr_emissions.shape[-1]

    # Step 4: walk and collect chunks.
    chunks = list(_walk_chunks(
        per_instr_type, rvs.n_instructions, max_chunk_len))
    if not chunks:
        # Empty batch; return zero-size T2Batch.
        return T2Batch(
            chunk_emissions=torch.zeros(0, max_chunk_len, d_out, device=device),
            chunk_lens=torch.zeros(0, dtype=torch.int64, device=device),
            valid_mask=torch.zeros(0, dtype=torch.bool, device=device),
            reg_delta=torch.zeros(0, rvs.per_instr_regs.shape[2], 32,
                                  dtype=torch.int32, device=device),
            chunk_type=torch.zeros(0, dtype=torch.int8, device=device),
            sequence_idx=torch.zeros(0, dtype=torch.int32, device=device),
            instr_start=torch.zeros(0, dtype=torch.int32, device=device),
            instr_end=torch.zeros(0, dtype=torch.int32, device=device),
        )

    n_chunks = len(chunks)
    n_inputs = rvs.per_instr_regs.shape[2]

    # Step 5: pack into T2Batch tensors.
    chunk_emissions = torch.zeros(
        n_chunks, max_chunk_len, d_out, device=device)
    chunk_lens = np.zeros(n_chunks, dtype=np.int64)
    reg_delta_np = np.zeros((n_chunks, n_inputs, 32), dtype=np.int32)
    chunk_type_np = np.zeros(n_chunks, dtype=np.int8)
    sequence_idx_np = np.zeros(n_chunks, dtype=np.int32)
    instr_start_np = np.zeros(n_chunks, dtype=np.int32)
    instr_end_np = np.zeros(n_chunks, dtype=np.int32)

    for c, info in enumerate(chunks):
        b = info['sequence_idx']
        s = info['instr_start']
        e = info['instr_end']
        L = e - s

        # Slice the T1 emissions for this chunk.
        chunk_emissions[c, :L] = per_instr_emissions[b, s:e]

        chunk_lens[c] = L
        chunk_type_np[c] = info['chunk_type']
        sequence_idx_np[c] = b
        instr_start_np[c] = s
        instr_end_np[c] = e

        # Register state delta over the chunk: regs after the last
        # instruction minus regs before the first.
        regs_before = rvs.per_instr_regs[b, s, :, :]      # (n_inputs, 32)
        regs_after = rvs.per_instr_regs[b, e, :, :]       # (n_inputs, 32)
        reg_delta_np[c] = regs_after - regs_before

    return T2Batch(
        chunk_emissions=chunk_emissions,
        chunk_lens=torch.from_numpy(chunk_lens).to(device),
        valid_mask=torch.ones(n_chunks, dtype=torch.bool, device=device),
        reg_delta=torch.from_numpy(reg_delta_np).to(device),
        chunk_type=torch.from_numpy(chunk_type_np).to(device),
        sequence_idx=torch.from_numpy(sequence_idx_np).to(device),
        instr_start=torch.from_numpy(instr_start_np).to(device),
        instr_end=torch.from_numpy(instr_end_np).to(device),
    )


# ---------------------------------------------------------------------------
# Invalid-chunk augmentation
# ---------------------------------------------------------------------------

# Invalidity type codes (continue numbering from TYPE_*).
INVALID_SPANNING = 7
INVALID_MULTI    = 8
INVALID_OVERLONG = 9

DEFAULT_INVALIDITY_WEIGHTS = {
    'spanning': 0.45,
    'multi':    0.35,
    'overlong': 0.20,
}


def _build_invalidity_table(weights):
    """Normalize a {type_name: weight} dict into a [(weight, name)] list."""
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


def _gen_invalid_multi(t2, rng, max_total_len):
    """Concatenate two complete valid chunks. Total length ≤ max_total_len."""
    n = int(t2.chunk_emissions.shape[0])
    for _ in range(20):
        a, b = rng.integers(0, n), rng.integers(0, n)
        len_a = int(t2.chunk_lens[a])
        len_b = int(t2.chunk_lens[b])
        L = len_a + len_b
        if 2 <= L <= max_total_len:
            emissions = torch.cat([
                t2.chunk_emissions[a, :len_a],
                t2.chunk_emissions[b, :len_b],
            ])
            return emissions, L, INVALID_MULTI
    # Fallback: just shrink. Rare under typical batch sizes.
    a = int(rng.integers(0, n))
    len_a = int(t2.chunk_lens[a])
    return t2.chunk_emissions[a, :len_a], len_a, INVALID_MULTI


def _gen_invalid_spanning(t2, rng, max_total_len):
    """Tail of chunk A (including its terminator) + head of chunk B.

    j ∈ [1, len(A)], k ∈ [1, len(B)]; excludes the (j=len(A), k=len(B))
    case which is just multi.
    """
    n = int(t2.chunk_emissions.shape[0])
    for _ in range(20):
        a, b = rng.integers(0, n), rng.integers(0, n)
        len_a = int(t2.chunk_lens[a])
        len_b = int(t2.chunk_lens[b])
        if len_a < 1 or len_b < 1:
            continue
        j = int(rng.integers(1, len_a + 1))   # [1, len_a]
        k = int(rng.integers(1, len_b + 1))   # [1, len_b]
        if j == len_a and k == len_b:
            continue  # full + full = multi
        L = j + k
        if not (2 <= L <= max_total_len):
            continue
        # tail of A (last j positions): A[len_a - j : len_a]
        # head of B (first k positions): B[0 : k]
        emissions = torch.cat([
            t2.chunk_emissions[a, len_a - j : len_a],
            t2.chunk_emissions[b, :k],
        ])
        return emissions, L, INVALID_SPANNING
    # Fallback to multi.
    return _gen_invalid_multi(t2, rng, max_total_len)


def _gen_invalid_overlong(t2, rng, max_total_len, min_overlong=17):
    """Concatenate enough valid chunks to exceed the validity length cap.

    If max_total_len < min_overlong, falls back to multi (we can't fit
    an overlong chunk in the storage allotted).
    """
    if max_total_len < min_overlong:
        return _gen_invalid_multi(t2, rng, max_total_len)
    n = int(t2.chunk_emissions.shape[0])
    target_L = int(rng.integers(min_overlong, max_total_len + 1))
    parts = []
    cur_L = 0
    for _ in range(20):
        if cur_L >= target_L:
            break
        ci = int(rng.integers(0, n))
        len_c = int(t2.chunk_lens[ci])
        if len_c == 0:
            continue
        # If adding this chunk would overshoot, take a prefix of it.
        take = min(len_c, target_L - cur_L)
        parts.append(t2.chunk_emissions[ci, :take])
        cur_L += take
    if not parts or cur_L < min_overlong:
        # Couldn't reach overlong with available chunks; fall back to multi.
        return _gen_invalid_multi(t2, rng, max_total_len)
    emissions = torch.cat(parts, dim=0)
    return emissions, cur_L, INVALID_OVERLONG


_INVALIDITY_GENERATORS = {
    'spanning': _gen_invalid_spanning,
    'multi':    _gen_invalid_multi,
    'overlong': _gen_invalid_overlong,
}


def augment_t2_with_invalid(
    t2: T2Batch,
    invalidity_rate: float = 0.2,
    type_weights=None,
    storage_max_chunk_len: int = 24,
    rng=None,
) -> T2Batch:
    """Add invalid T2 chunks to an all-valid T2Batch.

    invalidity_rate: target fraction of invalid chunks in the
        returned batch. n_invalid = round(n_valid * r / (1 - r))
        so that n_invalid / (n_valid + n_invalid) ≈ r.
    type_weights: dict over {spanning, multi, overlong}.
        Defaults to DEFAULT_INVALIDITY_WEIGHTS.
    storage_max_chunk_len: tensor padding cap for chunk_emissions
        in the returned batch. Must accommodate the longest invalid
        chunk plus all valid chunks (which are bounded by the
        chunker's max_chunk_len, typically 16).

    The returned batch's chunk_emissions has shape
    (n_valid + n_invalid, storage_max_chunk_len, d_out). All other
    fields are zero on invalid rows except valid_mask=False,
    chunk_lens, chunk_type (INVALID_*), and chunk_emissions.
    """
    if type_weights is None:
        type_weights = DEFAULT_INVALIDITY_WEIGHTS
    if rng is None:
        rng = np.random.default_rng()

    n_valid = int(t2.chunk_emissions.shape[0])
    if n_valid == 0 or invalidity_rate <= 0:
        # Nothing to augment from / no invalids requested. Return
        # a re-padded version if storage_max_chunk_len differs.
        return _repad(t2, storage_max_chunk_len)

    n_invalid = int(round(n_valid * invalidity_rate / (1 - invalidity_rate)))
    if n_invalid <= 0:
        return _repad(t2, storage_max_chunk_len)

    if storage_max_chunk_len < t2.chunk_emissions.shape[1]:
        raise ValueError(
            f'storage_max_chunk_len={storage_max_chunk_len} is smaller '
            f'than the input batch padding {t2.chunk_emissions.shape[1]}')

    type_table = _build_invalidity_table(type_weights)
    device = t2.chunk_emissions.device
    d_out = t2.chunk_emissions.shape[-1]
    n_inputs = t2.reg_delta.shape[1]

    # Generate n_invalid invalid chunks.
    invalid_emissions = torch.zeros(
        n_invalid, storage_max_chunk_len, d_out, device=device)
    invalid_lens = np.zeros(n_invalid, dtype=np.int64)
    invalid_types = np.zeros(n_invalid, dtype=np.int8)
    for k in range(n_invalid):
        type_name = _sample_invalidity_type(rng, type_table)
        gen = _INVALIDITY_GENERATORS[type_name]
        emissions, L, type_code = gen(t2, rng, storage_max_chunk_len)
        invalid_emissions[k, :L] = emissions
        invalid_lens[k] = L
        invalid_types[k] = type_code

    # Repad valid chunks to storage_max_chunk_len.
    valid_padded = _repad(t2, storage_max_chunk_len)

    # Concatenate.
    final_emissions = torch.cat(
        [valid_padded.chunk_emissions, invalid_emissions], dim=0)
    final_lens = torch.cat([
        valid_padded.chunk_lens,
        torch.from_numpy(invalid_lens).to(device),
    ])
    final_valid = torch.cat([
        valid_padded.valid_mask,
        torch.zeros(n_invalid, dtype=torch.bool, device=device),
    ])
    final_reg_delta = torch.cat([
        valid_padded.reg_delta,
        torch.zeros(n_invalid, n_inputs, 32, dtype=torch.int32, device=device),
    ])
    final_chunk_type = torch.cat([
        valid_padded.chunk_type,
        torch.from_numpy(invalid_types).to(device),
    ])
    # Provenance fields are not meaningful for invalids; fill with -1.
    fill = torch.full((n_invalid,), -1, dtype=torch.int32, device=device)
    final_seq_idx = torch.cat([valid_padded.sequence_idx, fill])
    final_start = torch.cat([valid_padded.instr_start, fill])
    final_end = torch.cat([valid_padded.instr_end, fill])

    return T2Batch(
        chunk_emissions=final_emissions,
        chunk_lens=final_lens,
        valid_mask=final_valid,
        reg_delta=final_reg_delta,
        chunk_type=final_chunk_type,
        sequence_idx=final_seq_idx,
        instr_start=final_start,
        instr_end=final_end,
    )


def _repad(t2: T2Batch, storage_max_chunk_len: int) -> T2Batch:
    """Re-pad chunk_emissions to a possibly-larger max_chunk_len."""
    cur = t2.chunk_emissions.shape[1]
    if cur == storage_max_chunk_len:
        return t2
    if cur > storage_max_chunk_len:
        raise ValueError(
            f'cannot shrink padding from {cur} to {storage_max_chunk_len}')
    n, _, d = t2.chunk_emissions.shape
    repadded = torch.zeros(n, storage_max_chunk_len, d,
                            device=t2.chunk_emissions.device,
                            dtype=t2.chunk_emissions.dtype)
    repadded[:, :cur] = t2.chunk_emissions
    return T2Batch(
        chunk_emissions=repadded,
        chunk_lens=t2.chunk_lens,
        valid_mask=t2.valid_mask,
        reg_delta=t2.reg_delta,
        chunk_type=t2.chunk_type,
        sequence_idx=t2.sequence_idx,
        instr_start=t2.instr_start,
        instr_end=t2.instr_end,
    )
