"""Unified training batch.

A Batch carries chunks (some valid, some invalid) plus their supervision
targets. Losses see different views of the same batch:

  - Reconstruction CE (decoder):
        for c in chunks if c.valid:
            loss += CE(decode(encode(c.tokens)), c.tokens)
  - Validity magnitude:
        for c in chunks:
            loss += (||encode(c.tokens)|| - (1.0 if c.valid else 0.0)) ** 2
  - Value prediction: predict each output slot's canonical value per anchor
    from the vector + canonical input values (the out_regs oracle target).
  - Register-identity aux + value prediction read the per-row aux targets
    (live_in/out masks, slot regs, out_regs).

Storage shape: encoder consumes flat (B, max_tokens) tokens with a
padding mask, so the binary format stores tokens flatly. Per-instruction
segmentation isn't preserved across the wire — relabeling happens at
generation time when the source list[Instruction] is still in hand.
"""

import struct
from dataclasses import dataclass

import numpy as np

from tokenizer import PAD, encode_instruction
from tokenizer.tokenizer import MAX_INSTR_TOKENS

from .compare import (
    AUX_CE_IGNORE, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS, N_REGS,
    precompute_chunk,
)
from .generate import (
    collect_groups, random_instruction, random_relabel,
    sample_injection_tuples,
)


# ===========================================================================
# Generic binary-format machinery
# ===========================================================================

@dataclass(frozen=True)
class BodyField:
    """One field in a batch's serialized body. shape entries are header
    field names (str — looked up at I/O time) or constant ints."""
    name: str
    dtype: np.dtype
    shape: tuple


class BinaryFormat:
    """Stream format for a sequence of dataclass-shaped batches.

    No stream header — every batch is self-describing, so RVT streams
    concatenate and repeat with plain `cat` (cat a.rvt b.rvt, or
    cat corpus.rvt corpus.rvt for a second epoch).

    Per batch:
      prefix: 4-byte magic + 1-byte version + N dtype chars
      header: one uint32 per symbolic dimension
      body:   the body fields as contiguous raw arrays
    """

    def __init__(self, magic, version, header_fields, body_fields):
        if len(magic) != 4:
            raise ValueError('magic must be exactly 4 bytes')
        self.magic = magic
        self.version = version
        self.header_fields = list(header_fields)
        self.body_fields = list(body_fields)
        self._header_struct = struct.Struct('<' + 'I' * len(header_fields))
        self._dtype_chars = b''.join(
            f.dtype.char.encode() for f in body_fields)
        # Per-batch self-describing prefix: magic + version + dtype
        # signature. There is no stream header — every batch carries this,
        # which is what lets RVT streams concatenate/repeat with `cat`.
        self._prefix_struct = struct.Struct(
            f'<4sB{len(self._dtype_chars)}s')

    @property
    def name(self):
        return self.magic.rstrip(b'\x00').decode().lower()

    @property
    def batch_header(self):
        return self._header_struct

    @property
    def batch_prefix_size(self):
        """Bytes of the self-describing prefix (magic+version+dtype) that
        precede each batch's dimension header."""
        return self._prefix_struct.size

    @property
    def dtype_chars(self):
        return self._dtype_chars

    def _shape(self, fld, header):
        return tuple(header[s] if isinstance(s, str) else s
                     for s in fld.shape)

    def body_size(self, *header_values):
        header = dict(zip(self.header_fields, header_values))
        total = 0
        for fld in self.body_fields:
            shape = self._shape(fld, header)
            n = 1
            for s in shape:
                n *= s
            total += n * fld.dtype.itemsize
        return total

    def _extract_header(self, batch):
        header = {}
        for fld in self.body_fields:
            arr = getattr(batch, fld.name)
            for i, s in enumerate(fld.shape):
                if isinstance(s, str) and s not in header:
                    header[s] = int(arr.shape[i])
        return header

    def _read_prefix(self, f):
        """Read + validate one batch's self-describing prefix (magic,
        version, dtype). Returns the prefix bytes, or None at clean EOF."""
        buf = f.read(self._prefix_struct.size)
        if len(buf) == 0:
            return None
        if len(buf) < self._prefix_struct.size:
            raise EOFError(f'Truncated batch prefix ({len(buf)} bytes)')
        magic, version, dtype_chars = self._prefix_struct.unpack(buf)
        if magic != self.magic:
            raise ValueError(
                f'Bad magic: {magic!r} (expected {self.magic!r})')
        if version != self.version:
            raise ValueError(
                f'Unsupported {self.name} version: {version}')
        if dtype_chars != self._dtype_chars:
            raise ValueError(f'Dtype mismatch: {dtype_chars!r}')
        return buf

    def write_batch(self, f, batch):
        header = self._extract_header(batch)
        f.write(self._prefix_struct.pack(
            self.magic, self.version, self._dtype_chars))
        f.write(self._header_struct.pack(
            *[header[n] for n in self.header_fields]))
        for fld in self.body_fields:
            f.write(getattr(batch, fld.name).tobytes())

    def read_batch(self, f, batch_class):
        if self._read_prefix(f) is None:
            return None
        header_buf = f.read(self._header_struct.size)
        if len(header_buf) < self._header_struct.size:
            raise EOFError(
                f'Truncated batch header ({len(header_buf)} bytes)')
        header = dict(zip(self.header_fields,
                          self._header_struct.unpack(header_buf)))
        kwargs = {}
        for fld in self.body_fields:
            shape = self._shape(fld, header)
            n_bytes = int(np.prod(shape)) * fld.dtype.itemsize
            buf = f.read(n_bytes) if n_bytes > 0 else b''
            if n_bytes > 0 and len(buf) < n_bytes:
                raise EOFError(
                    f'Truncated body for {fld.name} '
                    f'(got {len(buf)}, expected {n_bytes})')
            kwargs[fld.name] = np.frombuffer(
                buf, dtype=fld.dtype).reshape(shape).copy()
        return batch_class(**kwargs)

    def read_batch_bytes(self, f):
        prefix = self._read_prefix(f)
        if prefix is None:
            return None
        header_buf = f.read(self._header_struct.size)
        if len(header_buf) < self._header_struct.size:
            raise EOFError(
                f'Truncated batch header ({len(header_buf)} bytes)')
        header_vals = self._header_struct.unpack(header_buf)
        body_size = self.body_size(*header_vals)
        body = f.read(body_size) if body_size > 0 else b''
        if body_size > 0 and len(body) < body_size:
            raise EOFError(
                f'Truncated body (got {len(body)}, expected {body_size})')
        return prefix + header_buf + body

    def reader(self, f, batch_class):
        while True:
            batch = self.read_batch(f, batch_class)
            if batch is None:
                break
            yield batch


# ===========================================================================
# In-memory chunk + Batch
# ===========================================================================

@dataclass
class Chunk:
    """One row of a batch.

    tokens:        the encoder input (a flat token sequence).
    valid:         True if this is a complete, well-formed thought —
                   for T1, exactly one decodable instruction.
    instructions:  source instructions if available; None for invalid
                   windows whose tokens don't decode (partial / spanning
                   / bogus). Even decodable invalid windows (multi) carry
                   None because we never operate on their structure.
    """
    tokens: list
    valid: bool
    instructions: list | None


@dataclass
class Batch:
    """Serialized batch: flat tokens + validity + per-instruction
    segmentation + supervision payloads.

    All arrays sized for B chunks; instr_lens sized for max_n_instrs
    columns (0 past actual count, 0 for invalid).
    """
    tokens:         np.ndarray   # (B, max_tokens) int8
    token_lens:     np.ndarray   # (B,) int32
    valid:          np.ndarray   # (B,) bool — complete instruction window
    instr_lens:     np.ndarray   # (B, max_n_instrs) int32
    # T2 aux register-identity targets — always sized B; rows without a
    # valid Precomputed (invalid windows, mem-op chunks) get zero
    # masks and AUX_CE_IGNORE in the slot arrays.
    live_in_mask:   np.ndarray   # (B, N_REGS) bool — behavioral inputs
    live_out_mask:  np.ndarray   # (B, N_REGS) bool — actually-modified regs
    pc_writes:      np.ndarray   # (B,) bool — pc_explicit flag
    in_slot_regs:   np.ndarray   # (B, MAX_INPUT_SLOTS) int8 — reg or AUX_CE_IGNORE
    out_slot_regs:  np.ndarray   # (B, MAX_OUTPUT_SLOTS) int8
    # Full post-execution register file per anchor — the value-prediction
    # oracle target. Computed in the (parallel) gen workers from the same
    # Precomputed as the binding aux, so the trainer reads it instead of
    # recomputing single-threaded. OB = B when shipped (T2 twin mode), 0
    # otherwise; out_regs_valid marks decodable non-mem rows.
    out_regs:       np.ndarray   # (OB, n_anchors, N_REGS) int32
    out_regs_valid: np.ndarray   # (OB,) bool


RVT_FORMAT = BinaryFormat(
    magic=b'RVT\x00', version=8,
    header_fields=['B', 'max_tokens', 'max_n_instrs', 'n_anchors', 'OB'],
    body_fields=[
        BodyField('tokens', np.dtype(np.int8), ('B', 'max_tokens')),
        BodyField('token_lens', np.dtype(np.int32), ('B',)),
        BodyField('valid', np.dtype(np.bool_), ('B',)),
        BodyField('instr_lens', np.dtype(np.int32), ('B', 'max_n_instrs')),
        BodyField('out_regs', np.dtype(np.int32), ('OB', 'n_anchors', N_REGS)),
        BodyField('out_regs_valid', np.dtype(np.bool_), ('OB',)),
        BodyField('live_in_mask', np.dtype(np.bool_), ('B', N_REGS)),
        BodyField('live_out_mask', np.dtype(np.bool_), ('B', N_REGS)),
        BodyField('pc_writes', np.dtype(np.bool_), ('B',)),
        BodyField('in_slot_regs', np.dtype(np.int8),
                  ('B', MAX_INPUT_SLOTS)),
        BodyField('out_slot_regs', np.dtype(np.int8),
                  ('B', MAX_OUTPUT_SLOTS)),
    ],
)


# ===========================================================================
# Pack / unpack
# ===========================================================================

@dataclass
class AuxPayload:
    """Per-row T2 register-identity targets. Length = actual_B; rows
    without a valid Precomputed (invalid windows, mem-op chunks) get
    zero masks, pc_writes=False, and AUX_CE_IGNORE in the slot arrays."""
    live_in_mask: np.ndarray    # (actual_B, N_REGS) bool
    live_out_mask: np.ndarray   # (actual_B, N_REGS) bool
    pc_writes: np.ndarray       # (actual_B,) bool
    in_slot_regs: np.ndarray    # (actual_B, MAX_INPUT_SLOTS) int8
    out_slot_regs: np.ndarray   # (actual_B, MAX_OUTPUT_SLOTS) int8


def _empty_aux(n):
    """Build an AuxPayload sized for n rows with no behavioral targets
    (used when callers don't have Precomputed objects available; T2
    aux losses skip via the AUX_CE_IGNORE sentinel)."""
    return AuxPayload(
        live_in_mask=np.zeros((n, N_REGS), dtype=bool),
        live_out_mask=np.zeros((n, N_REGS), dtype=bool),
        pc_writes=np.zeros((n,), dtype=bool),
        in_slot_regs=np.full((n, MAX_INPUT_SLOTS), AUX_CE_IGNORE,
                             dtype=np.int8),
        out_slot_regs=np.full((n, MAX_OUTPUT_SLOTS), AUX_CE_IGNORE,
                              dtype=np.int8),
    )


def _aux_from_precomputeds(precomputeds):
    """Build an AuxPayload from a list of (Precomputed | None). None
    entries produce zero-mask / AUX_CE_IGNORE rows."""
    n = len(precomputeds)
    aux = _empty_aux(n)
    for i, pre in enumerate(precomputeds):
        if pre is None:
            continue
        aux.live_in_mask[i] = pre.live_in_mask
        aux.live_out_mask[i] = pre.live_out_mask
        aux.pc_writes[i] = pre.pc_explicit
        aux.in_slot_regs[i] = pre.in_slot_regs
        aux.out_slot_regs[i] = pre.out_slot_regs
    return aux


def _out_regs_from_precomputeds(precomputeds, n_anchors):
    """Full post-execution register file per anchor for each row, from the
    already-computed Precomputeds. None rows (invalid / mem-op) → zeros with
    valid=False. Same result the trainer would recompute single-threaded;
    computing it here ships it from the (parallel) gen workers instead."""
    n = len(precomputeds)
    out = np.zeros((n, n_anchors, N_REGS), dtype=np.int32)
    valid = np.zeros(n, dtype=bool)
    for i, pre in enumerate(precomputeds):
        if pre is None:
            continue
        out[i] = pre.out_regs
        valid[i] = True
    return out, valid


def pack_batch(chunks, *,
               target_B, target_max_tokens,
               target_max_n_instrs,
               out_regs=None, out_regs_valid=None,
               aux_payload=None):
    """Build a Batch from chunks with the canonical out_regs oracle.

    chunks:    list[Chunk]

    target_*:  required fixed shape parameters. Arrays are padded out to
               these exact sizes (raises ValueError on overflow), so every
               batch in a run shares one shape — otherwise downstream
               PyTorch caching allocators fragment over training runs of
               any meaningful length. collect_into_batches derives them
               from the chunking rule's length cap. Padding rows have
               valid=False, token_lens=0, instr_lens=0 — naturally zero
               loss, no special handling needed in training.

    aux_payload:
        Optional AuxPayload of T2 register-identity targets. When
        None, all aux fields are zero-mask / AUX_CE_IGNORE — T2 aux
        losses on those rows are skipped via the sentinel.
    """
    if not chunks:
        raise ValueError('pack_batch requires at least one chunk')

    actual_B = len(chunks)
    actual_max_tokens = max(len(c.tokens) for c in chunks)

    n_instrs_per_chunk = [
        len(c.instructions) if c.valid and c.instructions is not None
        else 0
        for c in chunks
    ]
    actual_max_n_instrs = max(max(n_instrs_per_chunk, default=0), 1)

    # Fixed output shape: pad the actual data up to the caller-supplied
    # targets (always set — collect_into_batches derives them from the
    # rule's length cap). The overflow checks below reject data that
    # doesn't fit.
    B = target_B
    max_tokens = target_max_tokens
    max_n_instrs = target_max_n_instrs

    if actual_B > B:
        raise ValueError(
            f'B={actual_B} exceeds target_B={B}')
    if actual_max_tokens > max_tokens:
        raise ValueError(
            f'max_tokens={actual_max_tokens} exceeds '
            f'target_max_tokens={max_tokens}')
    if actual_max_n_instrs > max_n_instrs:
        raise ValueError(
            f'max_n_instrs={actual_max_n_instrs} exceeds '
            f'target_max_n_instrs={max_n_instrs}')

    tokens = np.full((B, max_tokens), PAD, dtype=np.int8)
    token_lens = np.zeros(B, dtype=np.int32)
    valid = np.zeros(B, dtype=bool)
    instr_lens = np.zeros((B, max_n_instrs), dtype=np.int32)
    for i, c in enumerate(chunks):
        n = len(c.tokens)
        tokens[i, :n] = c.tokens
        token_lens[i] = n
        valid[i] = c.valid
        if c.valid and c.instructions is not None:
            for j, instr in enumerate(c.instructions):
                instr_lens[i, j] = len(encode_instruction(instr))

    # Anchor count comes from the out_regs payload (every rule ships it now).
    n_anchors = int(out_regs.shape[1]) if out_regs is not None else 0

    # out_regs payload (OB = B in T2 mode, 0 otherwise).
    if out_regs is None:
        out_regs_arr = np.zeros((0, n_anchors, N_REGS), dtype=np.int32)
        out_regs_valid_arr = np.zeros((0,), dtype=bool)
    else:
        if out_regs.shape[0] != actual_B:
            raise ValueError(
                f'out_regs first dim ({out_regs.shape[0]}) must equal '
                f'actual_B ({actual_B})')
        out_regs_arr = np.zeros((B, n_anchors, N_REGS), dtype=np.int32)
        out_regs_arr[:actual_B] = out_regs
        out_regs_valid_arr = np.zeros((B,), dtype=bool)
        out_regs_valid_arr[:actual_B] = out_regs_valid

    aux = aux_payload if aux_payload is not None else _empty_aux(actual_B)
    if aux.live_in_mask.shape[0] != actual_B:
        raise ValueError(
            f'aux_payload first dim ({aux.live_in_mask.shape[0]}) must '
            f'equal actual_B ({actual_B})')
    live_in_arr = np.zeros((B, N_REGS), dtype=bool)
    live_in_arr[:actual_B] = aux.live_in_mask
    live_out_arr = np.zeros((B, N_REGS), dtype=bool)
    live_out_arr[:actual_B] = aux.live_out_mask
    pc_writes_arr = np.zeros((B,), dtype=bool)
    pc_writes_arr[:actual_B] = aux.pc_writes
    in_slot_arr = np.full((B, MAX_INPUT_SLOTS), AUX_CE_IGNORE, dtype=np.int8)
    in_slot_arr[:actual_B] = aux.in_slot_regs
    out_slot_arr = np.full((B, MAX_OUTPUT_SLOTS), AUX_CE_IGNORE, dtype=np.int8)
    out_slot_arr[:actual_B] = aux.out_slot_regs

    return Batch(tokens=tokens, token_lens=token_lens, valid=valid,
                 instr_lens=instr_lens,
                 out_regs=out_regs_arr,
                 out_regs_valid=out_regs_valid_arr,
                 live_in_mask=live_in_arr,
                 live_out_mask=live_out_arr,
                 pc_writes=pc_writes_arr,
                 in_slot_regs=in_slot_arr,
                 out_slot_regs=out_slot_arr)


def padding_mask(batch):
    """Return a (B, max_tokens) bool array, True where padded."""
    B, max_tokens = batch.tokens.shape
    arange = np.arange(max_tokens)
    return arange[None, :] >= batch.token_lens[:, None]


# ===========================================================================
# Generation: chunks -> twins -> Batch
# ===========================================================================

# Memory ops have no Precomputed in V1 scope; twin/aux construction
# drops chunks containing them.
_MEM_OPS = frozenset({'LB', 'LBU', 'LH', 'LHU', 'LW', 'SB', 'SH', 'SW'})


def _has_mem_ops(instrs):
    return any(i.opcode in _MEM_OPS for i in instrs)


def _instr_chunk_to_tokens(instrs):
    out = []
    for i in instrs:
        out.extend(encode_instruction(i))
    return out


def _make_valid_chunk(instrs):
    return Chunk(tokens=_instr_chunk_to_tokens(instrs),
                 valid=True, instructions=list(instrs))


def _make_invalid_chunk(tokens):
    return Chunk(tokens=list(tokens), valid=False, instructions=None)


def build_twins(chunks, twins, anchor_states, rng):
    """Add `twins` relabeled copies of each valid source chunk and build
    the parallel T2 register-identity aux targets.

    Cluster structure: each valid source plus its twins form a cluster
    of size (twins + 1) — twins are equivalence-preserving relabelings.
    Memory-op chunks keep no twins (their Precomputed is unavailable in
    V1 scope) and contribute an empty aux row.

    Returns (chunks_out, aux, out_regs, out_regs_valid) where chunks_out
    extends `chunks` with the new twins, aux is the parallel AuxPayload, and
    out_regs/out_regs_valid are the per-row post-exec register file (the
    value-prediction oracle target, shipped from the gen workers).
    """
    # Filter out memory-op chunks at the source list — twins inherit
    # their parent's filtering.
    valid_sources = []
    for ch in chunks:
        if not ch.valid:
            valid_sources.append(None)  # placeholder; no twins
            continue
        if _has_mem_ops(ch.instructions):
            valid_sources.append(None)
            continue
        valid_sources.append(ch)

    # Build the chunks_out list: original chunks + twins after each
    # valid source. Track which output rows have a usable Precomputed.
    chunks_out = []
    has_pre = []           # one entry per output row
    for original, source in zip(chunks, valid_sources):
        chunks_out.append(original)
        if source is None:
            has_pre.append(False)
            continue
        has_pre.append(True)
        for _ in range(twins):
            twin_instrs = random_relabel(source.instructions, rng)
            chunks_out.append(_make_valid_chunk(twin_instrs))
            has_pre.append(True)

    # Aux targets: rows with a usable source get their Precomputed;
    # mem-op / invalid rows get None → empty aux row.
    aux_pre = [
        precompute_chunk(ch.instructions, anchor_states) if hp else None
        for ch, hp in zip(chunks_out, has_pre)
    ]
    aux = _aux_from_precomputeds(aux_pre)
    out_regs, out_regs_valid = _out_regs_from_precomputeds(
        aux_pre, anchor_states.shape[0])

    return chunks_out, aux, out_regs, out_regs_valid


# ===========================================================================
# Top-level: instruction stream → Batch stream
# ===========================================================================

def _yield_random_instructions(rng, opcode_table=None):
    while True:
        yield random_instruction(rng, opcode_table=opcode_table)


def generate_chunks(rule, rng, *, opcode_table=None,
                    eq_rate=0.0, eq_max_per_class=8, eq_min_per_class=0,
                    eq_boost=None):
    """Yield Chunk objects (all valid) from a random-instruction stream
    grouped by `rule`.

    If eq_rate > 0, after each grouped chunk emit, with probability
    eq_rate, also emit a fresh tuple of MANIFEST-equivalent instructions
    as additional chunks. This biases the stream toward chunks that the
    encoder must learn to collapse together.
    """
    instr_iter = _yield_random_instructions(rng, opcode_table=opcode_table)
    for group in collect_groups(instr_iter, rule):
        yield _make_valid_chunk(group)
        if eq_rate > 0 and rng.random() < eq_rate:
            tuple_instrs = sample_injection_tuples(
                target_count=2, max_per_class=eq_max_per_class, rng=rng,
                min_per_class=eq_min_per_class, boost=eq_boost)
            for instr in tuple_instrs:
                yield _make_valid_chunk([instr])


def collect_into_batches(chunks_iter, *, batch_size, twins,
                         anchor_states, rng,
                         invalid_rate=0.0, invalid_provider=None,
                         max_invalid_window=None,
                         max_chunk_len):
    """Consume a Chunk stream, build Batches of `batch_size` rows.

    invalid_rate>0 mixes invalid windows in at the given fraction of
    `batch_size`. invalid_provider() must return a list[int] token
    sequence (typically datagen.invalidity.generate_invalid).

    Each batch ends up with `n_invalid = round(batch_size * invalid_rate)`
    invalid rows + `batch_size - n_invalid` valid source slots; twins
    are added on top of the valid-source slots, growing the batch.

    Every rule (single → T1, multi-instr → T2) ships the canonical out_regs
    oracle via build_twins (the T1/T2 unification).

    max_chunk_len: maximum number of instructions per chunk under the
                   chunking rule (the rule's length cap; 1 for 'single').
                   Required: every emitted batch has identical shape — B,
                   max_tokens, max_n_instrs all padded to deterministic
                   upper bounds derived from it. This prevents PyTorch's
                   caching allocator from fragmenting over a long training
                   run with stochastically-varying batch shapes (a real
                   OOM trigger we hit at ~step 500 of t1_cosine_full).
    """
    if invalid_rate > 0 and invalid_provider is None:
        raise ValueError('invalid_rate > 0 requires invalid_provider')

    n_invalid = int(round(batch_size * invalid_rate))
    n_valid_sources = batch_size - n_invalid

    # Fixed-shape upper bounds derived from max_chunk_len. These match
    # the batch shape with no mem-op filtering: all valid sources
    # contribute (twins+1) rows; invalids contribute 1 row each. Mem-op
    # filtering and length variation produce smaller actual shapes, which
    # then get padded up to these targets.
    cluster_size = twins + 1
    target_B = n_valid_sources * cluster_size + n_invalid
    target_max_n_instrs = max_chunk_len
    target_max_tokens = max_chunk_len * MAX_INSTR_TOKENS
    if max_invalid_window is not None:
        target_max_tokens = max(target_max_tokens, max_invalid_window)

    buf = []
    for ch in chunks_iter:
        buf.append(ch)
        if len(buf) >= n_valid_sources:
            valid_sources = buf[:n_valid_sources]
            buf = buf[n_valid_sources:]
            invalids = [_make_invalid_chunk(invalid_provider())
                        for _ in range(n_invalid)]
            mixed = valid_sources + invalids
            chunks_out, aux, out_regs, out_regs_valid = build_twins(
                mixed, twins=twins,
                anchor_states=anchor_states, rng=rng)
            yield pack_batch(
                chunks_out,
                target_B=target_B,
                target_max_tokens=target_max_tokens,
                target_max_n_instrs=target_max_n_instrs,
                out_regs=out_regs,
                out_regs_valid=out_regs_valid,
                aux_payload=aux)
