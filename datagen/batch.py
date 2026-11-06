"""Unified training batch.

A Batch carries chunks (some valid, some invalid) plus their supervision
targets. Losses see different views of the same batch:

  - Reconstruction CE:
        for c in chunks if c.valid:
            loss += CE(decode(encode(c.tokens)), c.tokens)
  - Validity magnitude:
        for c in chunks:
            loss += (||encode(c.tokens)|| - (1.0 if c.valid else 0.0)) ** 2
  - Value prediction: predict every output's value per anchor from the
    essence and the chunk's input values in canonical role order (out_values /
    in_roles / in_role_values).
  - Register-identity aux reads the per-row aux targets (live_in/out masks,
    slot regs).

Storage shape: encoder consumes flat (B, max_tokens) tokens with a
padding mask, so the binary format stores tokens flatly. Per-instruction
segmentation isn't preserved across the wire — relabeling happens at
generation time when the source list[Instruction] is still in hand.
"""

import struct
import zlib
from dataclasses import dataclass

import numpy as np

from tokenizer import PAD, encode_instruction
from tokenizer.tokenizer import MAX_INSTR_TOKENS

from .compare import (
    AUX_CE_IGNORE, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS, N_REGS, N_WITNESS,
    n_positions, placements, precompute_chunk,
)
from .generate import (
    chain_group, collect_groups, random_instruction,
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
      prefix: 4-byte magic + N dtype chars
      header: one uint32 per symbolic dimension
      length: uint64, the compressed body's size
      body:   the body fields as contiguous raw arrays, zlib-compressed
              (level 1). The distance fields repeat shared witness columns
              across positions and placements, so a len-2 batch shrinks about
              18x, and a trainer's main thread moves only the compressed bytes.
    """

    _LEN = struct.Struct('<Q')

    def __init__(self, magic, header_fields, body_fields, header_bounds=None):
        if len(magic) != 4:
            raise ValueError('magic must be exactly 4 bytes')
        self.magic = magic
        self.header_fields = list(header_fields)
        self.body_fields = list(body_fields)
        # Per-field (lo, hi) sanity bounds on the dimension header, checked on
        # every read (see _check_header_bounds). Structural validation: a torn
        # or corrupt stream can present a syntactically-valid header with insane
        # dims, and we must reject it before trusting it to size a read.
        self.header_bounds = dict(header_bounds or {})
        self._header_struct = struct.Struct('<' + 'I' * len(header_fields))
        self._dtype_chars = b''.join(
            f.dtype.char.encode() for f in body_fields)
        # Per-batch self-describing prefix: magic + dtype signature. There is
        # no stream header — every batch carries this, which is what lets RVT
        # streams concatenate/repeat with `cat`. No version field: the format
        # is singular and global; any change to it is immediate everywhere, so
        # there is no compatibility to negotiate. The dtype signature still
        # rejects a schema-mismatched stream (a body-field change shifts it).
        self._prefix_struct = struct.Struct(
            f'<4s{len(self._dtype_chars)}s')

    @property
    def name(self):
        return self.magic.rstrip(b'\x00').decode().lower()

    @property
    def batch_header(self):
        return self._header_struct

    @property
    def batch_prefix_size(self):
        """Bytes of the self-describing prefix (magic+dtype) that precede each
        batch's dimension header."""
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
        """Read + validate one batch's self-describing prefix (magic, dtype).
        Returns the prefix bytes, or None at clean EOF."""
        buf = f.read(self._prefix_struct.size)
        if len(buf) == 0:
            return None
        if len(buf) < self._prefix_struct.size:
            raise EOFError(f'Truncated batch prefix ({len(buf)} bytes)')
        magic, dtype_chars = self._prefix_struct.unpack(buf)
        if magic != self.magic:
            raise ValueError(
                f'Bad magic: {magic!r} (expected {self.magic!r})')
        if dtype_chars != self._dtype_chars:
            raise ValueError(f'Dtype mismatch: {dtype_chars!r}')
        return buf

    def _check_header_bounds(self, header_values):
        """Reject a structurally-invalid dimension header before we trust it to
        size a read. A field whose value falls outside its declared (lo, hi)
        bound means the stream is torn/corrupt, not a different version."""
        for name, val in zip(self.header_fields, header_values):
            bounds = self.header_bounds.get(name)
            if bounds is None:
                continue
            lo, hi = bounds
            if not (lo <= val <= hi):
                dims = dict(zip(self.header_fields, header_values))
                raise ValueError(
                    f'Invalid {self.name} header: {name}={val} outside '
                    f'[{lo}, {hi}] ({dims})')

    def peek_magic(self, f):
        """Confirm the stream starts with this format's magic WITHOUT consuming
        it (the batch is read whole afterward). Raises ValueError otherwise.
        Requires a buffered/peekable stream."""
        n = len(self.magic)
        buf = b''
        while len(buf) < n:
            prev = len(buf)
            buf = f.peek(n)
            if len(buf) <= prev:
                raise ValueError(
                    f'Truncated stream: {len(buf)} bytes before magic')
        magic = bytes(buf[:n])
        if magic != self.magic:
            raise ValueError(
                f'Bad magic: {magic!r} (expected {self.magic!r})')

    def write_batch(self, f, batch):
        header = self._extract_header(batch)
        f.write(self._prefix_struct.pack(self.magic, self._dtype_chars))
        f.write(self._header_struct.pack(
            *[header[n] for n in self.header_fields]))
        body = zlib.compress(b''.join(
            np.ascontiguousarray(getattr(batch, fld.name)).tobytes()
            for fld in self.body_fields), 1)
        f.write(self._LEN.pack(len(body)))
        f.write(body)

    def _read_body(self, f, header_vals):
        """The compressed body after a header, validated against the size the
        header implies."""
        lb = f.read(self._LEN.size)
        if len(lb) < self._LEN.size:
            raise EOFError(f'Truncated body length ({len(lb)} bytes)')
        (n,) = self._LEN.unpack(lb)
        size = self.body_size(*header_vals)
        if n > size + 1024 + size // 100:
            raise ValueError(f'Invalid {self.name} body length {n} for a '
                             f'{size}-byte body')
        body = f.read(n) if n else b''
        if len(body) < n:
            raise EOFError(f'Truncated body (got {len(body)}, expected {n})')
        return lb + body

    def read_batch(self, f, batch_class):
        if self._read_prefix(f) is None:
            return None
        header_buf = f.read(self._header_struct.size)
        if len(header_buf) < self._header_struct.size:
            raise EOFError(
                f'Truncated batch header ({len(header_buf)} bytes)')
        header_vals = self._header_struct.unpack(header_buf)
        self._check_header_bounds(header_vals)
        header = dict(zip(self.header_fields, header_vals))
        raw = zlib.decompress(self._read_body(f, header_vals)[self._LEN.size:])
        if len(raw) != self.body_size(*header_vals):
            raise ValueError(f'Invalid {self.name} body: {len(raw)} bytes, '
                             f'header implies {self.body_size(*header_vals)}')
        kwargs, off = {}, 0
        for fld in self.body_fields:
            shape = self._shape(fld, header)
            n_bytes = int(np.prod(shape)) * fld.dtype.itemsize
            kwargs[fld.name] = np.frombuffer(
                raw, dtype=fld.dtype, count=int(np.prod(shape)),
                offset=off).reshape(shape).copy()
            off += n_bytes
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
        self._check_header_bounds(header_vals)
        return prefix + header_buf + self._read_body(f, header_vals)

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
    live_out_mask:  np.ndarray   # (B, N_REGS) bool — written regs
    pc_writes:      np.ndarray   # (B,) bool — pc_explicit flag
    in_slot_regs:   np.ndarray   # (B, MAX_INPUT_SLOTS) int8 — reg or AUX_CE_IGNORE
    out_slot_regs:  np.ndarray   # (B, MAX_OUTPUT_SLOTS) int8
    # Per output slot, the external input registers it depends on, in the
    # order the evaluator gathers their values (AUX_CE_IGNORE padded).
    out_inputs:     np.ndarray   # (B, MAX_OUTPUT_SLOTS, MAX_INPUT_SLOTS) int8
    # The chunk's inputs in canonical role order (`datagen.roles`;
    # AUX_CE_IGNORE padded), and at each anchor state each output's value after
    # the chunk (aligned with out_slot_regs) — the value-prediction target —
    # and the values of those inputs, 0 past the last. Each row is executed
    # from one structured draw placed on its own input registers; these are
    # everything value prediction reads of that execution. Computed in the
    # gen workers. OB = B; out_regs_valid marks decodable non-mem rows.
    in_roles:         np.ndarray   # (B, n_positions) int8
    out_values:       np.ndarray   # (OB, n_anchors, max_n_instrs) int32
    in_role_values:   np.ndarray   # (OB, n_anchors, n_positions) int32
    out_regs_valid:   np.ndarray   # (OB,) bool
    # The behavioral distance's fields (`datagen.compare.Precomputed.rf_*`):
    # each row's register file after the chunk at the shared witness states,
    # under every placement of its canonical inputs on the n_positions
    # register positions — the final value on each position, and of each
    # register written but not read — with the bits each column never changes
    # and each written-only register's squared distance from its old value.
    # n_wo counts the written-only registers. All 0 on rows without targets.
    rf_pos:      np.ndarray   # (B, n_places, n_positions, N_WITNESS) int32
    rf_pos_zero: np.ndarray   # (B, n_places, n_positions) int64
    rf_pos_one:  np.ndarray   # (B, n_places, n_positions) int64
    rf_wo:       np.ndarray   # (B, n_places, max_n_instrs, N_WITNESS) int32
    rf_wo_zero:  np.ndarray   # (B, max_n_instrs) int64
    rf_wo_one:   np.ndarray   # (B, max_n_instrs) int64
    rf_wo_free:  np.ndarray   # (B, max_n_instrs) float32
    n_wo:        np.ndarray   # (B,) int8


RVT_FORMAT = BinaryFormat(
    magic=b'RVT\x00',
    header_fields=['B', 'max_tokens', 'max_n_instrs', 'n_anchors', 'OB',
                   'n_places', 'n_positions'],
    body_fields=[
        BodyField('tokens', np.dtype(np.int8), ('B', 'max_tokens')),
        BodyField('token_lens', np.dtype(np.int32), ('B',)),
        BodyField('valid', np.dtype(np.bool_), ('B',)),
        BodyField('instr_lens', np.dtype(np.int32), ('B', 'max_n_instrs')),
        BodyField('in_roles', np.dtype(np.int8), ('B', 'n_positions')),
        BodyField('out_values', np.dtype(np.int32),
                  ('OB', 'n_anchors', 'max_n_instrs')),
        BodyField('in_role_values', np.dtype(np.int32),
                  ('OB', 'n_anchors', 'n_positions')),
        BodyField('out_regs_valid', np.dtype(np.bool_), ('OB',)),
        BodyField('live_in_mask', np.dtype(np.bool_), ('B', N_REGS)),
        BodyField('live_out_mask', np.dtype(np.bool_), ('B', N_REGS)),
        BodyField('pc_writes', np.dtype(np.bool_), ('B',)),
        BodyField('in_slot_regs', np.dtype(np.int8),
                  ('B', MAX_INPUT_SLOTS)),
        BodyField('out_slot_regs', np.dtype(np.int8),
                  ('B', MAX_OUTPUT_SLOTS)),
        BodyField('out_inputs', np.dtype(np.int8),
                  ('B', MAX_OUTPUT_SLOTS, MAX_INPUT_SLOTS)),
        BodyField('rf_pos', np.dtype(np.int32),
                  ('B', 'n_places', 'n_positions', N_WITNESS)),
        BodyField('rf_pos_zero', np.dtype(np.int64), ('B', 'n_places', 'n_positions')),
        BodyField('rf_pos_one', np.dtype(np.int64), ('B', 'n_places', 'n_positions')),
        BodyField('rf_wo', np.dtype(np.int32),
                  ('B', 'n_places', 'max_n_instrs', N_WITNESS)),
        BodyField('rf_wo_zero', np.dtype(np.int64), ('B', 'max_n_instrs')),
        BodyField('rf_wo_one', np.dtype(np.int64), ('B', 'max_n_instrs')),
        BodyField('rf_wo_free', np.dtype(np.float32), ('B', 'max_n_instrs')),
        BodyField('n_wo', np.dtype(np.int8), ('B',)),
    ],
    # Sanity bounds on the dimension header — reject a torn/corrupt stream
    # whose dims are garbage before sizing a read off them.
    header_bounds={
        'B': (1, 1_000_000),
        'max_tokens': (1, 100_000),
        'max_n_instrs': (1, 100_000),
        'n_anchors': (0, 256),
        'OB': (0, 1_000_000),
        'n_places': (1, 40320),
        'n_positions': (1, MAX_INPUT_SLOTS),
    },
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
    out_inputs: np.ndarray      # (actual_B, MAX_OUTPUT_SLOTS, MAX_INPUT_SLOTS) int8
    precomputed: list           # per row, its Precomputed (in_roles, rf_*) or None


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
        out_inputs=np.full((n, MAX_OUTPUT_SLOTS, MAX_INPUT_SLOTS),
                           AUX_CE_IGNORE, dtype=np.int8),
        precomputed=[None] * n,
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
        aux.out_inputs[i] = pre.out_inputs
        aux.precomputed[i] = pre
    return aux


def _values_from_precomputeds(precomputeds, n_anchors, max_n_instrs):
    """Each row's per-output values and role-ordered input values at its
    anchors, padded to (n, n_anchors, max_n_instrs) and (n, n_anchors,
    n_positions). None rows (invalid / mem-op) are zeros with valid=False."""
    n, P = len(precomputeds), n_positions(max_n_instrs)
    values = np.zeros((n, n_anchors, max_n_instrs), dtype=np.int32)
    inputs = np.zeros((n, n_anchors, P), dtype=np.int32)
    valid = np.zeros(n, dtype=bool)
    for i, pre in enumerate(precomputeds):
        if pre is None:
            continue
        values[i, :, :pre.out_values.shape[1]] = pre.out_values
        inputs[i, :, :len(pre.in_roles)] = pre.in_role_values
        valid[i] = True
    return values, inputs, valid


def pack_batch(chunks, *,
               target_B, target_max_tokens,
               target_max_n_instrs,
               out_values=None, in_role_values=None, out_regs_valid=None,
               aux_payload=None):
    """Build a Batch from chunks and their per-output value targets.

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

    P = n_positions(max_n_instrs)
    if out_values is None:
        values_arr = np.zeros((0, 0, max_n_instrs), dtype=np.int32)
        inputs_arr = np.zeros((0, 0, P), dtype=np.int32)
        out_regs_valid_arr = np.zeros((0,), dtype=bool)
    else:
        n_anchors = out_values.shape[1]
        if (out_values.shape != (actual_B, n_anchors, max_n_instrs)
                or in_role_values.shape != (actual_B, n_anchors, P)):
            raise ValueError(
                f'out_values {out_values.shape} / in_role_values '
                f'{in_role_values.shape} must be (actual_B={actual_B}, '
                f'n_anchors, {max_n_instrs} / {P})')
        values_arr = np.zeros((B, n_anchors, max_n_instrs), dtype=np.int32)
        values_arr[:actual_B] = out_values
        inputs_arr = np.zeros((B, n_anchors, P), dtype=np.int32)
        inputs_arr[:actual_B] = in_role_values
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
    out_inputs_arr = np.full((B, MAX_OUTPUT_SLOTS, MAX_INPUT_SLOTS),
                             AUX_CE_IGNORE, dtype=np.int8)
    out_inputs_arr[:actual_B] = aux.out_inputs
    S = len(placements(P))
    in_roles = np.full((B, P), AUX_CE_IGNORE, dtype=np.int8)
    for i, pre in enumerate(aux.precomputed):
        if pre is not None:
            in_roles[i, :len(pre.in_roles)] = pre.in_roles
    rf = dict(rf_pos=np.zeros((B, S, P, N_WITNESS), dtype=np.int32),
              rf_pos_zero=np.zeros((B, S, P), dtype=np.int64),
              rf_pos_one=np.zeros((B, S, P), dtype=np.int64),
              rf_wo=np.zeros((B, S, max_n_instrs, N_WITNESS), dtype=np.int32),
              rf_wo_zero=np.zeros((B, max_n_instrs), dtype=np.int64),
              rf_wo_one=np.zeros((B, max_n_instrs), dtype=np.int64),
              rf_wo_free=np.zeros((B, max_n_instrs), dtype=np.float32),
              n_wo=np.zeros(B, dtype=np.int8))
    for i, pre in enumerate(aux.precomputed):
        if pre is None:
            continue
        for k in rf:
            rf[k][i] = getattr(pre, k)

    return Batch(tokens=tokens, token_lens=token_lens, valid=valid,
                 instr_lens=instr_lens,
                 out_values=values_arr,
                 in_roles=in_roles,
                 in_role_values=inputs_arr,
                 out_regs_valid=out_regs_valid_arr,
                 live_in_mask=live_in_arr,
                 live_out_mask=live_out_arr,
                 pc_writes=pc_writes_arr,
                 in_slot_regs=in_slot_arr,
                 out_slot_regs=out_slot_arr,
                 out_inputs=out_inputs_arr,
                 **rf)


def padding_mask(batch):
    """Return a (B, max_tokens) bool array, True where padded."""
    B, max_tokens = batch.tokens.shape
    arange = np.arange(max_tokens)
    return arange[None, :] >= batch.token_lens[:, None]


# ===========================================================================
# Generation: chunks -> targets -> Batch
# ===========================================================================

# Memory ops have no Precomputed in V1 scope; target construction skips
# chunks containing them.
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


def build_targets(chunks, draw_anchors, max_n_instrs, aim=None):
    """Execute each valid row at its own anchor draw and build its targets.

    Memory-op chunks get no targets (their Precomputed is unavailable in V1
    scope) and contribute an empty aux row, as invalid windows do.

    draw_anchors(k) supplies the structured draws of k valid chunks as one
    (k, n_anchors, 32) array, which `precompute_chunk` places on each row's
    own input registers. aim(instructions, states), when given, rewrites a
    row's draw to straddle that chunk's boundaries (`datagen.aim`) before it
    is executed.

    Returns (aux, out_values, in_role_values, out_regs_valid): aux is the
    parallel AuxPayload; the rest is `_values_from_precomputeds`, padded to
    max_n_instrs output slots.
    """
    usable = [ch.valid and not _has_mem_ops(ch.instructions) for ch in chunks]
    drawn = draw_anchors(max(sum(usable), 1))
    aux_pre, n_drawn = [], 0
    for ch, ok in zip(chunks, usable):
        if ok:
            states = drawn[n_drawn]
            if aim is not None:
                states = aim(ch.instructions, states)
            aux_pre.append(precompute_chunk(ch.instructions, states, max_n_instrs))
            n_drawn += 1
        else:
            aux_pre.append(None)
    aux = _aux_from_precomputeds(aux_pre)
    return (aux, *_values_from_precomputeds(aux_pre, drawn.shape[1],
                                            max_n_instrs))


# ===========================================================================
# Top-level: instruction stream → Batch stream
# ===========================================================================

def _yield_random_instructions(rng, opcode_table=None, dest_regs=None):
    while True:
        yield random_instruction(rng, opcode_table=opcode_table,
                                 dest_regs=dest_regs)


def generate_chunks(rule, rng, *, opcode_table=None, dest_regs=None,
                    chain_rate=0.0):
    """Yield Chunk objects (all valid) from a random-instruction stream
    grouped by `rule`.

    If chain_rate > 0, that fraction of groups is rewritten so each
    instruction reads its predecessor's result (`chain_group`), raising the
    density of dataflow-composed chunks above the ~4.7% a uniform register
    draw gives. Opcodes are untouched, so the config's opcode distribution
    holds either way.
    """
    instr_iter = _yield_random_instructions(rng, opcode_table=opcode_table,
                                            dest_regs=dest_regs)
    for group in collect_groups(instr_iter, rule):
        if chain_rate > 0 and rng.random() < chain_rate:
            group = chain_group(group, rng)
        yield _make_valid_chunk(group)


def collect_into_batches(chunks_iter, *, batch_size, draw_anchors, aim=None,
                         partner_rate=0.0, partner_sampler=None,
                         invalid_rate=0.0, invalid_provider=None,
                         max_invalid_window=None,
                         max_chunk_len):
    """Consume a Chunk stream, build Batches of `batch_size` rows.

    invalid_rate>0 mixes invalid windows in at the given fraction of
    `batch_size`. invalid_provider() must return a list[int] token
    sequence (typically datagen.invalidity.generate_invalid).

    Each batch ends up with `n_invalid = round(batch_size * invalid_rate)`
    invalid rows + `batch_size - n_invalid` valid rows.

    partner_rate>0 makes that fraction of each batch pairs of rows computing
    one function, two different spellings each, drawn by `partner_sampler()`
    -> (list[Instruction], list[Instruction]), each at most max_chunk_len
    long: the equivalent pairs the collapse objective needs, which a naturally
    drawn batch almost never holds. The rest of the valid rows come from
    `chunks_iter`.

    Every rule (single → T1, multi-instr → T2) ships the value targets via
    build_targets (the T1/T2 unification).

    draw_anchors(k) supplies k chunks' anchor states — a fresh draw per chunk
    for training, one fixed set for a scoring corpus — and every row's states
    ship with it. `aim` is passed to build_targets.

    max_chunk_len: maximum number of instructions per chunk under the
                   chunking rule (the rule's length cap; 1 for 'single').
                   Required: every emitted batch has identical shape — B,
                   max_tokens, max_n_instrs all padded to deterministic
                   upper bounds derived from it. This prevents PyTorch's
                   caching allocator from fragmenting over a long training
                   run with stochastically-varying batch shapes, which
                   leads to OOM.
    """
    if invalid_rate > 0 and invalid_provider is None:
        raise ValueError('invalid_rate > 0 requires invalid_provider')

    if partner_rate > 0 and partner_sampler is None:
        raise ValueError('partner_rate > 0 requires partner_sampler')

    n_invalid = int(round(batch_size * invalid_rate))
    n_pairs = int(round(batch_size * partner_rate / 2))
    n_valid = batch_size - n_invalid - 2 * n_pairs

    # Fixed-shape upper bounds derived from max_chunk_len. Length variation
    # produces smaller actual shapes, which get padded up to these targets.
    target_B = batch_size
    target_max_n_instrs = max_chunk_len
    target_max_tokens = max_chunk_len * MAX_INSTR_TOKENS
    if max_invalid_window is not None:
        target_max_tokens = max(target_max_tokens, max_invalid_window)

    buf = []
    for ch in chunks_iter:
        buf.append(ch)
        if len(buf) >= n_valid:
            valid = buf[:n_valid]
            buf = buf[n_valid:]
            invalids = [_make_invalid_chunk(invalid_provider())
                        for _ in range(n_invalid)]
            partners = [_make_valid_chunk(instrs) for _ in range(n_pairs)
                        for instrs in partner_sampler()]
            mixed = valid + partners + invalids
            aux, out_values, in_role_values, out_regs_valid = build_targets(
                mixed, draw_anchors=draw_anchors,
                max_n_instrs=target_max_n_instrs, aim=aim)
            yield pack_batch(
                mixed,
                target_B=target_B,
                target_max_tokens=target_max_tokens,
                target_max_n_instrs=target_max_n_instrs,
                out_values=out_values,
                in_role_values=in_role_values,
                out_regs_valid=out_regs_valid,
                aux_payload=aux)
