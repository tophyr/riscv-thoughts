"""Unified training batch.

A Batch carries chunks (some valid, some invalid) plus an optional pair
structure. Three losses see three views of the same batch:

  - Pair-MSE (encoder):
        for (i, j), d in zip(pair_indices, distances):
            loss += (||encode(c_i) - encode(c_j)|| - d) ** 2
  - Reconstruction CE (decoder):
        for c in chunks if c.valid:
            loss += CE(decode(encode(c.tokens)), c.tokens)
  - Validity magnitude:
        for c in chunks:
            loss += (||encode(c.tokens)|| - (1.0 if c.valid else 0.0)) ** 2

Pair indices reference only valid chunks. If pair_indices is empty,
no pair-MSE supervision exists in the batch (decoder/validity-only).

Storage shape: encoder consumes flat (B, max_tokens) tokens with a
padding mask, so the binary format stores tokens flatly. Per-instruction
segmentation isn't preserved across the wire — relabeling happens at
generation time when the source list[Instruction] is still in hand.
"""

import struct
from dataclasses import dataclass

import numpy as np

from emulator import Instruction
from tokenizer import PAD, encode_instruction
from tokenizer.tokenizer import MAX_INSTR_TOKENS, decode_instruction

from .compare import (
    AUX_CE_IGNORE, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS, N_REGS,
    behavioral_distance_cached, make_anchor_states, precompute_chunk,
    precompute_row_outputs,
)
from .generate import (
    DEFAULT_DISTRIBUTION, _build_opcode_table, validate_distribution,
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

    Stream layout:
      Stream header: 4-byte magic + 1-byte version + N dtype chars
      Per batch:     uint32-per-header-field struct + body fields as
                     contiguous raw arrays
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
        self._stream_header_struct = struct.Struct(
            f'<4sB{len(self._dtype_chars)}s')

    @property
    def name(self):
        return self.magic.rstrip(b'\x00').decode().lower()

    @property
    def batch_header(self):
        return self._header_struct

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

    def write_stream_header(self, f):
        f.write(self._stream_header_struct.pack(
            self.magic, self.version, self._dtype_chars))

    def read_stream_header(self, f):
        buf = f.read(self._stream_header_struct.size)
        if len(buf) < self._stream_header_struct.size:
            raise ValueError('Missing stream header')
        magic, version, dtype_chars = self._stream_header_struct.unpack(buf)
        if magic != self.magic:
            raise ValueError(
                f'Bad magic: {magic!r} (expected {self.magic!r})')
        if version != self.version:
            raise ValueError(
                f'Unsupported {self.name} version: {version}')
        if dtype_chars != self._dtype_chars:
            raise ValueError(f'Dtype mismatch: {dtype_chars!r}')

    def write_batch(self, f, batch):
        header = self._extract_header(batch)
        f.write(self._header_struct.pack(
            *[header[n] for n in self.header_fields]))
        for fld in self.body_fields:
            f.write(getattr(batch, fld.name).tobytes())

    def read_batch(self, f, batch_class):
        header_buf = f.read(self._header_struct.size)
        if len(header_buf) == 0:
            return None
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
        header_buf = f.read(self._header_struct.size)
        if len(header_buf) == 0:
            return None
        if len(header_buf) < self._header_struct.size:
            raise EOFError(
                f'Truncated batch header ({len(header_buf)} bytes)')
        header_vals = self._header_struct.unpack(header_buf)
        body_size = self.body_size(*header_vals)
        body = f.read(body_size) if body_size > 0 else b''
        if body_size > 0 and len(body) < body_size:
            raise EOFError(
                f'Truncated body (got {len(body)}, expected {body_size})')
        return header_buf + body

    def reader(self, f, batch_class):
        self.read_stream_header(f)
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
    segmentation + one of two pair-distance payloads.

    Two mutually exclusive pair-distance modes (training-side
    branches on which is populated):

      Pair mode (multi-instruction chunks):
        pair_indices (P, 2) and distances (P,) carry pre-computed
        scalar pair distances from per-pair CPU loop. row_* fields
        all have first-dim = 0.

      Row-outputs mode (single-instruction T1 chunks):
        Per-row canonical execution data carried in the row_*
        fields, sized RB = B. The (B, B) target distance matrix
        forms at training time on-device via
        `pairwise_distance_canonical`. pair_indices/distances are
        empty (P = 0). pair_valid masks which rows are
        behaviorally-meaningful (False for invalid windows and
        mem-op rows whose distance is undefined in V1 scope).

    All other arrays sized for B chunks; instr_lens sized for
    max_n_instrs columns (0 past actual count, 0 for invalid).
    """
    tokens:         np.ndarray   # (B, max_tokens) int8
    token_lens:     np.ndarray   # (B,) int32
    valid:          np.ndarray   # (B,) bool — complete instruction window
    instr_lens:     np.ndarray   # (B, max_n_instrs) int32
    pair_indices:   np.ndarray   # (P, 2) int32 — pair-mode only
    distances:      np.ndarray   # (P,) float32 — pair-mode only
    # Row-outputs mode payload (RB=B if active, RB=0 otherwise):
    row_outputs:    np.ndarray   # (RB, K, n_anchors, n_channels) float32
    row_n_inputs:   np.ndarray   # (RB,) int8 — count of behavioral inputs
    row_input_mags: np.ndarray   # (RB, K, max_in) float32 — canonical-order
    row_has_rd:     np.ndarray   # (RB,) bool
    row_rd_mag:     np.ndarray   # (RB,) float32
    pair_valid:     np.ndarray   # (RB,) bool — meaningful row_outputs
    # T2 aux register-identity targets — always sized B; rows without a
    # valid Precomputed (invalid windows, mem-op chunks) get zero
    # masks and AUX_CE_IGNORE in the slot arrays.
    live_in_mask:   np.ndarray   # (B, N_REGS) bool — behavioral inputs
    live_out_mask:  np.ndarray   # (B, N_REGS) bool — actually-modified regs
    pc_writes:      np.ndarray   # (B,) bool — pc_explicit flag
    in_slot_regs:   np.ndarray   # (B, MAX_INPUT_SLOTS) int8 — reg or AUX_CE_IGNORE
    out_slot_regs:  np.ndarray   # (B, MAX_OUTPUT_SLOTS) int8


RVT_FORMAT = BinaryFormat(
    magic=b'RVT\x00', version=5,
    header_fields=['B', 'max_tokens', 'max_n_instrs', 'P',
                   'RB', 'K', 'n_anchors', 'n_channels', 'max_in'],
    body_fields=[
        BodyField('tokens', np.dtype(np.int8), ('B', 'max_tokens')),
        BodyField('token_lens', np.dtype(np.int32), ('B',)),
        BodyField('valid', np.dtype(np.bool_), ('B',)),
        BodyField('instr_lens', np.dtype(np.int32), ('B', 'max_n_instrs')),
        BodyField('pair_indices', np.dtype(np.int32), ('P', 2)),
        BodyField('distances', np.dtype(np.float32), ('P',)),
        BodyField('row_outputs', np.dtype(np.float32),
                  ('RB', 'K', 'n_anchors', 'n_channels')),
        BodyField('row_n_inputs', np.dtype(np.int8), ('RB',)),
        BodyField('row_input_mags', np.dtype(np.float32),
                  ('RB', 'K', 'max_in')),
        BodyField('row_has_rd', np.dtype(np.bool_), ('RB',)),
        BodyField('row_rd_mag', np.dtype(np.float32), ('RB',)),
        BodyField('pair_valid', np.dtype(np.bool_), ('RB',)),
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
class RowOutputsPayload:
    """Per-row arrays produced by build_row_outputs and consumed by
    pack_batch when in row-outputs mode. All sized for actual rows
    (length-actual_B); pack_batch pads to target_B."""
    row_outputs: np.ndarray     # (actual_B, K, n_anchors, n_channels) float
    n_inputs: np.ndarray        # (actual_B,) int8
    input_mags: np.ndarray      # (actual_B, K, max_in) float
    has_rd: np.ndarray          # (actual_B,) bool
    rd_mag: np.ndarray          # (actual_B,) float
    pair_valid: np.ndarray      # (actual_B,) bool


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


def pack_batch(chunks, pairs, distances, *,
               target_B=None, target_max_tokens=None,
               target_max_n_instrs=None, target_P=None,
               row_outputs_payload=None,
               aux_payload=None):
    """Build a Batch from chunks + (parallel) pair lists, optionally
    with row-outputs payload.

    chunks:    list[Chunk]
    pairs:     iterable of (i, j) into chunks (must reference valid rows).
               Pass empty in row-outputs mode.
    distances: iterable of float aligned with pairs. Pass empty in
               row-outputs mode.

    target_*:  optional fixed shape parameters. If specified, arrays
               are padded out to these exact sizes (raises ValueError
               on overflow). When None, computed from the data —
               which means each batch can have a different shape, and
               downstream PyTorch caching allocators fragment over
               training runs of any meaningful length. Pass these
               from collect_into_batches for fixed-shape output.
               Padding rows have valid=False, token_lens=0,
               instr_lens=0; padding pairs are (0, 0) with distance
               0.0 — naturally zero loss, no special handling needed
               in training.

    row_outputs_payload:
        Optional RowOutputsPayload. When supplied, the row-* fields
        are populated (RB = target_B). When None, RB = 0 and the
        row-* body fields are empty — pair-indices mode.

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

    pairs_list = list(pairs)
    distances_list = list(distances)
    actual_P = len(pairs_list)
    if len(distances_list) != actual_P:
        raise ValueError(
            f'pairs ({actual_P}) and distances ({len(distances_list)}) '
            f'length mismatch')

    # Resolve target shapes (data-driven if not specified).
    B = target_B if target_B is not None else actual_B
    max_tokens = (target_max_tokens if target_max_tokens is not None
                  else actual_max_tokens)
    max_n_instrs = (target_max_n_instrs if target_max_n_instrs is not None
                    else actual_max_n_instrs)
    P = target_P if target_P is not None else actual_P

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
    if actual_P > P:
        raise ValueError(
            f'P={actual_P} exceeds target_P={P}')

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

    pair_arr = np.zeros((P, 2), dtype=np.int32)
    dist_arr = np.zeros(P, dtype=np.float32)
    if pairs_list:
        pair_arr[:actual_P] = np.asarray(
            pairs_list, dtype=np.int32).reshape(-1, 2)
        dist_arr[:actual_P] = np.asarray(distances_list, dtype=np.float32)

    if row_outputs_payload is None:
        # pair-indices mode: row-outputs fields are empty.
        row_outputs_arr = np.zeros((0, 0, 0, 0), dtype=np.float32)
        n_inputs_arr = np.zeros((0,), dtype=np.int8)
        input_mags_arr = np.zeros((0, 0, 0), dtype=np.float32)
        has_rd_arr = np.zeros((0,), dtype=bool)
        rd_mag_arr = np.zeros((0,), dtype=np.float32)
        pair_valid_arr = np.zeros((0,), dtype=bool)
    else:
        # row-outputs mode: RB = B; pad payload's actual_B to B.
        p = row_outputs_payload
        if p.row_outputs.shape[0] != actual_B:
            raise ValueError(
                f'row_outputs_payload.row_outputs first dim '
                f'({p.row_outputs.shape[0]}) must equal '
                f'actual_B ({actual_B})')
        K, n_anchors, n_channels = p.row_outputs.shape[1:]
        max_in = p.input_mags.shape[2]

        row_outputs_arr = np.zeros((B, K, n_anchors, n_channels),
                                   dtype=np.float32)
        row_outputs_arr[:actual_B] = p.row_outputs.astype(
            np.float32, copy=False)

        n_inputs_arr = np.zeros((B,), dtype=np.int8)
        n_inputs_arr[:actual_B] = p.n_inputs.astype(np.int8, copy=False)

        input_mags_arr = np.zeros((B, K, max_in), dtype=np.float32)
        input_mags_arr[:actual_B] = p.input_mags.astype(
            np.float32, copy=False)

        has_rd_arr = np.zeros((B,), dtype=bool)
        has_rd_arr[:actual_B] = p.has_rd

        rd_mag_arr = np.zeros((B,), dtype=np.float32)
        rd_mag_arr[:actual_B] = p.rd_mag.astype(np.float32, copy=False)

        pair_valid_arr = np.zeros((B,), dtype=bool)
        pair_valid_arr[:actual_B] = p.pair_valid

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
                 pair_indices=pair_arr, distances=dist_arr,
                 row_outputs=row_outputs_arr,
                 row_n_inputs=n_inputs_arr,
                 row_input_mags=input_mags_arr,
                 row_has_rd=has_rd_arr,
                 row_rd_mag=rd_mag_arr,
                 pair_valid=pair_valid_arr,
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


def unpack_chunks(batch):
    """Yield per-row token sequences as plain lists (decoder eval, etc)."""
    for i in range(batch.tokens.shape[0]):
        n = int(batch.token_lens[i])
        yield batch.tokens[i, :n].tolist()


# ===========================================================================
# Generation: chunks -> twins -> pairs -> Batch
# ===========================================================================

# Memory ops aren't supported by behavioral_distance V1; pair construction
# filters chunks containing them.
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


def build_pairs(chunks, twins, partners, anchor_states, rng):
    """Add `twins` relabeled copies of each valid source chunk and
    construct a flat pair-index list with computed target distances.

    Cluster structure: each valid source plus its twins form a cluster
    of size (twins + 1). Within a cluster every (a, b) pair has
    distance 0 (relabeling is equivalence-preserving). Across clusters,
    `partners - twins` random non-cluster partners per row get their
    distance computed via behavioral_distance.

    Memory-op chunks are dropped (behavioral_distance V1 doesn't support them).

    Returns (chunks_out, pair_indices, distances) where chunks_out
    extends `chunks` with the new twins.
    """
    if partners < twins:
        raise ValueError(
            f'partners={partners} < twins={twins}')

    # Filter out memory-op chunks at the source list — twins inherit
    # their parent's filtering, and we need cluster_id continuity.
    valid_sources = []
    for ch in chunks:
        if not ch.valid:
            valid_sources.append(None)  # placeholder; not paired
            continue
        if _has_mem_ops(ch.instructions):
            valid_sources.append(None)
            continue
        valid_sources.append(ch)

    # Build the chunks_out list: original chunks + twins after each
    # paired source (clusters are contiguous for clean cluster_id).
    chunks_out = []
    cluster_id = []        # one entry per output row, -1 = unpaired
    next_cluster = 0
    for original, source in zip(chunks, valid_sources):
        chunks_out.append(original)
        if source is None:
            cluster_id.append(-1)
            continue
        cid = next_cluster
        next_cluster += 1
        cluster_id.append(cid)
        for _ in range(twins):
            twin_instrs = random_relabel(source.instructions, rng)
            chunks_out.append(_make_valid_chunk(twin_instrs))
            cluster_id.append(cid)

    # Build cluster -> indices map.
    n_clusters = next_cluster
    if n_clusters == 0 or partners == 0:
        # Still need aux for whatever rows we have (invalid windows,
        # mem-op chunks). Empty cluster_id array → all None.
        return chunks_out, [], [], _aux_from_precomputeds(
            [None] * len(chunks_out))

    cluster_members = {c: [] for c in range(n_clusters)}
    for idx, c in enumerate(cluster_id):
        if c >= 0:
            cluster_members[c].append(idx)

    # Precompute SSA + anchor execution per row that participates in
    # pairs. Cache result for cross-cluster distance computation.
    precomputed = {}
    def _pre(idx):
        if idx not in precomputed:
            precomputed[idx] = precompute_chunk(
                chunks_out[idx].instructions, anchor_states)
        return precomputed[idx]

    pair_indices = []
    distances = []
    n_random = partners - twins
    paired_rows = [i for i, c in enumerate(cluster_id) if c >= 0]
    pool_per_cluster = {}
    if n_random > 0:
        for c in range(n_clusters):
            pool_per_cluster[c] = [i for i in paired_rows
                                   if cluster_id[i] != c]

    for i in paired_rows:
        c = cluster_id[i]
        siblings = [j for j in cluster_members[c] if j != i]
        for j in siblings:
            pair_indices.append((i, j))
            distances.append(0.0)
        if n_random == 0:
            continue
        pool = pool_per_cluster[c]
        if not pool:
            continue
        replace = len(pool) < n_random
        picks = rng.choice(pool, size=n_random, replace=replace)
        for j in picks.tolist():
            d = float(behavioral_distance_cached(
                _pre(i), _pre(int(j)), anchor_states))
            # Pairs whose canonical-positions overflow on both
            # directions return inf; drop rather than poison training.
            if d == float('inf'):
                continue
            pair_indices.append((i, int(j)))
            distances.append(d)

    # Aux targets: every paired row gets its Precomputed; mem-op /
    # invalid rows get None → empty aux row.
    aux_pre = [_pre(i) if cluster_id[i] >= 0 else None
               for i in range(len(chunks_out))]
    aux = _aux_from_precomputeds(aux_pre)

    return chunks_out, pair_indices, distances, aux


# Row-outputs mode constants. K=2 covers the rs1↔rs2 swap; max_in=2
# covers single-instruction max input cardinality. Pinned in the wire
# format so consumers don't need a per-batch reshape.
_ROW_K = 2
_ROW_MAX_IN = 2
_ROW_N_CHANNELS = 2  # (rd_value, pc_value)


def build_row_outputs(chunks, twins, anchor_states, rng):
    """Add `twins` relabeled copies of each valid non-mem-op single-
    instruction source chunk and compute the per-row canonical outputs
    payload for every row.

    Like `build_pairs` but for the row-outputs mode: the per-pair
    distance computation is deferred to training-time on-GPU via
    `pairwise_distance_canonical`. This function only produces the
    per-row data.

    All chunks must be single-instruction (or invalid windows). Memory
    ops are kept in the chunks list (so the encoder still sees their
    tokens for magnitude-validity training) but their row_outputs is
    zero and pair_valid is False — distance loss skips them.

    Returns (chunks_out, payload, aux) where chunks_out extends
    `chunks` with twin rows, payload is a RowOutputsPayload sized for
    chunks_out, and aux is the parallel AuxPayload of T2 register-
    identity targets.
    """
    chunks_out = []
    rows = []
    pres = []   # parallel to chunks_out; None for invalid/mem-op rows

    n_anchors = anchor_states.shape[0]
    zero_row_outputs = np.zeros(
        (_ROW_K, n_anchors, _ROW_N_CHANNELS), dtype=np.float64)
    zero_input_mags = np.zeros((_ROW_K, _ROW_MAX_IN), dtype=np.float64)
    empty_row = (zero_row_outputs, 0, zero_input_mags,
                 False, 0.0, False)

    def _process(ch):
        if (not ch.valid
                or ch.instructions is None
                or len(ch.instructions) != 1
                or _has_mem_ops(ch.instructions)):
            return empty_row, None
        instr = ch.instructions[0]
        pre = precompute_chunk(ch.instructions, anchor_states)
        ro = precompute_row_outputs(instr, anchor_states, pre=pre)
        return ((ro.row_outputs, ro.n_inputs, ro.input_mags,
                 ro.has_rd, ro.rd_mag, True), pre)

    for ch in chunks:
        chunks_out.append(ch)
        row, pre = _process(ch)
        rows.append(row)
        pres.append(pre)

        if pre is None:
            # Invalid / mem-op / multi-instr: no twins.
            continue

        for _ in range(twins):
            twin_instrs = random_relabel(ch.instructions, rng)
            twin_chunk = _make_valid_chunk(twin_instrs)
            chunks_out.append(twin_chunk)
            twin_row, twin_pre = _process(twin_chunk)
            rows.append(twin_row)
            pres.append(twin_pre)

    payload = RowOutputsPayload(
        row_outputs=np.stack([r[0] for r in rows]),
        n_inputs=np.array([r[1] for r in rows], dtype=np.int8),
        input_mags=np.stack([r[2] for r in rows]),
        has_rd=np.array([r[3] for r in rows], dtype=bool),
        rd_mag=np.array([r[4] for r in rows], dtype=np.float32),
        pair_valid=np.array([r[5] for r in rows], dtype=bool),
    )
    aux = _aux_from_precomputeds(pres)
    return chunks_out, payload, aux


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


def collect_into_batches(chunks_iter, *, batch_size, twins, partners,
                         anchor_states, rng,
                         invalid_rate=0.0, invalid_provider=None,
                         max_invalid_window=None,
                         max_chunk_len=None,
                         row_outputs_mode=False):
    """Consume a Chunk stream, build Batches of `batch_size` rows.

    invalid_rate>0 mixes invalid windows in at the given fraction of
    `batch_size`. invalid_provider() must return a list[int] token
    sequence (typically datagen.invalidity.generate_invalid).

    Each batch ends up with `n_invalid = round(batch_size * invalid_rate)`
    invalid rows + `batch_size - n_invalid` valid source slots; twins
    are added on top of the valid-source slots, growing the batch.

    max_chunk_len: maximum number of instructions per chunk under the
                   chunking rule. When specified, every emitted batch
                   has identical shape — B, max_tokens, max_n_instrs,
                   and P all padded to deterministic upper bounds.
                   This prevents PyTorch's caching allocator from
                   fragmenting over a long training run with stochas-
                   tically-varying batch shapes (a real OOM trigger
                   we hit at ~step 500 of t1_cosine_full). When None,
                   shapes are data-driven (legacy behavior).

    row_outputs_mode: if True, switch to single-instruction T1 row-
                   outputs path. partners is ignored (no per-pair
                   distance computation); per-row canonical outputs
                   are computed and shipped instead, and the (B, B)
                   target distance matrix forms at training time.
                   Requires single-instruction chunks (e.g. --rule
                   single).
    """
    if invalid_rate > 0 and invalid_provider is None:
        raise ValueError('invalid_rate > 0 requires invalid_provider')

    n_invalid = int(round(batch_size * invalid_rate))
    n_valid_sources = batch_size - n_invalid

    # Compute fixed-shape upper bounds when max_chunk_len is given.
    # These match the batch shape that would be produced if EVERY
    # valid source were paired (no mem-op filtering): all valid
    # sources contribute (twins+1) rows; invalids contribute 1 row
    # each. Mem-op filtering and length variation produce smaller
    # actual shapes, which then get padded up to these targets.
    if max_chunk_len is not None:
        cluster_size = twins + 1
        target_B = n_valid_sources * cluster_size + n_invalid
        target_max_n_instrs = max_chunk_len
        target_max_tokens = max_chunk_len * MAX_INSTR_TOKENS
        if max_invalid_window is not None:
            target_max_tokens = max(target_max_tokens, max_invalid_window)
        target_P = (0 if row_outputs_mode
                    else n_valid_sources * cluster_size * partners)
    else:
        target_B = None
        target_max_n_instrs = None
        target_max_tokens = None
        target_P = None

    buf = []
    for ch in chunks_iter:
        buf.append(ch)
        if len(buf) >= n_valid_sources:
            valid_sources = buf[:n_valid_sources]
            buf = buf[n_valid_sources:]
            invalids = [_make_invalid_chunk(invalid_provider())
                        for _ in range(n_invalid)]
            mixed = valid_sources + invalids
            if row_outputs_mode:
                chunks_out, payload, aux = build_row_outputs(
                    mixed, twins=twins,
                    anchor_states=anchor_states, rng=rng)
                yield pack_batch(
                    chunks_out, [], [],
                    target_B=target_B,
                    target_max_tokens=target_max_tokens,
                    target_max_n_instrs=target_max_n_instrs,
                    target_P=target_P,
                    row_outputs_payload=payload,
                    aux_payload=aux)
            else:
                chunks_out, pairs, dists, aux = build_pairs(
                    mixed, twins=twins, partners=partners,
                    anchor_states=anchor_states, rng=rng)
                yield pack_batch(
                    chunks_out, pairs, dists,
                    target_B=target_B,
                    target_max_tokens=target_max_tokens,
                    target_max_n_instrs=target_max_n_instrs,
                    target_P=target_P,
                    aux_payload=aux)
