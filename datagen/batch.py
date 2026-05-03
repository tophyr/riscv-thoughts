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
from tokenizer.tokenizer import decode_instruction

from .compare import (
    chunk_distance_cached, make_anchor_states, precompute_chunk,
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
    """Serialized batch: flat tokens + validity flags + pair structure.

    All arrays sized for B chunks; pair_indices and distances sized
    for P pairs (P can be 0).
    """
    tokens:       np.ndarray   # (B, max_tokens) int64
    token_lens:   np.ndarray   # (B,) int32
    valid:        np.ndarray   # (B,) bool
    pair_indices: np.ndarray   # (P, 2) int32
    distances:    np.ndarray   # (P,) float32


RVT_FORMAT = BinaryFormat(
    magic=b'RVT\x00', version=1,
    header_fields=['B', 'max_tokens', 'P'],
    body_fields=[
        BodyField('tokens', np.dtype(np.int64), ('B', 'max_tokens')),
        BodyField('token_lens', np.dtype(np.int32), ('B',)),
        BodyField('valid', np.dtype(np.bool_), ('B',)),
        BodyField('pair_indices', np.dtype(np.int32), ('P', 2)),
        BodyField('distances', np.dtype(np.float32), ('P',)),
    ],
)


# ===========================================================================
# Pack / unpack
# ===========================================================================

def pack_batch(chunks, pairs, distances):
    """Build a Batch from chunks + (parallel) pair lists.

    chunks:    list[Chunk]
    pairs:     iterable of (i, j) into chunks (must reference valid rows)
    distances: iterable of float aligned with pairs
    """
    if not chunks:
        raise ValueError('pack_batch requires at least one chunk')
    B = len(chunks)
    max_tokens = max(len(c.tokens) for c in chunks)
    tokens = np.full((B, max_tokens), PAD, dtype=np.int64)
    token_lens = np.zeros(B, dtype=np.int32)
    valid = np.zeros(B, dtype=bool)
    for i, c in enumerate(chunks):
        n = len(c.tokens)
        tokens[i, :n] = c.tokens
        token_lens[i] = n
        valid[i] = c.valid
    pairs = list(pairs)
    distances = list(distances)
    if pairs:
        pair_arr = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        dist_arr = np.asarray(distances, dtype=np.float32)
        if dist_arr.shape[0] != pair_arr.shape[0]:
            raise ValueError(
                f'pairs ({pair_arr.shape[0]}) and distances '
                f'({dist_arr.shape[0]}) length mismatch')
    else:
        pair_arr = np.zeros((0, 2), dtype=np.int32)
        dist_arr = np.zeros(0, dtype=np.float32)
    return Batch(tokens=tokens, token_lens=token_lens, valid=valid,
                 pair_indices=pair_arr, distances=dist_arr)


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

# Memory ops aren't supported by chunk_distance V1; pair construction
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
    distance computed via chunk_distance.

    Memory-op chunks are dropped (chunk_distance V1 doesn't support them).

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
        return chunks_out, [], []

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
            pair_indices.append((i, int(j)))
            distances.append(float(chunk_distance_cached(
                _pre(i), _pre(int(j)), anchor_states)))

    return chunks_out, pair_indices, distances


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
                         max_invalid_window=None):
    """Consume a Chunk stream, build Batches of `batch_size` rows.

    invalid_rate>0 mixes invalid windows in at the given fraction of
    `batch_size`. invalid_provider() must return a list[int] token
    sequence (typically datagen.invalidity.generate_invalid).

    Each batch ends up with `n_invalid = round(batch_size * invalid_rate)`
    invalid rows + `batch_size - n_invalid` valid source slots; twins
    are added on top of the valid-source slots, growing the batch.
    """
    if invalid_rate > 0 and invalid_provider is None:
        raise ValueError('invalid_rate > 0 requires invalid_provider')

    n_invalid = int(round(batch_size * invalid_rate))
    n_valid_sources = batch_size - n_invalid

    buf = []
    for ch in chunks_iter:
        buf.append(ch)
        if len(buf) >= n_valid_sources:
            valid_sources = buf[:n_valid_sources]
            buf = buf[n_valid_sources:]
            invalids = [_make_invalid_chunk(invalid_provider())
                        for _ in range(n_invalid)]
            mixed = valid_sources + invalids
            chunks_out, pairs, dists = build_pairs(
                mixed, twins=twins, partners=partners,
                anchor_states=anchor_states, rng=rng)
            yield pack_batch(chunks_out, pairs, dists)
