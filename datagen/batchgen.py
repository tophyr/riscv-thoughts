"""Single-instruction batch generator (RVB format).

Generates batches of individual instructions (not sequences), executes
each on multiple random register states, and captures the computed
output value and final PC. This is the modernized replacement for the
original datagen/datagen.py.

Binary format (RVB):
  Stream header: 4-byte magic "RVB\x00" + 1-byte version (1)
  Per batch:     12-byte header (B, max_len, n_inputs as uint32)
                 + raw array data in field order
"""

import struct
from dataclasses import dataclass

import numpy as np

from emulator import (
    Instruction, run as run_instruction, make_ctx, random_regs,
    B_TYPE, STORE_TYPE, MEM_WIDTH,
)
from tokenizer import encode_instruction, PAD
from .instrgen import random_instruction, validate_distribution, _build_opcode_table
from .invalidity import (
    DEFAULT_TYPE_WEIGHTS, build_type_table, generate_invalid,
)


# ---------------------------------------------------------------------------
# Instruction output extraction
# ---------------------------------------------------------------------------

def dest_type(instr: Instruction) -> int:
    """Return 0 for register-writing instructions, 1 for memory-writing (stores).

    Branches return 0 (they write neither, but we classify them with
    registers for dest_reg=0 convention).
    """
    if instr.opcode in STORE_TYPE:
        return 1
    return 0


def dest_reg(instr: Instruction) -> int:
    """Return the destination register index, or 0 for stores/branches."""
    if instr.opcode in STORE_TYPE or instr.opcode in B_TYPE:
        return 0
    return instr.args[0]


def extract_data_val(instr: Instruction,
                     final_regs: np.ndarray,
                     final_mem) -> int:
    """Extract the computed output value for a single instruction.

    Returns:
      - Branches: 0 (no data output)
      - Stores: the stored value read back from final_mem at rs1+imm
      - Everything else: final_regs[rd] (destination register value)
    """
    if instr.opcode in B_TYPE:
        return 0

    if instr.opcode in STORE_TYPE:
        rs2, imm, rs1 = instr.args
        addr = (int(final_regs[rs1]) + imm) & 0xFFFFFFFF
        width = MEM_WIDTH[instr.opcode]
        val = 0
        for i in range(width):
            val |= int(final_mem[addr + i]) << (8 * i)
        return val

    return int(final_regs[instr.args[0]])


# ---------------------------------------------------------------------------
# Batch container
# ---------------------------------------------------------------------------

@dataclass
class InstructionBatch:
    """A batch of single instructions or invalid token windows.

    Each valid instruction is executed on n_inputs shared random
    register states; data_vals and pc_vals capture the per-input
    outputs. Invalid windows (partial, spanning, multi-instruction,
    bogus) have their execution-state fields zeroed and are marked
    via valid_mask=False; the training loop is responsible for
    filtering.

    instructions is populated during generation but set to None after
    deserialization (the binary format does not store Instruction
    objects). Entries may be None even during generation when the
    row is an invalid window.
    """
    instructions: list          # list of Instruction | None
    token_ids: np.ndarray       # (B, max_len) int64
    padding_mask: np.ndarray    # (B, max_len) bool -- True = padding
    data_vals: np.ndarray       # (B, n_inputs) int64 -- zero for invalid rows
    pc_vals: np.ndarray         # (B, n_inputs) int64 -- zero for invalid rows
    dest_types: np.ndarray      # (B,) int64 -- zero for invalid rows
    dest_regs: np.ndarray       # (B,) int64 -- zero for invalid rows
    valid_mask: np.ndarray      # (B,) bool -- True = valid single instruction


# ---------------------------------------------------------------------------
# Batch production
# ---------------------------------------------------------------------------

def produce_instruction_batch(batch_size: int, n_inputs: int,
                              rng: np.random.Generator,
                              dist=None,
                              max_window: int = 32) -> InstructionBatch:
    """Generate one batch of single instructions and/or invalid windows.

    dist: optional config dict (from load_distribution or
          DEFAULT_DISTRIBUTION). Controls:
          - instruction type mix (opcode-category keys)
          - equivalence-tuple injection rate ('equivalences' sub-dict)
          - invalid-window augmentation ('invalidity' sub-dict with
            'rate' and per-type weights). Invalid rows have
            valid_mask=False and zero execution-state fields.
    max_window: token-length cap for invalid windows. Must accommodate
                the longest instruction tokenization (9 tokens). Default
                matches the T1 encoder's default pos-embedding size.
    """
    if dist is not None:
        validate_distribution(dist)
        opcode_table = _build_opcode_table(dist)
        eq_config = dist.get('equivalences', {})
        inv_config = dist.get('invalidity', {})
    else:
        opcode_table = None
        eq_config = {}
        inv_config = {}

    # --- Equivalence injection (valid tuples) ---
    inject_rate = eq_config.get('rate', 0.0)
    max_per_class = eq_config.get('max_per_class', 8)
    min_per_class = eq_config.get('min_per_class', 0)
    boost = eq_config.get('boost', None)

    target_inject = int(round(batch_size * inject_rate))

    # --- Invalidity augmentation ---
    invalid_rate = inv_config.get('rate', 0.0)
    invalid_types = inv_config.get('types', DEFAULT_TYPE_WEIGHTS)
    target_invalid = int(round(batch_size * invalid_rate))
    if target_invalid > 0:
        type_table = build_type_table(invalid_types)
    else:
        type_table = None

    # Cap so that random instructions + injected + invalid sum to batch_size.
    max_valid_inject = max(0, batch_size - target_invalid - 1)
    if target_inject > max_valid_inject:
        target_inject = max_valid_inject

    if target_inject > 0:
        from datagen.equivalences import sample_injection_tuples
        injected = sample_injection_tuples(
            target_inject, max_per_class, rng,
            min_per_class=min_per_class, boost=boost)
        if len(injected) > max_valid_inject:
            injected = injected[:max_valid_inject]
    else:
        injected = []

    n_random = batch_size - len(injected) - target_invalid
    instructions: list = []
    encoded: list[list[int]] = []
    valid_flags: list[bool] = []

    for _ in range(n_random):
        instr = random_instruction(rng, opcode_table=opcode_table)
        instructions.append(instr)
        encoded.append(encode_instruction(instr))
        valid_flags.append(True)
    for instr in injected:
        instructions.append(instr)
        encoded.append(encode_instruction(instr))
        valid_flags.append(True)
    for _ in range(target_invalid):
        toks, _type_name = generate_invalid(
            rng, opcode_table, max_window, type_table)
        instructions.append(None)
        encoded.append(toks)
        valid_flags.append(False)

    max_len = max(len(toks) for toks in encoded)

    token_ids = np.full((batch_size, max_len), PAD, dtype=np.int64)
    padding_mask = np.ones((batch_size, max_len), dtype=np.bool_)

    for b, toks in enumerate(encoded):
        n = len(toks)
        token_ids[b, :n] = toks
        padding_mask[b, :n] = False

    valid_mask = np.array(valid_flags, dtype=np.bool_)

    data_vals = np.zeros((batch_size, n_inputs), dtype=np.int64)
    pc_vals = np.zeros((batch_size, n_inputs), dtype=np.int64)
    dest_types = np.zeros(batch_size, dtype=np.int64)
    dest_regs_arr = np.zeros(batch_size, dtype=np.int64)

    for b, instr in enumerate(instructions):
        if instr is None:
            continue
        dest_types[b] = dest_type(instr)
        dest_regs_arr[b] = dest_reg(instr)

    # Execute only the valid rows. Invalid rows leave data_vals /
    # pc_vals at zero — the training loop ignores them via valid_mask.
    ctx = make_ctx()
    for s in range(n_inputs):
        regs = random_regs(rng)
        pc = int(rng.integers(0, 1024)) * 4

        for b, instr in enumerate(instructions):
            if instr is None:
                continue
            state, final_pc, final_mem = run_instruction(
                [instr], regs=regs, pc=pc, rng=rng, _ctx=ctx, max_steps=1)
            data_vals[b, s] = extract_data_val(instr, state.regs, final_mem)
            pc_vals[b, s] = final_pc

    return InstructionBatch(
        instructions=instructions,
        token_ids=token_ids,
        padding_mask=padding_mask,
        data_vals=data_vals,
        pc_vals=pc_vals,
        dest_types=dest_types,
        dest_regs=dest_regs_arr,
        valid_mask=valid_mask,
    )


# ---------------------------------------------------------------------------
# Binary I/O -- RVB format
#
# Stream format:
#   Stream header (once):  4-byte magic "RVB\x00" + 1-byte version (2)
#                          + 7 dtype chars (one per field)
#   Per batch:             12-byte header (B, max_len, n_inputs as uint32)
#                          + raw array data in field order:
#                            token_ids, padding_mask, data_vals,
#                            pc_vals, dest_types, dest_regs, valid_mask
#
# Version history:
#   1: 6 fields, valid-only batches.
#   2: 7 fields, adds valid_mask to support invalid-window training
#      examples (partial, spanning, multi, bogus). V1 files are no
#      longer readable — regenerate.
# ---------------------------------------------------------------------------

_MAGIC = b'RVB\x00'
_VERSION = 2
_STREAM_HEADER = struct.Struct('<4sB7s')  # magic, version, 7 dtype chars
_BATCH_HEADER = struct.Struct('<III')     # B, max_len, n_inputs

_FIELD_DTYPES = (
    np.dtype(np.int64),   # token_ids
    np.dtype(np.bool_),   # padding_mask
    np.dtype(np.int64),   # data_vals
    np.dtype(np.int64),   # pc_vals
    np.dtype(np.int64),   # dest_types
    np.dtype(np.int64),   # dest_regs
    np.dtype(np.bool_),   # valid_mask
)
_DTYPE_CHARS = b''.join(dt.char.encode() for dt in _FIELD_DTYPES)


def _batch_body_size(B: int, max_len: int, n_inputs: int) -> int:
    """Compute the byte size of a batch body (excluding header)."""
    return (
        B * max_len * _FIELD_DTYPES[0].itemsize +    # token_ids
        B * max_len * _FIELD_DTYPES[1].itemsize +    # padding_mask
        B * n_inputs * _FIELD_DTYPES[2].itemsize +   # data_vals
        B * n_inputs * _FIELD_DTYPES[3].itemsize +   # pc_vals
        B * _FIELD_DTYPES[4].itemsize +              # dest_types
        B * _FIELD_DTYPES[5].itemsize +              # dest_regs
        B * _FIELD_DTYPES[6].itemsize                # valid_mask
    )


def write_stream_header(f):
    """Write the RVB stream header."""
    f.write(_STREAM_HEADER.pack(_MAGIC, _VERSION, _DTYPE_CHARS))


def read_stream_header(f):
    """Read and validate the RVB stream header.

    V1 files (pre-invalidity) are no longer readable; regenerate
    with the current pipeline.
    """
    buf = f.read(_STREAM_HEADER.size)
    if len(buf) < _STREAM_HEADER.size:
        raise ValueError('Missing stream header')
    magic, version, dtype_chars = _STREAM_HEADER.unpack(buf)
    if magic != _MAGIC:
        raise ValueError(f'Bad magic: {magic!r} (expected {_MAGIC!r})')
    if version != _VERSION:
        raise ValueError(
            f'Unsupported RVB version: {version} '
            f'(expected {_VERSION}). V1 files lack valid_mask; '
            f'regenerate with gen_instr_batches.py.')
    if dtype_chars != _DTYPE_CHARS:
        raise ValueError(f'Dtype mismatch: {dtype_chars!r}')


def write_batch(f, batch: InstructionBatch):
    """Write an InstructionBatch to a binary stream."""
    B, max_len = batch.token_ids.shape
    n_inputs = batch.data_vals.shape[1]
    f.write(_BATCH_HEADER.pack(B, max_len, n_inputs))
    f.write(batch.token_ids.tobytes())
    f.write(batch.padding_mask.tobytes())
    f.write(batch.data_vals.tobytes())
    f.write(batch.pc_vals.tobytes())
    f.write(batch.dest_types.tobytes())
    f.write(batch.dest_regs.tobytes())
    f.write(batch.valid_mask.tobytes())


def read_batch(f) -> InstructionBatch | None:
    """Read an InstructionBatch from a binary stream. Returns None at clean EOF."""
    header = f.read(_BATCH_HEADER.size)
    if len(header) == 0:
        return None
    if len(header) < _BATCH_HEADER.size:
        raise EOFError(f'Truncated batch header ({len(header)} bytes)')
    B, max_len, n_inputs = _BATCH_HEADER.unpack(header)

    def _read_array(dtype, shape):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        buf = f.read(nbytes)
        if len(buf) < nbytes:
            raise EOFError(f'Truncated batch data (got {len(buf)}, expected {nbytes})')
        return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()

    return InstructionBatch(
        instructions=None,
        token_ids=_read_array(_FIELD_DTYPES[0], (B, max_len)),
        padding_mask=_read_array(_FIELD_DTYPES[1], (B, max_len)),
        data_vals=_read_array(_FIELD_DTYPES[2], (B, n_inputs)),
        pc_vals=_read_array(_FIELD_DTYPES[3], (B, n_inputs)),
        dest_types=_read_array(_FIELD_DTYPES[4], (B,)),
        dest_regs=_read_array(_FIELD_DTYPES[5], (B,)),
        valid_mask=_read_array(_FIELD_DTYPES[6], (B,)),
    )


def read_batch_bytes(f) -> bytes | None:
    """Read one complete batch as raw bytes. For pass-through use."""
    header = f.read(_BATCH_HEADER.size)
    if len(header) == 0:
        return None
    if len(header) < _BATCH_HEADER.size:
        raise EOFError(f'Truncated batch header ({len(header)} bytes)')
    B, max_len, n_inputs = _BATCH_HEADER.unpack(header)
    body_size = _batch_body_size(B, max_len, n_inputs)
    body = f.read(body_size)
    if len(body) < body_size:
        raise EOFError(f'Truncated batch data (got {len(body)}, expected {body_size})')
    return header + body


class InstructionBatchReader:
    """Reads pre-generated instruction batches from a binary stream."""

    def __init__(self, f):
        self._f = f
        read_stream_header(f)

    def __iter__(self):
        return self

    def __next__(self):
        batch = read_batch(self._f)
        if batch is None:
            raise StopIteration
        return batch
