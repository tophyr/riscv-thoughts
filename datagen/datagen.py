"""Training data generation and batch I/O for the T0→T1 compressor.

Core function is produce_batch() which generates one batch synchronously.
Parallelism is handled externally via gen_batches.py + mux_batches.py.

Binary format: write_batch/read_batch serialize Batch objects for piping.
BatchReader wraps a binary stream as an iterator for the training loop.
"""

import struct
from dataclasses import dataclass

import numpy as np

from emulator import (
    Instruction, run as run_instruction, make_ctx, random_regs,
    R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE, MEM_WIDTH,
)
from tokenizer import encode_instruction, PAD


# ---------------------------------------------------------------------------
# Batch data container
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    instructions: list         # list of Instruction objects
    token_ids: np.ndarray      # (B, max_len) int64
    padding_mask: np.ndarray   # (B, max_len) bool
    data_vals: np.ndarray      # (B, n_inputs) int64
    pc_vals: np.ndarray        # (B, n_inputs) int64
    dest_types: np.ndarray     # (B,) int64
    dest_regs: np.ndarray      # (B,) int64


# ---------------------------------------------------------------------------
# Instruction generation — all RV32I types
# ---------------------------------------------------------------------------

_DEST_REGS = list(range(1, 32))
_SRC_REGS = list(range(0, 32))

_ALU_R_OPS = ['ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU']
_ALU_I_OPS = ['ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI', 'SLTI', 'SLTIU']
_SHIFT_I_OPS = {'SLLI', 'SRLI', 'SRAI'}
_LOAD_OPS = ['LB', 'LBU', 'LH', 'LHU', 'LW']
_STORE_OPS = ['SB', 'SH', 'SW']
_BRANCH_OPS = ['BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU']


def random_instruction(rng: np.random.Generator) -> Instruction:
    """Generate a single random RV32I instruction of any type."""
    roll = rng.random()

    if roll < 0.25:
        op = rng.choice(_ALU_R_OPS)
        return Instruction(op, int(rng.choice(_DEST_REGS)),
                           int(rng.choice(_SRC_REGS)),
                           int(rng.choice(_SRC_REGS)))
    elif roll < 0.50:
        op = rng.choice(_ALU_I_OPS)
        rd = int(rng.choice(_DEST_REGS))
        rs1 = int(rng.choice(_SRC_REGS))
        imm = int(rng.integers(0, 32)) if op in _SHIFT_I_OPS else int(rng.integers(-2048, 2048))
        return Instruction(op, rd, rs1, imm)
    elif roll < 0.625:
        op = rng.choice(_LOAD_OPS)
        return Instruction(op, int(rng.choice(_DEST_REGS)),
                           int(rng.integers(-2048, 2048)),
                           int(rng.choice(_SRC_REGS)))
    elif roll < 0.70:
        op = rng.choice(_STORE_OPS)
        return Instruction(op, int(rng.choice(_SRC_REGS)),
                           int(rng.integers(-2048, 2048)),
                           int(rng.choice(_SRC_REGS)))
    elif roll < 0.85:
        op = rng.choice(_BRANCH_OPS)
        return Instruction(op, int(rng.choice(_SRC_REGS)),
                           int(rng.choice(_SRC_REGS)),
                           int(rng.integers(-2048, 2048)) * 2)
    elif roll < 0.925:
        op = rng.choice(['LUI', 'AUIPC'])
        return Instruction(op, int(rng.choice(_DEST_REGS)),
                           int(rng.integers(0, 0x100000)))
    elif roll < 0.9625:
        return Instruction('JAL', int(rng.choice(_DEST_REGS)),
                           int(rng.integers(-524288, 524288)) * 2)
    else:
        return Instruction('JALR', int(rng.choice(_DEST_REGS)),
                           int(rng.choice(_SRC_REGS)),
                           int(rng.integers(-2048, 2048)))


# ---------------------------------------------------------------------------
# Destination / data-value extraction
# ---------------------------------------------------------------------------

def dest_type(instr: Instruction) -> int:
    """0 = register destination, 1 = memory destination."""
    return 1 if instr.opcode in STORE_TYPE else 0


def dest_reg(instr: Instruction) -> int:
    """Destination register number. 0 for stores and branches."""
    if instr.opcode in STORE_TYPE or instr.opcode in B_TYPE:
        return 0
    return instr.args[0]


def extract_data_val(instr: Instruction, final_regs: np.ndarray,
                     final_mem) -> int:
    """Extract the data output value from execution results."""
    if instr.opcode in B_TYPE:
        return 0
    if instr.opcode in STORE_TYPE:
        rs2, imm, rs1 = instr.args
        addr = int(final_regs[rs1]) + imm
        width = MEM_WIDTH[instr.opcode]
        val = 0
        for i in range(width):
            val |= int(final_mem[addr + i]) << (8 * i)
        return val
    return int(final_regs[instr.args[0]])


# ---------------------------------------------------------------------------
# Single-batch production (the core logic, no threading)
# ---------------------------------------------------------------------------

def produce_batch(batch_size: int, n_inputs: int,
                  rng: np.random.Generator) -> Batch:
    """Generate one batch of training data. Pure, synchronous, no side effects."""
    instructions = [random_instruction(rng) for _ in range(batch_size)]
    B = len(instructions)

    dt = np.array([dest_type(instr) for instr in instructions], dtype=np.int64)
    dr = np.array([dest_reg(instr) for instr in instructions], dtype=np.int64)

    data_vals = np.zeros((B, n_inputs), dtype=np.int64)
    pc_vals = np.zeros((B, n_inputs), dtype=np.int64)
    ctx = make_ctx()

    for s in range(n_inputs):
        regs = random_regs(rng)
        pc = int(rng.integers(0, 1024)) * 4
        for i, instr in enumerate(instructions):
            state, final_pc, final_mem = run_instruction(
                [instr], regs=regs, pc=pc, rng=rng, _ctx=ctx)
            data_vals[i, s] = extract_data_val(instr, state.regs, final_mem)
            pc_vals[i, s] = final_pc

    encoded = [encode_instruction(instr) for instr in instructions]
    max_len = max(len(e) for e in encoded)
    token_ids = np.full((B, max_len), PAD, dtype=np.int64)
    padding_mask = np.ones((B, max_len), dtype=np.bool_)
    for i, enc in enumerate(encoded):
        token_ids[i, :len(enc)] = enc
        padding_mask[i, :len(enc)] = False

    return Batch(instructions, token_ids, padding_mask, data_vals, pc_vals, dt, dr)


def produce_focused_batch(batch_size: int, n_inputs: int,
                          rng: np.random.Generator) -> Batch:
    """Generate a batch focused on branch behavior.

    Half branch instructions, half non-branches. Register states have
    many equal pairs so branches actually trigger. Gives the model
    clear signal that branching instructions behave differently from
    comparison/ALU instructions.
    """
    half = batch_size // 2
    instructions = []

    # Half branches.
    for _ in range(half):
        op = rng.choice(_BRANCH_OPS)
        instructions.append(Instruction(
            op, int(rng.choice(_SRC_REGS)),
            int(rng.choice(_SRC_REGS)),
            int(rng.integers(-2048, 2048)) * 2))

    # Half non-branches (comparison + ALU mix for contrast).
    for _ in range(batch_size - half):
        roll = rng.random()
        if roll < 0.3:
            # SLT/SLTU — the instructions we want to separate from branches.
            op = rng.choice(['SLT', 'SLTU'])
            instructions.append(Instruction(
                op, int(rng.choice(_DEST_REGS)),
                int(rng.choice(_SRC_REGS)),
                int(rng.choice(_SRC_REGS))))
        else:
            instructions.append(random_instruction(rng))

    B = len(instructions)
    dt = np.array([dest_type(instr) for instr in instructions], dtype=np.int64)
    dr = np.array([dest_reg(instr) for instr in instructions], dtype=np.int64)

    data_vals = np.zeros((B, n_inputs), dtype=np.int64)
    pc_vals = np.zeros((B, n_inputs), dtype=np.int64)
    ctx = make_ctx()

    for s in range(n_inputs):
        regs = random_regs(rng)
        # Force many registers to share a value so branches trigger.
        n_equal = int(rng.integers(15, 26))
        idxs = rng.choice(range(1, 32), size=n_equal, replace=False)
        val = regs[idxs[0]]
        for k in idxs:
            regs[k] = val

        pc = int(rng.integers(0, 1024)) * 4
        for i, instr in enumerate(instructions):
            state, final_pc, final_mem = run_instruction(
                [instr], regs=regs, pc=pc, rng=rng, _ctx=ctx)
            data_vals[i, s] = extract_data_val(instr, state.regs, final_mem)
            pc_vals[i, s] = final_pc

    encoded = [encode_instruction(instr) for instr in instructions]
    max_len = max(len(e) for e in encoded)
    token_ids = np.full((B, max_len), PAD, dtype=np.int64)
    padding_mask = np.ones((B, max_len), dtype=np.bool_)
    for i, enc in enumerate(encoded):
        token_ids[i, :len(enc)] = enc
        padding_mask[i, :len(enc)] = False

    return Batch(instructions, token_ids, padding_mask, data_vals, pc_vals, dt, dr)


# ---------------------------------------------------------------------------
# Binary batch I/O
#
# Stream format:
#   Stream header (once):  4-byte magic "RVB\x00" + 1-byte version (1)
#                          + 6 dtype chars (token_ids, padding_mask,
#                          data_vals, pc_vals, dest_types, dest_regs)
#   Per batch:             12-byte header (B, max_len, n_inputs as uint32)
#                          + raw array data in field order
# ---------------------------------------------------------------------------

_MAGIC = b'RVB\x00'
_VERSION = 1
_STREAM_HEADER = struct.Struct('<4sB6s')  # magic, version, 6 dtype chars
_BATCH_HEADER = struct.Struct('<III')     # B, max_len, n_inputs

# Dtype chars for each field, in order.
_FIELD_DTYPES = (
    np.dtype(np.int64),   # token_ids
    np.dtype(np.bool_),   # padding_mask
    np.dtype(np.int64),   # data_vals
    np.dtype(np.int64),   # pc_vals
    np.dtype(np.int64),   # dest_types
    np.dtype(np.int64),   # dest_regs
)
_DTYPE_CHARS = b''.join(dt.char.encode() for dt in _FIELD_DTYPES)


def _batch_body_size(B, max_len, n_inputs):
    """Compute the byte size of a batch body given its header values."""
    return (
        B * max_len * _FIELD_DTYPES[0].itemsize +  # token_ids
        B * max_len * _FIELD_DTYPES[1].itemsize +   # padding_mask
        B * n_inputs * _FIELD_DTYPES[2].itemsize +   # data_vals
        B * n_inputs * _FIELD_DTYPES[3].itemsize +   # pc_vals
        B * _FIELD_DTYPES[4].itemsize +               # dest_types
        B * _FIELD_DTYPES[5].itemsize                 # dest_regs
    )


def write_stream_header(f):
    """Write the stream header. Call once before writing batches."""
    f.write(_STREAM_HEADER.pack(_MAGIC, _VERSION, _DTYPE_CHARS))


def read_stream_header(f):
    """Read and validate the stream header. Call once before reading batches."""
    buf = f.read(_STREAM_HEADER.size)
    if len(buf) < _STREAM_HEADER.size:
        raise ValueError('Missing stream header')
    magic, version, dtype_chars = _STREAM_HEADER.unpack(buf)
    if magic != _MAGIC:
        raise ValueError(f'Bad magic: {magic!r} (expected {_MAGIC!r})')
    if version != _VERSION:
        raise ValueError(f'Unsupported version: {version} (expected {_VERSION})')
    if dtype_chars != _DTYPE_CHARS:
        raise ValueError(f'Dtype mismatch: {dtype_chars!r} (expected {_DTYPE_CHARS!r})')


def write_batch(f, batch):
    """Write a Batch to a binary stream (after stream header)."""
    B, max_len = batch.token_ids.shape
    n_inputs = batch.data_vals.shape[1]
    f.write(_BATCH_HEADER.pack(B, max_len, n_inputs))
    f.write(batch.token_ids.tobytes())
    f.write(batch.padding_mask.tobytes())
    f.write(batch.data_vals.tobytes())
    f.write(batch.pc_vals.tobytes())
    f.write(batch.dest_types.tobytes())
    f.write(batch.dest_regs.tobytes())


def read_batch(f):
    """Read a Batch from a binary stream. Returns None at clean EOF."""
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

    return Batch(
        instructions=None,
        token_ids=_read_array(_FIELD_DTYPES[0], (B, max_len)),
        padding_mask=_read_array(_FIELD_DTYPES[1], (B, max_len)),
        data_vals=_read_array(_FIELD_DTYPES[2], (B, n_inputs)),
        pc_vals=_read_array(_FIELD_DTYPES[3], (B, n_inputs)),
        dest_types=_read_array(_FIELD_DTYPES[4], (B,)),
        dest_regs=_read_array(_FIELD_DTYPES[5], (B,)),
    )


def read_batch_bytes(f):
    """Read one complete batch as raw bytes. Returns None at clean EOF.

    For pass-through use (e.g. muxing) where parsing is unnecessary.
    """
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


class BatchReader:
    """Reads pre-generated batches from a binary stream.

    Validates the stream header, then yields Batch objects until EOF.
    """

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
