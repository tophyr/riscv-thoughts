"""Training data for the streaming compressor.

Generates structured synthetic instruction sequences (basic blocks)
with internal data flow, executes them on random input states,
captures per-instruction register snapshots, and serializes to a
binary stream format (RVS) for the training pipeline.
"""

import struct
from dataclasses import dataclass

import numpy as np

from emulator import (
    Instruction, run as run_instruction, make_ctx, random_regs,
    SparseMemory,
    R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE,
)
from tokenizer import encode_instruction, BOS, EOS, PAD


# ---------------------------------------------------------------------------
# Opcode, register, and immediate helpers
# ---------------------------------------------------------------------------

_DEST_REGS = list(range(1, 32))
_SRC_REGS = list(range(0, 32))

_ALU_R_OPS = ['ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU']
_ALU_I_OPS = ['ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI', 'SLTI', 'SLTIU']
_SHIFT_I_OPS = {'SLLI', 'SRLI', 'SRAI'}
_LOAD_OPS = ['LB', 'LBU', 'LH', 'LHU', 'LW']
_STORE_OPS = ['SB', 'SH', 'SW']
_BRANCH_OPS = ['BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU']


def _random_imm12(rng: np.random.Generator) -> int:
    """Random 12-bit signed immediate [-2048, 2048)."""
    return int(rng.integers(-2048, 2048))


def _i_type_imm(rng: np.random.Generator, op: str) -> int:
    """Random I-type immediate: 5-bit shift amount or 12-bit signed."""
    return int(rng.integers(0, 32)) if op in _SHIFT_I_OPS else _random_imm12(rng)


def _random_branch_offset(rng: np.random.Generator) -> int:
    """Random branch offset (13-bit signed, 2-byte aligned)."""
    return int(rng.integers(-2048, 2048)) * 2


def _random_upper_imm(rng: np.random.Generator) -> int:
    """Random upper immediate (20-bit unsigned)."""
    return int(rng.integers(0, 0x100000))


# ---------------------------------------------------------------------------
# Batch container
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    """A batch of instruction sequences with execution snapshots.

    Token layout: each sequence is a flat token stream with optional
    BOS/EOS markers. token_instr_idx[b, t] gives the instruction
    index that token t belongs to (-1 for BOS/EOS/padding).

    Execution layout: per_instr_regs[b, i, s, r] is the value of
    register r BEFORE instruction i, on input state s. The final
    state (after the last instruction) is at index n_instructions[b].
    """
    token_ids: np.ndarray         # (B, max_tokens) int64
    padding_mask: np.ndarray      # (B, max_tokens) bool — True = padding
    token_instr_idx: np.ndarray   # (B, max_tokens) int32 — instr each tok belongs to, -1 if none
    n_instructions: np.ndarray    # (B,) int32 — instructions per sequence
    per_instr_regs: np.ndarray    # (B, max_instrs+1, n_inputs, 32) int32
    per_instr_pcs: np.ndarray     # (B, max_instrs+1, n_inputs) int64


# ---------------------------------------------------------------------------
# Basic block generation with internal data flow
# ---------------------------------------------------------------------------

def _data_flow_instruction(rng: np.random.Generator,
                           live_regs: set[int],
                           force_branch: bool = False) -> Instruction:
    """Generate an instruction that, with high probability, reads from
    the live register set (creating a data dependency).

    If force_branch is True, generate a branch instruction.
    """
    live_list = sorted(live_regs)  # deterministic ordering for rng.choice

    if force_branch:
        op = str(rng.choice(_BRANCH_OPS))
        if len(live_list) >= 2:
            rs1, rs2 = rng.choice(live_list, size=2, replace=False)
        elif len(live_list) == 1:
            rs1 = live_list[0]
            rs2 = int(rng.choice(_SRC_REGS))
        else:
            rs1 = int(rng.choice(_SRC_REGS))
            rs2 = int(rng.choice(_SRC_REGS))
        return Instruction(op, int(rs1), int(rs2), _random_branch_offset(rng))

    use_live = len(live_list) > 0 and rng.random() < 0.6

    def _pick_src():
        return int(rng.choice(live_list)) if use_live else int(rng.choice(_SRC_REGS))

    roll = rng.random()
    if roll < 0.4:
        # R-type ALU
        op = str(rng.choice(_ALU_R_OPS))
        rd = int(rng.choice(_DEST_REGS))
        rs1 = _pick_src()
        rs2 = int(rng.choice(_SRC_REGS))
        return Instruction(op, rd, rs1, rs2)
    elif roll < 0.75:
        # I-type ALU
        op = str(rng.choice(_ALU_I_OPS))
        rd = int(rng.choice(_DEST_REGS))
        rs1 = _pick_src()
        return Instruction(op, rd, rs1, _i_type_imm(rng, op))
    elif roll < 0.85:
        # Load
        op = str(rng.choice(_LOAD_OPS))
        rd = int(rng.choice(_DEST_REGS))
        rs1 = _pick_src()
        return Instruction(op, rd, _random_imm12(rng), rs1)
    elif roll < 0.95:
        # Store
        op = str(rng.choice(_STORE_OPS))
        rs2 = _pick_src()
        rs1 = int(rng.choice(_SRC_REGS))
        return Instruction(op, rs2, _random_imm12(rng), rs1)
    else:
        # LUI
        rd = int(rng.choice(_DEST_REGS))
        return Instruction('LUI', rd, _random_upper_imm(rng))


def _instr_dest_reg(instr: Instruction) -> int | None:
    """Return the destination register if instr writes to a register,
    None otherwise (stores, branches don't write registers)."""
    if instr.opcode in STORE_TYPE or instr.opcode in B_TYPE:
        return None
    return instr.args[0]


def random_basic_block(rng: np.random.Generator,
                       max_length: int = 10) -> list[Instruction]:
    """Generate a basic block with internal data flow.

    A basic block is a sequence of non-control-flow instructions
    terminated by a branch. Each instruction has a high probability
    of reading from a register written by a previous instruction
    in the block (data dependency).

    The block length is drawn uniformly from [2, max_length].
    """
    length = int(rng.integers(2, max_length + 1))

    instructions = []
    live_regs: set[int] = set()

    for _ in range(length - 1):
        instr = _data_flow_instruction(rng, live_regs)
        instructions.append(instr)
        rd = _instr_dest_reg(instr)
        if rd is not None and rd != 0:
            live_regs.add(rd)

    branch = _data_flow_instruction(rng, live_regs, force_branch=True)
    instructions.append(branch)

    return instructions


# ---------------------------------------------------------------------------
# Sequence execution with per-instruction snapshots
# ---------------------------------------------------------------------------

def execute_sequence(instructions: list[Instruction],
                     regs: np.ndarray,
                     pc: int,
                     rng: np.random.Generator,
                     ctx) -> tuple[np.ndarray, np.ndarray]:
    """Execute a sequence and capture per-instruction register snapshots.

    A single SparseMemory is shared across all instructions in the
    sequence, so stores persist to subsequent loads. Memory is
    zero-initialized; uninitialized addresses read as 0.

    Returns:
        regs_snapshots: (n_instructions+1, 32) int32 — register state
            BEFORE each instruction, plus the final state at the end.
        pc_snapshots: (n_instructions+1,) int64 — PC before each
            instruction, plus the final PC at the end.
    """
    n = len(instructions)
    regs_snapshots = np.zeros((n + 1, 32), dtype=np.int32)
    pc_snapshots = np.zeros(n + 1, dtype=np.int64)

    cur_regs = np.array(regs, dtype=np.int32)
    cur_regs[0] = 0
    cur_pc = pc
    mem = SparseMemory()

    for i, instr in enumerate(instructions):
        regs_snapshots[i] = cur_regs
        pc_snapshots[i] = cur_pc

        state, _, mem = run_instruction(
            [instr], regs=cur_regs, pc=0, rng=rng, _ctx=ctx,
            max_steps=1, mem=mem)
        cur_regs = state.regs
        cur_pc = cur_pc + 4

    regs_snapshots[n] = cur_regs
    pc_snapshots[n] = cur_pc

    return regs_snapshots, pc_snapshots


# ---------------------------------------------------------------------------
# Tokenization with instruction-index tracking
# ---------------------------------------------------------------------------

def _encode_with_instr_idx(instructions: list[Instruction]
                           ) -> tuple[list[int], list[int]]:
    """Encode a sequence to tokens and track which instruction each
    token belongs to.

    Returns (token_ids, token_instr_idx). token_instr_idx[t] is the
    instruction index for token t, or -1 for BOS/EOS markers.
    """
    tokens = [BOS]
    instr_idx = [-1]
    for i, instr in enumerate(instructions):
        instr_tokens = encode_instruction(instr)
        tokens.extend(instr_tokens)
        instr_idx.extend([i] * len(instr_tokens))
    tokens.append(EOS)
    instr_idx.append(-1)
    return tokens, instr_idx


# ---------------------------------------------------------------------------
# Batch production
# ---------------------------------------------------------------------------

def produce_batch(batch_size: int, n_inputs: int, max_block_len: int,
                  rng: np.random.Generator) -> Batch:
    """Generate one batch of instruction sequences with executions."""
    sequences: list[list[Instruction]] = []
    encoded: list[tuple[list[int], list[int]]] = []

    for _ in range(batch_size):
        instrs = random_basic_block(rng, max_length=max_block_len)
        sequences.append(instrs)
        encoded.append(_encode_with_instr_idx(instrs))

    max_tokens = max(len(toks) for toks, _ in encoded)
    max_instrs = max(len(s) for s in sequences)

    token_ids = np.full((batch_size, max_tokens), PAD, dtype=np.int64)
    padding_mask = np.ones((batch_size, max_tokens), dtype=np.bool_)
    token_instr_idx = np.full((batch_size, max_tokens), -1, dtype=np.int32)
    n_instructions = np.array([len(s) for s in sequences], dtype=np.int32)

    for b, (toks, idx) in enumerate(encoded):
        n = len(toks)
        token_ids[b, :n] = toks
        padding_mask[b, :n] = False
        token_instr_idx[b, :n] = idx

    per_instr_regs = np.zeros((batch_size, max_instrs + 1, n_inputs, 32),
                              dtype=np.int32)
    per_instr_pcs = np.zeros((batch_size, max_instrs + 1, n_inputs),
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

    return Batch(
        token_ids=token_ids,
        padding_mask=padding_mask,
        token_instr_idx=token_instr_idx,
        n_instructions=n_instructions,
        per_instr_regs=per_instr_regs,
        per_instr_pcs=per_instr_pcs,
    )


# ---------------------------------------------------------------------------
# Binary I/O — RVS format
#
# Stream format:
#   Stream header (once):  4-byte magic "RVS\x00" + 1-byte version (1)
#                          + 6 dtype chars
#   Per batch:             16-byte header (B, max_tokens, max_instrs,
#                          n_inputs as uint32)
#                          + raw array data in field order
# ---------------------------------------------------------------------------

_MAGIC = b'RVS\x00'
_VERSION = 1
_STREAM_HEADER = struct.Struct('<4sB6s')
_BATCH_HEADER = struct.Struct('<IIII')  # B, max_tokens, max_instrs, n_inputs

_FIELD_DTYPES = (
    np.dtype(np.int64),   # token_ids
    np.dtype(np.bool_),   # padding_mask
    np.dtype(np.int32),   # token_instr_idx
    np.dtype(np.int32),   # n_instructions
    np.dtype(np.int32),   # per_instr_regs
    np.dtype(np.int64),   # per_instr_pcs
)
_DTYPE_CHARS = b''.join(dt.char.encode() for dt in _FIELD_DTYPES)


def _batch_body_size(B, max_tokens, max_instrs, n_inputs):
    """Compute the byte size of a batch body."""
    return (
        B * max_tokens * _FIELD_DTYPES[0].itemsize +     # token_ids
        B * max_tokens * _FIELD_DTYPES[1].itemsize +     # padding_mask
        B * max_tokens * _FIELD_DTYPES[2].itemsize +     # token_instr_idx
        B * _FIELD_DTYPES[3].itemsize +                  # n_instructions
        B * (max_instrs + 1) * n_inputs * 32
            * _FIELD_DTYPES[4].itemsize +                # per_instr_regs
        B * (max_instrs + 1) * n_inputs
            * _FIELD_DTYPES[5].itemsize                  # per_instr_pcs
    )


def write_stream_header(f):
    """Write the stream header."""
    f.write(_STREAM_HEADER.pack(_MAGIC, _VERSION, _DTYPE_CHARS))


def read_stream_header(f):
    """Read and validate the stream header."""
    buf = f.read(_STREAM_HEADER.size)
    if len(buf) < _STREAM_HEADER.size:
        raise ValueError('Missing stream header')
    magic, version, dtype_chars = _STREAM_HEADER.unpack(buf)
    if magic != _MAGIC:
        raise ValueError(f'Bad magic: {magic!r} (expected {_MAGIC!r})')
    if version != _VERSION:
        raise ValueError(f'Unsupported version: {version}')
    if dtype_chars != _DTYPE_CHARS:
        raise ValueError(f'Dtype mismatch: {dtype_chars!r}')


def write_batch(f, batch: Batch):
    """Write a Batch to a binary stream."""
    B, max_tokens = batch.token_ids.shape
    max_instrs = batch.per_instr_regs.shape[1] - 1
    n_inputs = batch.per_instr_regs.shape[2]
    f.write(_BATCH_HEADER.pack(B, max_tokens, max_instrs, n_inputs))
    f.write(batch.token_ids.tobytes())
    f.write(batch.padding_mask.tobytes())
    f.write(batch.token_instr_idx.tobytes())
    f.write(batch.n_instructions.tobytes())
    f.write(batch.per_instr_regs.tobytes())
    f.write(batch.per_instr_pcs.tobytes())


def read_batch(f) -> Batch | None:
    """Read a Batch from a binary stream. Returns None at clean EOF."""
    header = f.read(_BATCH_HEADER.size)
    if len(header) == 0:
        return None
    if len(header) < _BATCH_HEADER.size:
        raise EOFError(f'Truncated batch header ({len(header)} bytes)')
    B, max_tokens, max_instrs, n_inputs = _BATCH_HEADER.unpack(header)

    def _read_array(dtype, shape):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        buf = f.read(nbytes)
        if len(buf) < nbytes:
            raise EOFError(f'Truncated batch data (got {len(buf)}, expected {nbytes})')
        return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()

    return Batch(
        token_ids=_read_array(_FIELD_DTYPES[0], (B, max_tokens)),
        padding_mask=_read_array(_FIELD_DTYPES[1], (B, max_tokens)),
        token_instr_idx=_read_array(_FIELD_DTYPES[2], (B, max_tokens)),
        n_instructions=_read_array(_FIELD_DTYPES[3], (B,)),
        per_instr_regs=_read_array(_FIELD_DTYPES[4],
                                    (B, max_instrs + 1, n_inputs, 32)),
        per_instr_pcs=_read_array(_FIELD_DTYPES[5],
                                   (B, max_instrs + 1, n_inputs)),
    )


def read_batch_bytes(f) -> bytes | None:
    """Read one complete batch as raw bytes. For pass-through use."""
    header = f.read(_BATCH_HEADER.size)
    if len(header) == 0:
        return None
    if len(header) < _BATCH_HEADER.size:
        raise EOFError(f'Truncated batch header ({len(header)} bytes)')
    B, max_tokens, max_instrs, n_inputs = _BATCH_HEADER.unpack(header)
    body_size = _batch_body_size(B, max_tokens, max_instrs, n_inputs)
    body = f.read(body_size)
    if len(body) < body_size:
        raise EOFError(f'Truncated batch data (got {len(body)}, expected {body_size})')
    return header + body


class BatchReader:
    """Reads pre-generated batches from a binary stream."""

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
