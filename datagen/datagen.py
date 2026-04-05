"""Training data producers for the T0→T1 compressor.

Two implementations of the same iterator interface:
- InlineProducer: synchronous, single-threaded. For tests and debugging.
- ParallelProducer: fans out across processes. For real training.

Both yield Batch objects.
"""

from dataclasses import dataclass
from collections import deque

import numpy as np

from emulator import (
    Instruction, run as run_instruction, random_regs,
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

    for s in range(n_inputs):
        regs = random_regs(rng)
        pc = int(rng.integers(0, 1024)) * 4
        for i, instr in enumerate(instructions):
            state, final_pc, final_mem = run_instruction(
                [instr], regs=regs, pc=pc, rng=rng)
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


def _produce_batch_from_seed(args):
    """Entry point for ProcessPoolExecutor. Unpacks args and calls produce_batch."""
    seed, batch_size, n_inputs = args
    return produce_batch(batch_size, n_inputs, np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# Producer implementations
# ---------------------------------------------------------------------------

class InlineProducer:
    """Synchronous batch producer. Yields Batch objects one at a time.

    No threading, no multiprocessing. Suitable for tests and debugging.
    """

    def __init__(self, batch_size: int, n_inputs: int,
                 n_batches: int, seed: int):
        self._batch_size = batch_size
        self._n_inputs = n_inputs
        self._n_batches = n_batches
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self._n_batches):
            yield produce_batch(self._batch_size, self._n_inputs, self._rng)


class ParallelProducer:
    """Asynchronous batch producer backed by a ProcessPoolExecutor.

    Maintains a prefetch queue of futures so the training loop never
    waits for data. Must be closed after use.

    Fork the pool BEFORE initializing CUDA or torch threads.
    """

    def __init__(self, batch_size: int, n_inputs: int,
                 n_batches: int, seed: int,
                 n_workers: int, prefetch: int):
        from concurrent.futures import ProcessPoolExecutor

        self._batch_size = batch_size
        self._n_inputs = n_inputs
        self._rng = np.random.default_rng(seed)
        self._remaining = n_batches
        self._pool = ProcessPoolExecutor(max_workers=n_workers)
        self._pending = deque()

        for _ in range(min(prefetch, n_batches)):
            self._submit()

    def _submit(self):
        seed = int(self._rng.integers(0, 2**63))
        self._pending.append(
            self._pool.submit(_produce_batch_from_seed,
                              (seed, self._batch_size, self._n_inputs)))
        self._remaining -= 1

    def __iter__(self):
        return self

    def __next__(self):
        if not self._pending:
            raise StopIteration
        batch = self._pending.popleft().result()
        if self._remaining > 0:
            self._submit()
        return batch

    def close(self):
        self._pool.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
