"""Thin wrapper around TinyFive for executing RV32I instruction sequences.

Provides a clean interface: set initial register state, execute a sequence
of instructions, return final register state. Hides TinyFive's internals
behind an API that can be swapped for a custom emulator later.
"""

import warnings
import numpy as np
from tinyfive.machine import machine

# TinyFive uses numpy int32 arithmetic which wraps on overflow — correct
# RISC-V behavior, but numpy warns about it.
warnings.filterwarnings('ignore', message='overflow', category=RuntimeWarning,
                        module='tinyfive')

# Instructions grouped by signature for dispatch.
# R-type: (rd, rs1, rs2)
R_TYPE = {
    'ADD', 'SUB', 'XOR', 'OR', 'AND',
    'SLL', 'SRL', 'SRA',
    'SLT', 'SLTU',
}

# I-type: (rd, rs1, imm)
I_TYPE = {
    'ADDI', 'XORI', 'ORI', 'ANDI',
    'SLLI', 'SRLI', 'SRAI',
    'SLTI', 'SLTIU',
}

# Branch: (rs1, rs2, imm)
B_TYPE = {
    'BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU',
}

# Load: (rd, imm, rs1) — note TinyFive's argument order
LOAD_TYPE = {
    'LB', 'LBU', 'LH', 'LHU', 'LW',
}

# Store: (rs2, imm, rs1) — note TinyFive's argument order
STORE_TYPE = {
    'SB', 'SH', 'SW',
}

# All supported opcodes for validation.
ALL_OPCODES = R_TYPE | I_TYPE | B_TYPE | LOAD_TYPE | STORE_TYPE | {
    'LUI', 'AUIPC', 'JAL', 'JALR',
}

# Memory size for the TinyFive machine. 64KB is plenty for short sequences.
DEFAULT_MEM_SIZE = 65536

# Default base address for a memory region that load/store instructions can
# use safely. We pick the middle of the memory space so that small positive
# and negative offsets stay in bounds.
DEFAULT_MEM_BASE = DEFAULT_MEM_SIZE // 2


class RV32IState:
    """Immutable snapshot of the integer register file."""

    __slots__ = ('regs',)

    def __init__(self, regs: np.ndarray):
        self.regs = np.array(regs, dtype=np.int32, copy=True)
        assert self.regs.shape == (32,)
        self.regs[0] = 0  # x0 is always zero

    def distance(self, other: 'RV32IState') -> dict:
        """Compute several distance metrics between two states."""
        diff = self.regs.astype(np.int64) - other.regs.astype(np.int64)
        neq = self.regs != other.regs
        return {
            'regs_differ': int(np.sum(neq)),
            'l1': int(np.sum(np.abs(diff))),
            'l2': float(np.sqrt(np.sum(diff.astype(np.float64) ** 2))),
            'hamming': int(np.sum(np.unpackbits(
                (self.regs.view(np.uint8) ^ other.regs.view(np.uint8))
            ))),
            'which_differ': np.where(neq)[0].tolist(),
        }

    def __eq__(self, other):
        if not isinstance(other, RV32IState):
            return NotImplemented
        return np.array_equal(self.regs, other.regs)

    def __repr__(self):
        nonzero = {f'x{i}': int(v) for i, v in enumerate(self.regs) if v != 0}
        return f'RV32IState({nonzero})'


class Instruction:
    """A single RV32I instruction in structural form."""

    __slots__ = ('opcode', 'args')

    def __init__(self, opcode: str, *args: int):
        assert opcode in ALL_OPCODES, f'unknown opcode: {opcode}'
        self.opcode = opcode
        self.args = args

    def __repr__(self):
        args_str = ', '.join(
            f'x{a}' if i < self._num_reg_args() else str(a)
            for i, a in enumerate(self.args)
        )
        return f'{self.opcode} {args_str}'

    def _num_reg_args(self):
        if self.opcode in R_TYPE:
            return 3
        if self.opcode in I_TYPE:
            return 2  # rd, rs1 are regs; imm is not
        if self.opcode in B_TYPE:
            return 2  # rs1, rs2 are regs; imm is not
        if self.opcode in LOAD_TYPE:
            return 1  # rd is reg; imm and rs1 follow TinyFive order
        if self.opcode in STORE_TYPE:
            return 1  # rs2 is reg; imm and rs1 follow TinyFive order
        if self.opcode == 'LUI' or self.opcode == 'AUIPC':
            return 1  # rd
        if self.opcode == 'JAL':
            return 1  # rd
        if self.opcode == 'JALR':
            return 2  # rd, rs1
        return 0


class Executor:
    """Reusable executor for RV32I instructions.

    Holds a TinyFive machine and dispatch table, resetting state between
    executions instead of recreating them. Use this for batch workloads
    where execute() is called thousands of times.
    """

    def __init__(self, mem_size: int = DEFAULT_MEM_SIZE):
        self._m = machine(mem_size)
        self._dispatch = self._build_dispatch()

    def _build_dispatch(self) -> dict:
        m = self._m

        def _u32(x):
            return np.uint32(np.int32(x))

        def _i32(x):
            return np.int32(np.uint32(x))

        def _jal(rd, imm):
            m.x[rd] = int(m.pc[0]) + 4
            m.ipc(imm)

        def _jalr(rd, rs1, imm):
            t = int(m.pc[0]) + 4
            m.pc[0] = (int(m.x[rs1]) + imm) & ~1
            m.x[rd] = t
            m.x[0] = 0

        def _srl(rd, rs1, rs2):
            m.x[rd] = _i32(_u32(m.x[rs1]) >> (m.x[rs2] & 0x1f))
            m.ipc()

        def _srli(rd, rs1, imm):
            m.x[rd] = _i32(_u32(m.x[rs1]) >> imm)
            m.ipc()

        def _sltu(rd, rs1, rs2):
            m.x[rd] = 1 if _u32(m.x[rs1]) < _u32(m.x[rs2]) else 0
            m.ipc()

        def _sltiu(rd, rs1, imm):
            m.x[rd] = 1 if _u32(m.x[rs1]) < _u32(imm) else 0
            m.ipc()

        dispatch = {op: getattr(m, op) for op in ALL_OPCODES}
        dispatch['JAL'] = _jal
        dispatch['JALR'] = _jalr
        dispatch['SRL'] = _srl
        dispatch['SRLI'] = _srli
        dispatch['SLTU'] = _sltu
        dispatch['SLTIU'] = _sltiu
        return dispatch

    def run(
        self,
        instructions: list[Instruction],
        initial_regs: np.ndarray | None = None,
        initial_mem: np.ndarray | None = None,
        max_steps: int | None = None,
    ) -> RV32IState:
        """Execute instructions and return the final state.

        Resets the machine state before execution. Same semantics as the
        module-level execute() function.
        """
        m = self._m

        # Reset state.
        m.x[:] = 0
        m.pc[0] = 0
        m.mem[:] = 0

        if initial_regs is not None:
            m.x[:] = np.asarray(initial_regs, dtype=np.int32)
            m.x[0] = 0

        if initial_mem is not None:
            n = min(len(initial_mem), len(m.mem))
            m.mem[:n] = np.asarray(initial_mem[:n], dtype=np.uint8)

        if max_steps is None:
            max_steps = len(instructions) * 10

        dispatch = self._dispatch
        pc_start = 0
        n_instr = len(instructions)

        steps = 0
        while steps < max_steps:
            idx = (int(m.pc[0]) - pc_start) // 4
            if idx < 0 or idx >= n_instr:
                break
            instr = instructions[idx]
            dispatch[instr.opcode](*instr.args)
            steps += 1

        return RV32IState(m.x)


def execute(
    instructions: list[Instruction],
    initial_regs: np.ndarray | None = None,
    initial_mem: np.ndarray | None = None,
    mem_size: int = DEFAULT_MEM_SIZE,
    max_steps: int | None = None,
) -> RV32IState:
    """Execute a sequence of RV32I instructions and return the final state.

    Convenience function that creates a fresh Executor per call. For batch
    workloads, use Executor directly to avoid repeated setup.

    Args:
        instructions: List of Instruction objects to execute.
        initial_regs: Initial register file (32 x int32). Zeros if None.
        initial_mem: Initial memory contents (uint8 array). Zeros if None.
        mem_size: Size of memory in bytes.
        max_steps: Maximum instructions to execute (for loops). Defaults to
            10x the instruction count.

    Returns:
        RV32IState with the final register file contents.
    """
    return Executor(mem_size).run(
        instructions, initial_regs, initial_mem, max_steps,
    )


def random_regs(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random initial register state."""
    if rng is None:
        rng = np.random.default_rng()
    regs = rng.integers(
        low=np.iinfo(np.int32).min,
        high=np.iinfo(np.int32).max,
        size=32,
        dtype=np.int32,
        endpoint=True,
    )
    regs[0] = 0  # x0 is hardwired to zero
    return regs
