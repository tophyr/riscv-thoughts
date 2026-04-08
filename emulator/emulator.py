"""RV32I instruction execution.

Wraps TinyFive for the actual instruction logic, with overrides for
numpy int32/uint32 boundary issues in newer numpy versions. TinyFive
is used as a state container (registers, memory, PC) and for the ~17
instructions whose computations stay within numpy's type ranges. The
remaining ~20 instructions are overridden in the dispatch table.
"""

import warnings
import numpy as np
from tinyfive.machine import machine

# TinyFive uses numpy int32 arithmetic which wraps on overflow — correct
# RISC-V behavior, but numpy warns about it.
warnings.filterwarnings('ignore', message='overflow', category=RuntimeWarning,
                        module='tinyfive')

# Instructions grouped by signature for dispatch.
R_TYPE = {
    'ADD', 'SUB', 'XOR', 'OR', 'AND',
    'SLL', 'SRL', 'SRA',
    'SLT', 'SLTU',
}

I_TYPE = {
    'ADDI', 'XORI', 'ORI', 'ANDI',
    'SLLI', 'SRLI', 'SRAI',
    'SLTI', 'SLTIU',
}

B_TYPE = {
    'BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU',
}

LOAD_TYPE = {
    'LB', 'LBU', 'LH', 'LHU', 'LW',
}

STORE_TYPE = {
    'SB', 'SH', 'SW',
}

ALL_OPCODES = R_TYPE | I_TYPE | B_TYPE | LOAD_TYPE | STORE_TYPE | {
    'LUI', 'AUIPC', 'JAL', 'JALR',
}


MEM_WIDTH = {
    'LB': 1, 'LBU': 1, 'SB': 1,
    'LH': 2, 'LHU': 2, 'SH': 2,
    'LW': 4, 'SW': 4,
}


class SparseMemory:
    """Dict-backed memory supporting arbitrary 32-bit addresses.

    Drop-in replacement for TinyFive's numpy byte array for the subset
    of operations TinyFive uses: integer get/set, slice reset, and len().
    """

    def __init__(self):
        self._data = {}

    def __getitem__(self, addr):
        if isinstance(addr, slice):
            return self
        return np.uint8(self._data.get(int(addr), 0))

    def __setitem__(self, addr, val):
        if isinstance(addr, slice):
            self._data.clear()
            return
        self._data[int(addr)] = int(val) & 0xFF

    def __len__(self):
        return 2**32

    def fill_random(self, addr, n_bytes, rng):
        """Initialize n_bytes of random data starting at addr."""
        for i in range(n_bytes):
            self._data[int(addr) + i] = int(rng.integers(0, 256))


class RV32IState:
    """Immutable snapshot of the integer register file."""

    __slots__ = ('regs',)

    def __init__(self, regs: np.ndarray):
        self.regs = np.array(regs, dtype=np.int32, copy=True)
        assert self.regs.shape == (32,)
        self.regs[0] = 0

    def distance(self, other: 'RV32IState') -> dict:
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
            return 2
        if self.opcode in B_TYPE:
            return 2
        if self.opcode in LOAD_TYPE:
            return 1
        if self.opcode in STORE_TYPE:
            return 1
        if self.opcode in ('LUI', 'AUIPC'):
            return 1
        if self.opcode == 'JAL':
            return 1
        if self.opcode == 'JALR':
            return 2
        return 0


def _build_dispatch(m):
    """Build instruction dispatch table for a TinyFive machine.

    Starts with TinyFive's own methods, then overrides the ones that
    have numpy int32/uint32 boundary issues with wrappers that compute
    in Python int space and convert back.
    """
    def _u32(x):
        return np.uint32(np.int32(x))

    def _i32(x):
        return np.int32(np.uint32(x))

    def _safe_ipc(incr):
        m.x[0] = 0
        m.pc[0] = (int(m.pc[0]) + incr) & 0xFFFFFFFF

    def _jal(rd, imm):
        m.x[rd] = int(m.pc[0]) + 4
        _safe_ipc(imm)

    def _jalr(rd, rs1, imm):
        t = int(m.pc[0]) + 4
        m.pc[0] = (int(m.x[rs1]) + imm) & ~1 & 0xFFFFFFFF
        m.x[rd] = t
        m.x[0] = 0

    def _srl(rd, rs1, rs2):
        m.x[rd] = _i32(_u32(m.x[rs1]) >> (m.x[rs2] & 0x1f))
        _safe_ipc(4)

    def _srli(rd, rs1, imm):
        m.x[rd] = _i32(_u32(m.x[rs1]) >> imm)
        _safe_ipc(4)

    def _sltu(rd, rs1, rs2):
        m.x[rd] = 1 if _u32(m.x[rs1]) < _u32(m.x[rs2]) else 0
        _safe_ipc(4)

    def _sltiu(rd, rs1, imm):
        m.x[rd] = 1 if _u32(m.x[rs1]) < _u32(imm) else 0
        _safe_ipc(4)

    def _auipc(rd, imm):
        m.x[rd] = _i32((int(m.pc[0]) + (imm << 12)) & 0xFFFFFFFF)
        _safe_ipc(4)

    def _lui(rd, imm):
        m.x[rd] = _i32((imm << 12) & 0xFFFFFFFF)
        _safe_ipc(4)

    def _beq(rs1, rs2, imm):  _safe_ipc(imm if m.x[rs1] == m.x[rs2] else 4)
    def _bne(rs1, rs2, imm):  _safe_ipc(imm if m.x[rs1] != m.x[rs2] else 4)
    def _blt(rs1, rs2, imm):  _safe_ipc(imm if m.x[rs1] <  m.x[rs2] else 4)
    def _bge(rs1, rs2, imm):  _safe_ipc(imm if m.x[rs1] >= m.x[rs2] else 4)
    def _bltu(rs1, rs2, imm): _safe_ipc(imm if _u32(m.x[rs1]) <  _u32(m.x[rs2]) else 4)
    def _bgeu(rs1, rs2, imm): _safe_ipc(imm if _u32(m.x[rs1]) >= _u32(m.x[rs2]) else 4)

    dispatch = {op: getattr(m, op) for op in ALL_OPCODES}
    dispatch['JAL'] = _jal
    dispatch['JALR'] = _jalr
    dispatch['SRL'] = _srl
    dispatch['SRLI'] = _srli
    dispatch['SLTU'] = _sltu
    dispatch['SLTIU'] = _sltiu
    dispatch['AUIPC'] = _auipc
    dispatch['LUI'] = _lui
    dispatch['BEQ'] = _beq
    dispatch['BNE'] = _bne
    dispatch['BLT'] = _blt
    dispatch['BGE'] = _bge
    dispatch['BLTU'] = _bltu
    dispatch['BGEU'] = _bgeu
    return dispatch


def make_ctx():
    """Create a reusable (machine, dispatch) context for run().

    The machine object is stateful (registers, PC, memory), but run()
    resets it before each execution. Reusing the context avoids
    recreating the machine and dispatch table on every call.
    """
    m = machine(64)
    return (m, _build_dispatch(m))


def run(
    instructions: list[Instruction],
    regs: np.ndarray | None = None,
    pc: int = 0,
    mem=None,
    rng: np.random.Generator | None = None,
    max_steps: int | None = None,
    _ctx=None,
) -> tuple[RV32IState, int, object]:
    """Execute instructions and return final state.

    Args:
        instructions: List of Instruction objects to execute.
        regs: Initial register file (32 x int32). Zeros if None.
        pc: Initial program counter value.
        mem: Memory object — SparseMemory for arbitrary addressing,
             numpy uint8 array for fixed-size, or None for a fresh
             SparseMemory with random fill for loads.
        rng: Random generator for SparseMemory fill when mem is None.
        max_steps: Maximum instructions to execute. Defaults to
            10x the instruction count.
        _ctx: Reusable (machine, dispatch) from make_ctx(). Created
            fresh if None.

    Returns:
        (RV32IState, final_pc, final_mem) tuple.
    """
    if _ctx is not None:
        m, dispatch = _ctx
    else:
        m, dispatch = make_ctx()

    # Registers.
    if regs is not None:
        m.x[:] = np.asarray(regs, dtype=np.int32)
        m.x[0] = 0

    # PC.
    m.pc[0] = pc & 0xFFFFFFFF

    # Memory.
    if mem is None:
        mem = SparseMemory()
        if rng is not None:
            for instr in instructions:
                if instr.opcode in LOAD_TYPE:
                    _, imm, rs1 = instr.args
                    addr = int(m.x[rs1]) + imm
                    width = MEM_WIDTH[instr.opcode]
                    mem.fill_random(addr, width, rng)
    m.mem = mem

    if max_steps is None:
        max_steps = len(instructions) * 10

    pc_start = pc
    n_instr = len(instructions)
    steps = 0
    while steps < max_steps:
        idx = (int(m.pc[0]) - pc_start) // 4
        if idx < 0 or idx >= n_instr:
            break
        instr = instructions[idx]
        dispatch[instr.opcode](*instr.args)
        steps += 1

    return RV32IState(m.x), int(m.pc[0]), m.mem



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
    regs[0] = 0
    return regs
