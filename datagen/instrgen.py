"""Shared instruction generation infrastructure.

Opcode/register constants, immediate helpers, and the instruction
builder used by both seqgen.py and batchgen.py.
"""

import json
import numpy as np

from emulator import (
    Instruction,
    R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE,
)


# ---------------------------------------------------------------------------
# Opcode and register sets
# ---------------------------------------------------------------------------

_DEST_REGS = list(range(1, 32))
_SRC_REGS = list(range(0, 32))

_ALU_R_OPS = ['ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU']
_ALU_I_OPS = ['ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI', 'SLTI', 'SLTIU']
_SHIFT_I_OPS = {'SLLI', 'SRLI', 'SRAI'}
_LOAD_OPS = ['LB', 'LBU', 'LH', 'LHU', 'LW']
_STORE_OPS = ['SB', 'SH', 'SW']
_BRANCH_OPS = ['BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU']


# ---------------------------------------------------------------------------
# Immediate helpers
# ---------------------------------------------------------------------------

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


def _random_jal_offset(rng: np.random.Generator) -> int:
    """Random JAL offset (21-bit signed, 2-byte aligned)."""
    return int(rng.integers(-524288, 524288)) * 2


# ---------------------------------------------------------------------------
# Instruction builder
# ---------------------------------------------------------------------------

def _make_instruction(rng, opcode, rd=None, rs1=None, rs2=None, imm=None):
    """Build an instruction with the given opcode, filling in random
    values for any unspecified operands.

    This is the single point of truth for instruction construction.
    All instruction generators call this rather than constructing
    Instruction objects directly.
    """
    if opcode in R_TYPE:
        return Instruction(opcode,
                           rd if rd is not None else int(rng.choice(_DEST_REGS)),
                           rs1 if rs1 is not None else int(rng.choice(_SRC_REGS)),
                           rs2 if rs2 is not None else int(rng.choice(_SRC_REGS)))
    elif opcode in I_TYPE:
        return Instruction(opcode,
                           rd if rd is not None else int(rng.choice(_DEST_REGS)),
                           rs1 if rs1 is not None else int(rng.choice(_SRC_REGS)),
                           imm if imm is not None else _i_type_imm(rng, opcode))
    elif opcode in LOAD_TYPE:
        return Instruction(opcode,
                           rd if rd is not None else int(rng.choice(_DEST_REGS)),
                           imm if imm is not None else _random_imm12(rng),
                           rs1 if rs1 is not None else int(rng.choice(_SRC_REGS)))
    elif opcode in STORE_TYPE:
        return Instruction(opcode,
                           rs2 if rs2 is not None else int(rng.choice(_SRC_REGS)),
                           imm if imm is not None else _random_imm12(rng),
                           rs1 if rs1 is not None else int(rng.choice(_SRC_REGS)))
    elif opcode in B_TYPE:
        return Instruction(opcode,
                           rs1 if rs1 is not None else int(rng.choice(_SRC_REGS)),
                           rs2 if rs2 is not None else int(rng.choice(_SRC_REGS)),
                           imm if imm is not None else _random_branch_offset(rng))
    elif opcode in ('LUI', 'AUIPC'):
        return Instruction(opcode,
                           rd if rd is not None else int(rng.choice(_DEST_REGS)),
                           imm if imm is not None else _random_upper_imm(rng))
    elif opcode == 'JAL':
        return Instruction('JAL',
                           rd if rd is not None else int(rng.choice(_DEST_REGS)),
                           imm if imm is not None else _random_jal_offset(rng))
    elif opcode == 'JALR':
        return Instruction('JALR',
                           rd if rd is not None else int(rng.choice(_DEST_REGS)),
                           rs1 if rs1 is not None else int(rng.choice(_SRC_REGS)),
                           imm if imm is not None else _random_imm12(rng))
    else:
        raise ValueError(f'Unknown opcode: {opcode}')


# ---------------------------------------------------------------------------
# Random instruction generation
# ---------------------------------------------------------------------------

DEFAULT_DISTRIBUTION = {
    'R_ALU': 0.25,
    'I_ALU': 0.25,
    'LOAD': 0.125,
    'STORE': 0.075,
    'BRANCH': 0.15,
    'UPPER': 0.0825,
    'JAL': 0.04,
    'JALR': 0.0275,
}

_CATEGORY_OPS = {
    'R_ALU': _ALU_R_OPS,
    'I_ALU': _ALU_I_OPS,
    'LOAD': _LOAD_OPS,
    'STORE': _STORE_OPS,
    'BRANCH': _BRANCH_OPS,
    'UPPER': ['LUI', 'AUIPC'],
    'JAL': ['JAL'],
    'JALR': ['JALR'],
}


def validate_distribution(dist):
    """Validate an opcode distribution dict. Raises ValueError if
    keys are unrecognized or weights don't sum to 1.0."""
    unknown = set(dist.keys()) - set(_CATEGORY_OPS.keys())
    if unknown:
        raise ValueError(f'Unknown instruction categories: {unknown}')
    total = sum(dist.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f'Distribution weights sum to {total}, not 1.0')


def load_distribution(path):
    """Load an opcode distribution from a JSON file."""
    with open(path) as f:
        dist = json.load(f)
    validate_distribution(dist)
    return dist


def _build_opcode_table(dist):
    """Convert a distribution dict to the internal (weight, ops) list."""
    return [(dist[cat], _CATEGORY_OPS[cat]) for cat in dist]


# Validate and build default table at import time.
validate_distribution(DEFAULT_DISTRIBUTION)
_OPCODE_DISTRIBUTION = _build_opcode_table(DEFAULT_DISTRIBUTION)


def random_instruction(rng: np.random.Generator,
                       opcode_table=None) -> Instruction:
    """Generate a single random RV32I instruction.

    opcode_table: optional pre-built (weight, ops) list from
        _build_opcode_table. Uses the default distribution if None.
    """
    table = opcode_table if opcode_table is not None else _OPCODE_DISTRIBUTION
    roll = rng.random()
    cumulative = 0.0
    for weight, ops in table:
        cumulative += weight
        if roll < cumulative:
            op = str(rng.choice(ops))
            return _make_instruction(rng, op)
    return _make_instruction(rng, 'JALR')
