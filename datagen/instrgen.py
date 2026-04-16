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

_DEST_REGS = list(range(0, 32))  # x0 included: write-to-x0 NOPs are a real semantic class
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


_CONFIG_EXTRA_KEYS = {'equivalences'}
_EQUIVALENCES_KEYS = {'rate', 'max_per_class', 'min_per_class', 'boost'}


def validate_distribution(dist):
    """Validate a config dict.

    Required: opcode-category keys with weights summing to 1.0.
    Optional: 'equivalences' sub-dict with 'rate' (float in [0,1])
    and 'max_per_class' (positive int).
    """
    keys = set(dist.keys())
    unknown = keys - set(_CATEGORY_OPS.keys()) - _CONFIG_EXTRA_KEYS
    if unknown:
        raise ValueError(f'Unknown config keys: {unknown}')
    weights = {k: v for k, v in dist.items() if k in _CATEGORY_OPS}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f'Distribution weights sum to {total}, not 1.0')
    if 'equivalences' in dist:
        eq = dist['equivalences']
        if not isinstance(eq, dict):
            raise ValueError("'equivalences' must be a dict")
        eq_unknown = set(eq.keys()) - _EQUIVALENCES_KEYS
        if eq_unknown:
            raise ValueError(f'Unknown equivalences keys: {eq_unknown}')
        if 'rate' in eq and not (0 <= eq['rate'] <= 1):
            raise ValueError(
                f"equivalences.rate must be in [0,1], got {eq['rate']}")
        if 'max_per_class' in eq and eq['max_per_class'] < 1:
            raise ValueError(
                "equivalences.max_per_class must be >= 1, got "
                f"{eq['max_per_class']}")
        if 'boost' in eq and not isinstance(eq['boost'], dict):
            raise ValueError("equivalences.boost must be a dict")


def load_distribution(path):
    """Load a batch config from a JSON file. May include opcode
    weights and an optional 'equivalences' section."""
    with open(path) as f:
        dist = json.load(f)
    validate_distribution(dist)
    return dist


def _build_opcode_table(dist):
    """Convert the opcode-distribution portion of a config dict to
    the internal (weight, ops) list. Ignores non-opcode keys."""
    return [(dist[cat], _CATEGORY_OPS[cat])
            for cat in dist if cat in _CATEGORY_OPS]


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
