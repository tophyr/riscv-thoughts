"""Synthesis primitives: random instructions, equivalence-preserving
transforms, curated equivalence classes, and instruction-stream
collection rules.

Three kinds of "synthesis" live here, layered:

  Atoms — `random_instruction` (parameterized by an opcode-category
    distribution) and the related immediate/register helpers.

  Transforms — `relabel` (equivalence-preserving register permutation)
    and the manifest of curated equivalence classes (XOR x,x ≡ ADDI x,0,0,
    commutativity, etc.). Both produce equivalent variants of a chunk.

  Collection — `collect_groups(instr_iter, termination_rule)` walks an
    instruction iterable and emits groups whenever the rule fires.
    Termination rules are first-class composable predicates:
    `single()`, `until_branch()`, `until_transformation()`,
    `length_cap(n)`, and `either(*rules)`.

The same `collect_groups` function services both T1 single-instruction
batching (`single()`), T2 chunk extraction (`until_transformation()
| length_cap(n)`), and any future grouping rule.
"""

import json
import zlib
from dataclasses import dataclass, field

import numpy as np

from emulator import (
    Instruction,
    R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE,
)


# ===========================================================================
# Atoms — opcode classes, immediate helpers, instruction builder
# ===========================================================================

_DEST_REGS = list(range(0, 32))   # x0 included: write-to-x0 NOPs are real
_SRC_REGS = list(range(0, 32))

_ALU_R_OPS = ['ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU']
_ALU_I_OPS = ['ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI', 'SLTI', 'SLTIU']
_SHIFT_I_OPS = {'SLLI', 'SRLI', 'SRAI'}
_LOAD_OPS = ['LB', 'LBU', 'LH', 'LHU', 'LW']
_STORE_OPS = ['SB', 'SH', 'SW']
_BRANCH_OPS = ['BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU']


def _random_imm12(rng):
    """12-bit signed immediate [-2048, 2048)."""
    return int(rng.integers(-2048, 2048))


def _i_type_imm(rng, op):
    """5-bit shift amount or 12-bit signed immediate."""
    return int(rng.integers(0, 32)) if op in _SHIFT_I_OPS else _random_imm12(rng)


def _random_branch_offset(rng):
    """13-bit signed, 2-byte aligned."""
    return int(rng.integers(-2048, 2048)) * 2


def _random_upper_imm(rng):
    """20-bit unsigned."""
    return int(rng.integers(0, 0x100000))


def _random_jal_offset(rng):
    """21-bit signed, 2-byte aligned."""
    return int(rng.integers(-524288, 524288)) * 2


def _make_instruction(rng, opcode, rd=None, rs1=None, rs2=None, imm=None,
                      dest_regs=None):
    """Build an instruction with the given opcode, filling unspecified
    operands with random valid values. Single point of truth for
    instruction construction — all generators below call this.

    dest_regs: pool the random destination register (rd) is drawn from;
    defaults to _DEST_REGS (x0 included). Pass range(1,32) to exclude
    write-to-x0 (a corpus restriction for the register-writers-only tier).
    """
    dest = _DEST_REGS if dest_regs is None else dest_regs
    if opcode in R_TYPE:
        return Instruction(opcode,
                           rd if rd is not None else int(rng.choice(dest)),
                           rs1 if rs1 is not None else int(rng.choice(_SRC_REGS)),
                           rs2 if rs2 is not None else int(rng.choice(_SRC_REGS)))
    elif opcode in I_TYPE:
        return Instruction(opcode,
                           rd if rd is not None else int(rng.choice(dest)),
                           rs1 if rs1 is not None else int(rng.choice(_SRC_REGS)),
                           imm if imm is not None else _i_type_imm(rng, opcode))
    elif opcode in LOAD_TYPE:
        return Instruction(opcode,
                           rd if rd is not None else int(rng.choice(dest)),
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
                           rd if rd is not None else int(rng.choice(dest)),
                           imm if imm is not None else _random_upper_imm(rng))
    elif opcode == 'JAL':
        return Instruction('JAL',
                           rd if rd is not None else int(rng.choice(dest)),
                           imm if imm is not None else _random_jal_offset(rng))
    elif opcode == 'JALR':
        return Instruction('JALR',
                           rd if rd is not None else int(rng.choice(dest)),
                           rs1 if rs1 is not None else int(rng.choice(_SRC_REGS)),
                           imm if imm is not None else _random_imm12(rng))
    else:
        raise ValueError(f'Unknown opcode: {opcode}')


# ---------------------------------------------------------------------------
# Distribution config — controls the opcode-category mix in random_instruction
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

_CONFIG_EXTRA_KEYS = {'invalidity', 'exclude_x0_dest', 'chain', 'partners',
                      'aim'}
_AIM_KEYS = {'fraction'}
_PARTNERS_KEYS = {'rate', 'rewrites', 'mined'}
_CHAIN_KEYS = {'rate'}
_INVALIDITY_KEYS = {'rate', 'types'}


def validate_distribution(dist):
    """Validate a config dict.

    Required: opcode-category keys with weights summing to 1.0.
    Optional: 'invalidity' sub-dict {'rate': float, 'types': {name: w}}
              for invalid-window injection.
    Optional: 'partners' sub-dict {'rate': float, 'rewrites': [str],
              'mined': {'fraction': float, 'exclude': [str], 'reuse': int}}:
              the fraction of each batch made of equivalent pairs; for chunks
              the rewrites drawn from (`_REWRITES`; all of them when absent),
              and the fraction of pairs found instead by search and proved —
              each mined fresh and drawn at most `reuse` times from a rolling
              pool, renamed per draw (`datagen.mine.sample_pooled_pair`), no
              opcode of `exclude` in it.
    Optional: 'aim' sub-dict {'fraction': float}: the fraction of each
              row's anchor states aimed at its boundaries (`datagen.aim`).
    """
    keys = set(dist.keys())
    unknown = keys - set(_CATEGORY_OPS.keys()) - _CONFIG_EXTRA_KEYS
    if unknown:
        raise ValueError(f'Unknown config keys: {unknown}')
    weights = {k: v for k, v in dist.items() if k in _CATEGORY_OPS}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f'Distribution weights sum to {total}, not 1.0')
    if 'chain' in dist:
        ch = dist['chain']
        if not isinstance(ch, dict):
            raise ValueError("'chain' must be a dict")
        ch_unknown = set(ch.keys()) - _CHAIN_KEYS
        if ch_unknown:
            raise ValueError(f'Unknown chain keys: {ch_unknown}')
        if 'rate' in ch and not (0 <= ch['rate'] <= 1):
            raise ValueError(
                f"chain.rate must be in [0,1], got {ch['rate']}")
    if 'partners' in dist:
        pa = dist['partners']
        if not isinstance(pa, dict):
            raise ValueError("'partners' must be a dict")
        pa_unknown = set(pa.keys()) - _PARTNERS_KEYS
        if pa_unknown:
            raise ValueError(f'Unknown partners keys: {pa_unknown}')
        if 'rate' in pa and not (0 <= pa['rate'] <= 1):
            raise ValueError(
                f"partners.rate must be in [0,1], got {pa['rate']}")
        bad = set(pa.get('rewrites', [])) - set(_REWRITES)
        if bad:
            raise ValueError(f'Unknown partner rewrites: {bad}')
        mi = pa.get('mined')
        if mi is not None:
            if not isinstance(mi, dict) or set(mi) - {'fraction', 'exclude', 'reuse'}:
                raise ValueError("partners.mined must be {'fraction': float, "
                                 "'exclude': [opcode], 'reuse': int}")
            if int(mi.get('reuse', 1)) < 1:
                raise ValueError(f"partners.mined.reuse must be >= 1, got {mi['reuse']}")
            if not 0 <= mi.get('fraction', 1.0) <= 1:
                raise ValueError(f"partners.mined.fraction must be in [0,1], got {mi['fraction']}")
    if 'aim' in dist:
        am = dist['aim']
        if not isinstance(am, dict):
            raise ValueError("'aim' must be a dict")
        am_unknown = set(am.keys()) - _AIM_KEYS
        if am_unknown:
            raise ValueError(f'Unknown aim keys: {am_unknown}')
        if 'fraction' in am and not (0 <= am['fraction'] <= 1):
            raise ValueError(
                f"aim.fraction must be in [0,1], got {am['fraction']}")
    if 'invalidity' in dist:
        inv = dist['invalidity']
        if not isinstance(inv, dict):
            raise ValueError("'invalidity' must be a dict")
        inv_unknown = set(inv.keys()) - _INVALIDITY_KEYS
        if inv_unknown:
            raise ValueError(f'Unknown invalidity keys: {inv_unknown}')
        if 'rate' in inv and not (0 <= inv['rate'] <= 1):
            raise ValueError(
                f"invalidity.rate must be in [0,1], got {inv['rate']}")
        if 'types' in inv and not isinstance(inv['types'], dict):
            raise ValueError("invalidity.types must be a dict")


def load_distribution(path):
    """Load a config from a JSON file."""
    with open(path) as f:
        dist = json.load(f)
    validate_distribution(dist)
    return dist


def build_opcode_table(dist):
    """Convert opcode-distribution portion of a config dict to the
    internal (weight, ops) list. Ignores non-opcode keys."""
    return [(dist[cat], _CATEGORY_OPS[cat])
            for cat in dist if cat in _CATEGORY_OPS]


# Validate and build default table at import time.
validate_distribution(DEFAULT_DISTRIBUTION)
_OPCODE_DISTRIBUTION = build_opcode_table(DEFAULT_DISTRIBUTION)


# ---------------------------------------------------------------------------
# Random instruction generators
# ---------------------------------------------------------------------------

def random_instruction(rng, opcode_table=None, dest_regs=None):
    """Generate a single random RV32I instruction.

    opcode_table: optional pre-built (weight, ops) list from
        `build_opcode_table`. Uses the default distribution if None.
    dest_regs: optional rd pool (see _make_instruction) — pass range(1,32)
        to exclude write-to-x0.
    """
    table = opcode_table if opcode_table is not None else _OPCODE_DISTRIBUTION
    roll = rng.random()
    cumulative = 0.0
    for weight, ops in table:
        cumulative += weight
        if roll < cumulative:
            op = str(rng.choice(ops))
            return _make_instruction(rng, op, dest_regs=dest_regs)
    return _make_instruction(rng, 'JALR', dest_regs=dest_regs)


# ===========================================================================
# Transforms — equivalence-preserving register relabeling
# ===========================================================================

# Per-opcode-class register-slot layout: arg indices that are register
# references (non-listed slots are immediates).
_REG_SLOTS = {
    'R':         (0, 1, 2),  # rd, rs1, rs2
    'I':         (0, 1),     # rd, rs1
    'B':         (0, 1),     # rs1, rs2
    'LOAD':      (0, 2),     # rd, rs1   (imm sits between)
    'STORE':     (0, 2),     # rs2, rs1  (imm sits between)
    'LUI_AUIPC': (0,),       # rd
    'JAL':       (0,),       # rd
    'JALR':      (0, 1),     # rd, rs1
}


def _classify_op(opcode):
    if opcode in R_TYPE:
        return 'R'
    if opcode in I_TYPE:
        return 'I'
    if opcode in B_TYPE:
        return 'B'
    if opcode in LOAD_TYPE:
        return 'LOAD'
    if opcode in STORE_TYPE:
        return 'STORE'
    if opcode in ('LUI', 'AUIPC'):
        return 'LUI_AUIPC'
    if opcode == 'JAL':
        return 'JAL'
    if opcode == 'JALR':
        return 'JALR'
    raise ValueError(f'unknown opcode: {opcode}')


def random_perm(rng):
    """Sample a uniformly random register permutation of x1..x31, with
    x0 fixed (it's the constant zero, not a regular register).

    Returns int32 ndarray of length 32 with `perm[0] == 0` and
    `perm[1:]` a permutation of [1..31].
    """
    nonzero = np.arange(1, 32, dtype=np.int32)
    rng.shuffle(nonzero)
    perm = np.zeros(32, dtype=np.int32)
    perm[1:] = nonzero
    return perm


def relabel(instructions, perm):
    """Apply register permutation `perm` to every register reference
    in `instructions`. Returns a new list; input not mutated.

    perm must satisfy `perm[0] == 0`. Caller is responsible for
    ensuring perm is a bijection on x1..x31 for equivalence to hold
    (use `random_perm` for a valid sample).
    """
    if int(perm[0]) != 0:
        raise ValueError('perm[0] must equal 0 (x0 is fixed under relabeling)')
    out = []
    for instr in instructions:
        slots = _REG_SLOTS[_classify_op(instr.opcode)]
        new_args = list(instr.args)
        for s in slots:
            new_args[s] = int(perm[new_args[s]])
        out.append(Instruction(instr.opcode, *new_args))
    return out


# ===========================================================================
# Equivalence manifest — curated cross-syntax equivalence classes
# ===========================================================================

@dataclass
class InstructionTemplate:
    """A pattern with free variables (string names) and literals (ints)."""
    op: str
    args: tuple


@dataclass
class EquivalenceClass:
    """A set of instruction templates that are claimed to be
    semantically equivalent under any consistent variable binding."""
    name: str
    description: str
    canonical: list                # list[InstructionTemplate]
    contrast: list = field(default_factory=list)
    constraints: list = field(default_factory=list)


def _sample_value(name, rng):
    if name.startswith('imm'):
        return int(rng.integers(-2048, 2048))
    if name == 'shamt':
        return int(rng.integers(0, 32))
    return int(rng.integers(1, 32))  # register, excluding x0


def _free_vars(klass):
    names = set()
    for tpl in list(klass.canonical) + list(klass.contrast):
        for arg in tpl.args:
            if isinstance(arg, str):
                names.add(arg)
    return names


def sample_binding(klass, rng, max_tries=100):
    """Draw variable values satisfying the class's constraints."""
    names = _free_vars(klass)
    for _ in range(max_tries):
        binding = {name: _sample_value(name, rng) for name in names}
        if all(c(binding) for c in klass.constraints):
            return binding
    raise RuntimeError(
        f'Could not satisfy constraints for {klass.name} in {max_tries} tries')


def materialize(template, binding):
    """Turn a template + binding into a concrete Instruction."""
    args = tuple(binding[a] if isinstance(a, str) else a for a in template.args)
    return Instruction(template.op, *args)


def _t(op, *args):
    return InstructionTemplate(op, args)







MANIFEST = [
    # Operand-order commutativity.
    EquivalenceClass(
        name='commutative_add',
        description='ADD operand swap.',
        canonical=[_t('ADD', 'rd', 'rs1', 'rs2'), _t('ADD', 'rd', 'rs2', 'rs1')],
        contrast=[_t('SUB', 'rd', 'rs1', 'rs2')],
        constraints=[lambda b: b['rs1'] != b['rs2']],
    ),
    EquivalenceClass(
        name='commutative_and',
        description='AND operand swap.',
        canonical=[_t('AND', 'rd', 'rs1', 'rs2'), _t('AND', 'rd', 'rs2', 'rs1')],
        contrast=[_t('SUB', 'rd', 'rs1', 'rs2')],
        constraints=[lambda b: b['rs1'] != b['rs2']],
    ),
    EquivalenceClass(
        name='commutative_or',
        description='OR operand swap.',
        canonical=[_t('OR', 'rd', 'rs1', 'rs2'), _t('OR', 'rd', 'rs2', 'rs1')],
        contrast=[_t('SUB', 'rd', 'rs1', 'rs2')],
        constraints=[lambda b: b['rs1'] != b['rs2']],
    ),
    EquivalenceClass(
        name='commutative_xor',
        description='XOR operand swap.',
        canonical=[_t('XOR', 'rd', 'rs1', 'rs2'), _t('XOR', 'rd', 'rs2', 'rs1')],
        contrast=[_t('SUB', 'rd', 'rs1', 'rs2')],
        constraints=[lambda b: b['rs1'] != b['rs2']],
    ),

    # Self-operand identities.
    EquivalenceClass(
        name='self_op_zero',
        description='Self-op XOR/SUB/SLT/SLTU all compute 0.',
        canonical=[
            _t('XOR',  'rd', 'rs', 'rs'),
            _t('SUB',  'rd', 'rs', 'rs'),
            _t('SLT',  'rd', 'rs', 'rs'),
            _t('SLTU', 'rd', 'rs', 'rs'),
        ],
        contrast=[_t('XOR', 'rd', 'rs1', 'rs2')],
        constraints=[lambda b: b['rs1'] != b['rs2']],
    ),
    EquivalenceClass(
        name='self_op_identity',
        description='AND y,y ≡ OR y,y (both preserve rs).',
        canonical=[_t('AND', 'rd', 'rs', 'rs'), _t('OR', 'rd', 'rs', 'rs')],
        contrast=[_t('XOR', 'rd', 'rs', 'rs')],
    ),

    # Move via several syntactic routes.
    EquivalenceClass(
        name='mv_via_zero_reg',
        description='Move x[rs] to x[rd] via many routes.',
        canonical=[
            _t('ADDI', 'rd', 'rs',  0),
            _t('ADD',  'rd',  0,   'rs'),
            _t('OR',   'rd',  0,   'rs'),
            _t('XOR',  'rd',  0,   'rs'),
            _t('ORI',  'rd', 'rs',  0),
            _t('XORI', 'rd', 'rs',  0),
            _t('ANDI', 'rd', 'rs', -1),
            _t('SUB',  'rd', 'rs',  0),
            _t('SLLI', 'rd', 'rs',  0),
            _t('SRLI', 'rd', 'rs',  0),
            _t('SRAI', 'rd', 'rs',  0),
            _t('ADD',  'rd', 'rs',  0),
            _t('OR',   'rd', 'rs',  0),
            _t('XOR',  'rd', 'rs',  0),
            _t('SLL',  'rd', 'rs',  0),
            _t('SRL',  'rd', 'rs',  0),
            _t('SRA',  'rd', 'rs',  0),
        ],
        contrast=[_t('ADDI', 'rd', 'rs', 1)],
    ),

    # Compound idioms.
    EquivalenceClass(
        name='neg_via_sub_x0',
        description='SUB rd,x0,rs = -rs.',
        canonical=[_t('SUB', 'rd', 0, 'rs')],
        contrast=[_t('ADD', 'rd', 0, 'rs')],
    ),
    EquivalenceClass(
        name='not_via_xori_neg1',
        description='XORI rd,rs,-1 = ~rs.',
        canonical=[_t('XORI', 'rd', 'rs', -1)],
        contrast=[_t('XORI', 'rd', 'rs', 0)],
    ),
    EquivalenceClass(
        name='double_is_shl1',
        description='ADD rd,rs,rs ≡ SLLI rd,rs,1 (both compute 2·rs).',
        canonical=[_t('ADD', 'rd', 'rs', 'rs'), _t('SLLI', 'rd', 'rs', 1)],
        contrast=[_t('SLLI', 'rd', 'rs', 2)],
    ),

    # Write zero to rd (many syntactic routes).
    EquivalenceClass(
        name='write_zero_to_rd',
        description='Instructions that always write 0 to rd.',
        canonical=[
            _t('ADDI', 'rd',  0,    0),
            _t('XORI', 'rd',  0,    0),
            _t('ORI',  'rd',  0,    0),
            _t('AND',  'rd',  0,   'rs'),
            _t('ANDI', 'rd', 'rs',  0),
            _t('SLLI', 'rd',  0,   'shamt'),
            _t('SRLI', 'rd',  0,   'shamt'),
            _t('SRAI', 'rd',  0,   'shamt'),
            _t('XOR',  'rd', 'rs', 'rs'),
            _t('SUB',  'rd', 'rs', 'rs'),
            _t('SLT',  'rd', 'rs', 'rs'),
            _t('SLTU', 'rd', 'rs', 'rs'),
            _t('LUI',  'rd',  0),
            _t('ADD',  'rd',  0,    0),
            _t('AND',  'rd', 'rs',  0),
            _t('SLL',  'rd',  0,   'rs'),
            _t('SRL',  'rd',  0,   'rs'),
            _t('SRA',  'rd',  0,   'rs'),
            _t('SLTU', 'rd', 'rs',  0),
            _t('SLTIU', 'rd', 'rs', 0),
            _t('ANDI', 'rd',  0,   'imm'),
        ],
        contrast=[_t('ADDI', 'rd', 0, 1)],
    ),

    # Write-to-x0 NOPs.
    EquivalenceClass(
        name='x0_write_nop',
        description='Any register-writing instruction with rd=x0.',
        canonical=[
            _t('ADD',   0, 'rs1', 'rs2'),
            _t('SUB',   0, 'rs1', 'rs2'),
            _t('AND',   0, 'rs1', 'rs2'),
            _t('OR',    0, 'rs1', 'rs2'),
            _t('XOR',   0, 'rs1', 'rs2'),
            _t('SLL',   0, 'rs1', 'rs2'),
            _t('SRL',   0, 'rs1', 'rs2'),
            _t('SRA',   0, 'rs1', 'rs2'),
            _t('SLT',   0, 'rs1', 'rs2'),
            _t('SLTU',  0, 'rs1', 'rs2'),
            _t('ADDI',  0, 'rs1', 'imm'),
            _t('ANDI',  0, 'rs1', 'imm'),
            _t('ORI',   0, 'rs1', 'imm'),
            _t('XORI',  0, 'rs1', 'imm'),
            _t('SLTI',  0, 'rs1', 'imm'),
            _t('SLTIU', 0, 'rs1', 'imm'),
            _t('SLLI',  0, 'rs1', 'shamt'),
            _t('SRLI',  0, 'rs1', 'shamt'),
            _t('SRAI',  0, 'rs1', 'shamt'),
            _t('LUI',   0, 'imm'),
            _t('AUIPC', 0, 'imm'),
            _t('LB',    0, 'imm', 'rs1'),
            _t('LH',    0, 'imm', 'rs1'),
            _t('LW',    0, 'imm', 'rs1'),
            _t('LBU',   0, 'imm', 'rs1'),
            _t('LHU',   0, 'imm', 'rs1'),
        ],
        contrast=[_t('ADD', 'rd', 'rs1', 'rs2')],
    ),

    # Branch always/never taken (rs1=rs2 forces the condition).
    EquivalenceClass(
        name='branch_always_taken',
        description='Unconditional forward jump.',
        canonical=[
            _t('BEQ',  'rs', 'rs', 'imm'),
            _t('BGE',  'rs', 'rs', 'imm'),
            _t('BGEU', 'rs', 'rs', 'imm'),
            _t('JAL',   0,   'imm'),
        ],
        contrast=[_t('BNE', 'rs', 'rs', 'imm')],
    ),
    EquivalenceClass(
        name='branch_never_taken',
        description='BNE/BLT/BLTU with rs1=rs2 never take.',
        canonical=[
            _t('BNE',  'rs', 'rs', 'imm'),
            _t('BLT',  'rs', 'rs', 'imm'),
            _t('BLTU', 'rs', 'rs', 'imm'),
        ],
        contrast=[_t('BEQ', 'rs', 'rs', 'imm')],
    ),

    # Sign test.
    EquivalenceClass(
        name='sign_test',
        description='SLTI rd,rs,0 ≡ SLT rd,rs,x0 ≡ SRLI rd,rs,31 (rs < 0).',
        canonical=[_t('SLTI', 'rd', 'rs', 0), _t('SLT', 'rd', 'rs', 0),
                   _t('SRLI', 'rd', 'rs', 31)],
        contrast=[_t('SLTI', 'rd', 'rs', 1)],
    ),

    # Load a sign-extended 12-bit constant.
    EquivalenceClass(
        name='load_constant',
        description='ADDI/ORI/XORI rd,x0,imm all write sext(imm).',
        canonical=[_t('ADDI', 'rd', 0, 'imm'), _t('ORI', 'rd', 0, 'imm'),
                   _t('XORI', 'rd', 0, 'imm')],
        contrast=[_t('ADDI', 'rd', 0, 'imm2')],
        constraints=[lambda b: b['imm'] != b['imm2']],
    ),
]


# ===========================================================================
# Partner groups — the manifest's classes merged by the function computed
# ===========================================================================

# Each group is one register function: every canonical template of every
# listed class computes it under any consistent binding, so any two make an
# equivalent pair. Classes overlap (`self_op_zero` is part of writing zero),
# which is why partners are drawn by group and not by class. Opcodes the
# writers corpora never draw are left out.
_PARTNER_GROUP_CLASSES = {
    'copy': ('mv_via_zero_reg', 'self_op_identity'),
    'zero': ('write_zero_to_rd', 'self_op_zero'),
    'double': ('double_is_shl1',),
    'sign': ('sign_test',),
    'constant': ('load_constant',),
}
_PARTNER_EXCLUDED_OPS = frozenset({'LUI'})


def _partner_group(name):
    by_name = {k.name: k for k in MANIFEST}
    classes = [by_name[c] for c in _PARTNER_GROUP_CLASSES[name]]
    canon, seen = [], set()
    for k in classes:
        for tpl in k.canonical:
            key = (tpl.op, tpl.args)
            if tpl.op not in _PARTNER_EXCLUDED_OPS and key not in seen:
                seen.add(key)
                canon.append(tpl)
    return EquivalenceClass(
        name=name, description=f'partner group {name}', canonical=canon,
        contrast=[t for k in classes for t in k.contrast],
        constraints=[c for k in classes for c in k.constraints])


PARTNER_GROUPS = [_partner_group(n) for n in _PARTNER_GROUP_CLASSES]


def sample_partner_pair(rng, dest_regs=None):
    """Two different spellings of one register function, bound to the same
    registers and literals: (Instruction, Instruction). `dest_regs` limits rd
    as the corpus's own draw does."""
    group = PARTNER_GROUPS[int(rng.integers(0, len(PARTNER_GROUPS)))]
    i, j = rng.choice(len(group.canonical), size=2, replace=False)
    binding = sample_binding(group, rng)
    if dest_regs is not None:
        binding['rd'] = int(rng.choice(list(dest_regs)))
    return (materialize(group.canonical[int(i)], binding),
            materialize(group.canonical[int(j)], binding))


def _imm12(v):
    return -2048 <= v <= 2047


def _split(total, rng):
    """Two 12-bit literals summing to `total` (a 12-bit literal)."""
    while True:
        a = int(rng.integers(-2048, 2048))
        if _imm12(total - a):
            return a, total - a


_SUFFIX_OPS = _ALU_R_OPS + _ALU_I_OPS

_REWRITES = ('swap', 'respell', 'fold', 'cancel', 'reassociate', 'identity')

# Same-value writes of register t: each leaves t holding what it held.
_IDENTITIES = (('ADDI', 'T', 'T', 0), ('ORI', 'T', 'T', 0), ('XORI', 'T', 'T', 0),
               ('ANDI', 'T', 'T', -1), ('SLLI', 'T', 'T', 0), ('SRLI', 'T', 'T', 0),
               ('SRAI', 'T', 'T', 0), ('ADD', 'T', 'T', 0), ('OR', 'T', 'T', 0),
               ('XOR', 'T', 'T', 0), ('SUB', 'T', 'T', 0), ('AND', 'T', 'T', 'T'),
               ('OR', 'T', 'T', 'T'))


def _identity(rng, t):
    op, *args = _IDENTITIES[int(rng.integers(len(_IDENTITIES)))]
    return Instruction(op, *(t if a == 'T' else a for a in args))


def _rewrite_pair(rng, max_len, pool, rewrites):
    """A chunk of random ALU context holding one equivalence-preserving
    rewrite, against the same chunk rewritten. The rewrite's operands are
    registers already in the chunk half the time — another instruction's
    input or result — so every operand context is drawn; context before and
    after it pads the chunk to a random length up to max_len."""
    name = str(rewrites[int(rng.integers(len(rewrites)))])
    if name == 'identity' and (max_len == 1 or rng.random() < 0.3):
        t = int(rng.choice(pool))
        while True:
            x, y = _identity(rng, t), _identity(rng, t)
            if x.opcode != y.opcode or x.args != y.args:
                return [x], [y]
    seen = []
    pick = lambda: (int(rng.choice(seen)) if seen and rng.random() < 0.5
                    else int(rng.integers(1, 32)))

    def context():
        op = str(rng.choice(_SUFFIX_OPS))
        ins = _make_instruction(rng, op, rd=int(rng.choice(pool)), rs1=pick(),
                                rs2=pick() if op in _ALU_R_OPS else None)
        seen.extend(r for r in (ins.args[:3] if op in _ALU_R_OPS else ins.args[:2]) if r)
        return ins

    size = {'swap': 1, 'respell': 1, 'identity': 1}.get(name, 2)
    L = int(rng.integers(size + (name == 'identity'), max_len + 1))
    before = int(rng.integers(0, L - size + 1))
    prefix = [context() for _ in range(before)]
    t = int(rng.choice(pool))
    while True:
        a, b, c = pick(), pick(), pick()
        if len({a, b, c, t}) == 4:
            break
        seen.append(int(rng.integers(1, 32)))
    if name == 'swap':
        op = str(rng.choice(['ADD', 'XOR', 'AND', 'OR']))
        x, y = [Instruction(op, t, a, b)], [Instruction(op, t, b, a)]
    elif name == 'identity':
        x, y = [_identity(rng, pick())], []
    elif name == 'respell':
        group = PARTNER_GROUPS[int(rng.integers(0, len(PARTNER_GROUPS)))]
        i, j = rng.choice(len(group.canonical), size=2, replace=False)
        binding = sample_binding(group, rng)
        binding['rd'] = t
        if 'rs' in binding:
            binding['rs'] = a
        x = [materialize(group.canonical[int(i)], binding)]
        y = [materialize(group.canonical[int(j)], binding)]
    elif name == 'fold':
        if rng.random() < 0.5:
            k1, k2 = _split(int(rng.integers(-2048, 2048)), rng)
            k3, k4 = _split(k1 + k2, rng)
            op = 'ADDI'
        else:
            k1, k2, k3 = (int(v) for v in rng.integers(-2048, 2048, 3))
            k4 = ((k1 ^ k2 ^ k3) & 0xFFF) - (((k1 ^ k2 ^ k3) & 0x800) << 1)
            op = 'XORI'
        x = [Instruction(op, t, a, k1), Instruction(op, t, t, k2)]
        y = [Instruction(op, t, a, k3), Instruction(op, t, t, k4)]
    elif name == 'cancel':
        k = int(rng.integers(-2047, 2048))
        op = 'ADDI' if rng.random() < 0.5 else 'XORI'
        x = [Instruction(op, t, a, k), Instruction(op, t, t, -k if op == 'ADDI' else k)]
        y = [Instruction('ADDI', t, a, 0)]
    else:
        op = str(rng.choice(['ADD', 'XOR', 'AND', 'OR', 'SUB']))
        if op == 'SUB':
            x = [Instruction('SUB', t, a, b), Instruction('SUB', t, t, c)]
            y = [Instruction('ADD', t, b, c), Instruction('SUB', t, a, t)]
        else:
            x = [Instruction(op, t, a, b), Instruction(op, t, t, c)]
            y = [Instruction(op, t, b, c), Instruction(op, t, t, a)]
    seen.extend([t, a, b, c])
    suffix = [context() for _ in range(L - before - size)]
    return prefix + x + suffix, prefix + y + suffix


def sample_chunk_partner_pair(rng, max_len, dest_regs=None, rewrites=None, mined=None):
    """Two different spellings of one register function, each a chunk of at
    most `max_len` instructions: (list[Instruction], list[Instruction]). At
    max_len 1 it is `sample_partner_pair`; otherwise, with probability
    mined['fraction'], a pair found by search and proved
    (`datagen.mine.sample_pooled_pair`),
    else one rewrite of `rewrites` (default every one of `_REWRITES`)
    embedded in random context (`_rewrite_pair`)."""
    if max_len == 1:
        x, y = sample_partner_pair(rng, dest_regs)
        return [x], [y]
    if mined is not None and max_len >= 2 and rng.random() < mined.get('fraction', 1.0):
        from .mine import sample_pooled_pair
        return sample_pooled_pair(rng, frozenset(mined.get('exclude', ())),
                                  int(mined.get('reuse', 1)))
    pool = list(dest_regs) if dest_regs is not None else list(range(1, 32))
    return _rewrite_pair(rng, max_len, pool, rewrites or _REWRITES)


# ===========================================================================
# Termination rules + group collection
# ===========================================================================

# Opcodes that end a "register-state-transformation block" (T2 chunk):
# memory access, branches, and jumps. Anything else is a pure ALU op
# that can extend the current chunk.
_TRANSFORMATION_TERMINATORS = LOAD_TYPE | STORE_TYPE | B_TYPE | {'JAL', 'JALR'}


class TerminationRule:
    """Predicate over (group_so_far, last_instr) → should we end this
    group after `last_instr`? Compose with `|` (which calls `either`)."""

    def should_terminate(self, group, last_instr):
        raise NotImplementedError

    def __or__(self, other):
        return either(self, other)

    @property
    def max_len(self):
        """Finite upper bound on group length this rule guarantees, or
        None if it imposes no bound (a pure terminator that may never
        fire on an arbitrary instruction stream). collect_groups refuses
        to run a rule whose max_len is None."""
        return None


class _Single(TerminationRule):
    def should_terminate(self, group, last_instr):
        return True

    @property
    def max_len(self):
        return 1


class _UntilBranch(TerminationRule):
    def should_terminate(self, group, last_instr):
        return last_instr.opcode in B_TYPE


class _UntilTransformation(TerminationRule):
    def should_terminate(self, group, last_instr):
        return last_instr.opcode in _TRANSFORMATION_TERMINATORS


class _LengthCap(TerminationRule):
    def __init__(self, n):
        if n < 1:
            raise ValueError('length_cap requires n >= 1')
        self.n = n

    def should_terminate(self, group, last_instr):
        return len(group) >= self.n

    @property
    def max_len(self):
        return self.n


class _Either(TerminationRule):
    def __init__(self, *rules):
        self.rules = rules

    def should_terminate(self, group, last_instr):
        return any(r.should_terminate(group, last_instr) for r in self.rules)

    @property
    def max_len(self):
        # Group terminates when ANY member fires, so the ceiling is the
        # tightest finite bound among members. None (unbounded) iff no
        # member imposes a bound — i.e. the rule set contains no cap.
        bounds = [r.max_len for r in self.rules if r.max_len is not None]
        return min(bounds) if bounds else None


def single():
    """Every instruction is its own group."""
    return _Single()


def until_branch():
    """Terminate when the last instruction is a branch (basic-block
    boundary)."""
    return _UntilBranch()


def until_transformation():
    """Terminate when the last instruction is a memory access, branch,
    or jump (T2 chunk boundary)."""
    return _UntilTransformation()


def length_cap(n):
    """Force termination when the group reaches `n` instructions."""
    return _LengthCap(n)


def either(*rules):
    """Terminate when ANY of the given rules fires."""
    return _Either(*rules)


def chain_group(group, rng):
    """Rewrite a group so each instruction reads its predecessor's result.

    Dataflow composition is what a chunk's register draw leaves to chance: with
    sources drawn uniformly, only ~4.7% of two-instruction chunks have the
    second reading the first's destination, and the sub-populations that
    combine composition with an immediate are rarer still — the cells that
    carry most of T2's value-prediction error (EXPERIMENT_LOG, "T2
    faithfulness"). This makes composition explicit so that population can be
    supervised at a usable density.

    Only ONE source operand of each later instruction is rewritten, to its
    predecessor's destination register. The opcode of every instruction is
    untouched, so the opcode distribution the config asks for is preserved and
    the only thing that changes is which register an operand reads. An
    instruction writing x0 is skipped as a producer: its result is discarded,
    so reading it composes nothing.
    """
    if len(group) < 2:
        return group
    out = [group[0]]
    for instr in group[1:]:
        prev_rd = out[-1].args[0] if out[-1].opcode not in _NO_DEST_OPS else 0
        if prev_rd == 0 or len(instr.args) < 2:
            out.append(instr)
            continue
        args = list(instr.args)
        # which positions are source REGISTERS for this opcode
        src_pos = _src_arg_positions(instr.opcode)
        if not src_pos:
            out.append(instr)
            continue
        args[src_pos[int(rng.integers(0, len(src_pos)))]] = prev_rd
        out.append(Instruction(instr.opcode, *args))
    return out


def _src_arg_positions(opcode):
    """Indices into Instruction.args that hold a SOURCE register, by opcode
    type. Mirrors compressor.model.instruction_wiring's argument orders."""
    if opcode in R_TYPE:
        return (1, 2)
    if opcode in I_TYPE or opcode == 'JALR':
        return (1,)
    if opcode in B_TYPE:
        return (0, 1)
    if opcode in LOAD_TYPE:
        return (2,)
    if opcode in STORE_TYPE:
        return (0, 2)
    return ()


_NO_DEST_OPS = frozenset(B_TYPE) | frozenset(STORE_TYPE)


def collect_groups(instr_iter, rule):
    """Walk `instr_iter` (iterable of Instruction), yielding groups
    (lists of Instruction) per the termination rule. The instruction
    that triggers termination IS included in its group. Any leftover
    instructions at the end become a final tail group.

    `instr_iter` is typically an infinite random stream, so `rule` MUST
    impose a finite length bound or a group could grow forever. We refuse
    a rule whose `rule.max_len` is None — for an `either(...)` composition
    that is exactly the set-membership check that the rule set contains a
    cap (length_cap)."""
    if rule.max_len is None:
        raise ValueError(
            'Termination rule is unbounded (no length cap); it can build '
            'an infinite group from the instruction stream. Compose it '
            'with length_cap(N).')
    group = []
    for instr in instr_iter:
        group.append(instr)
        if rule.should_terminate(group, instr):
            yield group
            group = []
    if group:
        yield group
