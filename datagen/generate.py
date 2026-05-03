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
| length_cap(n)`), and any future grouping rule. Basic-block generation
(seqgen-style: N-1 non-branch instructions followed by a forced branch)
is the one shape that doesn't fit the predicate model cleanly — it has
its own dedicated `random_basic_block` generator.
"""

import json
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


def _make_instruction(rng, opcode, rd=None, rs1=None, rs2=None, imm=None):
    """Build an instruction with the given opcode, filling unspecified
    operands with random valid values. Single point of truth for
    instruction construction — all generators below call this.
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


def _instr_dest_reg(instr):
    """Destination register if instr writes one, else None."""
    if instr.opcode in STORE_TYPE or instr.opcode in B_TYPE:
        return None
    return instr.args[0]


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

_CONFIG_EXTRA_KEYS = {'equivalences', 'invalidity'}
_EQUIVALENCES_KEYS = {'rate', 'max_per_class', 'min_per_class', 'boost'}
_INVALIDITY_KEYS = {'rate', 'types'}


def validate_distribution(dist):
    """Validate a config dict.

    Required: opcode-category keys with weights summing to 1.0.
    Optional: 'equivalences' sub-dict for manifest-injection settings.
    Optional: 'invalidity' sub-dict {'rate': float, 'types': {name: w}}
              for invalid-window injection.
    """
    keys = set(dist.keys())
    unknown = keys - set(_CATEGORY_OPS.keys()) - _CONFIG_EXTRA_KEYS
    if unknown:
        raise ValueError(f'Unknown config keys: {unknown}')
    weights = {k: v for k, v in dist.items() if k in _CATEGORY_OPS}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f'Distribution weights sum to {total}, not 1.0')
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


def _build_opcode_table(dist):
    """Convert opcode-distribution portion of a config dict to the
    internal (weight, ops) list. Ignores non-opcode keys."""
    return [(dist[cat], _CATEGORY_OPS[cat])
            for cat in dist if cat in _CATEGORY_OPS]


# Validate and build default table at import time.
validate_distribution(DEFAULT_DISTRIBUTION)
_OPCODE_DISTRIBUTION = _build_opcode_table(DEFAULT_DISTRIBUTION)


# ---------------------------------------------------------------------------
# Random instruction generators
# ---------------------------------------------------------------------------

def random_instruction(rng, opcode_table=None):
    """Generate a single random RV32I instruction.

    opcode_table: optional pre-built (weight, ops) list from
        `_build_opcode_table`. Uses the default distribution if None.
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


def _data_flow_instruction(rng, live_regs, force_branch=False):
    """Generate an instruction biased toward reading from `live_regs`
    (creating internal data flow). Used by `random_basic_block`.

    Type distribution is hardcoded (40/35/10/10/5 across R-ALU /
    I-ALU / LOAD / STORE / LUI). Never produces branches except via
    force_branch=True.
    """
    live_list = sorted(live_regs)

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
        return _make_instruction(rng, op, rs1=int(rs1), rs2=int(rs2))

    use_live = len(live_list) > 0 and rng.random() < 0.6

    def _pick_src():
        return int(rng.choice(live_list)) if use_live else int(rng.choice(_SRC_REGS))

    roll = rng.random()
    if roll < 0.4:
        return _make_instruction(rng, str(rng.choice(_ALU_R_OPS)), rs1=_pick_src())
    elif roll < 0.75:
        return _make_instruction(rng, str(rng.choice(_ALU_I_OPS)), rs1=_pick_src())
    elif roll < 0.85:
        return _make_instruction(rng, str(rng.choice(_LOAD_OPS)), rs1=_pick_src())
    elif roll < 0.95:
        return _make_instruction(rng, str(rng.choice(_STORE_OPS)), rs2=_pick_src())
    else:
        return _make_instruction(rng, 'LUI')


def random_basic_block(rng, max_length=10):
    """Generate a basic block: N-1 ALU/load/store/upper instructions
    followed by a forced branch. Length drawn uniformly from [2, N].
    Each instruction has high probability of reading from a register
    written by a previous instruction in the block.

    Distinct from `collect_groups(rule=...)` because the "force a
    branch as the last instruction" pattern doesn't fit the
    termination-predicate model — branches don't appear in the
    natural instruction stream from `_data_flow_instruction`, and
    they get appended explicitly at the end.
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
    instructions.append(_data_flow_instruction(rng, live_regs, force_branch=True))
    return instructions


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


def random_relabel(instructions, rng):
    """Convenience: relabel under a fresh random permutation."""
    return relabel(instructions, random_perm(rng))


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
    if name == 'imm':
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


def _select_templates(klass, max_per_class, rng):
    """Pick canonical templates from a class, capping at max_per_class."""
    canon = klass.canonical
    if len(canon) > max_per_class:
        indices = rng.choice(len(canon), size=max_per_class, replace=False)
        return [canon[int(i)] for i in indices]
    return list(canon)


def sample_injection_tuples(target_count, max_per_class, rng,
                            min_per_class=0, boost=None):
    """Sample canonical tuples from MANIFEST. Returns a flat list of
    materialized Instructions, always emitting whole tuples.

    Three passes: guaranteed (every class with >=2 templates gets at
    least min_per_class tuples), boost (classes named in `boost` get
    extra), random fill (until target_count instructions reached).
    """
    if target_count <= 0 or not MANIFEST:
        return []

    out = []

    if min_per_class > 0:
        for klass in MANIFEST:
            templates = _select_templates(klass, max_per_class, rng)
            if len(templates) < 2:
                continue
            for _ in range(min_per_class):
                binding = sample_binding(klass, rng)
                for tpl in _select_templates(klass, max_per_class, rng):
                    out.append(materialize(tpl, binding))

    if boost:
        by_name = {k.name: k for k in MANIFEST}
        for class_name, extra in boost.items():
            klass = by_name.get(class_name)
            if klass is None:
                continue
            templates = _select_templates(klass, max_per_class, rng)
            if len(templates) < 2:
                continue
            for _ in range(int(extra)):
                binding = sample_binding(klass, rng)
                for tpl in _select_templates(klass, max_per_class, rng):
                    out.append(materialize(tpl, binding))

    while len(out) < target_count:
        klass = MANIFEST[int(rng.integers(0, len(MANIFEST)))]
        templates = _select_templates(klass, max_per_class, rng)
        if len(templates) < 2:
            continue
        binding = sample_binding(klass, rng)
        for tpl in templates:
            out.append(materialize(tpl, binding))

    return out


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
        description='SLTI rd,rs,0 ≡ SLT rd,rs,x0 (both test rs < 0).',
        canonical=[_t('SLTI', 'rd', 'rs', 0), _t('SLT', 'rd', 'rs', 0)],
        contrast=[_t('SLTI', 'rd', 'rs', 1)],
    ),
]


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


class _Single(TerminationRule):
    def should_terminate(self, group, last_instr):
        return True


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


class _Either(TerminationRule):
    def __init__(self, *rules):
        self.rules = rules

    def should_terminate(self, group, last_instr):
        return any(r.should_terminate(group, last_instr) for r in self.rules)


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


def collect_groups(instr_iter, rule):
    """Walk `instr_iter` (iterable of Instruction), yielding groups
    (lists of Instruction) per the termination rule. The instruction
    that triggers termination IS included in its group. Any leftover
    instructions at the end become a final tail group."""
    group = []
    for instr in instr_iter:
        group.append(instr)
        if rule.should_terminate(group, instr):
            yield group
            group = []
    if group:
        yield group
