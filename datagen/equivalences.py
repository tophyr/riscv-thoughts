"""Equivalence manifest for T1 encoder training and evaluation.

Each EquivalenceClass is a *rule* parameterized by free variables
(register names like 'rd'/'rs1'/'rs2', or 'imm'/'shamt' for
immediates). Literal `0` in an argument position means the literal
register x0 (or literal immediate 0, depending on position). The
sampler instantiates concrete Instructions by drawing random values
for each free variable, subject to the class's constraints.

Usage:
    rng = np.random.default_rng(42)
    for klass in MANIFEST:
        binding = sample_binding(klass, rng)
        canonical_instrs = [materialize(tpl, binding) for tpl in klass.canonical]
        contrast_instrs = [materialize(tpl, binding) for tpl in klass.contrast]
        # ... encode, compute cohesion/separation

Variable naming convention:
    rd, rs, rs1, rs2, ra — register, sampled from {1..31}
    imm                 — immediate, sampled from [-2048, 2048)
    shamt               — shift amount, sampled from [0, 32)
    (int literal)       — used as-is (e.g., 0 means x0 in a reg slot)
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from emulator import Instruction


@dataclass
class InstructionTemplate:
    op: str
    args: tuple  # each element: str (free var name) or int (literal)


@dataclass
class EquivalenceClass:
    name: str
    description: str
    canonical: list           # list[InstructionTemplate]; should all collapse
    contrast: list = field(default_factory=list)  # must stay apart from canonical
    constraints: list = field(default_factory=list)  # list[Callable[[dict], bool]]


def _sample_value(name: str, rng: np.random.Generator) -> int:
    if name == 'imm':
        return int(rng.integers(-2048, 2048))
    if name == 'shamt':
        return int(rng.integers(0, 32))
    return int(rng.integers(1, 32))  # register, excluding x0


def _free_vars(klass: EquivalenceClass) -> set:
    names = set()
    for tpl in list(klass.canonical) + list(klass.contrast):
        for arg in tpl.args:
            if isinstance(arg, str):
                names.add(arg)
    return names


def sample_binding(klass: EquivalenceClass,
                   rng: np.random.Generator,
                   max_tries: int = 100) -> dict:
    """Draw variable values satisfying the class's constraints."""
    names = _free_vars(klass)
    for _ in range(max_tries):
        binding = {name: _sample_value(name, rng) for name in names}
        if all(c(binding) for c in klass.constraints):
            return binding
    raise RuntimeError(
        f'Could not satisfy constraints for {klass.name} in {max_tries} tries')


def materialize(template: InstructionTemplate, binding: dict) -> Instruction:
    """Turn a template + binding into a concrete Instruction."""
    args = tuple(binding[a] if isinstance(a, str) else a for a in template.args)
    return Instruction(template.op, *args)


# Shorthand for templates.
def _t(op, *args):
    return InstructionTemplate(op, args)


def _select_templates(klass, max_per_class, rng):
    """Pick canonical templates from a class, capping at max_per_class."""
    canon = klass.canonical
    if len(canon) > max_per_class:
        indices = rng.choice(len(canon),
                             size=max_per_class, replace=False)
        return [canon[int(i)] for i in indices]
    return list(canon)


def sample_injection_tuples(target_count: int,
                            max_per_class: int,
                            rng: np.random.Generator,
                            min_per_class: int = 0,
                            boost: dict | None = None) -> list:
    """Sample canonical tuples from MANIFEST. Returns a flat list of
    materialized Instructions, always emitting whole tuples.

    Three passes:
    1. Guaranteed: every class with >=2 canonical templates gets
       at least min_per_class tuples (each from a fresh binding).
    2. Boost: classes named in the boost dict get extra tuples
       (e.g., {"double_is_shl1": 60} adds 60 extra tuples for
       that class, to overcome opcode-cluster pressure).
    3. Random fill: picks random classes until total instruction
       count >= target_count.

    Classes with <2 templates after max_per_class subsampling are
    skipped (no co-occurrence value from a single instruction).
    """
    if target_count <= 0 or not MANIFEST:
        return []

    out: list = []

    # Guaranteed pass.
    if min_per_class > 0:
        for klass in MANIFEST:
            templates = _select_templates(klass, max_per_class, rng)
            if len(templates) < 2:
                continue
            for _ in range(min_per_class):
                binding = sample_binding(klass, rng)
                for tpl in _select_templates(klass, max_per_class, rng):
                    out.append(materialize(tpl, binding))

    # Boost pass.
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

    # Random fill.
    while len(out) < target_count:
        klass = MANIFEST[int(rng.integers(0, len(MANIFEST)))]
        templates = _select_templates(klass, max_per_class, rng)
        if len(templates) < 2:
            continue
        binding = sample_binding(klass, rng)
        for tpl in templates:
            out.append(materialize(tpl, binding))

    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST: list = [

    # -----------------------------------------------------------------
    # Operand-order commutativity.
    # -----------------------------------------------------------------

    EquivalenceClass(
        name='commutative_add',
        description='ADD operand swap produces identical state delta.',
        canonical=[
            _t('ADD', 'rd', 'rs1', 'rs2'),
            _t('ADD', 'rd', 'rs2', 'rs1'),
        ],
        contrast=[_t('SUB', 'rd', 'rs1', 'rs2')],
        constraints=[lambda b: b['rs1'] != b['rs2']],
    ),
    EquivalenceClass(
        name='commutative_and',
        description='AND operand swap produces identical state delta.',
        canonical=[
            _t('AND', 'rd', 'rs1', 'rs2'),
            _t('AND', 'rd', 'rs2', 'rs1'),
        ],
        contrast=[_t('SUB', 'rd', 'rs1', 'rs2')],
        constraints=[lambda b: b['rs1'] != b['rs2']],
    ),
    EquivalenceClass(
        name='commutative_or',
        description='OR operand swap produces identical state delta.',
        canonical=[
            _t('OR', 'rd', 'rs1', 'rs2'),
            _t('OR', 'rd', 'rs2', 'rs1'),
        ],
        contrast=[_t('SUB', 'rd', 'rs1', 'rs2')],
        constraints=[lambda b: b['rs1'] != b['rs2']],
    ),
    EquivalenceClass(
        name='commutative_xor',
        description='XOR operand swap produces identical state delta.',
        canonical=[
            _t('XOR', 'rd', 'rs1', 'rs2'),
            _t('XOR', 'rd', 'rs2', 'rs1'),
        ],
        contrast=[_t('SUB', 'rd', 'rs1', 'rs2')],
        constraints=[lambda b: b['rs1'] != b['rs2']],
    ),

    # -----------------------------------------------------------------
    # Self-operand identities (rs used in both slots = same register).
    # -----------------------------------------------------------------

    EquivalenceClass(
        name='self_op_zero',
        description='Self-operand XOR/SUB/SLT/SLTU all compute 0.',
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
        description='AND y,y ≡ OR y,y: both preserve rs (self = identity).',
        canonical=[
            _t('AND', 'rd', 'rs', 'rs'),
            _t('OR',  'rd', 'rs', 'rs'),
        ],
        contrast=[_t('XOR', 'rd', 'rs', 'rs')],  # XOR y,y = 0, NOT identity
    ),

    # -----------------------------------------------------------------
    # Move (copy rs → rd) via several syntactic routes.
    # -----------------------------------------------------------------

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
            _t('ANDI', 'rd', 'rs', -1),   # AND with all-ones is identity
            _t('SUB',  'rd', 'rs',  0),   # subtract x0 = no-op
            _t('SLLI', 'rd', 'rs',  0),   # shift-by-0 is identity
            _t('SRLI', 'rd', 'rs',  0),
            _t('SRAI', 'rd', 'rs',  0),
        ],
        contrast=[_t('ADDI', 'rd', 'rs', 1)],  # off-by-one imm must stay apart
    ),

    # -----------------------------------------------------------------
    # Standard compound idioms (NEG, NOT, double).
    # -----------------------------------------------------------------

    EquivalenceClass(
        name='neg_via_sub_x0',
        description='SUB rd,x0,rs = -rs (canonical NEG form).',
        canonical=[_t('SUB', 'rd', 0, 'rs')],
        contrast=[_t('ADD', 'rd', 0, 'rs')],  # ADD x0,rs = MV, not NEG
    ),
    EquivalenceClass(
        name='not_via_xori_neg1',
        description='XORI rd,rs,-1 = ~rs (canonical NOT form).',
        canonical=[_t('XORI', 'rd', 'rs', -1)],
        contrast=[_t('XORI', 'rd', 'rs', 0)],  # XOR with 0 is MV, not NOT
    ),
    EquivalenceClass(
        name='double_is_shl1',
        description='ADD rd,rs,rs ≡ SLLI rd,rs,1 (both compute 2·rs).',
        canonical=[
            _t('ADD',  'rd', 'rs', 'rs'),
            _t('SLLI', 'rd', 'rs', 1),
        ],
        contrast=[_t('SLLI', 'rd', 'rs', 2)],  # shift-by-2 is 4x, not 2x
    ),

    # -----------------------------------------------------------------
    # Write zero to rd (many syntactic routes to the same result).
    # -----------------------------------------------------------------

    EquivalenceClass(
        name='write_zero_to_rd',
        description='Instructions that write 0 to rd regardless of inputs. '
                    'All collapse under the scalar execution metric.',
        canonical=[
            _t('ADDI', 'rd',  0,    0),
            _t('XORI', 'rd',  0,    0),
            _t('ORI',  'rd',  0,    0),
            _t('AND',  'rd',  0,   'rs'),
            _t('ANDI', 'rd', 'rs',  0),
            _t('SLLI', 'rd',  0,   'shamt'),    # 0 shifted is 0
            _t('SRLI', 'rd',  0,   'shamt'),
            _t('SRAI', 'rd',  0,   'shamt'),
            _t('XOR',  'rd', 'rs', 'rs'),        # self-XOR = 0
            _t('SUB',  'rd', 'rs', 'rs'),
            _t('SLT',  'rd', 'rs', 'rs'),        # y < y = 0
            _t('SLTU', 'rd', 'rs', 'rs'),
            _t('LUI',  'rd',  0),                # LUI imm=0 writes 0
        ],
        contrast=[_t('ADDI', 'rd', 0, 1)],       # +1 must NOT collapse to 0
    ),

    # -----------------------------------------------------------------
    # Write-to-x0 NOPs: every register-writing instruction with rd=x0.
    # NOTE: currently blocked by _DEST_REGS excluding x0 from training
    # data. See TODO step 1 (coverage).
    # -----------------------------------------------------------------

    EquivalenceClass(
        name='x0_write_nop',
        description='Any register-writing instruction with rd=x0. '
                    'Zero register delta, pc+=4, dest_reg=0.',
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
            _t('LB',    0, 'imm', 'rs1'),   # LOAD arg order: (rd, imm, rs1)
            _t('LH',    0, 'imm', 'rs1'),
            _t('LW',    0, 'imm', 'rs1'),
            _t('LBU',   0, 'imm', 'rs1'),
            _t('LHU',   0, 'imm', 'rs1'),
        ],
        # Contrast: same op with rd!=0 (single-token change, opposite semantics).
        contrast=[_t('ADD', 'rd', 'rs1', 'rs2')],
    ),

    # -----------------------------------------------------------------
    # Branch always-taken / never-taken (rs1=rs2 forces the condition).
    # -----------------------------------------------------------------

    EquivalenceClass(
        name='branch_always_taken',
        description='Unconditional forward jump (pc+=offset, no register '
                    'write). BEQ/BGE/BGEU with rs1=rs2 always take; '
                    'JAL x0 is an unconditional jump with no link. All '
                    'produce identical state delta.',
        canonical=[
            _t('BEQ',  'rs', 'rs', 'imm'),
            _t('BGE',  'rs', 'rs', 'imm'),
            _t('BGEU', 'rs', 'rs', 'imm'),
            _t('JAL',   0,   'imm'),
        ],
        contrast=[_t('BNE', 'rs', 'rs', 'imm')],  # BNE y,y = never taken
    ),
    EquivalenceClass(
        name='branch_never_taken',
        description='BNE/BLT/BLTU with rs1=rs2 never take; pc+=4.',
        canonical=[
            _t('BNE',  'rs', 'rs', 'imm'),
            _t('BLT',  'rs', 'rs', 'imm'),
            _t('BLTU', 'rs', 'rs', 'imm'),
        ],
        contrast=[_t('BEQ', 'rs', 'rs', 'imm')],  # BEQ y,y = always taken
    ),

    # -----------------------------------------------------------------
    # Sign test: SLTI rd,rs,0 ≡ SLT rd,rs,x0 (both return rs < 0).
    # -----------------------------------------------------------------

    EquivalenceClass(
        name='sign_test',
        description='SLTI rd,rs,0 ≡ SLT rd,rs,x0 (both test if rs < 0).',
        canonical=[
            _t('SLTI', 'rd', 'rs', 0),
            _t('SLT',  'rd', 'rs', 0),
        ],
        contrast=[_t('SLTI', 'rd', 'rs', 1)],  # off-by-one imm
    ),
]
