"""Data generation for RV32I thought compression training.

Generates random instruction sequences, semantic-preserving transformations
(equivalence pairs), and controlled mutations (near-equivalence pairs with
known execution-state distances).

Currently limited to straight-line arithmetic/logic/shift code (no branches,
jumps, or memory operations) to keep generation simple and deterministic.
"""

import numpy as np
from emulator import Instruction, RV32IState, execute, random_regs

# Opcodes used for random generation. Straight-line, no memory, no control flow.
_ALU_R_OPS = ['ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU']
_ALU_I_OPS = ['ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI', 'SLTI', 'SLTIU']
_SHIFT_I_OPS = {'SLLI', 'SRLI', 'SRAI'}

# Registers available for generation. Skip x0 as destination (writes are
# discarded). We use x1-x15 to keep the register space small enough that
# data dependencies form naturally in short sequences.
_DEST_REGS = list(range(1, 16))
_SRC_REGS = list(range(0, 16))  # x0 is valid as a source (reads as 0)

# Immediate ranges
_IMM12_RANGE = (-2048, 2047)
_SHIFT_RANGE = (0, 31)

# Strength reduction pairs: (single_op, equivalent_pair_fn)
# add rd, rs, rs  <->  slli rd, rs, 1
_STRENGTH_PAIRS = {
    'ADD': lambda rd, rs1, rs2: (
        Instruction('SLLI', rd, rs1, 1) if rs1 == rs2 else None
    ),
    'SLLI': lambda rd, rs1, imm: (
        Instruction('ADD', rd, rs1, rs1) if imm == 1 else None
    ),
}

# Opcode mutation groups: mutating within a group produces semantically
# different but structurally similar code.
_R_MUTATION_GROUPS = [
    ['ADD', 'SUB'],
    ['XOR', 'OR', 'AND'],
    ['SLL', 'SRL', 'SRA'],
    ['SLT', 'SLTU'],
]
_I_MUTATION_GROUPS = [
    ['ADDI'],
    ['XORI', 'ORI', 'ANDI'],
    ['SLLI', 'SRLI', 'SRAI'],
    ['SLTI', 'SLTIU'],
]


def _random_r_instruction(rng: np.random.Generator) -> Instruction:
    op = rng.choice(_ALU_R_OPS)
    rd = rng.choice(_DEST_REGS)
    rs1 = rng.choice(_SRC_REGS)
    rs2 = rng.choice(_SRC_REGS)
    return Instruction(op, int(rd), int(rs1), int(rs2))


def _random_i_instruction(rng: np.random.Generator) -> Instruction:
    op = rng.choice(_ALU_I_OPS)
    rd = rng.choice(_DEST_REGS)
    rs1 = rng.choice(_SRC_REGS)
    if op in _SHIFT_I_OPS:
        imm = int(rng.integers(*_SHIFT_RANGE, endpoint=True))
    else:
        imm = int(rng.integers(*_IMM12_RANGE, endpoint=True))
    return Instruction(op, int(rd), int(rs1), imm)


def generate_sequence(
    length: int,
    rng: np.random.Generator | None = None,
    r_type_prob: float = 0.5,
) -> list[Instruction]:
    """Generate a random straight-line RV32I instruction sequence.

    Args:
        length: Number of instructions.
        rng: Random generator (created if None).
        r_type_prob: Probability of R-type vs I-type for each instruction.

    Returns:
        List of Instructions.
    """
    if rng is None:
        rng = np.random.default_rng()
    seq = []
    for _ in range(length):
        if rng.random() < r_type_prob:
            seq.append(_random_r_instruction(rng))
        else:
            seq.append(_random_i_instruction(rng))
    return seq


# ---------------------------------------------------------------------------
# Semantic-preserving transformations
# ---------------------------------------------------------------------------

def rename_registers(
    seq: list[Instruction],
    rng: np.random.Generator,
) -> tuple[list[Instruction], list[int]]:
    """Rename registers via a random permutation of x1-x15.

    x0 is left unchanged (hardwired zero).

    Returns:
        (renamed_sequence, permutation) where permutation[old_reg] = new_reg.
        Use permute_regs() to apply the permutation to a register state.
    """
    perm = list(range(16))
    # Shuffle x1-x15
    subset = perm[1:]
    rng.shuffle(subset)
    perm[1:] = subset

    def remap(reg: int) -> int:
        return perm[reg] if reg < 16 else reg

    result = []
    for instr in seq:
        op = instr.opcode
        args = instr.args
        if op in ('ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA',
                   'SLT', 'SLTU'):
            result.append(Instruction(op, remap(args[0]), remap(args[1]),
                                      remap(args[2])))
        elif op in ('ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI',
                     'SLTI', 'SLTIU'):
            result.append(Instruction(op, remap(args[0]), remap(args[1]),
                                      args[2]))
        else:
            result.append(instr)
    return result, perm


def permute_regs(regs: np.ndarray, perm: list[int]) -> np.ndarray:
    """Apply a register permutation to a register state array.

    perm[old_reg] = new_reg: value that was in old_reg moves to new_reg.
    """
    result = regs.copy()
    for old, new in enumerate(perm):
        result[new] = regs[old]
    return result


def _get_reads_writes(instr: Instruction) -> tuple[set[int], set[int]]:
    """Return (registers_read, registers_written) for an instruction."""
    op = instr.opcode
    args = instr.args
    if op in ('ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA',
               'SLT', 'SLTU'):
        return {args[1], args[2]}, {args[0]}
    elif op in ('ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI',
                 'SLTI', 'SLTIU'):
        return {args[1]}, {args[0]}
    return set(), set()


def _are_independent(a: Instruction, b: Instruction) -> bool:
    """Check if two instructions can be safely swapped."""
    a_reads, a_writes = _get_reads_writes(a)
    b_reads, b_writes = _get_reads_writes(b)
    # No WAR, WAW, or RAW hazards
    return (not (a_writes & b_reads)
            and not (b_writes & a_reads)
            and not (a_writes & b_writes))


def reorder_independent(
    seq: list[Instruction],
    rng: np.random.Generator,
    n_swaps: int | None = None,
) -> list[Instruction]:
    """Randomly swap adjacent independent instructions.

    Args:
        seq: Instruction sequence.
        rng: Random generator.
        n_swaps: Number of swap attempts. Defaults to len(seq).
    """
    if len(seq) < 2:
        return list(seq)
    result = list(seq)
    if n_swaps is None:
        n_swaps = len(seq)
    for _ in range(n_swaps):
        i = int(rng.integers(0, len(result) - 1))
        if _are_independent(result[i], result[i + 1]):
            result[i], result[i + 1] = result[i + 1], result[i]
    return result


def insert_nops(
    seq: list[Instruction],
    rng: np.random.Generator,
    count: int = 1,
) -> list[Instruction]:
    """Insert no-op instructions (add x0, x0, x0) at random positions."""
    nop = Instruction('ADD', 0, 0, 0)
    result = list(seq)
    for _ in range(count):
        pos = int(rng.integers(0, len(result) + 1))
        result.insert(pos, nop)
    return result


def remove_nops(seq: list[Instruction]) -> list[Instruction]:
    """Remove no-op instructions from a sequence."""
    return [instr for instr in seq
            if not (instr.opcode == 'ADD' and instr.args == (0, 0, 0))]


def strength_reduce(
    seq: list[Instruction],
    rng: np.random.Generator,
) -> list[Instruction] | None:
    """Apply a random strength reduction if possible. Returns None if no
    reduction is applicable."""
    candidates = []
    for i, instr in enumerate(seq):
        if instr.opcode in _STRENGTH_PAIRS:
            alt = _STRENGTH_PAIRS[instr.opcode](*instr.args)
            if alt is not None:
                candidates.append((i, alt))
    if not candidates:
        return None
    i, alt = candidates[int(rng.integers(len(candidates)))]
    result = list(seq)
    result[i] = alt
    return result


def decompose_immediate(
    seq: list[Instruction],
    rng: np.random.Generator,
) -> list[Instruction] | None:
    """Replace an ADDI with two ADDIs that sum to the same value.

    Only applies to ADDI instructions where the immediate can be split
    into two values that each fit in 12 bits.
    """
    candidates = []
    for i, instr in enumerate(seq):
        if instr.opcode == 'ADDI' and abs(instr.args[2]) >= 2:
            candidates.append(i)
    if not candidates:
        return None
    i = candidates[int(rng.integers(len(candidates)))]
    instr = seq[i]
    rd, rs1, imm = instr.args

    # Split the immediate into two parts
    lo = int(rng.integers(max(imm - 2047, -2048), min(imm + 1, 2048)))
    hi = imm - lo
    # Verify both halves fit in 12 bits
    if not (-2048 <= lo <= 2047 and -2048 <= hi <= 2047):
        return None

    result = list(seq)
    result[i:i+1] = [
        Instruction('ADDI', rd, rs1, lo),
        Instruction('ADDI', rd, rd, hi),
    ]
    return result


# ---------------------------------------------------------------------------
# Controlled mutations (near-equivalence)
# ---------------------------------------------------------------------------

def mutate_immediate(
    seq: list[Instruction],
    rng: np.random.Generator,
) -> list[Instruction] | None:
    """Change one immediate value by a small amount."""
    candidates = [i for i, instr in enumerate(seq)
                  if instr.opcode in _ALU_I_OPS]
    if not candidates:
        return None
    i = candidates[int(rng.integers(len(candidates)))]
    instr = seq[i]
    rd, rs1, imm = instr.args

    if instr.opcode in _SHIFT_I_OPS:
        delta = int(rng.choice([-1, 1]))
        new_imm = max(0, min(31, imm + delta))
        if new_imm == imm:
            return None
    else:
        delta = int(rng.integers(-10, 11))
        if delta == 0:
            delta = 1
        new_imm = imm + delta
        if not (-2048 <= new_imm <= 2047):
            return None

    result = list(seq)
    result[i] = Instruction(instr.opcode, rd, rs1, new_imm)
    return result


def mutate_opcode(
    seq: list[Instruction],
    rng: np.random.Generator,
) -> list[Instruction] | None:
    """Change one opcode to a related but different operation."""
    candidates = []
    for i, instr in enumerate(seq):
        groups = _R_MUTATION_GROUPS if instr.opcode in _ALU_R_OPS else _I_MUTATION_GROUPS
        for group in groups:
            if instr.opcode in group and len(group) > 1:
                alts = [op for op in group if op != instr.opcode]
                candidates.append((i, alts))
                break
    if not candidates:
        return None
    i, alts = candidates[int(rng.integers(len(candidates)))]
    new_op = alts[int(rng.integers(len(alts)))]
    result = list(seq)
    result[i] = Instruction(new_op, *seq[i].args)
    return result


# ---------------------------------------------------------------------------
# Training sample generation
# ---------------------------------------------------------------------------

def generate_equivalence_pair(
    length: int = 10,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate a pair of execution-equivalent instruction sequences.

    Returns dict with keys: seq_a, seq_b, initial_regs, state_a, state_b,
    transformation.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq_a = generate_sequence(length, rng)
    initial_regs = random_regs(rng)

    # Pick a random transformation.
    # rename_registers returns (seq, perm) and needs special handling:
    # the initial register state must be permuted to match the renaming,
    # so the renamed sequence reads the same values from renamed locations.
    initial_regs_b = initial_regs
    perm = None

    transforms = [
        'reorder_independent',
        'insert_nops',
        'rename_registers',
    ]
    optional = [
        'strength_reduce',
        'decompose_immediate',
    ]
    rng.shuffle(transforms)
    rng.shuffle(optional)
    all_names = list(optional) + list(transforms)

    seq_b = None
    transform_name = None
    for name in all_names:
        if name == 'rename_registers':
            seq_b, perm = rename_registers(seq_a, rng)
            initial_regs_b = permute_regs(initial_regs, perm)
            transform_name = name
            break
        elif name == 'reorder_independent':
            seq_b = reorder_independent(seq_a, rng)
            transform_name = name
            break
        elif name == 'insert_nops':
            seq_b = insert_nops(seq_a, rng)
            transform_name = name
            break
        elif name == 'strength_reduce':
            result = strength_reduce(seq_a, rng)
            if result is not None:
                seq_b = result
                transform_name = name
                break
        elif name == 'decompose_immediate':
            result = decompose_immediate(seq_a, rng)
            if result is not None:
                seq_b = result
                transform_name = name
                break

    if seq_b is None:
        # Fallback: always-applicable transform
        seq_b, perm = rename_registers(seq_a, rng)
        initial_regs_b = permute_regs(initial_regs, perm)
        transform_name = 'rename_registers'

    state_a = execute(seq_a, initial_regs=initial_regs)
    state_b = execute(seq_b, initial_regs=initial_regs_b)

    # For register renaming, map state_b back to the original register
    # namespace so that state_a and state_b are directly comparable.
    if perm is not None:
        inv_perm = [0] * len(perm)
        for old, new in enumerate(perm):
            inv_perm[new] = old
        state_b = RV32IState(permute_regs(state_b.regs, inv_perm))

    return {
        'seq_a': seq_a,
        'seq_b': seq_b,
        'initial_regs': initial_regs,
        'state_a': state_a,
        'state_b': state_b,
        'transformation': transform_name,
    }


def generate_near_equivalence_pair(
    length: int = 10,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate a pair of nearly-equivalent sequences with known distance.

    Returns dict with keys: seq_a, seq_b, initial_regs, state_a, state_b,
    distance, mutation.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq_a = generate_sequence(length, rng)
    initial_regs = random_regs(rng)

    mutations = [
        ('mutate_immediate', lambda s: mutate_immediate(s, rng)),
        ('mutate_opcode', lambda s: mutate_opcode(s, rng)),
    ]
    rng.shuffle(mutations)

    seq_b = None
    mutation_name = None
    for name, fn in mutations:
        result = fn(seq_a)
        if result is not None:
            seq_b = result
            mutation_name = name
            break

    if seq_b is None:
        # Force a mutation: change the first I-type immediate
        for i, instr in enumerate(seq_a):
            if instr.opcode in _ALU_I_OPS:
                rd, rs1, imm = instr.args
                new_imm = imm + 1
                if instr.opcode in _SHIFT_I_OPS:
                    new_imm = min(31, imm + 1)
                elif new_imm > 2047:
                    new_imm = imm - 1
                seq_b = list(seq_a)
                seq_b[i] = Instruction(instr.opcode, rd, rs1, new_imm)
                mutation_name = 'mutate_immediate_forced'
                break
        if seq_b is None:
            # All R-type: change first opcode
            seq_b = list(seq_a)
            for group in _R_MUTATION_GROUPS:
                if seq_a[0].opcode in group:
                    alt = [op for op in group if op != seq_a[0].opcode][0]
                    seq_b[0] = Instruction(alt, *seq_a[0].args)
                    mutation_name = 'mutate_opcode_forced'
                    break

    state_a = execute(seq_a, initial_regs=initial_regs)
    state_b = execute(seq_b, initial_regs=initial_regs)
    distance = state_a.distance(state_b)

    return {
        'seq_a': seq_a,
        'seq_b': seq_b,
        'initial_regs': initial_regs,
        'state_a': state_a,
        'state_b': state_b,
        'distance': distance,
        'mutation': mutation_name,
    }


def generate_training_sample(
    length: int = 10,
    rng: np.random.Generator | None = None,
    equiv_prob: float = 0.5,
) -> dict:
    """Generate a single training sample: either an equivalence pair or a
    near-equivalence pair.

    Returns dict with 'label' key ('equivalent' or 'near_equivalent') plus
    the keys from the respective generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < equiv_prob:
        sample = generate_equivalence_pair(length, rng)
        sample['label'] = 'equivalent'
    else:
        sample = generate_near_equivalence_pair(length, rng)
        sample['label'] = 'near_equivalent'
    return sample
