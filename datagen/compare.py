"""SSA analysis + canonical-baseline execution over RV32I chunks.

Owns the Precomputed schema — structural labels decided at fixed witness
states, and the canonical out_regs executed at a chunk's anchor states — that
training consumes (`precompute_chunk`), and the anchor-state draws.

Shared infrastructure:
  - SSA construction (to_ssa) — converts an instruction list into a
    dataflow DAG with INPUT leaves for live-in registers, anchored
    leaves for PC and memory state, and op nodes for each instruction.
  - DCE (live_nodes) — backward reachability from outputs.
  - SSA-numpy evaluator (_eval_ssa_numpy) — evaluates an SSA graph on
    N anchor states using vectorized numpy ops. ~10× faster than per-
    state Python emulator dispatch and bit-equal for chunker-respecting
    chunks.

V1 scope: ALU R/I-type, LUI, AUIPC, B-type, JAL, JALR. Memory ops
(LOAD/STORE) raise NotImplementedError.
"""

import functools
from dataclasses import dataclass, field

import numpy as np

from emulator import (
    Instruction, R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE,
)
from .signature import output_inputs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Special pseudo-register slots, out of 0..31 so they don't collide
# with real registers. PC and MEM are anchored — they always identify
# with their counterpart in the other chunk, never permuted.
PC_REG = 32
MEM_REG = 33

# Branch ops produce a boolean comparator node tagged with the
# branch's condition. We name these `<OP>_COND` so they're distinct
# in SSA from the branch itself (which becomes a SELECT_PC).
_BRANCH_COND_OP = {
    'BEQ': 'BEQ_COND',  'BNE': 'BNE_COND',
    'BLT': 'BLT_COND',  'BGE': 'BGE_COND',
    'BLTU': 'BLTU_COND', 'BGEU': 'BGEU_COND',
}

# JALR clears the low bit of the computed target: AND with 0xFFFFFFFE.
_JALR_TARGET_MASK = 0xFFFFFFFE

# Full RV32I op set we support in SSA. Anything outside this raises.
SUPPORTED_OPS = (
    R_TYPE | I_TYPE | B_TYPE | LOAD_TYPE | STORE_TYPE
    | {'LUI', 'AUIPC', 'JAL', 'JALR'}
)

# Lower 5 bits of shift amounts per RV32I spec.
_SHIFT_MASK = np.int32(0x1f)

N_REGS = 32

# Canonical anchor-position list for register input sourcing during SSA
# evaluation. A chunk's behavioral inputs (in syntactic first-read order) are
# sourced from these positions instead of from the row's actual register
# positions, so out_regs does not leak register names — the
# rename-invariant canonical baseline. Position 0 stays reserved for x0.
_CANON_POSITIONS = list(range(1, N_REGS))

# Per-row aux target shapes. T2's slot-positional CE heads supervise
# which actual register is at each structural slot — input slot i is
# the i-th behavioral input in syntactic-first-read order; output
# slot i is the i-th SSA-write-order output. Sized for 16-instruction
# chunks with margin; rare to fill more than ~8 slots in practice.
MAX_INPUT_SLOTS = 32
MAX_OUTPUT_SLOTS = 16
AUX_CE_IGNORE = -100


class UnsupportedOpError(ValueError):
    """Raised when a chunk contains an instruction the SSA layer
    doesn't support. With full RV32I coverage this should be
    unreachable in practice — kept as a defensive guard."""


# ---------------------------------------------------------------------------
# SSA dataflow representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SSANode:
    """A node in the SSA dataflow graph.

    op identifies the node's kind:
      - 'INPUT'      : regular register read before any write (payload = phys reg)
      - 'PC_IN'      : anchored PC input leaf (no payload)
      - 'MEM_IN'     : anchored memory-state input leaf (no payload)
      - 'CONST_IMM'  : immediate value (payload = int)
      - 'CONST_ZERO' : constant 0 (payload = 0); used for x0 reads
      - 'LUI'        : payload = imm value (constant load)
      - any RV32I R/I-type opcode      : ALU op (operands are SSA ids)
      - 'BEQ_COND'..'BGEU_COND'        : branch comparator (boolean)
      - 'SELECT_PC'                    : conditional PC = (cond, taken, not_taken)
      - any LOAD opcode (LB, LBU, ...) : LOAD_w(MEM, addr) -> value
      - any STORE opcode (SB, SH, SW)  : STORE_w(MEM, addr, value) -> new MEM
    """
    op: str
    operands: tuple = ()
    payload: object = None


@dataclass
class SSAGraph:
    """SSA representation of a chunk.

    nodes:           list of SSANode by id
    input_regs:      regular phys regs read before written, in order
                     of first read. Each appears at most once. PC/MEM
                     are NOT here (they're anchored).
    input_node_id:   regular phys reg → SSA id of its INPUT leaf
    output_versions: phys reg (or PC_REG / MEM_REG) → SSA id of the
                     register's final value, only for slots that the
                     chunk modifies. x0 is never here.
    """
    nodes: list = field(default_factory=list)
    input_regs: list = field(default_factory=list)
    input_node_id: dict = field(default_factory=dict)
    output_versions: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SSA construction
# ---------------------------------------------------------------------------

def to_ssa(instrs):
    """Convert a list[Instruction] to an SSAGraph."""
    g = SSAGraph()
    initial_id = {}  # phys reg → first-read leaf id (regular + PC/MEM)

    def new_node(op, operands=(), payload=None):
        nid = len(g.nodes)
        g.nodes.append(SSANode(op=op, operands=operands, payload=payload))
        return nid

    # x0 is always zero. Reads of x0 reuse this single node.
    zero_id = new_node('CONST_ZERO', payload=0)
    current = {0: zero_id}

    def read(r):
        if r in current:
            return current[r]
        if r == PC_REG:
            nid = new_node('PC_IN')
        elif r == MEM_REG:
            nid = new_node('MEM_IN')
        else:
            nid = new_node('INPUT', payload=r)
            g.input_node_id[r] = nid
            g.input_regs.append(r)
        current[r] = nid
        initial_id[r] = nid
        return nid

    def const(value):
        return new_node('CONST_IMM', payload=int(value))

    def write(r, nid):
        if r == 0:
            return  # x0 writes are nops
        current[r] = nid

    def pc_at(position):
        """SSA id for PC at instruction position i. PC at position i
        is PC_IN + 4*i. Position 0 returns the raw PC_IN leaf; later
        positions wrap it in an ADD with a constant offset."""
        pc_in = read(PC_REG)
        if position == 0:
            return pc_in
        return new_node('ADD', operands=(pc_in, const(4 * position)))

    for i, instr in enumerate(instrs):
        op = instr.opcode
        if op not in SUPPORTED_OPS:
            raise UnsupportedOpError(f'opcode {op!r} not supported')

        if op in R_TYPE:
            rd, rs1, rs2 = instr.args
            v1 = read(rs1)
            v2 = read(rs2)
            write(rd, new_node(op, operands=(v1, v2)))
        elif op in I_TYPE:
            rd, rs1, imm = instr.args
            v1 = read(rs1)
            v_imm = const(imm)
            write(rd, new_node(op, operands=(v1, v_imm)))
        elif op == 'LUI':
            rd, imm = instr.args
            write(rd, new_node('LUI', payload=int(imm)))
        elif op == 'AUIPC':
            rd, imm = instr.args
            # rd = PC_at_i + (imm << 12). Combined into one ADD.
            pc_i = pc_at(i)
            write(rd, new_node('ADD',
                               operands=(pc_i, const(int(imm) << 12))))
        elif op == 'JAL':
            rd, imm = instr.args
            # rd = address of next instruction = PC_in + 4*(i+1).
            write(rd, pc_at(i + 1))
            # PC' = PC_at_i + imm.
            write(PC_REG, new_node('ADD',
                                   operands=(pc_at(i), const(imm))))
        elif op == 'JALR':
            rd, rs1, imm = instr.args
            write(rd, pc_at(i + 1))
            v1 = read(rs1)
            sum_id = new_node('ADD', operands=(v1, const(imm)))
            write(PC_REG, new_node('AND',
                                   operands=(sum_id, const(_JALR_TARGET_MASK))))
        elif op in B_TYPE:
            rs1, rs2, imm = instr.args
            v1 = read(rs1)
            v2 = read(rs2)
            cond = new_node(_BRANCH_COND_OP[op], operands=(v1, v2))
            pc_i = pc_at(i)
            taken = new_node('ADD', operands=(pc_i, const(imm)))
            not_taken = new_node('ADD', operands=(pc_i, const(4)))
            write(PC_REG, new_node('SELECT_PC',
                                   operands=(cond, taken, not_taken)))
        elif op in LOAD_TYPE:
            rd, imm, rs1 = instr.args
            v1 = read(rs1)
            addr = new_node('ADD', operands=(v1, const(imm)))
            mem = read(MEM_REG)
            write(rd, new_node(op, operands=(mem, addr)))
        elif op in STORE_TYPE:
            rs2, imm, rs1 = instr.args
            v1 = read(rs1)
            val = read(rs2)
            addr = new_node('ADD', operands=(v1, const(imm)))
            mem = read(MEM_REG)
            write(MEM_REG, new_node(op, operands=(mem, addr, val)))
        else:
            raise UnsupportedOpError(f'opcode {op!r} fell through dispatch')

    # Outputs: any slot whose final SSA id differs from its initial-leaf
    # id (or which never had an initial leaf — written without being
    # read first). x0 never counts.
    for r, nid in current.items():
        if r == 0:
            continue
        if r in initial_id and initial_id[r] == nid:
            continue
        g.output_versions[r] = nid

    return g


# ---------------------------------------------------------------------------
# DCE + SSA helpers
# ---------------------------------------------------------------------------

def live_nodes(g):
    """Return SSA node ids reachable backward from outputs; the rest cannot
    affect any output, so evaluation skips them."""
    live = set()
    frontier = list(g.output_versions.values())
    while frontier:
        nid = frontier.pop()
        if nid in live:
            continue
        live.add(nid)
        frontier.extend(g.nodes[nid].operands)
    return live


def live_input_regs(g, live):
    """Regular phys regs whose INPUT leaf is in `live`."""
    return [r for r in g.input_regs if g.input_node_id[r] in live]


def _has_anchored_inputs(g, live):
    """(has_pc_in, has_mem_in) — anchored-input membership."""
    has_pc = any(g.nodes[nid].op == 'PC_IN' for nid in live)
    has_mem = any(g.nodes[nid].op == 'MEM_IN' for nid in live)
    return has_pc, has_mem


def _split_outputs(g):
    """Split output_versions into (regular_dict, pc_id_or_None,
    mem_id_or_None)."""
    reg = {r: nid for r, nid in g.output_versions.items()
           if r != PC_REG and r != MEM_REG}
    pc = g.output_versions.get(PC_REG)
    mem = g.output_versions.get(MEM_REG)
    return reg, pc, mem


# ---------------------------------------------------------------------------
# Anchor states + SSA-numpy evaluator
# ---------------------------------------------------------------------------

def draw_anchor_states(rng, n_states, mix=None, copies=None):
    """Anchor states: (n_states, 32) int32 register files, x0 = 0.

    Drawn fresh for every source chunk, so the vp head never sees an input
    point twice and cannot store answers at it. Each register's value is drawn
    from one of the classes in `mix`, and some registers then (`copies`) copy a
    LOWER-numbered register exactly, to within a few units, or negated.
    Behavioral input k is read from register k+1, so the copies are what make
    two inputs equal, adjacent or opposite: without them `x == y`, `x < y` near
    its boundary and `x + y` carrying through all 32 bits are almost never
    exercised.
    """
    n = n_states
    mix = _ANCHOR_MIX if mix is None else mix
    p_equal, p_near, p_negated = _ANCHOR_COPIES if copies is None else copies
    p = np.array([w for _, w in mix])
    cls = rng.choice(len(p), size=(n, N_REGS), p=p / p.sum())
    pow2 = ((np.int64(1) << rng.integers(0, 31, size=(n, N_REGS)))
            * rng.choice([-1, 1], size=(n, N_REGS))
            + rng.integers(-1, 2, size=(n, N_REGS)))
    by_class = {
        'zero': np.zeros((n, N_REGS), dtype=np.int64),
        'small': rng.integers(-64, 64, size=(n, N_REGS)),
        'mid': rng.integers(-4096, 4096, size=(n, N_REGS)),
        'boundary': rng.choice(_BOUNDARY_VALUES, size=(n, N_REGS)),
        'pow2': pow2,
        'uniform': rng.integers(np.iinfo(np.int32).min,
                                np.iinfo(np.int32).max,
                                size=(n, N_REGS), endpoint=True).astype(np.int64),
    }
    v = np.zeros((n, N_REGS), dtype=np.int64)
    for i, (name, _) in enumerate(mix):
        v = np.where(cls == i, by_class[name], v)
    # Register r copies a uniformly chosen register in 1..r-1 (as drawn above,
    # so copies do not chain).
    r = np.arange(N_REGS)
    src = 1 + (rng.random((n, N_REGS)) * np.maximum(r - 1, 1)).astype(np.int64)
    copied = np.take_along_axis(v, src, axis=1)
    u = rng.random((n, N_REGS))
    u[:, :2] = 1.0                      # x0 and x1 have no lower register
    delta = rng.integers(-3, 4, size=(n, N_REGS))
    v = np.where(u < p_equal, copied,
                 np.where(u < p_equal + p_near, copied + delta,
                          np.where(u < p_equal + p_near + p_negated,
                                   delta - copied, v)))
    v = (v & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    v[:, 0] = 0
    return v


# Value classes and their weights for one anchor register.
_ANCHOR_MIX = (('zero', 0.06), ('small', 0.12), ('mid', 0.06),
               ('boundary', 0.08), ('pow2', 0.08), ('uniform', 0.60))
# Probabilities that a register copies a lower one exactly, to within +-3, or
# negated to within +-3.
_ANCHOR_COPIES = (0.15, 0.05, 0.05)
_BOUNDARY_VALUES = np.array(
    [0, 1, -1, 2, -2, 2**31 - 1, -2**31, 2**30, -2**30,
     2047, -2048, 2048, -2049, 31, 32], dtype=np.int64)

# The fixed anchor set every score is reported at, from a seed no generator
# uses, so no model trains on it. Its value classes are pinned here rather than
# following _ANCHOR_MIX, so tuning the training draw cannot move the criterion.
HELDOUT_N_ANCHORS = 64
_HELDOUT_MIX = (('zero', 0.10), ('small', 0.20), ('mid', 0.10),
                ('boundary', 0.10), ('pow2', 0.10), ('uniform', 0.40))
_HELDOUT_COPIES = (0.15, 0.05, 0.0)
_HELDOUT_SEED = 0x5EED0FF


def heldout_anchor_states(n_states=HELDOUT_N_ANCHORS):
    """The first n_states of the one fixed held-out set, so a smaller scoring
    corpus is a subset of a larger one."""
    assert n_states <= HELDOUT_N_ANCHORS, n_states
    return draw_anchor_states(np.random.default_rng(_HELDOUT_SEED),
                              HELDOUT_N_ANCHORS, _HELDOUT_MIX,
                              _HELDOUT_COPIES)[:n_states].copy()


def _canonical_state(anchor_states, inputs, canon_positions):
    """Place a structured draw on a chunk's input registers: inputs[i] takes
    its value from anchor_states[:, canon_positions[i]]. The draw's copies and
    near-immediate values sit at low positions, so this is what makes two of
    the chunk's inputs equal, adjacent or opposite whatever registers they
    occupy. The chunk executes on the returned register file as it is.
    """
    state = anchor_states.copy()
    for actual_reg, canon_reg in zip(inputs, canon_positions):
        if actual_reg == 0:
            continue
        state[:, actual_reg] = anchor_states[:, canon_reg]
    state[:, 0] = 0
    return state


def _eval_ssa_numpy(ssa, chunk_len, anchor_states, live=None):
    """Evaluate an SSA graph on N input states using numpy ops.

    Walks node ids in order (topologically valid by construction).
    Each node produces an (N,) int32 or bool array; one numpy op
    per node, vectorized across all N states.

    For chunker-respecting chunks (only the last instruction may be
    control flow), bit-equal to the Python emulator. For chunks with
    intra-chunk control flow the semantics differ (SSA evaluates every
    node regardless of runtime PC dispatch).
    """
    n_states = anchor_states.shape[0]
    values = {}
    nodes = ssa.nodes

    for nid, node in enumerate(nodes):
        if live is not None and nid not in live:
            continue
        op = node.op

        if op == 'INPUT':
            values[nid] = anchor_states[:, node.payload].copy()
        elif op == 'CONST_ZERO':
            values[nid] = np.zeros(n_states, dtype=np.int32)
        elif op == 'CONST_IMM':
            values[nid] = np.full(
                n_states, int(node.payload), dtype=np.int64
            ).astype(np.int32)
        elif op == 'LUI':
            values[nid] = np.full(
                n_states, int(node.payload) << 12, dtype=np.int64
            ).astype(np.int32)
        elif op == 'PC_IN':
            values[nid] = np.zeros(n_states, dtype=np.int32)
        elif op == 'MEM_IN':
            raise NotImplementedError('SSA-numpy eval does not support memory')

        elif op in ('ADD', 'ADDI'):
            a, b = node.operands
            # uint32 modular arithmetic: bit-identical to RV32 ADD's
            # two's-complement wrap, but the int64 round-trip's
            # bandwidth and cast cost is gone. .view() is a free
            # bit-reinterpret (no copy).
            values[nid] = (values[a].view(np.uint32)
                           + values[b].view(np.uint32)).view(np.int32)
        elif op == 'SUB':
            a, b = node.operands
            values[nid] = (values[a].view(np.uint32)
                           - values[b].view(np.uint32)).view(np.int32)
        elif op in ('XOR', 'XORI'):
            a, b = node.operands
            values[nid] = values[a] ^ values[b]
        elif op in ('OR', 'ORI'):
            a, b = node.operands
            values[nid] = values[a] | values[b]
        elif op in ('AND', 'ANDI'):
            a, b = node.operands
            values[nid] = values[a] & values[b]

        elif op in ('SLL', 'SLLI'):
            a, b = node.operands
            shift = (values[b] & _SHIFT_MASK).astype(np.uint32)
            values[nid] = (values[a].astype(np.uint32)
                           << shift).astype(np.int32)
        elif op in ('SRL', 'SRLI'):
            a, b = node.operands
            shift = (values[b] & _SHIFT_MASK).astype(np.uint32)
            values[nid] = (values[a].astype(np.uint32)
                           >> shift).astype(np.int32)
        elif op in ('SRA', 'SRAI'):
            a, b = node.operands
            shift = (values[b] & _SHIFT_MASK)
            values[nid] = (values[a] >> shift).astype(np.int32)

        elif op in ('SLT', 'SLTI'):
            a, b = node.operands
            values[nid] = (values[a] < values[b]).astype(np.int32)
        elif op in ('SLTU', 'SLTIU'):
            a, b = node.operands
            values[nid] = (values[a].astype(np.uint32)
                           < values[b].astype(np.uint32)).astype(np.int32)

        elif op == 'BEQ_COND':
            a, b = node.operands
            values[nid] = (values[a] == values[b])
        elif op == 'BNE_COND':
            a, b = node.operands
            values[nid] = (values[a] != values[b])
        elif op == 'BLT_COND':
            a, b = node.operands
            values[nid] = (values[a] < values[b])
        elif op == 'BGE_COND':
            a, b = node.operands
            values[nid] = (values[a] >= values[b])
        elif op == 'BLTU_COND':
            a, b = node.operands
            values[nid] = (values[a].astype(np.uint32)
                           < values[b].astype(np.uint32))
        elif op == 'BGEU_COND':
            a, b = node.operands
            values[nid] = (values[a].astype(np.uint32)
                           >= values[b].astype(np.uint32))

        elif op == 'SELECT_PC':
            cond, taken, not_taken = node.operands
            values[nid] = np.where(
                values[cond], values[taken], values[not_taken]
            ).astype(np.int32)

        else:
            raise ValueError(f'unsupported SSA op {op!r} in numpy eval')

    out_regs = anchor_states.copy()
    out_pcs = None
    for r, nid in ssa.output_versions.items():
        if r == PC_REG:
            out_pcs = (values[nid].astype(np.int64) & 0xFFFFFFFF)
        elif r == MEM_REG:
            pass
        else:
            out_regs[:, r] = values[nid]

    if out_pcs is None:
        out_pcs = np.full(n_states, 4 * chunk_len, dtype=np.int64)

    return out_regs, out_pcs


def _has_memory_ops(chunk):
    return any(instr.opcode in LOAD_TYPE or instr.opcode in STORE_TYPE
               for instr in chunk)


# ---------------------------------------------------------------------------
# Structural labels — decided on FIXED witness states, never on the anchors
# ---------------------------------------------------------------------------
#
# Which registers a chunk reads and writes is a property of the chunk, so it
# must not depend on which anchor states a batch drew: with per-batch anchors,
# `SLTIU x4,x5,40` would read x5 only in batches where some anchor puts a value
# below 40 there, and its binding target would flip between batches.

# Every pair of these is tried on an instruction's (at most two) sources. This
# decides read relevance for every RV32I ALU form exactly as several thousand
# rich states do: zero witnesses the unsigned compares, the small values the
# signed ones, and the generic value the shifts and masks.
_WITNESS_VALUES = (0, 1, -1, 0x5A3C96E1)
_READ_WITNESS = np.zeros((len(_WITNESS_VALUES) ** 2, N_REGS), dtype=np.int32)
_READ_WITNESS[:, 1] = np.repeat(_WITNESS_VALUES, len(_WITNESS_VALUES))
_READ_WITNESS[:, 2] = np.tile(_WITNESS_VALUES, len(_WITNESS_VALUES))


def _source_regs(instr):
    op = instr.opcode
    if op in R_TYPE:
        return instr.args[1:3]
    if op in B_TYPE:
        return instr.args[0:2]
    if op in I_TYPE or op == 'JALR':
        return instr.args[1:2]
    return ()


def _written_regs(instr):
    """The registers an instruction writes (PC aside)."""
    op = instr.opcode
    if op in B_TYPE or op in STORE_TYPE or instr.args[0] == 0:
        return ()
    return (instr.args[0],)


@functools.lru_cache(maxsize=None)
def _canonical_form_reads(opcode, args):
    """Relevant source registers of an instruction whose sources are x1/x2 (in
    read order) and whose destination is x3 unless it aliases a source. Keyed
    by value: an Instruction hashes by identity."""
    ssa = to_ssa([Instruction(opcode, *args)])
    live = live_nodes(ssa)
    base_regs, base_pcs = _eval_ssa_numpy(ssa, 1, _READ_WITNESS, live)
    outs = [r for r in ssa.output_versions if r < N_REGS]
    relevant = []
    for r in ssa.input_regs:
        state = _READ_WITNESS.copy()
        state[:, r] = ~state[:, r]
        regs, pcs = _eval_ssa_numpy(ssa, 1, state, live)
        if ((regs[:, outs] != base_regs[:, outs]).any()
                or (PC_REG in ssa.output_versions and (pcs != base_pcs).any())):
            relevant.append(r)
    return tuple(relevant)


def instruction_reads(instr):
    """The source registers an instruction's outputs depend on, in read order.

    Decided once per instruction FORM — the opcode, the immediate and which
    sources are x0 or alias each other — by flipping each source at the fixed
    witness states, so no per-opcode rule is written down and no anchor draw
    can change the answer.
    """
    canon, inverse = {0: 0}, {0: 0}
    for r in _source_regs(instr):
        if r not in canon:
            canon[r] = len(canon)
            inverse[canon[r]] = r
    args = list(instr.args)
    op = instr.opcode
    pos = {'R': (1, 2), 'B': (0, 1)}.get(
        'R' if op in R_TYPE else 'B' if op in B_TYPE else '', (1,))
    if op in R_TYPE or op in B_TYPE or op in I_TYPE or op == 'JALR':
        for i in pos:
            args[i] = canon[args[i]]
    if op not in B_TYPE:
        rd = instr.args[0]
        args[0] = canon.get(rd, 0 if rd == 0 else 3)
    return [inverse[r] for r in _canonical_form_reads(op, tuple(args))]


def _input_order(chunk):
    """The chunk's inputs in first-read order: every read relevant at its own
    instruction of a register not yet written, whether or not that instruction's
    result survives. That is the order T2's read ranks number them in, given
    T1's emitted wiring; a dead instruction's reads are shown to the encoder, so
    the target must count them. Whether dead code is equivalent to its absence
    is for the essence to learn, not for the targets to impose."""
    order, written = [], set()
    for instr in chunk:
        for r in instruction_reads(instr):
            if r not in written and r not in order:
                order.append(r)
        written.update(_written_regs(instr))
    return order


# ---------------------------------------------------------------------------
# Chunk analysis — Precomputed schema + aux targets
# ---------------------------------------------------------------------------

@dataclass
class Precomputed:
    """Per-chunk structural labels and canonical execution at the anchors.

    pc_explicit indicates whether some live SSA op writes PC.
    """
    inputs: list                 # all live regular inputs (syntactic)
    behavioral_inputs: list      # relevant reads in first-read order, dead included
    reg_outs: list               # written regular output regs
    anchors: np.ndarray          # (n_states, 32) the register files executed from
    out_regs: np.ndarray         # (n_states, 32) the register files after execution
    # Per output slot (aligned with out_slot_regs), the external input
    # registers that output depends on, in the structural-colour order the
    # evaluator gathers their values in (datagen.signature.output_inputs).
    out_inputs: np.ndarray       # (MAX_OUTPUT_SLOTS, MAX_INPUT_SLOTS) int8
    pc_explicit: bool            # does any live SSA op write PC
    # Per-row aux targets (T2 register-identity supervision).
    live_in_mask: np.ndarray     # (N_REGS,) bool — behavioral inputs as mask
    live_out_mask: np.ndarray    # (N_REGS,) bool — written regs as mask
    in_slot_regs: np.ndarray     # (MAX_INPUT_SLOTS,) int8 — reg or AUX_CE_IGNORE
    out_slot_regs: np.ndarray    # (MAX_OUTPUT_SLOTS,) int8 — reg or AUX_CE_IGNORE


def precompute_chunk(chunk, anchor_states):
    """Structural labels from the fixed witness states, and execution at
    `anchor_states` for the value-prediction target.

    `anchor_states` is a structured draw whose copies, near-immediate and
    boundary values sit at positions 1, 2, …; it is placed onto the chunk's
    own input registers (input i takes position i+1) and the chunk is executed
    on that register file as drawn, register names intact. The evaluator reads
    each output's value at its own register and gathers its inputs' values at
    theirs, so a renamed twin executes on its own placement of the same draw.

    Raises NotImplementedError on memory ops (V1 scope).
    """
    if _has_memory_ops(chunk):
        raise NotImplementedError(
            'precompute_chunk V1 does not support memory ops (LOAD/STORE)')

    ssa = to_ssa(chunk)
    live = live_nodes(ssa)
    _, has_mem_in = _has_anchored_inputs(ssa, live)
    reg_out, pc_out, mem_out = _split_outputs(ssa)
    assert mem_out is None and not has_mem_in

    inputs = live_input_regs(ssa, live)
    pc_explicit = pc_out is not None

    # Outputs: every register the chunk writes, even with the value it already
    # held. Calling `ADD x5,x5,x0` a no-op would make the binding targets impose
    # an equivalence (identity == empty chunk) that is the essence's to learn.
    actual_reg_outs = sorted(reg_out.keys())

    behavioral_inputs = _input_order(chunk)

    # The structured draw lands on the chunk's actual inputs; execution is on
    # that register file as it is, so the targets are raw register values.
    state = _canonical_state(anchor_states, behavioral_inputs, _CANON_POSITIONS)
    out_regs, _ = _eval_ssa_numpy(ssa, len(chunk), state, live)

    out_inputs = np.full((MAX_OUTPUT_SLOTS, MAX_INPUT_SLOTS), AUX_CE_IGNORE,
                         dtype=np.int8)

    # Aux targets: live-in/out masks + per-slot register IDs.
    live_in_mask = np.zeros(N_REGS, dtype=bool)
    for r in behavioral_inputs:
        live_in_mask[r] = True
    live_out_mask = np.zeros(N_REGS, dtype=bool)
    for r in actual_reg_outs:
        live_out_mask[r] = True

    in_slot_regs = np.full(MAX_INPUT_SLOTS, AUX_CE_IGNORE, dtype=np.int8)
    for i, r in enumerate(behavioral_inputs[:MAX_INPUT_SLOTS]):
        in_slot_regs[i] = r

    # SSA-write-order: sort the written regs by their final-
    # write SSA node id (later id = later write).
    write_ordered = sorted(actual_reg_outs,
                           key=lambda r: ssa.output_versions[r])
    out_slot_regs = np.full(MAX_OUTPUT_SLOTS, AUX_CE_IGNORE, dtype=np.int8)
    deps = output_inputs(chunk)
    for i, r in enumerate(write_ordered[:MAX_OUTPUT_SLOTS]):
        out_slot_regs[i] = r
        for j, src in enumerate(deps[r][:MAX_INPUT_SLOTS]):
            out_inputs[i, j] = src

    return Precomputed(
        inputs=inputs,
        behavioral_inputs=behavioral_inputs,
        reg_outs=actual_reg_outs,
        anchors=state,
        out_regs=out_regs,
        out_inputs=out_inputs,
        pc_explicit=pc_explicit,
        live_in_mask=live_in_mask,
        live_out_mask=live_out_mask,
        in_slot_regs=in_slot_regs,
        out_slot_regs=out_slot_regs,
    )
