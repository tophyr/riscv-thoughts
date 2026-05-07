"""Equivalence and distance reasoning over RV32I instruction chunks.

Two outputs from the same machinery, differing only in granularity:

  gvn_equivalent(a, b) -> bool
      Yes/no: are these chunks GVN-equivalent (isomorphic dataflow
      DAGs under register relabeling, commutativity, and dead-write
      elimination)?

  behavioral_distance(a, b) -> float >= 0
      Continuous: how different are these chunks behaviorally?
      Equivalence-preserving augmentations (relabeling, commutative
      arg swap, dead writes, cross-syntax aliases like XOR x,x ≡
      ADDI x,0,0) all collapse to 0; structurally and behaviorally
      different chunks get a smooth nonzero distance.

Shared infrastructure:
  - SSA construction (to_ssa) — converts an instruction list into a
    dataflow DAG with INPUT leaves for live-in registers, anchored
    leaves for PC and memory state, and op nodes for each instruction.
  - DCE (live_nodes) — backward reachability from outputs.
  - Value numbering (value_number) — assigns recursive structural
    keys; two nodes share a key iff they compute the same function
    under the chosen input bijection. Used by gvn_equivalent.
  - Partial bijection enumeration (partial_bijections) — yields
    candidate input mappings for both equivalence and distance search.
  - SSA-numpy evaluator (eval_ssa_numpy) — evaluates an SSA graph on
    N anchor states using vectorized numpy ops. ~10× faster than per-
    state Python emulator dispatch and bit-equal for chunker-respecting
    chunks. Used by behavioral_distance for behavioral evaluation.

V1 scope: ALU R/I-type, LUI, AUIPC, B-type, JAL, JALR. Memory ops
(LOAD/STORE) raise NotImplementedError in behavioral_distance; gvn_equivalent
handles them at the SSA level via anchored MEM_IN leaves.
"""

from dataclasses import dataclass, field
from itertools import combinations, permutations

import numexpr as ne
import numpy as np
from scipy.optimize import linear_sum_assignment

from emulator import (
    R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Special pseudo-register slots, out of 0..31 so they don't collide
# with real registers. PC and MEM are anchored — they always identify
# with their counterpart in the other chunk, never permuted.
PC_REG = 32
MEM_REG = 33

# Truly commutative ops: f(a, b) = f(b, a) under all input values.
# - ADD, XOR, OR, AND: R/I-type ALU symmetries (I-type variants are
#   NOT commutative because the immediate operand is structurally
#   distinct from the register operand).
# - BEQ_COND / BNE_COND: equality and inequality comparators are
#   symmetric.
COMMUTATIVE_OPS = frozenset({
    'ADD', 'XOR', 'OR', 'AND',
    'BEQ_COND', 'BNE_COND',
})

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

# An input register's behavioral magnitude must exceed this to enter
# the bijection search. Set just above floating-point noise so that
# "no observable effect on any output" reliably filters out.
BEHAVIORAL_RELEVANCE_THRESHOLD = 1e-9

# Cap on the bijection search size. The number of partial bijections
# is P(max(|a|, |b|), min(|a|, |b|)) — factorial in the smaller set.
# 8! = 40320 is fast and bounded; above that we fall back to the
# empty bijection (no input symmetry discovery; distance is an upper
# bound for those pathological chunks).
MAX_BIJECTION_SIZE = 8

N_REGS = 32

# Canonical anchor-position list for register input sourcing during
# SSA evaluation. A row's behavioral inputs are sourced from these
# positions instead of from the row's actual register positions, so
# the residual and magnitudes don't leak the row's register names.
# All 31 non-x0 register slots are usable. A paired evaluation needs
# up to |inputs_a| + |inputs_b| - matched distinct positions; with
# branch+cap=8 chunks producing up to 16 inputs each, that approaches
# 32. Position 0 stays reserved for x0 (always reads zero); pairs whose
# unaligned input count exceeds 31 raise CanonPositionOverflow so the
# caller can skip rather than silently aliasing positions.
_CANON_POSITIONS = list(range(1, N_REGS))

# Per-row aux target shapes. T2's slot-positional CE heads supervise
# which actual register is at each structural slot — input slot i is
# the i-th behavioral input in syntactic-first-read order; output
# slot i is the i-th SSA-write-order output. Sized for 16-instruction
# chunks with margin; rare to fill more than ~8 slots in practice.
MAX_INPUT_SLOTS = 32
MAX_OUTPUT_SLOTS = 16
AUX_CE_IGNORE = -100


class CanonPositionOverflow(Exception):
    """Raised when a pair's combined unaligned input count exceeds
    the 31 available canonical positions (positions 1..31; position 0
    is reserved for x0). Pairs that overflow can't be evaluated under
    the GVN-invariant canonical metric and should be skipped."""


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
    """Return SSA node ids reachable backward from outputs. Dead nodes
    don't contribute to the chunk's externally-observable effect, so
    equivalence and distance must ignore them."""
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
# Value numbering
# ---------------------------------------------------------------------------

def value_number(g, input_abstract_id, live=None):
    """Assign a recursive structural VN key to each (live) SSA node.

    Two nodes from different graphs share a VN key iff they compute the
    same function under the given input mapping. Used by gvn_equivalent
    to test structural equivalence under a candidate input bijection.

    input_abstract_id: dict mapping regular phys reg → abstract input
                       id (the bijection's choice). Anchored leaves
                       (PC_IN, MEM_IN) get fixed keys.
    live:              optional set of node ids to compute keys for.
    """
    keys = {}
    for nid, node in enumerate(g.nodes):
        if live is not None and nid not in live:
            continue
        op = node.op
        if op == 'INPUT':
            key = ('INPUT', input_abstract_id[node.payload])
        elif op == 'PC_IN':
            key = ('PC_IN',)
        elif op == 'MEM_IN':
            key = ('MEM_IN',)
        elif op == 'CONST_ZERO':
            key = ('CONST', 0)
        elif op == 'CONST_IMM':
            key = ('CONST', node.payload)
        elif op == 'LUI':
            key = ('LUI', node.payload)
        elif op in COMMUTATIVE_OPS:
            op_keys = tuple(sorted(keys[o] for o in node.operands))
            key = (op,) + op_keys
        else:
            op_keys = tuple(keys[o] for o in node.operands)
            key = (op,) + op_keys
        keys[nid] = key
    return keys


# ---------------------------------------------------------------------------
# Partial bijection enumeration — shared by gvn_equivalent and behavioral_distance
# ---------------------------------------------------------------------------

def partial_bijections(set_a, set_b):
    """Yield all partial bijections of size k = min(|a|, |b|) as
    paired ordered tuples (a_subset, b_subset) where a_subset[i] is
    matched to b_subset[i]."""
    a = tuple(set_a)
    b = tuple(set_b)
    k = min(len(a), len(b))
    if k == 0:
        yield (), ()
        return
    if len(a) <= len(b):
        for chosen in combinations(b, k):
            for perm in permutations(chosen):
                yield a, perm
    else:
        for chosen in combinations(a, k):
            for perm in permutations(chosen):
                yield perm, b


# ---------------------------------------------------------------------------
# GVN equivalence — binary yes/no
# ---------------------------------------------------------------------------

def gvn_equivalent(instrs_a, instrs_b):
    """Return True iff the two chunks are GVN-equivalent.

    Equivalent iff their dataflow DAGs are isomorphic under some
    bijection between live regular input register sets, with
    commutative ops matching modulo argument order, and PC/MEM
    inputs/outputs matched on identity (not interchangeable with
    regular regs).
    """
    a = to_ssa(instrs_a)
    b = to_ssa(instrs_b)
    live_a = live_nodes(a)
    live_b = live_nodes(b)

    inputs_a = live_input_regs(a, live_a)
    inputs_b = live_input_regs(b, live_b)
    if len(inputs_a) != len(inputs_b):
        return False
    if _has_anchored_inputs(a, live_a) != _has_anchored_inputs(b, live_b):
        return False

    reg_out_a, pc_out_a, mem_out_a = _split_outputs(a)
    reg_out_b, pc_out_b, mem_out_b = _split_outputs(b)
    if len(reg_out_a) != len(reg_out_b):
        return False
    if (pc_out_a is None) != (pc_out_b is None):
        return False
    if (mem_out_a is None) != (mem_out_b is None):
        return False

    if not a.output_versions and not b.output_versions:
        return True

    n = len(inputs_a)
    b_abstract = {r: i for i, r in enumerate(inputs_b)}
    keys_b = value_number(b, b_abstract, live=live_b)
    reg_out_keys_b = sorted(keys_b[reg_out_b[r]] for r in reg_out_b)
    pc_key_b = keys_b[pc_out_b] if pc_out_b is not None else None
    mem_key_b = keys_b[mem_out_b] if mem_out_b is not None else None

    perms = permutations(range(n)) if n > 0 else [()]
    for perm in perms:
        a_abstract = {inputs_a[i]: perm[i] for i in range(n)}
        keys_a = value_number(a, a_abstract, live=live_a)
        reg_out_keys_a = sorted(keys_a[reg_out_a[r]] for r in reg_out_a)
        if reg_out_keys_a != reg_out_keys_b:
            continue
        if pc_out_a is not None and keys_a[pc_out_a] != pc_key_b:
            continue
        if mem_out_a is not None and keys_a[mem_out_a] != mem_key_b:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Anchor states + SSA-numpy evaluator
# ---------------------------------------------------------------------------

def make_anchor_states(n_states, seed):
    """Generate shared anchor states for distance computation. Two
    chunks compared via behavioral_distance must use the same anchor states."""
    rng = np.random.default_rng(seed)
    anchor_states = rng.integers(
        np.iinfo(np.int32).min, np.iinfo(np.int32).max,
        size=(n_states, 32), dtype=np.int32)
    anchor_states[:, 0] = 0
    return anchor_states


def _canonical_state(anchor_states, inputs, canon_positions):
    """Build an anchor-state copy where each row input register is
    re-sourced from a canonical anchor position. inputs[i] takes its
    value from anchor_states[:, canon_positions[i]]. Two rows whose
    inputs are placed at the same canonical positions then read
    identical values regardless of which actual register names they
    use, so SSA evaluation is GVN-invariant under register relabeling.
    """
    state = anchor_states.copy()
    for actual_reg, canon_reg in zip(inputs, canon_positions):
        if actual_reg == 0:
            continue
        state[:, actual_reg] = anchor_states[:, canon_reg]
    state[:, 0] = 0
    return state


def _pair_canon_positions(inputs_a, inputs_b, a_keys, b_keys):
    """Assign canonical positions for a pair under bijection σ
    (a_keys[i] ↔ b_keys[i]). Matched inputs share canon positions
    0..m-1; a's unaligned inputs use m..k_a-1; b's unaligned inputs
    use k_a..k_a+k_b-m-1. Returns two lists in the syntactic input
    order of inputs_a and inputs_b. Raises CanonPositionOverflow if
    the combined position count exceeds the 31 available slots."""
    m = len(a_keys)
    a_to_pos = {a_keys[i]: _CANON_POSITIONS[i] for i in range(m)}
    b_to_pos = {b_keys[i]: _CANON_POSITIONS[i] for i in range(m)}
    next_pos = m
    n_avail = len(_CANON_POSITIONS)
    for r in inputs_a:
        if r not in a_to_pos:
            if next_pos >= n_avail:
                raise CanonPositionOverflow(
                    f'pair needs >{n_avail} canon positions '
                    f'(|a|={len(inputs_a)} |b|={len(inputs_b)} m={m})')
            a_to_pos[r] = _CANON_POSITIONS[next_pos]
            next_pos += 1
    for r in inputs_b:
        if r not in b_to_pos:
            if next_pos >= n_avail:
                raise CanonPositionOverflow(
                    f'pair needs >{n_avail} canon positions '
                    f'(|a|={len(inputs_a)} |b|={len(inputs_b)} m={m})')
            b_to_pos[r] = _CANON_POSITIONS[next_pos]
            next_pos += 1
    return ([a_to_pos[r] for r in inputs_a],
            [b_to_pos[r] for r in inputs_b])


def _eval_ssa_numpy(ssa, chunk_len, anchor_states, live=None):
    """Evaluate an SSA graph on N input states using numpy ops.

    Walks node ids in order (topologically valid by construction).
    Each node produces an (N,) int32 or bool array; one numpy op
    per node, vectorized across all N states.

    For chunker-respecting chunks (only the last instruction may be
    control flow), bit-equal to the Python emulator. For chunks with
    intra-chunk control flow the semantics differ (SSA evaluates every
    node regardless of runtime PC dispatch) — behavioral_distance's contract
    excludes those cases.
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


# ---------------------------------------------------------------------------
# Chunk distance — continuous behavioral metric
# ---------------------------------------------------------------------------

def _loglog(x):
    """log1p(log1p(|x|)) on a numpy array."""
    return np.log1p(np.log1p(np.abs(x).astype(np.float64)))


# Forward declaration; defined later in this file. Used by precompute_row_outputs.
def _has_memory_ops(chunk):
    return any(instr.opcode in LOAD_TYPE or instr.opcode in STORE_TYPE
               for instr in chunk)


def _output_magnitudes(out_regs, in_states, output_regs):
    """For each output register, mean loglog'd |delta| across states.
    Used as smooth edit cost for unmatched outputs."""
    if not output_regs:
        return {}
    # int64 upcast: int32-int32 deltas can overflow to INT32_MIN, whose
    # abs is also INT32_MIN, which log1p turns into NaN.
    deltas = (out_regs[:, output_regs].astype(np.int64)
              - in_states[:, output_regs].astype(np.int64))
    mags = _loglog(deltas).mean(axis=0)
    return {r: float(m) for r, m in zip(output_regs, mags)}


def _actually_modified(out_regs, in_states, ssa_candidates):
    """Filter SSA's candidate output regs to those whose value
    actually differs from the initial state in at least one anchor
    state. SSA over-claims when intra-chunk control flow drops some
    writes; execution truth drives the matched-output set."""
    if not ssa_candidates:
        return []
    out_view = out_regs[:, ssa_candidates]
    in_view = in_states[:, ssa_candidates]
    any_diff = (out_view != in_view).any(axis=0)
    return [r for r, d in zip(ssa_candidates, any_diff) if d]


def _input_magnitudes(ssa, live, syntactic_inputs, anchor_states,
                      baseline_out_regs, baseline_out_pcs, chunk_len,
                      output_regs, has_pc_out):
    """Per syntactic input, behavioral magnitude: how much do outputs
    change when this input's values are perturbed?

    Magnitude = max-across-output-slots of mean-per-state loglog'd
    abs-output-difference. Magnitude near 0 → input is behaviorally
    irrelevant → filter out of bijection search. Magnitude > 0 →
    weights the unaligned-input edit cost.

    Perturbations and baseline both run on the canonical state, so
    magnitudes are register-name-invariant — two GVN-equivalent rows
    produce identical input_mags.
    """
    mags = {}
    for r in syntactic_inputs:
        perturbed = _canonical_state(
            anchor_states, syntactic_inputs, _CANON_POSITIONS)
        perturbed[:, r] = ~perturbed[:, r]
        perturbed[:, 0] = 0  # x0 always 0

        out_regs, out_pcs = _eval_ssa_numpy(ssa, chunk_len, perturbed, live)

        per_slot_means = []
        if output_regs:
            reg_diffs = (out_regs[:, output_regs].astype(np.int64) -
                         baseline_out_regs[:, output_regs].astype(np.int64))
            slot_means = _loglog(reg_diffs).mean(axis=0)
            per_slot_means.extend(slot_means.tolist())
        if has_pc_out:
            pc_diffs = out_pcs.astype(np.int64) - baseline_out_pcs.astype(np.int64)
            per_slot_means.append(float(_loglog(pc_diffs).mean()))

        mags[r] = max(per_slot_means) if per_slot_means else 0.0
    return mags


@dataclass
class Precomputed:
    """Per-chunk data cached across many distance computations.

    pc_explicit indicates whether some live SSA op writes PC. For
    chunks that don't write PC explicitly, the implicit final PC =
    4*chunk_len; these chunks contribute no PC residual when paired
    with another non-explicit chunk.
    """
    chunk: list                  # list[Instruction]
    ssa: SSAGraph                # for σ-permuted re-evaluation
    live: set                    # live SSA node ids after DCE
    inputs: list                 # all live regular inputs (syntactic)
    behavioral_inputs: list      # subset of `inputs` with magnitude > threshold
    input_mags: dict             # reg → behavioral magnitude
    reg_outs: list               # actually-modified regular output regs
    out_regs: np.ndarray         # (n_states, 32) anchor exec result
    out_pcs: np.ndarray          # (n_states,) final PC per state
    out_mags: dict               # reg → loglog'd output magnitude
    pc_explicit: bool            # does any live SSA op write PC
    # Per-row aux targets (T2 register-identity supervision).
    live_in_mask: np.ndarray     # (N_REGS,) bool — behavioral inputs as mask
    live_out_mask: np.ndarray    # (N_REGS,) bool — actually-modified regs as mask
    in_slot_regs: np.ndarray     # (MAX_INPUT_SLOTS,) int8 — reg or AUX_CE_IGNORE
    out_slot_regs: np.ndarray    # (MAX_OUTPUT_SLOTS,) int8 — reg or AUX_CE_IGNORE


def precompute_chunk(chunk, anchor_states):
    """SSA + DCE + anchor execution + sensitivity analysis.

    Raises NotImplementedError on memory ops (V1 distance scope).
    """
    if _has_memory_ops(chunk):
        raise NotImplementedError(
            'behavioral_distance V1 does not support memory ops (LOAD/STORE)')

    ssa = to_ssa(chunk)
    live = live_nodes(ssa)
    _, has_mem_in = _has_anchored_inputs(ssa, live)
    reg_out, pc_out, mem_out = _split_outputs(ssa)
    assert mem_out is None and not has_mem_in

    inputs = live_input_regs(ssa, live)
    pc_explicit = pc_out is not None

    # Baseline executes on a canonical state: the chunk's syntactic
    # inputs are sourced from canonical anchor positions (canon_0,
    # canon_1, ... in syntactic order). All outputs and magnitudes
    # below are register-name-invariant — two GVN-equivalent chunks
    # produce identical canonical out_regs / out_pcs / mags.
    canon_state = _canonical_state(anchor_states, inputs, _CANON_POSITIONS)
    out_regs, out_pcs = _eval_ssa_numpy(ssa, len(chunk), canon_state, live)

    ssa_reg_outs = sorted(reg_out.keys())
    actual_reg_outs = _actually_modified(out_regs, canon_state, ssa_reg_outs)
    out_mags = _output_magnitudes(out_regs, canon_state, actual_reg_outs)

    input_mags = _input_magnitudes(
        ssa, live, inputs, anchor_states,
        out_regs, out_pcs, len(chunk),
        actual_reg_outs, pc_explicit)
    behavioral_inputs = [r for r in inputs
                         if input_mags[r] > BEHAVIORAL_RELEVANCE_THRESHOLD]

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

    # SSA-write-order: sort the actually-modified regs by their final-
    # write SSA node id (later id = later write).
    write_ordered = sorted(actual_reg_outs,
                           key=lambda r: ssa.output_versions[r])
    out_slot_regs = np.full(MAX_OUTPUT_SLOTS, AUX_CE_IGNORE, dtype=np.int8)
    for i, r in enumerate(write_ordered[:MAX_OUTPUT_SLOTS]):
        out_slot_regs[i] = r

    return Precomputed(
        chunk=chunk,
        ssa=ssa,
        live=live,
        inputs=inputs,
        behavioral_inputs=behavioral_inputs,
        input_mags=input_mags,
        reg_outs=actual_reg_outs,
        out_regs=out_regs,
        out_pcs=out_pcs,
        out_mags=out_mags,
        pc_explicit=pc_explicit,
        live_in_mask=live_in_mask,
        live_out_mask=live_out_mask,
        in_slot_regs=in_slot_regs,
        out_slot_regs=out_slot_regs,
    )


# ---------------------------------------------------------------------------
# Row-outputs path (single-instruction T1)
# ---------------------------------------------------------------------------
#
# For single-instruction T1 chunks, the bijection space is uniform
# (K <= 2: identity + rs1↔rs2 swap), the SSA execution is one numpy
# op per node, and the per-pair distance has a tractable closed form.
# So instead of the per-pair CPU loop pre-computing scalar distances,
# the generator can ship per-row data and the consumer (training
# script) forms the (B, B) target distance matrix as one fused
# tensor op via `pairwise_distance_canonical`.
#
# Math identity: `pairwise_distance_canonical` reproduces
# `behavioral_distance_cached` for single-instruction chunks. All
# four terms — matched_residual, pc_residual, unaligned_in_cost,
# unaligned_out_cost — are computed; the bijection min over K=2×K=2
# is the analog of `partial_bijections` over input register sets of
# size <= 2.
#
# Multi-instruction chunks keep the existing per-pair CPU loop. The
# bijection space and SSA shape are too irregular to vectorize cleanly
# (K up to 8! at MAX_BIJECTION_SIZE=8, variable graph shape per chunk).

@dataclass
class RowOutputs:
    """Per-row data for the canonical broadcast distance.

    Shape conventions:
      K              = 2 (max bijection candidates for single instr)
      max_in         = 2 (max behavioral inputs for single instr)
      n_anchors      = anchor_states.shape[0]
      n_channels     = 2 (rd_value, pc_value)
    """
    row_outputs: np.ndarray   # (K, n_anchors, n_channels) float64 — exec outputs
    n_inputs: int             # 0, 1, or 2 — count of behavioral inputs
    input_mags: np.ndarray    # (K, max_in) float64 — input mags in canonical
                              # position order under each k (zero past n_inputs)
    has_rd: bool              # does this row write a non-x0 register?
    rd_mag: float             # magnitude of rd output (0 if not has_rd)


def precompute_row_outputs(instr, anchor_states, *, pre=None):
    """Per-row data for single-instruction broadcast distance.

    Reuses precompute_chunk's canonical baseline for K=0; runs one
    additional SSA evaluation for K=1 (rs1↔rs2 swap) when the row has
    two distinct behavioral inputs.

    Pass `pre` to skip the internal precompute_chunk call when the
    caller already has it (avoids duplicate SSA work in build pipelines
    that also extract aux targets from the same Precomputed).

    Raises NotImplementedError on memory ops (V1 distance scope).
    """
    chunk = [instr]
    if pre is None:
        pre = precompute_chunk(chunk, anchor_states)

    behavioral = pre.behavioral_inputs
    n_inputs = len(behavioral)
    has_rd = bool(pre.reg_outs)
    rd = pre.reg_outs[0] if has_rd else None
    rd_mag = float(pre.out_mags[rd]) if has_rd else 0.0
    pc_explicit = pre.pc_explicit

    K = 2
    n_anchors = anchor_states.shape[0]
    row_outputs = np.zeros((K, n_anchors, 2), dtype=np.float64)

    pc_value = pre.out_pcs if pc_explicit else (4 * len(chunk))
    rd_value = pre.out_regs[:, rd] if rd is not None else 0.0
    row_outputs[0, :, 0] = rd_value
    row_outputs[0, :, 1] = pc_value

    if n_inputs >= 2:
        swapped = list(pre.inputs)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        state = _canonical_state(anchor_states, swapped, _CANON_POSITIONS)
        out_regs, out_pcs = _eval_ssa_numpy(
            pre.ssa, len(chunk), state, pre.live)
        row_outputs[1, :, 0] = (
            out_regs[:, rd] if rd is not None else 0.0)
        row_outputs[1, :, 1] = (
            out_pcs if pc_explicit else (4 * len(chunk)))
    else:
        row_outputs[1] = row_outputs[0]

    input_mags = np.zeros((K, 2), dtype=np.float64)
    syntactic_mags = [float(pre.input_mags[r]) for r in behavioral]
    for k in range(K):
        if k == 1 and n_inputs < 2:
            input_mags[k] = input_mags[0]
            continue
        for slot, mag in enumerate(syntactic_mags):
            pos = slot if k == 0 else (1 - slot)
            input_mags[k, pos] = mag

    return RowOutputs(
        row_outputs=row_outputs,
        n_inputs=n_inputs,
        input_mags=input_mags,
        has_rd=has_rd,
        rd_mag=rd_mag,
    )


def pairwise_distance_canonical(row_outputs, n_inputs, input_mags,
                                has_rd, rd_mag):
    """Form a (B, B) target distance matrix from packed per-row data.

    All inputs are batched arrays — first dim is B. Works on numpy or
    torch (broadcasting + the small set of element-wise ops common to
    both). For training, pass torch tensors on the model's device and
    the (B, B) result lives on-device, ready to consume in the loss.

    Args:
      row_outputs:  (B, K, n_anchors, n_channels) — exec outputs under
                    each bijection candidate. Channel 0 = rd_value,
                    channel 1 = pc_value (4*chunk_len for ALU
                    fall-through).
      n_inputs:     (B,) int — count of behavioral inputs per row.
      input_mags:   (B, K, max_in) — per-row input magnitudes in
                    canonical position order under each k.
      has_rd:       (B,) bool — does this row write a non-x0 register.
      rd_mag:       (B,) float — magnitude of the rd output (0 if not
                    has_rd, by convention).

    Returns:
      (B, B) float matrix where d[i, j] is the bijection-min sum of
      matched_residual_rd + pc_residual + unaligned_out_cost +
      unaligned_in_cost. Matches behavioral_distance_cached on
      single-instruction chunks.
    """
    is_torch = hasattr(row_outputs, 'log1p') and not isinstance(
        row_outputs, np.ndarray)
    xp = _xp_of(row_outputs)

    # Channel residuals: log1p(log1p(|Δ|)).mean(over anchors).
    # diff shape: (B, B, K, K, n_anchors, n_channels).
    diff = (row_outputs[:, None, :, None]
            - row_outputs[None, :, None, :])
    twice = _abs_log1p_log1p(diff, is_torch)
    chan_resid = twice.mean(dim=-2) if is_torch else twice.mean(axis=-2)
    # → (B, B, K, K, n_channels). Channel 0 = rd, channel 1 = pc.
    rd_resid = chan_resid[..., 0]
    pc_resid = chan_resid[..., 1]

    # rd contribution: matched_residual when both have rd, else
    # unaligned_out_cost = the present row's rd_mag (0 when neither).
    has_rd_a = has_rd[:, None, None, None]    # (B, 1, 1, 1)
    has_rd_b = has_rd[None, :, None, None]    # (1, B, 1, 1)
    rd_mag_a = rd_mag[:, None, None, None]
    rd_mag_b = rd_mag[None, :, None, None]
    both_rd = has_rd_a & has_rd_b
    rd_contrib = xp.where(both_rd, rd_resid, rd_mag_a + rd_mag_b)

    # Unaligned input cost: per canonical position p, if exactly one
    # row has an input at p, add that row's input_mags at p.
    max_in = input_mags.shape[-1]
    if is_torch:
        p_view = xp.arange(max_in, device=row_outputs.device).reshape(
            1, 1, 1, 1, max_in)
    else:
        p_view = xp.arange(max_in).reshape(1, 1, 1, 1, max_in)
    n_a = n_inputs[:, None, None, None, None]    # (B, 1, 1, 1, 1)
    n_b = n_inputs[None, :, None, None, None]    # (1, B, 1, 1, 1)
    has_in_a = p_view < n_a    # (B, 1, 1, 1, max_in)
    has_in_b = p_view < n_b    # (1, B, 1, 1, max_in)
    a_only = has_in_a & ~has_in_b    # (B, B, 1, 1, max_in)
    b_only = has_in_b & ~has_in_a

    mags_a = input_mags[:, None, :, None, :]   # (B, 1, K, 1, max_in)
    mags_b = input_mags[None, :, None, :, :]   # (1, B, 1, K, max_in)
    cost_from_a = (mags_a * a_only).sum(dim=-1) if is_torch else (
        mags_a * a_only).sum(axis=-1)   # (B, B, K, 1)
    cost_from_b = (mags_b * b_only).sum(dim=-1) if is_torch else (
        mags_b * b_only).sum(axis=-1)   # (B, B, 1, K)
    unaligned_in = cost_from_a + cost_from_b   # (B, B, K, K) by broadcast

    total_per_kk = rd_contrib + pc_resid + unaligned_in   # (B, B, K, K)

    if is_torch:
        return total_per_kk.amin(dim=(-1, -2))
    return total_per_kk.min(axis=(-1, -2))


def _xp_of(arr):
    """Return the array library (numpy or torch) for `arr`."""
    if isinstance(arr, np.ndarray):
        return np
    import torch
    return torch


def _abs_log1p_log1p(x, is_torch):
    """log1p(log1p(|x|)) — works on numpy or torch."""
    if is_torch:
        return x.abs().log1p().log1p()
    return np.log1p(np.log1p(np.abs(x).astype(np.float64)))


def behavioral_distance_cached(pre_a, pre_b, anchor_states):
    """Symmetric distance between two Precomputed chunks.

    Implementation note: the underlying directional distance (which
    permutes b's input state under σ but leaves a's at baseline) isn't
    symmetric in a/b. We average both directions to recover symmetry.
    The 2× cost vs a single direction is the price.
    """
    d_ab = _behavioral_distance_directional(pre_a, pre_b, anchor_states)
    d_ba = _behavioral_distance_directional(pre_b, pre_a, anchor_states)
    return 0.5 * (d_ab + d_ba)


def _behavioral_distance_directional(pre_a, pre_b, anchor_states):
    """One-directional distance under best partial bijection σ.

    Both rows are evaluated on σ-aware canonical states: matched
    inputs share canonical positions; unaligned inputs on each side
    use distinct positions. This makes the residual register-name-
    invariant — two GVN-equivalent chunks differing only in register
    naming produce identical canonical outputs.
    """
    if (not pre_a.reg_outs and not pre_b.reg_outs
            and not pre_a.pc_explicit and not pre_b.pc_explicit):
        return 0.0

    inputs_a = pre_a.behavioral_inputs
    inputs_b = pre_b.behavioral_inputs

    # Cap the bijection search. The number of partial bijections grows
    # as P(max(|a|,|b|), min(|a|,|b|)) — factorial in the smaller set.
    # Beyond ~8 inputs the search is both too slow per call and too
    # heavy in memory (a 12-input behavioral_distance call would allocate
    # ~335 GB). For chunks above the cap, fall back to the empty
    # bijection: distance is then an upper bound (no input symmetry
    # discovery), pair signal is noisier, but the call terminates.
    # The cap must check max(|a|,|b|): partial_bijections yields
    # C(max, min) * min! candidates, so a chunk with |a|=8 paired
    # against |b|=16 expands to ~519M bijections (8!*C(16,8)) and
    # hangs the worker. Both sides must be within MAX_BIJECTION_SIZE.
    if (len(inputs_a) <= MAX_BIJECTION_SIZE
            and len(inputs_b) <= MAX_BIJECTION_SIZE):
        bijections = partial_bijections(inputs_a, inputs_b)
    else:
        bijections = iter([((), ())])

    # PC is compared only when at least one chunk explicitly writes it.
    # If both are ALU-only (implicit PC = 4*chunk_len), PC values are
    # length-determined and not meaningful behavioral signal. This
    # preserves cross-syntax equivalence between chunks computing the
    # same register-state transformation with different lengths
    # (e.g., double-ADD ≡ SLLI must reach distance 0).
    compare_pc = pre_a.pc_explicit or pre_b.pc_explicit

    # Materialize bijections + canonical-state pairs in a single pass,
    # skipping any that overflow the canon position pool. We then run
    # SSA evaluation once across all bijections by stacking states
    # along a new leading axis (vectorized across the bijection loop).
    valid_bij = []        # list of (a_keys, b_keys)
    a_states_list = []    # each (n_states, 32)
    b_states_list = []
    for a_keys, b_keys in bijections:
        try:
            a_canon_pos, b_canon_pos = _pair_canon_positions(
                inputs_a, inputs_b, a_keys, b_keys)
        except CanonPositionOverflow:
            continue
        valid_bij.append((a_keys, b_keys))
        a_states_list.append(
            _canonical_state(anchor_states, inputs_a, a_canon_pos))
        b_states_list.append(
            _canonical_state(anchor_states, inputs_b, b_canon_pos))

    if not valid_bij:
        return float('inf')

    n_bij = len(valid_bij)
    n_states = anchor_states.shape[0]

    # Stack and flatten so _eval_ssa_numpy treats (B*n_states) as one
    # big batch of independent input states. SSA eval has no inter-state
    # coupling, so this is mathematically identical to running it
    # n_bij times — just amortizes numpy's per-op dispatch overhead.
    a_states_batch = np.stack(a_states_list, axis=0)  # (n_bij, n_states, 32)
    b_states_batch = np.stack(b_states_list, axis=0)
    a_flat = a_states_batch.reshape(n_bij * n_states, 32)
    b_flat = b_states_batch.reshape(n_bij * n_states, 32)

    a_out_flat, a_pcs_flat = _eval_ssa_numpy(
        pre_a.ssa, len(pre_a.chunk), a_flat, pre_a.live)
    b_out_flat, b_pcs_flat = _eval_ssa_numpy(
        pre_b.ssa, len(pre_b.chunk), b_flat, pre_b.live)

    a_out = a_out_flat.reshape(n_bij, n_states, 32)
    b_out = b_out_flat.reshape(n_bij, n_states, 32)
    a_pcs = a_pcs_flat.reshape(n_bij, n_states)
    b_pcs = b_pcs_flat.reshape(n_bij, n_states)

    # Vectorized cost matrix across bijections. Hungarian itself stays
    # serial per bijection (scipy has no batched solver and the cost
    # matrices are tiny — ≤8x8).
    #
    # fp32 throughout: int32 register values are exact in fp32 up to
    # 2^24 magnitude; above that ULP grows to ~256 at 2^31, but the
    # subsequent loglog compresses that entirely (derivative ~2.3e-11
    # at x=2e9). Halves memory vs fp64.
    if pre_a.reg_outs and pre_b.reg_outs:
        ra_vals = np.ascontiguousarray(
            a_out[:, :, pre_a.reg_outs]).astype(np.float32, copy=False)
        rb_vals = np.ascontiguousarray(
            b_out[:, :, pre_b.reg_outs]).astype(np.float32, copy=False)
        ra_b = ra_vals[:, :, :, None]
        rb_b = rb_vals[:, :, None, :]
        loglog_diff = ne.evaluate('log1p(log1p(abs(ra_b - rb_b)))')
        cost_batch = loglog_diff.mean(axis=1)
    else:
        cost_batch = None

    # PC residual vectorized: one scalar per bijection. Same-shape sub
    # (no broadcast), so plain numpy is cheap; not worth a torch hop.
    if compare_pc:
        a_pcs_f = a_pcs.astype(np.float32, copy=False)
        b_pcs_f = b_pcs.astype(np.float32, copy=False)
        diff = np.abs(a_pcs_f - b_pcs_f)
        pc_resid_batch = np.log1p(np.log1p(diff)).mean(axis=1)
    else:
        pc_resid_batch = np.zeros(n_bij, dtype=np.float64)

    best = float('inf')
    for i, (a_keys, b_keys) in enumerate(valid_bij):
        if cost_batch is not None:
            cost = cost_batch[i]
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_residual = float(cost[row_ind, col_ind].sum())
            matched_a = {pre_a.reg_outs[r] for r in row_ind}
            matched_b = {pre_b.reg_outs[c] for c in col_ind}
        else:
            matched_residual = 0.0
            matched_a = set()
            matched_b = set()

        unaligned_out_cost = (
            sum(pre_a.out_mags[r]
                for r in pre_a.reg_outs if r not in matched_a)
            + sum(pre_b.out_mags[r]
                  for r in pre_b.reg_outs if r not in matched_b)
        )
        unaligned_a_inputs = set(inputs_a) - set(a_keys)
        unaligned_b_inputs = set(inputs_b) - set(b_keys)
        unaligned_in_cost = (
            sum(pre_a.input_mags[r] for r in unaligned_a_inputs)
            + sum(pre_b.input_mags[r] for r in unaligned_b_inputs)
        )

        total = (matched_residual + unaligned_out_cost
                 + unaligned_in_cost + float(pc_resid_batch[i]))
        if total < best:
            best = total
        if best <= 0.0:
            break

    return best


def behavioral_distance(chunk_a, chunk_b, n_states=16, seed=0):
    """Compute the behavioral distance between two chunks.

    Returns >= 0. 0 iff behaviorally equivalent on the sampled anchor
    states (probabilistic). For batched use across many partner pairs,
    prefer precompute_chunk + behavioral_distance_cached so anchor execution
    + sensitivity analysis is amortized.

    Raises NotImplementedError if either chunk has memory ops.
    """
    anchor_states = make_anchor_states(n_states, seed)
    pre_a = precompute_chunk(chunk_a, anchor_states)
    pre_b = precompute_chunk(chunk_b, anchor_states)
    return behavioral_distance_cached(pre_a, pre_b, anchor_states)
