"""SSA analysis + canonical-baseline execution over RV32I chunks.

Owns the Precomputed/AuxPayload schema (SSA + DCE + anchor execution +
input-sensitivity magnitudes) — the canonical out_regs / binding targets
consumed by training (`precompute_chunk`).

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

from dataclasses import dataclass, field

import numpy as np

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

# An input register's behavioral magnitude must exceed this to be
# treated as behaviorally relevant. Set just above floating-point noise
# so that "no observable effect on any output" reliably filters out.
BEHAVIORAL_RELEVANCE_THRESHOLD = 1e-9

N_REGS = 32

# Canonical anchor-position list for register input sourcing during SSA
# evaluation. A chunk's behavioral inputs (in syntactic first-read order) are
# sourced from these positions instead of from the row's actual register
# positions, so out_regs / magnitudes don't leak register names — the
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
    """Return SSA node ids reachable backward from outputs. Dead nodes
    don't contribute to the chunk's externally-observable effect, so
    distance must ignore them."""
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

def make_anchor_states(n_states, seed):
    """Generate the shared anchor states (random register files) used as
    test inputs for value-prediction / canonical execution. gen and trainer
    must reconstruct the same states from matching (n_states, seed)."""
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


# ---------------------------------------------------------------------------
# Magnitude analysis (input perturbation + output diff)
# ---------------------------------------------------------------------------

def _loglog(x):
    """log1p(log1p(|x|)) on a numpy array."""
    return np.log1p(np.log1p(np.abs(x).astype(np.float64)))


def _has_memory_ops(chunk):
    return any(instr.opcode in LOAD_TYPE or instr.opcode in STORE_TYPE
               for instr in chunk)


def _output_magnitudes(out_regs, in_states, output_regs):
    """For each output register, mean loglog'd |delta| across states."""
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


# ---------------------------------------------------------------------------
# Chunk analysis — Precomputed schema + aux targets
# ---------------------------------------------------------------------------

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

    Builds a `Precomputed` from the SSA + numpy evaluator. Owns the
    supervision schema (live masks, slot registers).
    Raises NotImplementedError on memory ops (V1 distance scope).
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


