"""Structural signature of a chunk: the register function it computes, written
down without register names and without the order its instructions happen to
be listed in.

Two chunks are equivalent when their externally visible register inputs and
outputs are identical; the essence must then be the same point. The first-read
canonical frame numbers inputs by textual order and so separates reordered,
renumbered and dead-code-bearing equivalents (RISCV_DESIGN, "GVN"). This
module is the frame that replaces it, in oracle form:

  - the chunk is a dataflow graph: leaves (registers read before written),
    operation nodes, edges by operand role to the most recent writer; an
    output is a register's final writer;
  - dead instructions (overwritten before being read) are pruned by a backward
    pass from the final writers, so their reads leave no trace;
  - every node is coloured by role-aware Weisfeiler–Lehman refinement over the
    live graph, from the WIRING alone — no opcode, no immediate — with a
    commutative operation's two roles treated as one;
  - an output's DESCRIPTION is the canonical serialisation of its sub-graph,
    with opcodes and immediates at the nodes and leaves replaced by their
    colours; the chunk's SIGNATURE is the multiset of its outputs' descriptions.

Colouring from the wiring alone is deliberate: the encoder computes the same
colouring as its leaf seed from the wiring the lower tier emits
(`RegisterStateMachine._prepass`), with every role ordered — it is told
nothing about which operations commute, so a commutative swap is the
op-cell's to learn. `COMMUTATIVE` below serves the oracle only: the signature
LABELS a swap as equivalent, and `output_inputs` orders a symmetric pair
consistently. Opcodes reach the encoder only as content, so they belong in the
description (the op-cell's result), not in the seed. `test_signature.py` holds
the agreement with the emulator.

The signature is an equivalence LABEL — two chunks with equal signatures must
collapse — and never a training target: the classes it does not merge
(identity spellings, cancellations, immediate folds) are the ones the op-cell
must learn to fold, and a target would weld them apart.

Scope: register-writing ALU chunks. Memory and control-flow instructions raise
NotImplementedError, as `precompute_chunk` does.
"""
import hashlib

from emulator import R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE

# Operations whose two register operands may be exchanged without changing
# the result. A leaf feeding either role of one gets the same colour.
COMMUTATIVE = frozenset({'ADD', 'AND', 'OR', 'XOR'})

LEAF = 'L'

# WL refinement rounds. A node's colour after r rounds reflects the graph up
# to r edges away, so this must cover the deepest chunk the rules can build
# (`length_cap` ≤ 8 in every config); it is fixed rather than per-chunk so that
# identical live graphs in chunks of different lengths colour identically.
WL_ROUNDS = 8


def _reads(ins):
    """Operand registers by role (None for an immediate role, 0 for x0), in
    the same roles `compressor.model.instruction_wiring` routes them to."""
    op = ins.opcode
    if op in R_TYPE:
        return [ins.args[1], ins.args[2]]
    if op in I_TYPE or op == 'JALR':
        return [ins.args[1], None]
    if op in B_TYPE:
        return [ins.args[0], ins.args[1]]
    if op in ('LUI', 'AUIPC', 'JAL'):
        return [None, None]
    if op in LOAD_TYPE or op in STORE_TYPE:
        raise NotImplementedError(f'signature: memory op {op} is out of scope')
    raise NotImplementedError(f'signature: unknown opcode {op}')


def _dest(ins):
    """The GP register written, or None. A branch's PC effect is not an
    output (PC is not an independent output of a chunk)."""
    op = ins.opcode
    if op in B_TYPE or op in STORE_TYPE:
        return None
    return ins.args[0]


def _immediate(ins):
    return None if ins.opcode in R_TYPE else ins.args[2]


def dataflow(chunk):
    """The chunk's dataflow graph.

    Returns (edges, outputs): `edges[('op', i)]` is the list of children by
    role — `('leaf', reg)`, `('op', j)`, or None for x0 / an immediate — and
    `outputs[reg]` is the node that finally writes `reg`."""
    writer, edges = {}, {}
    for i, ins in enumerate(chunk):
        kids = []
        for r in _reads(ins):
            if r is None or r == 0:
                kids.append(None)
            elif r in writer:
                kids.append(writer[r])
            else:
                kids.append(('leaf', r))
        edges[('op', i)] = kids
        rd = _dest(ins)
        if rd:
            writer[rd] = ('op', i)
    return edges, dict(writer)


def live_nodes(edges, outputs):
    """Nodes reachable backwards from a final writer. An instruction whose
    result is overwritten before anyone reads it is absent, and so are the
    leaves only it read."""
    live, stack = set(), list(outputs.values())
    while stack:
        n = stack.pop()
        if n in live:
            continue
        live.add(n)
        if n[0] == 'op':
            stack.extend(k for k in edges[n] if k is not None)
    return live


def _symmetric(chunk, node, symmetric_ops):
    return chunk[node[1]].opcode in symmetric_ops


def colours(chunk, edges, live, symmetric_ops=COMMUTATIVE, rounds=None):
    """Role-aware WL colours of the live nodes, from the wiring alone.

    Each round a node's new colour is its old colour, the colours of what it
    reads (by role; as a multiset for a symmetric operation), and the colours
    of what reads it, tagged with the role it fills there. The round count is
    FIXED (`WL_ROUNDS`), not derived from the chunk: a colour after r rounds
    nests r levels of context, so two chunks compared at different depths would
    never share a colour even on identical live graphs."""
    rounds = WL_ROUNDS if rounds is None else rounds
    nodes = sorted(live)
    col = {n: LEAF if n[0] == 'leaf' else 'OP' for n in nodes}
    readers = {n: [] for n in nodes}
    for p, kids in edges.items():
        if p not in live:
            continue
        sym = _symmetric(chunk, p, symmetric_ops)
        for role, k in enumerate(kids):
            if k is not None:
                readers[k].append((p, 's' if sym else str(role)))
    for _ in range(rounds):
        new = {}
        for n in nodes:
            down = ''
            if n[0] == 'op':
                kids = [col[k] if k is not None else 'x0' for k in edges[n]]
                if _symmetric(chunk, n, symmetric_ops):
                    kids = sorted(kids)
                down = ','.join(kids)
            up = ','.join(sorted(f'{col[p]}@{role}' for p, role in readers[n]))
            new[n] = _digest(f'{col[n]}|{down}|{up}')
        col = new
    return col


def _digest(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def describe(node, chunk, edges, col, symmetric_ops=COMMUTATIVE):
    """Canonical serialisation of the sub-graph rooted at `node`: opcode and
    immediate at each operation, children by role (sorted for a symmetric
    operation), leaves as their colour."""
    if node is None:
        return 'x0'
    if node[0] == 'leaf':
        return LEAF + col[node]
    ins = chunk[node[1]]
    kids = [describe(k, chunk, edges, col, symmetric_ops) for k in edges[node]]
    if ins.opcode in symmetric_ops:
        kids = sorted(kids)
    return f'{ins.opcode}:{_immediate(ins)}({",".join(kids)})'


def signature(chunk, symmetric_ops=COMMUTATIVE):
    """The multiset of the chunk's output descriptions, as a sorted tuple."""
    edges, outputs = dataflow(chunk)
    live = live_nodes(edges, outputs)
    col = colours(chunk, edges, live, symmetric_ops)
    return tuple(sorted(describe(n, chunk, edges, col, symmetric_ops)
                        for n in outputs.values()))


def output_inputs(chunk, symmetric_ops=COMMUTATIVE):
    """Per output register, the external input registers its value depends
    on, in canonical order: by leaf colour, then by register number.

    This is the order an evaluator gathers a written slot's input values in.
    Two leaves of one colour are symmetric — exchanging them cannot change the
    output — so the register tiebreak decides nothing about the value, only
    which of two equal roles a register is listed under."""
    edges, outputs = dataflow(chunk)
    live = live_nodes(edges, outputs)
    col = colours(chunk, edges, live, symmetric_ops)
    out = {}
    for reg, node in outputs.items():
        leaves, stack, seen = set(), [node], set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n[0] == 'leaf':
                leaves.add(n[1])
            else:
                stack.extend(k for k in edges[n] if k is not None)
        out[reg] = sorted(leaves, key=lambda r: (col[('leaf', r)], r))
    return out
