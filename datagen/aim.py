"""Anchor states aimed at a chunk's inflection points.

A generic structured draw (`draw_anchor_states`) reaches the inputs where an
output changes discontinuously only by chance: `SLTI x,600` flips between
x = 599 and x = 600, and an add carries through h bits only where its result
crosses 2^h. Value prediction cannot learn a boundary it never sees, so a
fraction of each row's states is rewritten to straddle the chunk's own
boundaries, each as a PAIR of states holding the two values either side of
it at the node where the boundary lies.

A boundary on a computed value is moved to an input register by walking the
dataflow back through operations that can be solved for one operand (ADD,
ADDI, SUB, XOR, XORI), holding the other operand at its value in that state.
Every aimed state is executed afterwards and kept only if the node takes the
intended value there, so a walk that is wrong costs a generic state, never a
wrong label.

The targets are a function of the chunk's structure alone (instruction order,
opcodes, immediates, and which reads are leaves), so a register rename aims
the same draw columns with the same values.
"""

import numpy as np

from .compare import _input_order

_M = (1 << 32) - 1
_SIGN = 1 << 31

_OPS = {
    'ADD': lambda a, b: (a + b) & _M, 'ADDI': lambda a, b: (a + b) & _M,
    'SUB': lambda a, b: (a - b) & _M,
    'XOR': lambda a, b: a ^ b, 'XORI': lambda a, b: a ^ b,
    'OR': lambda a, b: a | b, 'ORI': lambda a, b: a | b,
    'AND': lambda a, b: a & b, 'ANDI': lambda a, b: a & b,
    'SLL': lambda a, b: (a << (b & 31)) & _M, 'SLLI': lambda a, b: (a << (b & 31)) & _M,
    'SRL': lambda a, b: a >> (b & 31), 'SRLI': lambda a, b: a >> (b & 31),
    'SRA': lambda a, b: (_signed(a) >> (b & 31)) & _M,
    'SRAI': lambda a, b: (_signed(a) >> (b & 31)) & _M,
    'SLT': lambda a, b: (_signed(a) < _signed(b)).astype(np.int64),
    'SLTI': lambda a, b: (_signed(a) < _signed(b)).astype(np.int64),
    'SLTU': lambda a, b: (a < b).astype(np.int64),
    'SLTIU': lambda a, b: (a < b).astype(np.int64),
}
_IMM_OPS = {'ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI', 'SLTI', 'SLTIU'}
_COMPARE_IMM = {'SLTI', 'SLTIU'}
_COMPARE_REG = {'SLT', 'SLTU'}
_SIGNED = {'SLT', 'SLTI'}
_CARRY = {'ADD', 'ADDI', 'SUB'}

# Carries are aimed at a result crossing 2^h for h drawn from this range: the
# generic draw already crosses low powers often (small and boundary values).
_CARRY_BITS = (4, 33)

# A comparison's second pair straddles its threshold at 2^u, u uniform in this
# range: up to the 12-bit immediate's span, where a readout that has only seen
# the threshold step misplaces it by hundreds (EXPERIMENT_LOG, phase 16).
_SCALE_BITS = (1, 11)


def _signed(a):
    return a - ((a & _SIGN) << 1)


def _graph(chunk):
    """Per instruction: (op, operand nodes, immediate or None). A node is
    ('L', reg) for a value read from outside the chunk, ('I', j) for
    instruction j's result, ('Z',) for x0. None when the chunk holds an
    operation outside the ALU set."""
    last, g = {}, []
    for ins in chunk:
        op, a = ins.opcode, ins.args
        if op not in _OPS:
            return None
        node = lambda r: ('Z',) if r == 0 else last.get(r, ('L', r))
        if op in _IMM_OPS:
            g.append((op, (node(a[1]),), a[2]))
        else:
            g.append((op, (node(a[1]), node(a[2])), None))
        if a[0]:
            last[a[0]] = ('I', len(g) - 1)
    return g, last


def _live(g, last):
    """Instructions whose result reaches a final write."""
    live = set(j for _, j in (n for n in last.values()))
    for j in range(len(g) - 1, -1, -1):
        if j in live:
            for n in g[j][1]:
                if n[0] == 'I':
                    live.add(n[1])
    return live


def _evaluate(g, regs):
    """Every instruction's result over the states, (K, n) int64 in [0, 2^32),
    from register files `regs` (n, 32) int64 in [0, 2^32)."""
    vals = []
    get = lambda n: (np.zeros(regs.shape[0], np.int64) if n[0] == 'Z'
                     else regs[:, n[1]] if n[0] == 'L' else vals[n[1]])
    for op, ops, imm in g:
        a = get(ops[0])
        b = np.full_like(a, imm & _M) if imm is not None else get(ops[1])
        vals.append(_OPS[op](a, b))
    return vals


def _leaves(g, n, memo):
    """The external registers node n depends on."""
    if n[0] == 'L':
        return frozenset((n[1],))
    if n[0] == 'Z':
        return frozenset()
    if n[1] not in memo:
        op, ops, _ = g[n[1]]
        memo[n[1]] = frozenset().union(*(_leaves(g, m, memo) for m in ops))
    return memo[n[1]]


def _solve(g, n, v, s, vals, regs, pos, memo, order):
    """{reg: value} placing value v at node n in state s, or None. An
    operation is solved through the operand whose inputs the other operand
    does not share, the other held at its value in state s."""
    if n[0] == 'L':
        return {n[1]: v} if n[1] in pos else None
    if n[0] == 'Z':
        return None
    op, ops, imm = g[n[1]]
    val = lambda m: (0 if m[0] == 'Z' else int(regs[s, m[1]]) if m[0] == 'L'
                     else int(vals[m[1]][s]))
    if imm is not None:
        if op == 'ADDI':
            return _solve(g, ops[0], (v - imm) & _M, s, vals, regs, pos, memo, order)
        if op == 'XORI':
            return _solve(g, ops[0], v ^ (imm & _M), s, vals, regs, pos, memo, order)
        return None
    if op not in ('ADD', 'SUB', 'XOR'):
        return None
    for side in order:
        mine, other = ops[side], ops[1 - side]
        if _leaves(g, mine, memo) & _leaves(g, other, memo):
            continue
        o = val(other)
        if op == 'ADD':
            want = v - o
        elif op == 'XOR':
            want = v ^ o
        else:
            want = v + o if side == 0 else o - v
        got = _solve(g, mine, want & _M, s, vals, regs, pos, memo, order)
        if got is not None:
            return got
    return None


def boundary_targets(g, live, rng):
    """The chunk's boundaries, most important first: per live instruction, a
    node and the two values either side of where an output changes. A
    relational target (`SLT a,b`) is resolved per state, so it is returned
    as ('rel', j, (d_below, d_at)): the two states put a - b at those
    offsets.

    A comparison is aimed twice: at the threshold exactly, and straddling it
    at a distance drawn log-uniformly up to the immediate range, so the
    readout is trained to place the threshold at every scale and not only
    at the single step where the output flips."""
    exact, scaled, ext, carry = [], [], [], []
    for j, (op, ops, imm) in enumerate(g):
        if j not in live:
            continue
        if op in _COMPARE_IMM | _COMPARE_REG:
            d = int(2 ** rng.uniform(*_SCALE_BITS))
        if op in _COMPARE_IMM:
            k = imm & _M
            if op == 'SLTI' or k:          # SLTIU x,0 is constant
                exact.append((ops[0], ((k - 1) & _M, k)))
                scaled.append((ops[0], ((k - d) & _M, (k - 1 + d) & _M)))
        elif op in _COMPARE_REG:
            exact.append(('rel', j, (-1, 0)))
            scaled.append(('rel', j, (-d, d - 1)))
        if op in _COMPARE_IMM | _COMPARE_REG:
            # A computed operand also flips the comparison where it wraps.
            wrap = ((_SIGN - 1, _SIGN) if op in _SIGNED else (_M, 0))
            for n in ops:
                if n[0] == 'I':
                    ext.append((n, wrap))
        elif op in _CARRY:
            h = int(rng.integers(*_CARRY_BITS))
            carry.append((('I', j), (((1 << h) - 1) & _M, (1 << h) & _M)))
    return exact + scaled + ext + carry


def aim_anchor_states(chunk, anchors, rng, fraction):
    """`anchors` (n, 32) int32, a structured draw, with up to `fraction` of
    its states rewritten in pairs to straddle the chunk's boundaries. Returns
    a new array; the draw is returned unchanged when nothing can be aimed."""
    built = _graph(chunk)
    n = anchors.shape[0]
    pairs = int(n * fraction) // 2
    if built is None or pairs == 0:
        return anchors
    g, last = built
    targets = boundary_targets(g, _live(g, last), rng)
    if not targets:
        return anchors
    inputs = _input_order(chunk)
    pos = {r: i + 1 for i, r in enumerate(inputs)}
    draw = anchors.astype(np.int64) & _M
    regs = draw.copy()
    for r, c in pos.items():
        regs[:, r] = draw[:, c]
    regs[:, 0] = 0
    vals = _evaluate(g, regs)
    memo = {}
    states = rng.permutation(n)
    aimed, used = [], 0
    for t in targets:
        if used == pairs:
            break
        s_pair = states[2 * used:2 * used + 2]
        order = (0, 1) if rng.random() < 0.5 else (1, 0)
        sols = []
        for s, side in zip(s_pair, (0, 1)):
            if t[0] == 'rel':
                sol = _solve_rel(g, t[1], t[2], side, s, vals, regs, pos, memo, order)
            else:
                node, pair = t
                sol = _solve(g, node, pair[side], s, vals, regs, pos, memo, order)
                sol = sol and (sol, node, pair[side])
            if sol is None:
                break
            sols.append((s, sol))
        if len(sols) < 2:
            continue
        for s, (assign, node, v) in sols:
            for r, x in assign.items():
                draw[s, pos[r]] = x
            aimed.append((s, node, v))
        used += 1
    if not aimed:
        return anchors
    # Execute the rewritten states and keep only those that hit their node.
    regs2 = draw.copy()
    for r, c in pos.items():
        regs2[:, r] = draw[:, c]
    regs2[:, 0] = 0
    vals2 = _evaluate(g, regs2)
    out = anchors.copy()
    for s, node, v in aimed:
        got = (0 if node[0] == 'Z' else regs2[s, node[1]] if node[0] == 'L'
               else vals2[node[1]][s])
        if got == v:
            out[s] = (draw[s] & _M).astype(np.uint32).view(np.int32)
    out[:, 0] = 0
    return out


def _solve_rel(g, j, offsets, side, s, vals, regs, pos, memo, order):
    """`a < b`: put a - b at offsets[side] in state s, solving whichever
    operand allows it with the other held. Returns (assignment, node, value)
    for the solved operand, or None."""
    a, b = g[j][1]
    if _leaves(g, a, memo) & _leaves(g, b, memo):
        return None
    val = lambda m: (0 if m[0] == 'Z' else int(regs[s, m[1]]) if m[0] == 'L'
                     else int(vals[m[1]][s]))
    d = offsets[side]
    for pick in order:
        node, v = (a, (val(b) + d) & _M) if pick == 0 else (b, (val(a) - d) & _M)
        sol = _solve(g, node, v, s, vals, regs, pos, memo, order)
        if sol is not None:
            return sol, node, v
    return None
