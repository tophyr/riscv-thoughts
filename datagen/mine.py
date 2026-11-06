"""Equivalent pairs found by search: a random two-instruction chain and a
single instruction with the same behaviour.

For a chain whose instructions both write one register t, every single ALU
instruction writing t from the chain's inputs is tried — each R-type operation
on every pair of inputs, each I-type operation with every immediate — and one
whose t equals the chain's at every search state, then proved equal by z3
(`datagen.prove`), is the partner. No equivalence is named in advance, so the
pairs are not limited to the edits a hand-written rewrite list
(`datagen.generate._REWRITES`) knows.
"""
import numpy as np

from emulator import Instruction

from .compare import _eval_ssa_numpy, draw_anchor_states, live_input_regs, live_nodes, to_ssa

SEARCH_SEED = 0x5EA2
N_SEARCH = 256
_STATES = []
_M = 0xFFFFFFFF

R_OPS = ('ADD', 'SUB', 'SLL', 'SLT', 'SLTU', 'XOR', 'SRL', 'SRA', 'OR', 'AND')
I_OPS = ('ADDI', 'SLTI', 'SLTIU', 'XORI', 'ORI', 'ANDI')
SH_OPS = ('SLLI', 'SRLI', 'SRAI')


def search_states():
    """(N_SEARCH, 32) int32 structured states, x0 = 0, drawn once."""
    if not _STATES:
        _STATES.append(draw_anchor_states(np.random.default_rng(SEARCH_SEED), N_SEARCH))
    return _STATES[0]


_I_BASE = {'ADDI': 'ADD', 'SLTI': 'SLT', 'SLTIU': 'SLTU', 'XORI': 'XOR',
           'ORI': 'OR', 'ANDI': 'AND', 'SLLI': 'SLL', 'SRLI': 'SRL', 'SRAI': 'SRA'}


_FIRST = 8          # states every candidate is tried at before the rest


def _matches(values, y):
    """Mask over a candidate axis: values (..., N) equal y (N,) everywhere,
    checked on the first _FIRST states before the rest."""
    ok = (values[..., :_FIRST] == y[:_FIRST]).all(-1)
    if ok.any():
        ok[ok] = (np.asarray(values)[ok] == y).all(-1)
    return ok


def _alu32(op, a, b):
    """The RV32 ALU on uint32 arrays (broadcasting): wrapping arithmetic native,
    signed views only where the operation is signed."""
    if op == 'ADD':
        return a + b
    if op == 'SUB':
        return a - b
    if op == 'XOR':
        return a ^ b
    if op == 'OR':
        return a | b
    if op == 'AND':
        return a & b
    if op == 'SLL':
        return a << (b & 31)
    if op == 'SRL':
        return a >> (b & 31)
    if op == 'SRA':
        return (a.view(np.int32) >> (b & 31).astype(np.int32)).view(np.uint32)
    if op == 'SLT':
        return (a.view(np.int32) < b.view(np.int32)).astype(np.uint32)
    if op == 'SLTU':
        return (a < b).astype(np.uint32)
    raise ValueError(op)


_IMM_U32 = (np.arange(-2048, 2048, dtype=np.int64) & _M).astype(np.uint32)
_SHAMT32 = np.arange(32, dtype=np.uint32)
_IMM_BLOCK = 512


def single_equivalents(chunk, t, states=None):
    """Every single ALU instruction writing t whose value of t equals the
    chunk's at every search state, reading only registers the chunk reads
    from outside (and x0): each operation family tried in one array operation
    on uint32 values, rejecting on the first _FIRST states."""
    s = search_states() if states is None else states
    ssa = to_ssa(chunk)
    live = live_nodes(ssa)
    y = _eval_ssa_numpy(ssa, len(chunk), s, live)[0][:, t].view(np.uint32)
    srcs = [0] + sorted(set(live_input_regs(ssa, live)) - {0})
    V = np.ascontiguousarray(s[:, srcs].T).view(np.uint32)            # (n, N)
    n, N = V.shape
    found = []
    A, B = V[:, None, :], V[None, :, :]
    for op in R_OPS:
        ok = _matches(np.broadcast_to(_alu32(op, A, B), (n, n, N)), y)
        found += [Instruction(op, t, srcs[a], srcs[b]) for a, b in zip(*np.nonzero(ok))]
    head_v, y_head = V[:, None, :_FIRST], y[:_FIRST]
    for op in I_OPS:
        base = _I_BASE[op]
        hits_a, hits_k = [], []
        # Immediates in blocks, so each step's arrays stay inside a core's L2.
        for lo in range(0, _IMM_U32.shape[0], _IMM_BLOCK):
            ai, ki = np.nonzero((_alu32(base, head_v, _IMM_U32[None, lo:lo + _IMM_BLOCK, None])
                                 == y_head).all(-1))
            hits_a.append(ai); hits_k.append(ki + lo)
        ai, ki = np.concatenate(hits_a), np.concatenate(hits_k)
        if len(ai):
            order = np.lexsort((ki, ai))                 # (input, immediate) order
            ai, ki = ai[order], ki[order]
            keep = (_alu32(base, V[ai], _IMM_U32[ki][:, None]) == y).all(-1)
            found += [Instruction(op, t, srcs[a], int(k) - 2048)
                      for a, k in zip(ai[keep], ki[keep])]
    for op in SH_OPS:
        ok = _matches(_alu32(_I_BASE[op], V[:, None, :], _SHAMT32[None, :, None]), y)
        found += [Instruction(op, t, srcs[a], int(k)) for a, k in zip(*np.nonzero(ok))]
    return found


_CHAIN_OPS = R_OPS + I_OPS + SH_OPS


# The fraction of chains with a constant t (common: a comparison of a
# comparison) that are kept, so the pairs are not mostly constants.
CONSTANT_KEEP = 0.15


def sample_mined_pair(rng, exclude=frozenset()):
    """A fresh equivalent pair found by search and proved: a two-instruction
    chain writing one register t twice, the second reading the first's
    result, against a single instruction computing the same t from the
    chain's inputs (`single_equivalents`), proved equal on every register
    file (`datagen.prove`). Opcodes in `exclude` appear in neither. A chain
    whose t is constant over the search states — common, a comparison of a
    comparison — is kept with probability CONSTANT_KEEP. Returns ([i1, i2], [single])."""
    from emulator import Instruction
    from .generate import _make_instruction
    from .prove import equivalent
    ops = [o for o in _CHAIN_OPS if o not in exclude]
    s = search_states()
    while True:
        t = int(rng.integers(1, 32))
        a, b, c = (int(x) for x in rng.integers(0, 32, 3))
        i1 = _make_instruction(rng, str(rng.choice(ops)), rd=t, rs1=a, rs2=b)
        op2 = str(rng.choice(ops))
        i2 = _make_instruction(rng, op2, rd=t, rs1=t, rs2=c)
        if op2 in R_OPS and rng.random() < 0.5:
            i2 = Instruction(op2, t, c, t)
        chain = [i1, i2]
        ssa = to_ssa(chain)
        live = live_nodes(ssa)
        # A single instruction reads at most two registers, so a chain reading
        # three matches one only where it ignores an input; it is skipped.
        if len(set(live_input_regs(ssa, live)) - {0}) > 2:
            continue
        y = _eval_ssa_numpy(ssa, 2, s, live)[0][:, t]
        if (y == y[0]).all() and rng.random() > CONSTANT_KEEP:
            continue
        cands = [x for x in single_equivalents(chain, t) if x.opcode not in exclude]
        for k in rng.permutation(len(cands))[:3]:
            if equivalent(chain, [cands[int(k)]]) is True:
                return chain, [cands[int(k)]]


POOL_SIZE = 64
_POOLS = {}


def sample_pooled_pair(rng, exclude=frozenset(), reuse=1):
    """A searched-and-proved pair (`sample_mined_pair`) drawn from a rolling
    per-process pool of POOL_SIZE, taken in turn and renamed afresh (x0
    fixed); a pair drawn `reuse` times is replaced by a newly mined one, so
    mining costs 1/`reuse` per draw. Taking the pool in turn keeps any
    POOL_SIZE consecutive draws distinct."""
    from .generate import random_perm, relabel
    key = (exclude, reuse)
    pool = _POOLS.get(key)
    if pool is None:
        pool = _POOLS[key] = {'pairs': [[*sample_mined_pair(rng, exclude), reuse]
                                        for _ in range(POOL_SIZE)], 'next': 0}
    i = pool['next']
    pool['next'] = (i + 1) % POOL_SIZE
    pool = pool['pairs']
    a, b, left = pool[i]
    if left <= 1:
        pool[i] = [*sample_mined_pair(rng, exclude), reuse]
    else:
        pool[i][2] = left - 1
    p = random_perm(rng)
    return relabel(a, p), relabel(b, p)

