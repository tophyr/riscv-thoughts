"""The reference behavioral distance d* between two chunks (RISCV_DESIGN,
"Collapse", "The ideal distance").

A chunk's behaviour is its map on the register file, up to a rename. d*(A, B)
is the minimum over renames π (x0 fixed) of the root-sum-square over registers
of the per-register distance between A's register file after s and B's after
π·s, each per-register distance the mean over states of the doubly-logarithmic
wrap-around difference, bounded by the largest difference the two value
columns allow (bits constant across the states, so the bound is a function of
behaviour, never of the spelling).

Exact in the rename and slow: it enumerates every injective map of B's
registers into A's, the rest landing on registers A does not touch. The states
are a structured draw with each state's registers shuffled, so the family
treats every register alike and fixing A's placement loses nothing. It is the
oracle the trained target is measured against, not a training path.
"""
import itertools

import numpy as np

from .compare import (N_REGS, PC_REG, MEM_REG, _eval_ssa_numpy, draw_anchor_states,
                      live_input_regs, live_nodes, to_ssa)
from .knownbits import column_distance, observed_bits

IDEAL_SEED = 0x1DEA
N_IDEAL_STATES = 256
_STATES = {}


def ideal_states(n=N_IDEAL_STATES):
    """(n, 32) int32 register files, x0 = 0: a structured draw whose registers
    1..31 are shuffled independently per state, so no register is special."""
    if n not in _STATES:
        rng = np.random.default_rng(IDEAL_SEED)
        s = draw_anchor_states(rng, n)
        for i in range(n):
            s[i, 1:] = s[i, 1 + rng.permutation(N_REGS - 1)]
        _STATES[n] = s
    return _STATES[n]


class Behaviour:
    """One chunk's register-file map, ready to be compared."""

    def __init__(self, chunk):
        self.chunk = list(chunk)
        self.ssa = to_ssa(self.chunk) if self.chunk else None
        if self.ssa is None:
            self.live, self.regs = None, []
        else:
            self.live = live_nodes(self.ssa)
            written = {r for r in self.ssa.output_versions if r not in (PC_REG, MEM_REG, 0)}
            read = set(live_input_regs(self.ssa, self.live)) - {0, PC_REG, MEM_REG}
            self.regs = sorted(written | read)

    def run(self, states):
        if self.ssa is None:
            return states.copy()
        return _eval_ssa_numpy(self.ssa, len(self.chunk), states, self.live)[0]


def _register_distance(a, b):
    """The per-register distance between two (N,) value columns
    (`datagen.knownbits.column_distance`)."""
    if np.array_equal(a, b):
        return 0.0
    return column_distance(a, observed_bits(a), b, observed_bits(b))


def _renames(regs_b, regs_a):
    """Every injective map of regs_b into regs_a ∪ fresh registers, as a full
    permutation of 0..31 with x0 fixed: perm[b] = π(b)."""
    fresh = [r for r in range(1, N_REGS) if r not in regs_a]
    nb = len(regs_b)
    for k in range(min(nb, len(regs_a)) + 1):
        for src in itertools.combinations(range(nb), k):
            for dst in itertools.permutations(regs_a, k):
                m = dict(zip((regs_b[i] for i in src), dst))
                rest = [regs_b[i] for i in range(nb) if i not in src]
                m.update(zip(rest, fresh))
                used = set(m.values())
                free_dst = [r for r in range(1, N_REGS) if r not in used]
                free_src = [r for r in range(1, N_REGS) if r not in m]
                m.update(zip(free_src, free_dst))
                perm = np.zeros(N_REGS, dtype=np.int64)
                for s_, d_ in m.items():
                    perm[s_] = d_
                yield perm


def ideal_distance(a, b, states=None, return_rename=False):
    """d*(A, B), the root-sum-square over registers of per-register distances,
    minimised over renames of B. `a`, `b`: chunks or Behaviours."""
    a = a if isinstance(a, Behaviour) else Behaviour(a)
    b = b if isinstance(b, Behaviour) else Behaviour(b)
    s = ideal_states() if states is None else states
    out_a = a.run(s)
    perms = list(_renames(b.regs, a.regs))
    # B relabelled by π reads register b from π(b): run B on s permuted, then
    # put its register b back at π(b).
    stacked = np.concatenate([s[:, p] for p in perms])
    out_t = b.run(stacked).reshape(len(perms), s.shape[0], N_REGS)
    best, best_p = None, None
    for p, ot in zip(perms, out_t):
        out_b = np.empty_like(ot)
        out_b[:, p] = ot
        regs = set(a.regs) | {int(p[r]) for r in b.regs}
        sq = sum(_register_distance(out_a[:, r], out_b[:, r]) ** 2 for r in regs)
        if best is None or sq < best:
            best, best_p = sq, p
            if best == 0.0:
                break
    d = float(np.sqrt(best))
    return (d, best_p) if return_rename else d
