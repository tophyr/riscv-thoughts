"""Canonical input roles: a chunk's inputs ordered by what its function does
with them, so two equivalent chunks order corresponding inputs alike whatever
their spelling or dataflow shape.

An input's fingerprint is the chunk's outputs over a fixed family of states
in which that input holds a_t and every other input b_t; because every other
input holds the same value, the fingerprint depends only on the function and
the input's role in it. Each ordered pair is fingerprinted the same way with
the second input at c_t. Inputs are partitioned by Weisfeiler–Lehman
refinement over these fingerprints, and a class left with several inputs has
one member individualised and the refinement repeated, until every input
stands alone. A class that is a genuine symmetry of the function gives the
same result whichever member is chosen.

Inputs the function ignores — read, but cancelled or masked away — are
dropped first, decided from the same fingerprints, so a chunk and its
equivalent that never reads them agree on every remaining position.
"""

import collections

import numpy as np

# The fingerprint states: a fixed structured family shared by every chunk.
ROLE_SEED = 0xF1A6
N_ROLE_STATES = 64
_FAMILY = []

_M = (1 << 32) - 1


def _family():
    if not _FAMILY:
        from .compare import draw_anchor_states
        _FAMILY.append(draw_anchor_states(np.random.default_rng(ROLE_SEED),
                                          N_ROLE_STATES))
    return _FAMILY[0]


def _key(outs):
    """A per-state multiset of output values, as bytes: a sort key that does
    not depend on how the outputs are numbered."""
    return np.sort(outs, axis=1).tobytes()


def _relabel(keys):
    order = sorted(set(keys.values()))
    return {i: order.index(k) for i, k in keys.items()}


def canonical_inputs(inputs, run):
    """The inputs the function depends on, in canonical order.

    inputs: candidate input registers; run(regs (n, 32) int32) -> (n, outs)
    int64 executes the chunk."""
    F = _family()
    T = F.shape[0]

    # Every assignment the fingerprints need, executed as one stacked call:
    # the baseline, each input at a, each at c, and each ordered pair.
    assigns = ([{}] + [{i: 1} for i in inputs] + [{j: 3} for j in inputs]
               + [{i: 1, j: 3} for i in inputs for j in inputs if i != j])
    regs = np.zeros((len(assigns), T, 32), np.int32)
    regs[:, :, inputs] = F[None, :, 2:3]
    for a, assign in enumerate(assigns):
        for r, col in assign.items():
            regs[a, :, r] = F[:, col]
    outs = run(regs.reshape(-1, 32)).reshape(len(assigns), T, -1)
    n = len(inputs)
    base = outs[0]
    U = dict(zip(inputs, outs[1:1 + n]))
    Q = dict(zip(inputs, outs[1 + n:1 + 2 * n]))
    P = dict(zip([(i, j) for i in inputs for j in inputs if i != j], outs[1 + 2 * n:]))
    # An input whose own variation and every paired variation leave the
    # outputs where the others put them is not something the function reads.
    keep = [i for i in inputs
            if not (np.array_equal(U[i], base)
                    and all(np.array_equal(P[(i, j)], Q[j]) for j in inputs if j != i))]
    U = {i: _key(U[i]) for i in keep}
    P = {(i, j): _key(P[(i, j)]) for i in keep for j in keep if i != j}

    def refine(cls):
        while True:
            new = _relabel({i: (cls[i], tuple(sorted((cls[j], P[(i, j)])
                                                     for j in keep if j != i)))
                            for i in keep})
            if len(set(new.values())) == len(set(cls.values())):
                return new
            cls = new

    cls = refine(_relabel(U))
    while len(set(cls.values())) < len(keep):
        sizes = collections.Counter(cls.values())
        c = min(k for k, v in sizes.items() if v > 1)
        # Any member of a class that is a symmetry gives the same result; a
        # class that is not one would make this choice register-dependent.
        chosen = min(i for i in keep if cls[i] == c)
        cls = refine(_relabel({i: (v, i == chosen) for i, v in cls.items()}))
    return sorted(keep, key=lambda i: cls[i])
