"""Ground-truth operator-equivalence oracle (the GVN bijection distance).

`behavioral_distance(a, b)` is the multi-instruction GVN bijection: it
matches the two chunks' input registers under a bijection and their
outputs under a Hungarian assignment, then measures the residual
behavioral difference. 0 iff the chunks are behaviorally equivalent on
the sampled anchor states.

It is rename-INVARIANT: it measures operator essence only and is blind
to register binding. For example
`bd(add r4,r4,r4 ; add r5,r5,r5) == 0` — the two chunks compute the
same operation, just on differently-named registers.

This oracle is NOT used in training or corpus generation. It exists for
offline validation of learned operator-equivalence and future decoder
reconstruction checks (a dormant ground-truth reference). It reuses the
SSA + evaluator primitives in `datagen.compare`.
"""

import numexpr as ne
import numpy as np
from scipy.optimize import linear_sum_assignment

from .compare import (
    make_anchor_states, precompute_chunk,
    partial_bijections, _pair_canon_positions, _canonical_state,
    _eval_ssa_numpy, CanonPositionOverflow, MAX_BIJECTION_SIZE,
)


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
