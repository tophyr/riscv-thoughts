# Phase 13: Pairwise behavioral geometry — operator-axis, twins-as-positives, equivariance

Phase 12 left T2 binding-dominated and ~13-effective-dim, with value
prediction unable to build an operator/effect axis, and proposed a pairwise
loss on the oracle's register-file outputs. This phase built that loss, then
twins-as-positives, then a rename-equivariance loss, and **set all three
aside**; none is part of the design that follows.

## Base pairwise loss (rename-sensitive register-file distance)

Match the cosine distance between chunk vectors to a behavioral distance from
their register-file outputs, compared register by register (no Hungarian
bijection). At 2M steps it opens the operator axis (SE−DE gap +0.011 → +0.044,
the Phase 12 probe), de-collapses the space (random-pair cosine 0.65 → 0.006),
raises effective rank (participation ratio 13 → 33), and un-starves out_slot
(0.16 → 0.004). **Binding still organizes the macro geometry**: renames are
pushed *far* (cosine 0.80 → 0.159), and operator separation appears only
within a fixed binding.

## Twins-as-positives — inverts the hierarchy, on an unsatisfying mechanism

For known rename twins (corpus pairs sharing a cluster label), override the
oracle: target a small margin instead of their large rename-sensitive
distance, since the difference is naming, not computation. Converged, this
**inverts the hierarchy** — rename pairs (cosine 0.573) sit closer than
different-effect same-binding pairs (0.453), and the operator becomes the macro
organizer (binding:operator from ~16:1 to ~1:1). Cost: out_slot degrades to
0.23, because pulling renames together squeezes the output-register subspace.
Set aside: operator-macro clustering was de-prioritized, and a margin pull
keyed on corpus twin labels has a known false-negative tail on coincidental
renames.

## Rename-equivariance (king−queen consistency) — fails

Make each shared rename π_j a *consistent* displacement `enc(π_j·X) − enc(X)`
across chunks (a variance penalty), so binding becomes a recoverable
displacement. This needs twins sharing one π and a per-twin slot label.
- **equivalence weight 1.0:** collapses the representation (participation
  ratio 23.7 → 3.1, out_slot 0.33, vp worse) — the cheap way to make
  displacements consistent is to flatten the space.
- **equivalence weight 0.1:** inert — geometry matches twins-as-positives
  (out_slot 0.012, participation ratio 14.3). This run had 2M steps against
  500k for twins-as-positives, so the comparison cannot separate the weight
  from training length.
- **Held-out-π generalization probe (the decider):** displacement consistency
  `R = ‖mean_X d‖ / mean_X‖d‖` ≈ **0.09 for every model** (vp only, base
  pairwise, equivalence weight 1.0, equivalence weight 0.1) — exactly the random
  baseline `1/√120`. The rename displacement is chunk-dependent and randomly
  directed; no consistent `D(π)` forms on trained or held-out π. King−queen
  rename arithmetic does not exist additively in this geometry.
- **Why:** register renaming acts by conjugation (`P·A·P⁻¹`), a non-abelian
  group action with no clean additive shadow, unlike word2vec's semantic
  relations. Representing renames needs a register-indexed layout in which a
  rename is a permutation of per-register blocks.

## Conclusion

Operator-macro clustering is achievable (twins-as-positives) but of unknown
value; rename equivariance does not form additively. The Phase 12 loss set
remains the T2 design. On a flat vector, **neither a pairwise behavioral target
nor an additive equivariance loss reshapes the binding-organized, ~13-dim
geometry** into an operator-organized, composable space. Handed to the next
phase: build the rename symmetry into a register-indexed representation rather
than coerce it with a loss.

---
