# Phase 13: Pairwise behavioral geometry — operator-axis, twins-as-positives, equivariance

Phase 12 proposed an oracle `out_regs` pairwise loss to reshape the
binding-dominated, ~13-effective-dim geometry (value-prediction alone never
built an operator/effect axis). This phase built and tested that line — a
base pairwise loss, then twins-as-positives, then rename-equivariance — and
**set them aside**. The code was reverted; this entry is the record.

## Base pairwise loss (rename-SENSITIVE `out_regs` distance)
Match cosine distance between chunk vectors to a behavioral distance from
their `out_regs` (registers compared position-by-position; no Hungarian
bijection). Result (2M): opened the operator axis (SE−DE gap +0.011 →
+0.044), de-collapsed the space (RANDOM 0.65 → 0.006), raised effective
rank (PR 13 → 33), and de-starved `out_slot` (0.16 → 0.004). **But binding
still organized the macro geometry** — renames were pushed *far* (0.80 →
0.159); operator separation only appeared *within* a fixed binding.

## #1 twins-as-positives — inverted the hierarchy, but smelly
Override the oracle for KNOWN rename-twins (shared `cluster_id`): target a
small `margin` instead of their large rename-sensitive distance ("the
difference is naming, not computation"). Converged result: **inverted the
hierarchy** — RENAME (0.573) closer than DE-SB (0.453), operator became the
macro organizer (binding had been ~16:1 dominant; now ~1:1). Cost:
`out_slot` degraded to 0.23 (pulling renames close squeezes the
output-register subspace). Set aside — operator-macro is de-prioritized,
and the mechanism (a margin-pull keyed on corpus twin-labels, with a known
false-negative tail on coincidental renames) is unsatisfying.

## #2 rename-equivariance (king−queen consistency) — failed
Make each shared rename π_j a CONSISTENT displacement `enc(π_j·X) − enc(X)`
across chunks (variance penalty), so binding becomes a recoverable
displacement. Required shared-π twins + a `twin_slot` field.
- **equiv_weight = 1.0:** collapsed the representation (PR 23.7 → 3.1,
  `out_slot` 0.33, vp hurt) — the cheap way to make displacements
  consistent is to flatten the space.
- **equiv_weight = 0.1:** inert — geometry matched #1 (`out_slot` 0.012, PR
  14.3), but with 2M steps instead of 500k, hard to distinguish success from
  training size or equiv weight.
- **Held-out-π generalization probe (the decider):** displacement
  consistency `R = ‖mean_X d‖ / mean_X‖d‖` ≈ **0.09 for ALL models**
  (vp-only, pair, equiv-1.0, equiv-0.1) — exactly the random baseline
  `1/√120`. The rename is a CHUNK-DEPENDENT, random-direction displacement;
  no consistent `D(π)` formed, on trained *or* held-out π. King−queen rename
  arithmetic does not exist additively in this geometry.
- **Why:** register renaming is conjugation (`P·A·P⁻¹`), non-abelian — no
  clean additive shadow, unlike word2vec semantic relations. The
  additive-displacement route is the wrong tool for renames; a
  register-indexed layout (rename = permutation of per-register blocks)
  would be required.

## Conclusion
Operator-macro clustering is achievable (#1) but de-prioritized and of
unknown value; king−queen rename-equivariance does not form additively (#2).
The Phase 12 vp + behavioral-binding loss set stands as the T2 design;
pairwise distances are used to improve separation and the equiv machinery is
reverted. Durable finding: the T2 geometry is currently binding-organized and
~13-effective-dim, and **neither a pairwise behavioral target nor an additive
equivariance loss cleanly reshapes it** into an operator-organized, composable
space.

---
