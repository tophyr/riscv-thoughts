# Phase 12: T2 loss-set convergence, value-prediction, and the operator-axis finding

The phases between 11 and here explored many head designs and output
geometries (per-register pair-MSE, syntactic dest/src CE heads, Sinkhorn
binding, register-indexed pools, shape/binding sub-vector splits,
unit-sphere normalization). They converged back to a **single open-space
vector with behavioral losses**. This section records the settled design
and the results that matter; the dead-ends are not re-narrated.

## Settled loss set — T1 is the single-instruction special case of T2

Removed: pair-MSE / `reg_effect_loss`, explicit `equivalence_loss`,
syntactic dest+src CE heads, `rename_equiv`. The behavioral-distance
machinery (`behavioral_distance` / Hungarian bijection) is demoted to a
dormant offline oracle — not used in generation or training.

Current losses (shared `compressor.train.binding_losses`, identical for
T1 and T2):
- **magnitude-validity**: `(‖v‖ − is_valid)²` — a *soft* loss, NOT
  `F.normalize`.
- **live_in / live_out**: BCE on which registers are read / written.
- **pc_writes**: BCE on whether the chunk writes PC.
- **in_slot / out_slot**: ListMLE (Plackett–Luce) over per-register
  scalar score heads → read/write ORDER. Duplicate-free by `argsort`.
- **value_predict (vp)**: per-anchor per-out-slot MSE — predict each
  output register's value from the vector + input-slot values, on 8
  sampled anchor states.

T1 uses the same losses and heads (one output register = a single output
slot). **Equivalence is now EMERGENT** — there is no equivalence loss;
equivalent instructions/chunks collapse from the behavioral targets alone.

## Architecture (settled)

Single open-space `d_out=512` vector. No shape/binding partition — the
merged vector learns everything (cap=4: in_slot 99.2% / out_slot 99.2% /
dup 0). Structural splitting was a diagnostic, not a keeper. Magnitude
carries validity (soft loss), direction carries semantics; binding heads
read `F.normalize(vec)`, vp reads the raw vector.

## Resolved: unit-sphere vs out_slot

A *hard* `F.normalize` on the output breaks out_slot (→ 3–7%); a *soft*
`(‖v‖−1)²` magnitude loss on an open-space output gives the same surface
geometry AND keeps out_slot ~99%. Train TOWARD the surface, don't project
ONTO it. (Supersedes the "unit-sphere is an open mystery" handoff note.)

## Slot difficulty is loss geometry, not capacity

out_slot is **intrinsically easy** — structural last-write ordering. In a
controlled tri-split with no vp on the bindings it reaches 0.005 (perfect)
while in_slot is dead; adding vp **flips** it (in_slot revives because it
depends on vp's input-sensitivity representation; out_slot starves under
vp's 10:1 weight). The long-standing "out_slot needs more capacity" read
was vp competition, not capacity.

## Infrastructure: GPU saturation

Full-step `torch.compile` (reduce-overhead / CUDA graphs) + pinned
non-blocking H2D + ListMLE truncation → **~95% GPU util** for T1, T2, and
the decoder (was ~32–50%). The compiled forward+loss+backward, optimizer,
scheduler, logging, and resume are lifted into one shared
`run_train_loop`; the decoder was made fixed-shape (pad to corpus-constant,
invalid rows CE-ignored) so its step is capturable. Trains until stdin
EOF; `--n-steps` shapes only the cosine schedule.

## Converged T2 run — `runs/t2_d512_cap4_2M`

2M steps, d=512 n=4, cap=4, vp weight 1.0 vs 0.1 per binding loss,
~10 ms/step, GPU ~96%. Final windowed losses:

| loss | start | final |
|------|-------|-------|
| value_predict | 2.15 | 0.18 |
| in_slot | 0.39 | 0.02 |
| out_slot | 0.49 | 0.16 |
| live_in | ~0.39 | 0.02 |
| pc_writes | ~0.4 | 0.00 |
| magnitude | — | 0.02 |

vp converged ~4× past the earlier 504k verification run
(`runs/t2_d512_cap4_500k_newt1`, vp ~0.72 → 0.18). out_slot plateaus ~0.16
(vp-starved, not failed).

## THE result: value-prediction does NOT build an operator/effect axis

Binding-controlled geometry probe (cosine of T2 vectors), holding the
register binding fixed and varying only the function:

| pair type | 504k | 2M |
|-----------|------|----|
| same effect, same binding (SE) | 0.964 | 0.986 |
| different effect, same binding (DE) | 0.935 | 0.975 |
| **operator-axis gap (SE − DE)** | **+0.029** | **+0.011** |
| rename (same op, different binding) | 0.754 | 0.801 |
| random (unrelated) | 0.579 | 0.652 |

Converging vp 4× did NOT grow the operator axis — it **shrank**
(+0.029 → +0.011). At fixed binding, flipping the function moves the
vector ~0.01; changing the binding moves it ~0.18. **Binding dominates the
geometry ~16× over operator-essence**, and more vp made everything more
similar, not more separated.

Structural reason: vp predicts outputs from `(vector, inputs)` through a
shared head — a per-chunk **reconstruction** objective with no term that
pulls behaviorally-equivalent chunks together. Two same-effect chunks can
hold different vectors as long as the head decodes each; nothing
co-locates them, and the binding heads actively separate by register. vp
cannot induce operator-clustering no matter how long it trains.

## VP hard probe — it works, as a reconstructor

312k `(chunk, anchor, out-slot)` triplets: compressed-space R² 0.96, corr
0.98. Decompressed, recovery is exact for small outputs (|val|≤15, 36% of
cases: 94.8% within 10%) but hopeless for the 59% of outputs >1M (random
int32 arithmetic). vp is a competent per-chunk predictor — which is
exactly why it cannot serve as the operator axis.

## Dimensionality — the vector is barely used

T2 output effective rank (participation ratio) ≈ **13 of 512**; 90% of
variance in 33 dims; only ~63 dims carry >0.1% variance (~87% dead).
Per-loss read width: in_score ~22, vp ~15, out_score / live_out ~6,
pc_writes ~3; the union of all heads ≈ 14 — they share one ~13-d subspace,
they don't expand it. Through the stack, rank peaks at the 2nd transformer
layer (~23) then **compresses** to ~13 at the output. **Capacity is not
the bottleneck anywhere** — d_out=512 is ~37× the dimensionality in use;
the geometry is starved of structure, not size. (Caveat: PR/variance is a
linear measure; counts are lower bounds.)

## Vestigial value compression

vp regresses against `_value_compress = sign(x)·log1p(log1p|x|)`. The
double-log is the `behavioral_distance` Hungarian-cost metric, carried
over now that behavioral_distance is dormant. It crushes 59% of targets into a
0.41-wide saturation band — a likely contributor to the low rank. A single
signed log would keep conditioning sane with ~18× more large-value
resolution; untested.

## Conclusion and indicated next step

The decisive lever is the **oracle `out_regs` pairwise loss**: push T2
vectors together/apart by their actual register-file outputs on the shared
anchors (rename-SENSITIVE behavioral distance, no Hungarian bijection —
cheap). It is the only proposed objective that forces behavioral
separation into the empty dimensions. vp convergence is *not* the missing
ingredient. Orthogonal tests queued: `out_slot_weight = vp_weight`
(starvation vs ratio), small-d to convergence (capacity vs optimization),
single-log vp.

---
