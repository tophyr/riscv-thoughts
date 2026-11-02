# Phase 12: T2 loss-set convergence, value-prediction, and the operator-axis finding

Phase 11's T2 did not discriminate operands. This phase tried many head
designs and output geometries for T2 (per-register pair MSE, syntactic
destination/source CE heads, Sinkhorn binding, register-indexed pools, split
shape/binding sub-vectors, unit-sphere normalization) and converged on a
**single open-space vector trained with behavioral losses**. This entry records
that design and the results that bear on it; the individual dead ends are not
recorded.

## The loss set — T1 is the single-instruction case of T2

The loss set contains no pair MSE, no explicit equivalence loss, no syntactic
destination/source heads and no rename-equivalence term; the behavioral
distance with a Hungarian register bijection is not used in generation or
training. The losses, identical for T1 and T2:
- **magnitude-validity**: `(‖v‖ − is_valid)²` — a *soft* loss, not
  `F.normalize`.
- **live_in / live_out**: BCE on which registers are read / written.
- **pc_writes**: BCE on whether the chunk writes PC.
- **in_slot / out_slot**: ListMLE (Plackett–Luce) over per-register scalar
  score heads, giving the order in which registers are first read / last
  written. `argsort` of the scores makes the order duplicate-free.
- **value_predict (vp)**: per-anchor, per-output-slot MSE — predict each output
  register's value from the vector plus the input-slot values, at 8 sampled
  anchor states.

T1 is the case with one instruction (one output slot). **Equivalence is
emergent**: no loss names equivalent pairs, so equivalent instructions and
chunks collapse only from the behavioral targets.

## Architecture

A single open-space vector, `d_out=512`, with no shape/binding partition — the
merged vector learns everything (chunk-length cap 4: in_slot 99.2% / out_slot
99.2% / duplicate slot predictions 0). Splitting the vector served as a
diagnostic only. Magnitude carries validity (soft loss) and direction carries
semantics; the binding heads read `F.normalize(vec)`, vp reads the raw vector.

## Unit sphere versus out_slot

A *hard* `F.normalize` on the output breaks out_slot (3–7%); a *soft*
`(‖v‖−1)²` magnitude loss on an open-space output gives the same surface
geometry and keeps out_slot at ~99%. Train toward the surface; do not project
onto it.

## Slot difficulty is loss geometry, not capacity

out_slot is **intrinsically easy** — it is structural last-write ordering. In a
controlled three-way split with no vp on the binding sub-vector it reaches loss
0.005 while in_slot does not learn; adding vp **flips** that: in_slot learns,
because it depends on vp's input-sensitive representation, and out_slot starves
under vp's 10:1 weight. A slow out_slot is vp competition, not missing
capacity.

## GPU saturation

Whole-step `torch.compile` (reduce-overhead, CUDA graphs), pinned non-blocking
host-to-device copies and truncated ListMLE bring T1, T2 and the decoder from
~32–50% to **~95% GPU utilization**. The decoder was made fixed-shape (padded
to a corpus-wide constant, invalid rows ignored by the CE) so its step can be
captured.

## Converged T2 run

2M steps, d=512, n=4, chunk cap 4, vp weight 1.0 against 0.1 per binding
loss, ~10 ms/step at ~96% GPU. Windowed losses:

| loss | start | final |
|------|-------|-------|
| value_predict | 2.15 | 0.18 |
| in_slot | 0.39 | 0.02 |
| out_slot | 0.49 | 0.16 |
| live_in | ~0.39 | 0.02 |
| pc_writes | ~0.4 | 0.00 |
| magnitude | — | 0.02 |

vp ends ~4× below a 504k-step verification run (~0.72 → 0.18).
out_slot plateaus at ~0.16, starved by vp rather than failed.

## Value prediction does not build an operator/effect axis

A geometry probe holding the register binding fixed and varying only the
function, reading the cosine between T2 vectors, at 504k and 2M steps:

| pair type | 504k | 2M |
|-----------|------|----|
| same effect, same binding (SE) | 0.964 | 0.986 |
| different effect, same binding (DE) | 0.935 | 0.975 |
| **operator-axis gap (SE − DE)** | **+0.029** | **+0.011** |
| rename (same op, different binding) | 0.754 | 0.801 |
| random (unrelated) | 0.579 | 0.652 |

Training vp 4× further *shrinks* the operator axis (+0.029 → +0.011). At fixed
binding, changing the function moves the vector ~0.01; changing the binding
moves it ~0.18. **Binding dominates the geometry ~16× over the operator**, and
more vp makes everything more similar, not more separated.

The reason is structural: vp predicts outputs from `(vector, inputs)` through a
shared head — a per-chunk **reconstruction** objective with no term pulling
behaviorally equivalent chunks together. Two same-effect chunks can hold
different vectors as long as the head decodes each, and the binding heads
actively separate by register. vp cannot induce operator clustering however
long it trains.

## vp as a predictor

On 312k `(chunk, anchor, output-slot)` triplets: R² 0.96 and correlation 0.98
in the compressed value space. Decompressed, recovery is accurate for small
outputs (|val| ≤ 15, 36% of cases: 94.8% within 10%) and fails for the 59% of
outputs above 1M (random int32 arithmetic). vp is a competent per-chunk
predictor — which is why it cannot serve as the operator axis.

## Dimensionality — the vector is barely used

The 2M T2's output effective rank (participation ratio) is **~13 of 512**; 90%
of variance lies in 33 dims; only ~63 dims carry >0.1% of variance (~87% dead).
Per-loss read width: in_slot score ~22, vp ~15, out_slot score / live_out ~6,
pc_writes ~3; the union of all heads is ~14 — they share one ~13-dim subspace
rather than expanding it. Through the stack, rank peaks at the second
transformer layer (~23) and **compresses** to ~13 at the output. Capacity is
not the bottleneck: d_out=512 is ~37× the dimensionality in use. Participation
ratio and variance are linear measures, so these counts are lower bounds.

## The vp value transform

vp regresses against `sign(x)·log1p(log1p|x|)`, the double log of the
Hungarian-bijection behavioral distance. It crushes 59% of targets into a
0.41-wide saturation band, a likely contributor to the low rank. A single
signed log would keep ~18× more resolution on large values; untested in this
phase.

## Handed to the next phase

The objective proposed to open the operator axis is a **pairwise loss on the
oracle's register-file outputs** at shared anchors: push T2 vectors together
or apart by their actual outputs, register by register (rename-sensitive, no
Hungarian bijection). vp convergence is not the missing ingredient.

---
