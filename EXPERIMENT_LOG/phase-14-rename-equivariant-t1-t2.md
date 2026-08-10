# Phase 14: Rename-equivariant T1/T2 — building the symmetry into the representation

Phase 13 ended on a wall: register renaming is a non-abelian permutation
group, and the only additive shadow of it is the trivial one (the torsion
theorem) — so no loss on a flat vector forms king−queen rename arithmetic, and
binding stays the macro organizer. This phase took the indicated turn: **stop
coercing the geometry with a loss; build the rename symmetry into the
representation.** The full design rationale now lives in RISCV_DESIGN.md
("Rename-Equivariant Compressor"); this entry records the build and the
validated results.

## The architecture (settled, then spike-validated before integration)

A **register-indexed state machine** (`compressor/model.py`, one `Compressor`
class): seed 32 register slots with per-chunk anonymous tags, then per
instruction `read operand slots → shared op-cell → write dest slot`, and emit
the final register-indexed state `T ∈ ℝ^{R×d}`. Essence = a
permutation-invariant attention pool over the register axis (the operator
axis); binding = the per-slot remainder, read by the binding heads. Two slot
channel groups: **value** (set on write, preserved on read — the threaded
dataflow state) and **event** (read/write order timestamps, so
live/in_slot/out_slot are recoverable). Because the only register-indexed
input is the anonymous tag and reads/writes are pure one-hot ops, a rename
permutes the slots exactly — **equivariance is by construction, no
per-register parameters.**

Validated on a standalone spike (4090) before any integration: state
equivariance `max‖T(π·C) − π·T(C)‖ = 0` (bit-exact), essence invariance ~6e-8
(float reduction noise), value-preserved-on-read = 0, and the full
forward+backward+opt captures as a single `reduce-overhead` CUDA graph.

## T1 is literally T2 with seq_len=1

One shared core. T1's per-step content is the embedded opcode+immediate tokens
(registers stripped to wiring, via a standalone `TokenEmbedder`); T2's per-step
content is the **frozen T1's per-instruction essence** (the tier contract — no
opcode re-embedding). Two operand-order fixes were needed for the binding
heads and apply to both tiers: a per-operand-role read bias (`read_role`, gives
within-instruction operand order → T1 in_slot 64.5%→100%) and operand-ranked
read/write timestamps (the cross-instruction analog → T2 out_slot via the
write timestamp, in_slot via an MLP score head for the relevance×order
product). All register-independent, so equivariance stays exact.

## The canonical value_predict bug (and fix)

`precompute_chunk.out_regs` is the **canonical** register file (inputs
relocated to canonical first-read positions, then executed — rename-invariant
by construction). The vp loss had been feeding the head the **raw** values at
the inputs' actual registers while supervising against this canonical target —
asking it to predict `f(canonical inputs)` from `raw inputs` = uncorrelated
numbers, a mathematically inconsistent task for any input-dependent output.
Fix: feed canonical input values (`anchor[:, k+1]` for input slot k) to match
the canonical target. Standardized both tiers on canonical vp + canonical
`out_regs`; the raw `row_outputs` path, `value_predict_loss`, and the RVT `RB`
wire fields were deleted (format **v7→v8**, 5-field header). This made vp a
valid training signal for the first time and is why the GVN collapse below
reflects *trained* value-numbering, not just the free structural collapse.

## T1 — validated (`runs/t1_equiv_d128_canonvp_250k`)

d128/n4/l2/d_out64, d_event16, 250k, `--rule single`, anchor-seed 0,
n-states 8, `instr_default.json`, 16-gen round-robin mux. Held-out
(`scripts/eval.py`):

| metric | value |
|---|---|
| equivariance state_max_err | `0.00e+00` (exact, trained model) |
| equivariance essence_max_err | `1.79e-07` (float noise) |
| tag-invariance cos | mean `1.0000` / min `0.9992` |
| live_in / live_out | 99.9% / 100% |
| pc | 100% |
| in_slot / out_slot | 100% / 100% |
| dup | 0 |

Its binding heads are valid under the current core — the prerequisite for
routing T2 off T1's predicted binding.

## Tier recursion — the real test (`runs/t2_equiv_d512_cap4_2M_principled`)

Earlier T2 runs (`t2_equiv_d512_cap4_2M`, `…_islfix`) reached/exceeded the
Phase-12 flat-T2 binding metrics with exact equivariance and GVN collapse, but
read each instruction's wiring `(in0,in1,out)` from **decoded RISC-V tokens** —
ground-truth dataflow handed in out-of-band. That tests the *architecture
given ground-truth wiring*, not the tier recursion: T2 would score identically
if T1's binding were garbage. The principled run routes T2's wiring off the
frozen T1's **predicted** binding (`argmax`/top-2 of its in/out-slot scores,
gated by predicted live counts, x0 excluded) — never tokens. `argmax` of an
equivariant readout permutes with renames, so equivariance stays exact.

2M steps (resumed 472k + 2M), `branch+cap=4`, d_out=512, d_event=16,
`instr_t2.json`, `--route binding`, `--pair-weight 0` (losses identical to
T1's). Held-out (`scripts/eval.py`, route=binding):

| metric | value | note |
|---|---|---|
| live_in / live_out | 99.7% / 100% | |
| pc | 98.4% | |
| in_slot / out_slot | **91.5%** / 99.9% | in_slot vs 93.8% on token wiring = honest cost of routing off *predicted* (imperfect) binding |
| dup | 0 | |
| equivariance (chunk) | `0.00e+00` | exact, composes over the chunk |
| GVN rename-twin | 0.0080 (ratio 0.012) | structural / free |
| **GVN behavioral (commutativity, non-rename)** | **0.0238 (ratio 0.036)** | TRAINED value-numbering |
| GVN distinct | 0.6593 | |

**The headline is the GVN split.** Non-rename behavioral equivalents
(commutative operand swaps) collapse ~28× tighter than distinct pairs — i.e.
the register-state machine performs *trained* value-numbering, not merely the
free rename-twin collapse (rename-twins dominate twins-heavy corpora, so the
split is essential to read the result honestly). This is the property the flat
T2 of Phases 11–13 could never deliver, and it answers the canonical-vp fix:
with vp a valid signal, the trained equivalences collapse. `vp`'s absolute
value remains a probe at an intrinsic floor (an opcode-oracle baseline of the
same width also fails to regress wrapped/bitwise R-type ops; the program-one-
hot oracle *overfits*, train 0.21 / held-out 2.7) — **GVN collapse, not vp
accuracy, is the metric that matters.**

**CORRECTION (Phase 15): there is no intrinsic vp floor.** Single-instruction
vp is exactly `0.000`, and the head is fed raw value bits, not the
log-compressed scalars this claim assumed. The opcode oracle cited here gave
opcodes but no dataflow, so it was never a full-information baseline, and the
program-one-hot result is just memorisation failing to generalise. vp is a real
objective that responds to architecture and training; treating it as
floor-limited delayed the Phase-15 diagnosis.

## Unification — the litmus test holds in code

After validation, a behavior-preserving refactor collapsed the T1/T2 code
divergences (each stage verified bit-identical; both validated checkpoints
still load and reproduce every eval number): one `Compressor` class
(pluggable `ingest='tokens'|'vectors'`), one `train_compressor` (the entry
scripts are thin presets; the only tier-specific code is the content
provider), one canonical vp loss, one data mode. The standard met: **T1 and T2
differ only in the ingestion front-end and the hparams** — in the code, not
just the prose.

## Throughput — the one-graph fold

The frozen-T1 encode + predicted-binding routing + T2 core + losses now fold
into ONE compiled CUDA-graph step at a fixed-padded instruction count (a dense
`(B, MAX_NI)` grid built host-side; padding gated via `active=0`). Verified
bit-identical to the ragged path (active-slot essence max |Δ| 1.8e-7,
padding-leak exactly 0). Result: **T2 at 8 ms/step @ 95–97% util** (was 11 ms
@ 66% with the eager per-step dispatch). Recorded as a lesson: never leave a
long run at low util — the fix is a pipeline/compile change, not a deferral.

## Status and open problems

The equivariance skeleton is validated end-to-end (T1, T2-on-frozen-T1, exact
equivariance, trained GVN). Scope is `branch+cap=4`; the additive extensions —
a non-renameable PC slot, memory ops (the real expressiveness risk: can the
essence capture value-dependent aliasing?), immediates, and integration with
the streaming shift-reduce parser for validity — remain open (RISCV_DESIGN.md
"Open problems"). The dormant `behavioral_distance` oracle was deleted: it is
rename-*invariant* (operator-only), so it is the wrong training target for a
rename-*equivariant* representation that must keep binding recoverable.
