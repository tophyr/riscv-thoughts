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

## Throughput

The frozen-T1 encode + predicted-binding routing + T2 core + losses fold into
ONE compiled CUDA-graph step at a fixed-padded instruction count (a dense
`(B, MAX_NI)` grid built host-side; padding gated via `active=0`), verified
bit-identical to the ragged path (active-slot essence max |Δ| 1.8e-7,
padding-leak exactly 0). That fold is what makes the step GPU-bound at all:
eager per-step dispatch ran at 66% util.

**Where the step time goes, and what was removed from it.** Measured on the
len=2 cell T2 runs at (`--d-out 512`, op-cell depth 4, 16 pool heads, vp head
depth 1, 384 rows per step, RTX 4090), fed from a fixed on-disk corpus so
generation cannot be the limit; steady-state ms/step from step 1500 onwards.
Each row adds one change to the row above it:

| change | ms/step | GPU util |
|---|---|---|
| checkpoint written at every 100-step log line | 8.14 | 73% |
| checkpoint every 10k steps (`--save-every`) | 6.02 | 96% |
| compile with `max-autotune` (benchmarked GEMM kernels) | 5.50 | 95% |
| vp head: essence term once per row, empty input slots skipped; pool projects only touched slots | 4.81 | 93% |
| prefetch queue 4 → 128 batches | 4.66 | 95% |
| in-slot score head on touched slots only | 4.47–4.56 | 95% |
| vp head: unsupervised output slots skipped | 4.44–4.48 | 94% |
| one pinned host→device copy per step instead of fourteen | 4.36 | 95% |
| gradient-clip scaling and Adam compiled together | 3.92 | 95% |
| prefetch workers pack each step into shared memory | 3.88 | 96% |
| whole step (forward, backward, clip, Adam) captured as one CUDA graph | 3.67–3.88 | 98% |

Two runs of one configuration differ by up to ~5% in ms/step, so the last rows
are separated only by their direction. The same step replayed on fixed inputs,
with no host work at all, runs at 3.75 ms: with the whole step captured, the
host issues one launch per step and the GPU never waits on it. Fed live from
odin the step runs at 3.96 ms and 97.6% GPU utilisation. Every change computes
the same function of the same inputs up to float rounding — it skips work whose
result is exactly known (a register slot no instruction touched is an exact
zero row, so any linear map of it is its bias), moves host work to another
process, or runs the same Adam update through different kernels. Seeded
300-step runs against the original code agree at step 0 to ≤1.5e-5 relative on
every loss and correlate at 0.9999999 over the run; on real checkpoints the
essence is bit-identical and the vp loss agrees to 2e-7. The T1 trainer runs
through the same captured step: from one initial state on one batch, its losses
agree with the original code's to 1.4e-5 relative compiled and 2.6e-7 eager.
Resuming the len=2 cell from a step-2000 checkpoint — one this code wrote, or
one the original fused-Adam code wrote — trains on indistinguishably from an
uninterrupted run: over the next 2000 steps, across one uninterrupted and three
resumed runs, each run's loss sits 0.30–0.36% on average from their per-step
median.

Seeded runs are not bit-reproducible: the compiled Adam update is not
order-stable, so two runs of identical code drift apart from the first few
steps, by ~1e-4 in parameters after 300 steps, the same size as the
difference from the original code. Without the compiled update (fused Adam)
the same comparison is bit-exact (3.7e-9); the compiled update is worth ~0.5 ms
of a ~4.3 ms step.

End to end, 1M-step cells on `configs/instr_alu_writers_chain010.json` over
the frozen `runs/t1_writers_250k_rank`, scored by `probes/vp_budget.py` on one
shared 1200-batch corpus (exact match; each pair is generator seeds 0 and 7):

| trainer | exact match |
|---|---|
| before these changes | 99.8969% (one cell) |
| through the single host→device copy | 99.9069%, 99.9007% |
| + compiled clip and Adam, shared-memory prefetch | 99.8731%, 99.8950% |
| the same with fused Adam (the control) | 99.8932%, 99.8936% |
| + whole step captured as one CUDA graph | 99.9020%, 99.8792% |

Every row computes the same function up to rounding, and the batch stream
differs between any two runs anyway (odin's pool interleaves its workers
nondeterministically), so the nine cells are draws from one distribution: mean
99.893%, standard deviation 0.011 points per cell. No row stands apart from
it. Binding accuracy is identical across them and the commutative-swap collapse
reads 0.038–0.041 throughout.

**Where the GPU time goes.** Nsight Systems over ~1,570 captured fp32 steps
puts the GPU at 3.63 ms of kernel time per step: matmuls 1.96 ms (54%), fused
elementwise and reduction kernels 1.26 ms (35%) — of which the compiled clip
and Adam is ~0.5 ms — the frozen T1's attention 0.24 ms (6.5%), the rest
0.18 ms. Under bf16 the matmuls fall to 1.32 ms while the elementwise kernels
barely move (1.16 ms); several halve exactly with their bytes, the signature of
memory-bound work. The card's own ceilings, measured on it: 157 TFLOPS on a
large bf16 matmul and 87 TFLOPS TF32, drawing ~420–465 W; 1.0 TB/s on a large
copy, drawing ~250–335 W. The trainer's ~340 W at 98% utilisation sits at the
memory-bound end of that range: the step is mostly data movement and matmuls
too small to approach peak, which is a property of the model's shape —
`d_out` 512, 384 rows, a 32-slot state rewritten every instruction — rather
than of the trainer. Per-kernel hardware counters (Nsight Compute) need the
Windows driver's "allow access to GPU performance counters" setting under WSL2.

**What does not help.** Two cells sharing the GPU run at 15.5 ms/step each —
129 steps/s together against 166 for one alone at the 6.02 ms configuration;
time-slicing costs more than it hides. More rows per step barely raise the
rate a row is trained at: with the whole step captured, 768 rows run at 7.07
ms and 1536 at 13.80 against 3.88 for 384 — 10% and 12% more rows per second —
so the GPU's time already scales almost with the work, and more rows would
change the training (fewer optimizer steps per row) for little. Compiling Adam
alone gains nothing (0.862 against 0.873 ms for fused Adam): the gain comes
from fusing it with the clip scaling. Running the frozen T1's attention through
plain matmuls instead of the memory-efficient kernel saves 2% and moves T1's
essence by up to 2.6e-4 (TF32 matmuls against the kernel's fp32), a precision
change for too little. Fewer intra-op threads is not free either way:
one is slower (5.08 against 4.66 ms), while four — the trainer's setting —
match sixteen file-fed and are faster live (below). Coordinate-descent kernel
tuning changes nothing (5.275 against 5.272 ms), and neither Python's garbage
collector nor its GIL switch interval costs anything measurable.

**Feeding it.** At 3.88 ms the trainer consumes ~260 batches/s, ~120 MB/s of
RVT. Generation is far from the limit — odin's pool delivers ~600 batches/s
over the gigabit link at a load of ~14 of its 128 cores, and lz4 holds the wire
to ~21 MB/s at 180 batches/s. The trainer's own per-instruction split is the
CPU-bound stage: one prefetch worker delivers ~240 batches/s, so the prefetcher
runs four (~540/s). The workers also build and pack each step's device inputs
and hand them over through a ring of slots in a `/dev/shm` file, so the main
thread only copies one buffer: sending the ~0.7 MB packed step through the
queue instead cost the main thread more to unpickle than the packing saved. The
queue holds 128 batches because the multinode stream has lulls of tens of
milliseconds, which a 4-batch queue passes straight to the GPU as idle time.
And the trainer caps its intra-op threads at four: with sixteen, their spinning
contends with the stream's receive path (nc, lz4, the local mux) and the live
step ran 7% slow.

**bf16 autocast — measured, not adopted.** Running everything after the
frozen-T1 front-end under bf16 autocast (T1 stays fp32, so T2 is fed the T1 it
was validated with) brings the captured step to 3.00 ms file-fed and 3.07 live,
against 3.88 / 3.96 in fp32/TF32 — 1.29x the rows per second — with its GPU
floor at 2.94 ms. On the shared corpus two bf16 cells read 99.9053% and
99.9074% exact against 99.9069% and 99.9007% for fp32 cells of the same code
(depth-2: 99.09% / 99.11% against 99.08% / 99.06%), all inside the 0.011
per-cell spread above; binding and the collapse rows are indistinguishable. It
changes the computation, so adopting it is a decision rather than a
measurement.

## Status and open problems

The equivariance skeleton is validated end-to-end (T1, T2-on-frozen-T1, exact
equivariance, trained GVN). Scope is `branch+cap=4`; the additive extensions —
a non-renameable PC slot, memory ops (the real expressiveness risk: can the
essence capture value-dependent aliasing?), immediates, and integration with
the streaming shift-reduce parser for validity — remain open (RISCV_DESIGN.md
"Open problems"). The dormant `behavioral_distance` oracle was deleted: it is
rename-*invariant* (operator-only), so it is the wrong training target for a
rename-*equivariant* representation that must keep binding recoverable.
