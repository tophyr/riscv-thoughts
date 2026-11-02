# Phase 14: Rename-equivariant T1/T2 — building the symmetry into the representation

Phase 13 showed that register renaming, a non-abelian permutation group, has
only the trivial additive shadow (every finite-order element maps to zero in a
vector space — the torsion argument, RISCV_DESIGN.md "The symmetry, and why a
rename cannot be a vector addition"), so no loss on a flat vector forms
king−queen rename arithmetic and binding stays the macro organizer. This phase
builds the rename symmetry into the representation instead of coercing the
geometry with a loss. The design rationale is in RISCV_DESIGN.md
("Rename-Equivariant Compressor").

## The architecture

A **register-indexed state machine**: 32 register slots seeded with per-chunk
anonymous tags; per instruction, `read operand slots → shared op-cell → write
destination slot`; the output is the final register-indexed state
`T ∈ ℝ^{R×d}`. The essence is a permutation-invariant attention pool over the
register axis (the operator axis); the binding is the per-slot remainder, read
by the binding heads. Each slot has two channel groups: **value** (set on
write, preserved on read — the threaded dataflow state) and **event** (read and
write order timestamps, from which live/in_slot/out_slot are recoverable).
Because the only register-indexed input is the anonymous tag and reads and
writes are pure one-hot operations, a rename permutes the slots exactly —
**equivariance holds by construction, with no per-register parameters.**

On a standalone build before integration: state equivariance
`max‖T(π·C) − π·T(C)‖ = 0` (bit-exact), essence invariance ~6e-8 (float
reduction noise), value preserved on read exactly (error 0), and the full
forward + backward + optimizer step captures as a single `reduce-overhead` CUDA
graph.

## T1 is T2 with seq_len=1

One shared core. T1's per-step content is the embedded opcode and immediate
tokens, with registers reduced to wiring; T2's per-step content is the
**frozen T1's per-instruction essence** (the tier contract — T2 never re-embeds
opcodes). Two operand-order components serve the binding heads at both tiers:
a per-operand-role read bias, which gives the order of operands within an
instruction (T1 in_slot 64.5% without it, 100% with it); and operand-ranked
read/write timestamps, the cross-instruction analog (T2 out_slot reads the
write timestamp; in_slot uses an MLP score head for the product of relevance
and order). Both are register-independent, so equivariance stays exact.

T1 and T2 share one model class (with token or vector ingestion), one trainer
and one vp loss; they differ only in the ingestion front-end and the
hyperparameters. This unification was verified bit-identical stage by stage,
and both validated checkpoints below reproduce every eval number under it.

## value_predict must read inputs in its target's frame

The vp target is the **canonical** register file: input values placed at
canonical first-read positions, then executed, which makes it rename-invariant
by construction. Feeding the head the **raw** values at the inputs' actual
registers while supervising against the canonical target asks it to predict
`f(canonical inputs)` from uncorrelated numbers — an inconsistent task for any
input-dependent output. Fed canonical input values (`anchor[:, k+1]` for input
slot k), vp becomes a valid training signal at both tiers; the raw-output target
path was deleted (the wire format, then versioned, went v7→v8). The GVN
collapse below is measured with canonical vp, so it reflects *trained* value
numbering, not only the structural rename collapse.

## T1

d_model 128, 4 heads, 2 layers, d_out 64, 16 event channels, 250k steps, single
instructions, 8 anchor states drawn from one fixed seed. Held out:

| metric | value |
|---|---|
| equivariance, state max error | `0.00e+00` (exact, trained model) |
| equivariance, essence max error | `1.79e-07` (float noise) |
| tag-invariance cosine | mean `1.0000` / min `0.9992` |
| live_in / live_out | 99.9% / 100% |
| pc | 100% |
| in_slot / out_slot | 100% / 100% |
| duplicate slot predictions | 0 |

Its binding heads are accurate enough for T2 to route off T1's predicted
binding.

## Tier recursion — T2 routed off T1's predicted binding

A T2 that reads each instruction's wiring `(in0, in1, out)` from decoded
RISC-V tokens is handed the ground-truth dataflow out of band: it tests the
architecture given true wiring, not the tier recursion, and would score
identically if T1's binding were garbage. The run below routes T2's wiring off
the frozen T1's **predicted** binding (`argmax`/top-2 of its in/out-slot
scores, gated by predicted live counts, x0 excluded), never tokens. `argmax` of
an equivariant readout permutes with renames, so equivariance stays exact.

472k steps, then resumed for 2M; branch-terminated chunks capped at 4
instructions, d_out 512, 16 event channels, no pairwise loss (losses identical
to T1's). Held out, routed off predicted binding:

| metric | value | note |
|---|---|---|
| live_in / live_out | 99.7% / 100% | |
| pc | 98.4% | |
| in_slot / out_slot | **91.5%** / 99.9% | 93.8% in_slot when routed off token wiring: the cost of imperfect predicted binding |
| duplicate slot predictions | 0 | |
| equivariance (chunk) | `0.00e+00` | exact, composes over the chunk |
| GVN rename-twin distance | 0.0080 (ratio 0.012) | structural |
| **GVN behavioral distance (commutative swap, no rename)** | **0.0238 (ratio 0.036)** | trained value numbering |
| GVN distinct distance | 0.6593 | |

Commutative operand swaps collapse ~28× tighter than distinct pairs: the
register-state machine performs trained value numbering, beyond the structural
rename collapse. Rename twins dominate twin-heavy corpora, so only the split
by kind shows this; the flat T2 of Phases 11–13 never did. **Scope:** these GVN
numbers come from a probe whose rename row drew independent random tags for
base and twin, so they are not comparable to any later GVN measurement
(Phase 15, finding 12).

vp's absolute value is not used as a metric in this phase. The two baselines
run against it do not bound it: an opcode oracle of the same width fails to
regress wrapped and bitwise R-type ops, but it is given opcodes without
dataflow; a program-one-hot oracle overfits (train 0.21 / held out 2.7), which
is memorisation failing to generalise. Single-instruction vp reaches exactly
0.000 (Phase 15).

## Throughput

The frozen-T1 encode, predicted-binding routing, T2 core and losses fold into
one compiled CUDA-graph step at a fixed padded instruction count (a dense
`(B, MAX_NI)` grid built on the host, padding gated by `active=0`), bit-identical
to the ragged path (active-slot essence max |Δ| 1.8e-7, padding leak exactly 0).
Eager per-step dispatch runs at 66% GPU utilization.

**Step time by change.** Measured on the len=2 T2 cell (d_out 512, op-cell
depth 4, 16 pool heads, vp head depth 1, 384 rows per step, RTX 4090), fed from
a fixed on-disk corpus so generation cannot be the limit; steady-state ms/step
from step 1500 onwards. Each row adds one change to the row above it:

| change | ms/step | GPU util |
|---|---|---|
| checkpoint written at every 100-step log line | 8.14 | 73% |
| checkpoint every 10k steps | 6.02 | 96% |
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
with no host work, runs at 3.75 ms: with the whole step captured, the host
issues one launch per step and the GPU never waits on it. Fed live over the
network from a remote generator host, the step runs at 3.96 ms and 97.6% GPU
utilization. Every change computes the same function of the same inputs up to
float rounding — it skips work whose result is exactly known (a register slot
no instruction touched is an exact zero row, so any linear map of it is its
bias), moves host work to another process, or runs the same Adam update
through different kernels. Seeded 300-step runs against the first row's code
agree at step 0 to ≤1.5e-5 relative on every loss and correlate at 0.9999999
over the run; on real checkpoints the essence is bit-identical and the vp loss
agrees to 2e-7. The T1 trainer runs through the same captured step: from one
initial state on one batch, its losses agree with the first row's code to
1.4e-5 relative compiled and 2.6e-7 eager. Resuming the len=2 cell from a
step-2000 checkpoint written by either code trains on indistinguishably from an
uninterrupted run: over the next 2000 steps, across one uninterrupted and three
resumed runs, each run's loss sits 0.30–0.36% on average from their per-step
median.

Seeded runs are not bit-reproducible: the compiled Adam update is not
order-stable, so two runs of identical code drift apart from the first few
steps, by ~1e-4 in parameters after 300 steps — the same size as the
difference from the first row's code. With fused Adam instead of the compiled
update the same comparison is bit-exact (3.7e-9); the compiled update is worth
~0.5 ms of a ~4.3 ms step.

End to end: 1M-step len=2 T2 cells on an ALU corpus with dataflow chains, over
one frozen T1, scored by exact 32-bit vp match on one shared 1200-batch corpus
(each pair is generator seeds 0 and 7):

| trainer | exact match |
|---|---|
| none of the step-time changes | 99.8969% (one cell) |
| through the single host→device copy | 99.9069%, 99.9007% |
| + compiled clip and Adam, shared-memory prefetch | 99.8731%, 99.8950% |
| the same with fused Adam (the control) | 99.8932%, 99.8936% |
| + whole step captured as one CUDA graph | 99.9020%, 99.8792% |

Every row computes the same function up to rounding, and the batch stream
differs between any two runs (the remote generator pool interleaves its workers
nondeterministically), so the nine cells are draws from one distribution: mean
99.893%, standard deviation 0.011 points per cell. No row stands apart.
Binding accuracy is identical across them and the commutative-swap collapse
reads 0.038–0.041 throughout.

**Where the GPU time goes.** Nsight Systems over ~1,570 captured fp32 steps
puts the GPU at 3.63 ms of kernel time per step: matmuls 1.96 ms (54%), fused
elementwise and reduction kernels 1.26 ms (35%) — of which the compiled clip
and Adam is ~0.5 ms — the frozen T1's attention 0.24 ms (6.5%), the rest
0.18 ms. Under bf16 the matmuls fall to 1.32 ms while the elementwise kernels
barely move (1.16 ms); several halve exactly with their bytes, the signature of
memory-bound work. The card's ceilings, measured on it: 157 TFLOPS on a large
bf16 matmul and 87 TFLOPS TF32, drawing ~420–465 W; 1.0 TB/s on a large copy,
drawing ~250–335 W. The trainer's ~340 W at 98% utilization sits at the
memory-bound end of that range: the step is mostly data movement and matmuls
too small to approach peak — a property of the model's shape (d_out 512, 384
rows, a 32-slot state rewritten every instruction), not of the trainer.

**What does not help.** Two cells sharing the GPU run at 15.5 ms/step each —
129 steps/s together against 166 for one alone at the 6.02 ms configuration;
time-slicing costs more than it hides. More rows per step barely raise the
rate a row is trained at: with the whole step captured, 768 rows run at 7.07
ms and 1536 at 13.80 against 3.88 for 384 — 10% and 12% more rows per second —
so GPU time already scales almost with the work, and more rows would change the
training (fewer optimizer steps per row) for little. Compiling Adam alone gains
nothing (0.862 against 0.873 ms for fused Adam): the gain comes from fusing it
with the clip scaling. Running the frozen T1's attention through plain matmuls
instead of the memory-efficient kernel saves 2% and moves T1's essence by up
to 2.6e-4 (TF32 matmuls against the kernel's fp32), a precision change for too
little. One intra-op thread is slower (5.08 against 4.66 ms); four match
sixteen file-fed and are faster live (below). Coordinate-descent kernel tuning
changes nothing (5.275 against 5.272 ms), and neither Python's garbage
collector nor its GIL switch interval costs anything measurable.

**Feeding it.** At 3.88 ms the trainer consumes ~260 batches/s, ~120 MB/s of
uncompressed batch data. Generation is far from the limit — the remote pool
delivers ~600 batches/s over a gigabit link at a load of ~14 of its 128 cores,
and lz4 holds the wire to ~21 MB/s at 180 batches/s. The trainer's own
per-instruction split is the CPU-bound stage: one prefetch worker delivers ~240
batches/s, so the prefetcher runs four (~540/s). The workers also build and
pack each step's device inputs into a ring of slots in a `/dev/shm` file, so
the main thread copies one buffer; sending the ~0.7 MB packed step through the
queue instead costs the main thread more to unpickle than the packing saves.
The queue holds 128 batches because the multi-host stream has lulls of tens of
milliseconds, which a 4-batch queue passes straight to the GPU as idle time.
Intra-op threads are capped at four: with sixteen, their spinning contends with
the stream's receive path (network receive, lz4, the local multiplexer) and the
live step runs 7% slow.

**bf16 autocast — measured, not adopted.** Running everything after the
frozen-T1 front-end under bf16 autocast (T1 stays fp32, so T2 is fed the T1 it
was validated with) brings the captured step to 3.00 ms file-fed and 3.07 live,
against 3.88 / 3.96 in fp32/TF32 — 1.29× the rows per second — with its GPU
floor at 2.94 ms. On the shared corpus two bf16 cells read 99.9053% and
99.9074% exact against 99.9069% and 99.9007% for fp32 cells of the same code
(depth-2: 99.09% / 99.11% against 99.08% / 99.06%), all inside the 0.011
per-cell spread above; binding and the collapse rows are indistinguishable. It
changes the computation, so adopting it is a decision rather than a
measurement.

## Scope and what it hands on

The equivariance skeleton holds end to end at chunks of up to 4 instructions
ending in a branch: T1, T2 on frozen T1, exact equivariance, trained value
numbering. A non-renameable PC slot, memory ops (whether the essence can
capture value-dependent aliasing), immediates, and integration with the
streaming shift-reduce parser for validity are outside it (RISCV_DESIGN.md,
"Open problems"). The Hungarian-bijection behavioral distance is
rename-*invariant* (operator-only), so it is the wrong training target for a
rename-*equivariant* representation that must keep binding recoverable; it is
not part of this design.

---
