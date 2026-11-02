# Phase 11: T2 Compressor — First Training Run

Phase 10 gave T1 magnitude-as-validity and defined the T2 unit. This phase
builds the T2 encoder, moves chunking out of the trainer into a CPU-side
pipeline stage, and trains T2 to completion. T2 lives in the unit ball with the
same magnitude-as-validity geometry as T1: it consumes the sequence of T1
vectors for a chunk (one per instruction) and emits one T2 vector.

## Architecture (Exp 35-37)

**Unit.** A T2 thought is a register-state transformation block: a maximal
contiguous run of instructions where only the LAST may be a memory access or a
control-flow change (Phase 10). A 16-instruction cap splits ALU runs that do
not terminate on their own.

**Aux heads.** Three heads on the T2 output, all reading the direction
T2 / ‖T2‖ (the Phase 10 consumer split):

- **Register-effect head**, 32-D linear: a per-register generalization of T1's
  scalar pair MSE. For each pair `(a, b)` of valid chunks and each register `r`,
  `|head(T2_dir[a])[r] − head(T2_dir[b])[r]|` is trained to match
  `mean_s log1p(log1p(|delta_r[a, s] − delta_r[b, s]|))` over probed input
  states s.
- **Modified-registers head**, 32-D BCE on whether each register was modified
  in any probed input state. It separates "agreed on a value" from "neither
  touched the register", which give the same zero pair distance.
- **Terminator-type head**, 5-class CE on the terminator category (ALU-only /
  LOAD / STORE / BRANCH / JUMP) — T1's destination type lifted to chunks.

**Loss (4 terms):** magnitude (MSE of ‖T2‖ against the valid mask, all rows) +
register-effect pair MSE + modified-registers BCE + terminator-type CE (the last
three on valid rows only).

## Pipeline (Exp 38-39)

Chunking (boundary detection and invalidity augmentation) is pure data
preparation, so it runs as its own CPU process between the instruction-sequence
generator and the trainer:

```
sequence generator  →  chunker (CPU)  →  T2 trainer (GPU; runs frozen T1
                                         on per-instruction tokens)
```

The trainer only runs T1 batched over each chunk's per-instruction tokens and
accumulates chunks across batches up to a target count of valid chunks per
step. The chunker runs in parallel with training, and several generators can
be multiplexed into one stream.

## Throughput (Exp 40)

With the chunker in-process the step was ~200 ms at a target of 1024 valid
chunks. Step time and GPU utilization by configuration:

| stage | step time at target=4K | GPU util |
|---|---|---|
| in-process chunker, no compile | 800-900ms | low |
| pipe-parallel chunker, no compile | 570ms | ~46% |
| pipe-parallel + bf16 autocast | 570ms | ~46% |
| pipe + bf16 + chunker compile + T2 compile | 388ms | ~58% |
| corpus replay (no live data feed) | 388ms | ~58% |
| **target=8K, chunked pair-MSE** | 1032ms | **99%** |
| target=16K, chunked pair-MSE | 3120ms | 99% |

- bf16 autocast halves the per-register pair-MSE memory.
- Computing the pair MSE in chunks makes its memory `O(chunk_size × B)` rather
  than `O(B²)`, using `torch.utils.checkpoint` for autograd, at ~8% wallclock.
  The unchunked pair MSE runs out of memory at target=4K (33GB spilled to shared
  memory) even under bf16.
- `torch.compile` on T2's forward gives ~15% at target=1K and more at larger
  batches.
- At target=8K the GPU is at 99%; 16K keeps utilization and trades per-step
  cost for cleaner pair-MSE gradients.

Four multiplexed generators feeding the live pipeline match corpus-replay speed
within noise, so one 16-core machine sustains the GPU without remote data
generation.

## Experiment 41: T2 first training run

d_model=256, n_heads=4, n_layers=2, d_out=256, max chunk length 24 tokens.
Frozen T1 with d_out=64. Four multiplexed generators → chunker → trainer.
16,384 valid chunks per step (pair MSE in chunks of 128). Invalidity rate 0.2
(spanning / multi / overlong). 5000 steps, cosine LR 3e-4 → 1e-6, ~4 hours.

**Final-step training metrics:**

| metric | value |
|--------|-------|
| total loss | 1.124 |
| magnitude | 0.004 |
| register-effect | 0.905 |
| modified-registers | 0.201 |
| terminator-type | 0.013 |
| ‖T2‖ valid (mean) | 0.99 |
| ‖T2‖ invalid (mean) | 0.09 |
| ‖T2‖ max | 1.20 |

Three of four losses settle near zero; the register-effect loss plateaus at
~0.91 early and barely moves.

## Experiment 42: T2 validity probe

500 chunks per class (valid, and three invalid types), encoded through the
frozen T1 and the trained T2:

| class | mean ‖T2‖ | std |
|-------|------|-----|
| valid | 0.979 | 0.121 |
| spanning | 0.091 | 0.047 |
| multi | 0.118 | 0.120 |
| overlong | 0.054 | 0.009 |

Magnitude threshold (‖T2‖ > 0.5): **99.0%** accuracy; a linear probe on raw T2:
98.8%. T1's threshold accuracy in Phase 10 is 99.8%: magnitude-as-validity
transfers from T1 to T2.

## Experiment 43: T2 equivalence probe

200 random pairs in each of five categories of instruction-level
transformation, plus random pairs across the batch for scale. The table gives
the within-pair directional distance `‖T2_dir[a] − T2_dir[b]‖`.

| class | mean | std | reading |
|-------|------|-----|---------|
| commutative_swap (rs1↔rs2 in commutative ops) | 0.003 | 0.002 | equivalent; collapsed |
| double_to_shl1 (`ADD x,y,y` ≡ `SLLI x,y,1`) | 0.003 | 0.001 | equivalent; collapsed |
| x0_writes_nop (different ops writing x0) | 0.026 | 0.013 | equivalent; partially collapsed |
| indep_reorder (independent ALU ops swapped) | 0.074 | 0.030 | different ordered I/O functions; separated |
| **one source register changed (control)** | **0.002** | **0.001** | **not equivalent; collapsed — wrong** |
| random pairs | 0.594 | 0.615 | separated |

Reordering independent instructions preserves behavior only with register
identity attached. Without it, the reorder permutes the first-read input order
and the write order, so the two chunks are different ordered I/O functions and a
nonzero distance is correct (RISCV_DESIGN.md, "GVN: the invariant essence is the
value-number").

**Result: T2 is nearly blind to source-register operand changes within a fixed
chunk shape.** The control pair differs in exactly one source register of one
instruction, and T2 places them 0.002 apart — within the spread of the truly
equivalent classes.

T2 does learn magnitude-as-validity (Exp 42); coarse chunk structure
(instruction order shifts T2 by ~0.07, opcode identity by ~0.03, shape by
~0.59); and two specific equivalences (commutative swap and double-to-shift
collapse). It does not learn operand-level semantics: changing which register
is read does not move T2.

## Candidate causes of the register-effect plateau (unmeasured)

The register-effect loss is the one that should separate per-register effects,
and it stalls at ~0.91. Three candidate causes, none isolated in this phase:

- **Loss-weight imbalance.** The structural losses saturate fast and dominate
  early optimizer steps; by the time they are done the learning rate has
  decayed.
- **Pair-target noise.** The target averages over only 4 random input states;
  for chunks that affect the same registers similarly, the per-state target
  varies widely.
- **Pair sparsity.** Even at ~13K valid chunks per step, pairs that differ only
  in one operand are rare; most pairs differ in many ways at once.

## Conclusions

Established: the recursive shift-reduce design works at level 2;
magnitude-as-validity transfers from T1 to T2; the three-stage pipeline
(generate → chunk → train) holds a single 16-core machine's GPU at 99%. Left
for the next phase: T2 does not discriminate operands, and whether the
register-effect plateau is a noise floor of the metric or encoder underfitting
is unresolved.

**Transferable lessons:**
- **Probe before declaring success.** The training metrics read as a clean run
  (clean magnitudes, three losses saturated); only the equivalence probe showed
  that T2 learned chunk shape rather than operand semantics.
- **Losses with different difficulty need active weighting.** With equal
  weights the easy losses are optimized first and the LR schedule decays
  before the hard one converges.
- **Decoupling data preparation into a pipe stage pays off**: it runs in
  parallel with training and makes the data path inspectable.

---
