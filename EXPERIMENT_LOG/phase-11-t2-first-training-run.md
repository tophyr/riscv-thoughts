# Phase 11: T2 Compressor — First Training Run

Builds the T2 encoder, splits the chunker into a CPU-side data-pipe
tool, and trains T2 to completion on the pipeline. T2 lives in the
unit ball with the same magnitude-as-validity geometry as T1. The
encoder consumes a sequence of T1 emission vectors (one per
instruction) and emits a single T2 vector per chunk.

## Architecture decisions (Exp 35-37)

**Unit definition.** A T2 thought is a register-state transformation
block: a maximal contiguous subsequence of instructions where only
the LAST may be a memory access (load/store) or control-flow change
(branch/jump). Termination at memory ops keeps T2 bounded — only
register state evolves inside a thought, which is finite. A
16-instruction max-length cap forces a split for ALU runs that
don't naturally terminate. See WHAT_IS_A_THOUGHT.md
"Method vs. Cognition" for the framing.

**Aux heads.** Three heads on T2 output, all read normalized
direction `T2 / ||T2||` (magnitude-invariant, same consumer-split
fix as T1):

- `reg_effect_head`: 32-D linear. Per-register pairwise MSE
  generalizes T1's scalar pair-MSE to a per-register dimension.
  For each pair `(a, b)` of valid chunks and each register `r`,
  the encoder is trained such that
  `|reg_effect_head(T2_dir[a])[r] - reg_effect_head(T2_dir[b])[r]|`
  matches `mean_s log1p(log1p(|delta_r[a, s] - delta_r[b, s]|))`.
- `modified_regs_head`: 32-D BCE on whether each register was
  modified in any probed input state. Disambiguates "agreed on a
  value" from "neither touched the register" (both produce zero
  pair-MSE distance otherwise).
- `terminator_type_head`: 5-class CE on terminator category
  (ALU-only / LOAD / STORE / BRANCH / JUMP). T1's `dest_type`
  analog lifted to chunks.

**Loss composition (4 terms):** `mag_loss` (MSE of `||T2||` against
valid_mask, all rows) + `reg_effect_loss` (per-register pair-MSE,
valid only) + `modified_regs_loss` (BCE, valid only) +
`term_type_loss` (CE, valid only).

## Pipeline architecture (Exp 38-39)

The chunker started in-process with the trainer, but pure data-prep
(boundary detection, invalidity augmentation) shouldn't consume
training-loop wallclock. Refactored into a three-stage pipe:

```
gen_seq_batches.py  →  chunk_t2.py  →  train_t2.py
   (RVS)              (CPU only,         (GPU; runs T1
                      RVC format)         on per-instr
                                          tokens)
```

- **`datagen/chunkgen.py`**: pure-CPU boundary detection +
  invalidity augmentation + RVC binary I/O. New format,
  registered in `scripts/_batch_util.py`.
- **`scripts/chunk_t2.py`**: pipe tool, RVS → RVC. Mux-friendly.
- **`compressor/chunker.py`**: shrunk to ~100 lines. Just
  `encode_chunkbatch(rvc_batch, t1_encoder)` — runs T1 batched on
  per-instruction tokens, returns a `T2Batch` of T1 emissions.
- **`compressor/train.py:train_t2()`**: reads RVC, accumulates
  chunks across batches until target_valid_chunks_per_step,
  trains.

This decoupling let the chunker run in parallel with training and
opened the door to multi-process generation via `mux_batches.py`.

## Where the throughput went (Exp 40)

The naive in-process chunker bottlenecked the training loop
(~200ms/step at target=1024). What each change bought, to 99% GPU
utilization:

| stage | step time at target=4K | GPU util |
|---|---|---|
| in-process chunker, no compile | 800-900ms | low |
| pipe-parallel chunker, no compile | 570ms | ~46% |
| pipe-parallel + bf16 autocast | 570ms | ~46% |
| pipe + bf16 + chunker compile + T2 compile | 388ms | ~58% |
| corpus replay (no live data feed) | 388ms | ~58% |
| **target=8K, chunked pair-MSE** | 1032ms | **99%** |
| target=16K, chunked pair-MSE | 3120ms | 99% |

**Key findings:**
- bf16 autocast halves per-register pair-MSE memory; T1's
  exec-distance metric already used bf16 — direct precedent.
- The chunked pair-MSE memory footprint scales as
  `O(chunk_size × B)` rather than `O(B²)`, allowing larger B.
  Implemented via `torch.utils.checkpoint` for autograd correctness;
  ~8% wallclock overhead.
- `torch.compile` on T2's forward gave ~15% step-time improvement
  at small batch (target=1K) and larger gains at higher batches.
- The unchunked full pair-MSE OOMs at target=4K (33GB shared
  memory spilled), even with bf16. Chunked path is the
  permanent answer.
- At target=8K chunked, the GPU is already at 99% util. Larger
  batches (16K) maintain util but trade per-step cost for cleaner
  pair-MSE gradients.

The live-pipeline performance with 4 parallel `gen_seq_batches`
workers via `mux_batches.py` matches corpus-replay speed within
noise, confirming this single 16-core machine can sustain the GPU
without external data generation (no need for the remote-data-gen
pattern T1 used).

## Experiment 41: T2 first training run

Configuration: T2Compressor with d_model=256, n_heads=4, n_layers=2,
d_out=256, max_chunk_len=24. Frozen T1 encoder (`runs/20260424_190609`,
d_out=64). Pipeline: 4 mux'd seq generators → chunker → trainer.
Target 16,384 valid chunks per training step (pair-MSE chunked at
chunk_size=128). Invalidity rate 0.2 (spanning/multi/overlong).
5000 steps with cosine-anneal LR 3e-4 → 1e-6. Total wall: ~4 hours.

Saved: `runs/20260427_022927_t2/t2.pt`.

**Final-step training metrics:**

| metric | value |
|--------|-------|
| total loss | 1.124 |
| mag_loss | 0.004 |
| reg_effect_loss | 0.905 |
| modified_regs_loss | 0.201 |
| term_type_loss | 0.013 |
| ‖T2‖ valid (mean) | 0.99 |
| ‖T2‖ invalid (mean) | 0.09 |
| ‖T2‖ max | 1.20 |

Three of four losses settled cleanly to near-zero. `reg_effect_loss`
plateaued at ~0.91 from early in training and barely moved.

## Experiment 42: T2 validity probe

Generated 500 chunks per class (one valid + three invalid types:
spanning, multi, overlong). Encoded each through frozen T1 + trained
T2; measured `‖T2‖`.

**Per-class magnitude statistics:**

| class | mean | std |
|-------|------|-----|
| valid | 0.979 | 0.121 |
| spanning | 0.091 | 0.047 |
| multi | 0.118 | 0.120 |
| overlong | 0.054 | 0.009 |

**Magnitude-threshold (`‖T2‖ > 0.5`) accuracy: 99.0%.**
Linear-probe-on-raw-T2 accuracy: 98.8%. Comparable to T1's 99.8%
threshold accuracy from Phase 10. The magnitude-as-validity geometry
transferred from T1 to T2 cleanly.

## Experiment 43: T2 equivalence probe

Constructed 200 randomized pairs each across five categories of
instruction-level transformations. The first four are
execution-equivalent transformations (T2 should collapse them);
the fifth is a negative control that should NOT collapse.

**Within-pair directional distance `||T2_dir[a] - T2_dir[b]||`:**

| class | mean | std | interpretation |
|-------|------|-----|----------------|
| commutative_swap (rs1↔rs2 in commutative ops) | 0.003 | 0.002 | tight collapse ✓ |
| double_to_shl1 (`ADD x,y,y` ≡ `SLLI x,y,1`) | 0.003 | 0.001 | tight collapse ✓ |
| x0_writes_nop (different op writes to x0) | 0.026 | 0.013 | partial collapse |
| indep_reorder (independent ALU op order) | 0.074 | 0.030 | partial collapse |
| **NON_EQUIV (one source register changed)** | **0.002** | **0.001** | **WRONG: collapsed** |
| random pairs across batch | 0.594 | 0.615 | (well-separated) |

**CORRECTION (later): `indep_reorder` was mis-categorized here.** It sits in
this table among classes expected to collapse, and its 0.074 is labelled
"partial collapse" — i.e. read as a shortfall. It is not one. Reordering
independent instructions is equivalent only *with register identity attached*;
in the canonical identity-free frame it permutes the first-read input order and
the write order, so the two chunks are different ordered I/O functions and the
canonical `out_regs` oracle gives them **different** per-output-slot vp targets
(verified directly; see RISCV_DESIGN.md "GVN: the invariant essence is the
value-number"). The supervision therefore separates them on purpose, and a
nonzero distance is correct behavior. Note the row immediately below it in the
"What T2 *did* learn" list already reads the same number the right way
("instruction order matters"), so this table and that list were in conflict.
The measurement stands; only the interpretation changes.

**Result: T2 has weak sensitivity to source-register operand changes
within a fixed chunk shape.** The non-equivalent control differs by
exactly one source register in one instruction, but T2 maps both
to nearly identical points (distance 0.002 — within noise of the
truly-equivalent classes).

What T2 *did* learn:
- Magnitude-as-validity (Exp 42).
- Coarse chunk structure: instruction order matters
  (indep_reorder shifts T2 by ~0.07), opcode identity matters
  (x0_writes_nop with different opcodes shifts by ~0.03), shape
  changes matter (random pairs at 0.59).
- Some specific structural equivalences: commutative-swap and
  double-to-shl1 collapse cleanly, suggesting T2 can learn
  per-equivalence-class structure when training signal supports it.

What T2 did *not* learn:
- Fine-grained operand-level semantics. Changing which register
  is read doesn't shift T2's output.

## Diagnosis: under-converged reg_effect_loss

The equivalence probe result is consistent with the training
trajectory: `reg_effect_loss` plateaued at 0.91 from early in
training and barely moved. That's the loss that should have pushed
T2 to discriminate per-register effects. It didn't drop because:

- **Loss-weight imbalance.** At end of training, `mag_loss=0.004`,
  `term_type_loss=0.013`, `modified_regs_loss=0.201`,
  `reg_effect_loss=0.905`. The structural aux losses saturated
  fast and dominated optimizer steps in early training; by the
  time they were done, learning rate had decayed and
  reg_effect_loss couldn't catch up.
- **Pair-MSE noise floor.** The execution-distance target is
  `mean_s log1p(log1p(|delta_r[a,s] - delta_r[b,s]|))` averaged
  over n_inputs=4 random input states. For pairs of chunks that
  affect the same registers similarly, the per-input target
  varies a lot — the mean is noisy.
- **Pair sampling sparsity.** Even at B_v ≈ 13K per step, any
  specific operand-level discrimination ("differ on rs2 only")
  is rare among pairs. Most pairs differ in many ways
  simultaneously, masking the operand-only signal.

## Phase 11 conclusions

**Architecturally validated:**
- Recursive shift-reduce works at level 2.
- Magnitude-as-validity transfers from T1 to T2.
- Chunker-as-pipe-tool architecture composes cleanly.
- Three-stage pipeline (gen → chunk → train) sustains GPU at
  99% utilization on a single 16-core machine.

**Open and worth investigating:**
- Operand-level semantic discrimination is undercooked in T2.
  Possible follow-ups: longer training; higher reg_effect_weight;
  more probed input states for cleaner pair-MSE targets; loss-
  curriculum that doesn't let aux losses drown the directional
  signal; richer per-register loss formulation.
- The `reg_effect_loss` plateau may also reflect a real noise
  floor of the metric. Measuring the distribution of pair-MSE
  targets vs predictions per-register could distinguish "metric
  noise" from "encoder underfitting".

**Transferable lessons from Phase 11:**
- **Probe-driven validation matters.** The raw training metrics
  showed a "successful" run (clean magnitudes, three losses
  saturated). Only the equivalence probe surfaced that T2 learned
  syntax (chunk shape) better than semantics (operand identity).
  Without a targeted probe of execution-equivalent vs non-equivalent
  pairs, the run would have been declared a clean success.
- **Loss weights need active management when goals differ across
  losses.** Magnitude-as-validity saturates fast; pair-MSE-style
  losses don't. With equal weights, the easy losses get optimized
  first, then the LR schedule decays before the harder loss
  converges.
- **Pipe-architecture decoupling pays off.** Splitting the chunker
  out of the trainer freed CPU work to run in parallel and made
  the data path inspectable + parallelizable. The same
  architectural pattern T1's batchgen tools follow.

---
