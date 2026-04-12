# Plan: Streaming Compressor Architecture

## Context

T0→T1 single-instruction compression is complete (17 experiments,
gradient decoder validated). The T1 space on S^127 has good geometry
but known context-free ambiguities (BEQ/SLT cluster, cross-syntax
equivalence) that dissolve with sequential context. Moving to a
multilevel streaming architecture with accept/emit/evict gates.

The existing T1Compressor, pairwise batch training, and batch I/O
infrastructure are superseded. New code replaces rather than extends
the old training pipeline.

## Approach: baby-steps experimentation

The gate training problem is genuinely hard. Every "simple" approach
has a flaw:
- Pairwise MSE alone: underdetermined gate timing, no signal to
  distinguish mid-instruction from boundary emissions
- Density regularization: imposes granularity bias
- Delta predictor: premature — needs a "good vector" target we
  don't have yet
- Reconstruction loss: more premature — needs a decoder, and must
  handle execution equivalences (not exact-match reconstruction)

Rather than committing to one design upfront, **Stage 0 is fixed
and Phase 1 is an experimental space**. We build the data pipeline
cleanly, then run small experiments with different gate/loss
configurations and observe. Committed design for Phase 1 comes
AFTER data narrows the decision.

## Stage 0: Sequence Training Data

### New files:
- `datagen/seqgen.py` — structured synthetic sequence generator
- `scripts/gen_seq_batches.py` — script to produce sequence batches
- `datagen/tests/test_seqgen.py` — tests

### Key components:

**`random_basic_block(rng, max_length) -> list[Instruction]`**:
Generate a block with internal data flow. Maintains a "live register"
set; ~50-70% of instructions read from a register written by a prior
instruction. Final instruction is a branch reading from live set.
Reuses `random_instruction()` from `datagen/datagen.py` for per-
instruction generation, with data-flow constraints on register
selection.

**`execute_sequence(instructions, regs, pc, rng, ctx)`**: Execute
a sequence instruction-by-instruction via `emulator.run()` with
`max_steps=1`, capturing per-instruction state deltas. Returns
final state + list of per-instruction (state, pc) snapshots.

**`SequenceBatch` dataclass**:
- `token_ids` (B, max_total_tokens) — flattened sequence tokens
- `padding_mask` (B, max_total_tokens)
- `token_instr_idx` (B, max_total_tokens) int32 — which instruction
  index each token belongs to (0-based within sequence)
- `n_instructions` (B,) — instructions per sequence
- `per_instr_data_vals` (B, max_instrs, n_inputs) — per-instruction
  data output values across input states
- `per_instr_pc_vals` (B, max_instrs, n_inputs) — per-instruction
  PC values across input states

`token_instr_idx` is the key for gate training: during the forward
pass, the model can track which complete instructions are in the
window by looking at which instruction indices are fully present.

**Binary format**: New magic `RVS\x00` version 1 with batch header
(B, max_tokens, max_instrs, n_inputs). Own `write_seq_batch/
read_seq_batch/read_seq_batch_bytes/SeqBatchReader` following the
same structural pattern as the existing batch I/O.

**Pipeline**: `gen_seq_batches.py` writes to stdout. Training script
reads from stdin. Mux support deferred until needed.

### Existing code reused:
- `emulator.run()`, `make_ctx()`, `random_regs()`, `SparseMemory`
- `tokenizer.encode_sequence()` (already exists)
- `datagen.random_instruction()` — called within `random_basic_block()`
- `datagen.extract_data_val()` — per-instruction output extraction
- `scripts/_batch_util.binary_stdout()`

### Success criteria:
1. Blocks have >50% data-flow dependencies between instructions
2. `execute_sequence` reproduces same final state as `emulator.run()`
   on full instruction list
3. Binary I/O roundtrips perfectly
4. Block lengths match configured distribution

## Stage 1: T1 Streaming Compressor with Gates

### New/rewritten files:
- `compressor/model.py` — rewrite: T1StreamingCompressor replaces
  T1Compressor
- `compressor/train.py` — rewrite: streaming training loop, new loss
- `scripts/train_compressor.py` — rewrite: reads sequence batches,
  trains streaming model
- `scripts/eval_compressor.py` — rewrite: evaluates streaming model
  (boundary detection, emission quality, context effects)
- `compressor/tests/test_compressor.py` — rewrite: tests for
  streaming model

### Architecture:

**T1StreamingCompressor**: processes flat token stream, maintains
gated window, emits compressed vectors on S^127.

- Token embeddings + positional embeddings (relative to window)
- Active window: fixed-capacity buffer (max ~16 tokens ≈ 2 instrs)
- Bidirectional self-attention encoder (2-layer transformer) over
  window contents
- Three gate heads reading from encoder output → scalar logits
- Output projection + F.normalize to S^(d_out - 1)

**Progressive gate unfreezing.** Accept and evict are fixed for
Phase 1; only emit is learned. Accept always fires (consume next
token). Evict uses a **structural policy**: when emit fires, evict
the tokens belonging to instructions that are fully in the window,
leave partial instructions in place. This uses `token_instr_idx`
ground truth as a cleanup policy without supervising the emit gate
itself. Phase 2 adds learned accept; Phase 3 adds learned evict.

This evict policy avoids destroying partial instructions on
mid-instruction emissions (the user's concern with auto-evict),
while still giving us a real streaming architecture (we evict
and move on, unlike no-eviction which produces redundant
overlapping emissions).

Gate implementation: Gumbel-sigmoid with straight-through
estimation. Each gate produces a sigmoid probability per step.
Forward pass uses hard threshold; backward pass routes gradients
through the continuous pre-threshold values.

**Force-emit rules**: force-emit at end of sequence. Force-emit at
max window capacity.

### Loss function:

**Execution distance metric for emissions**: the existing T0→T1
metric (`mean_s(log1p(|data_diff| + |pc_diff|))`) only works for
single instructions. For streaming we use **per-register state
deltas**:

```
delta_i(s, r) = regs_after_i(s, r) - regs_before_i(s, r)
exec_dist(i, j) = mean_s(mean_r(log1p(|delta_i(s,r) - delta_j(s,r)|)))
```

Each emission has effect `(n_inputs, 32)` — register changes from
running the instructions it covers, on each input state. The pairwise
distance compares these deltas across (input × register).

Properties:
- Single-instruction emission collapses to approximately the existing
  metric (only the dest register has nonzero delta)
- Multi-instruction emissions handle cancellation correctly: ADD
  followed by SUB to the same register has net delta zero, and a
  no-op-equivalent emission is correctly zero-distance from other
  no-ops
- Well-defined for any pair of emissions regardless of what they
  cover

**Connecting emissions to targets**: track token→instruction
membership via `token_instr_idx` from SequenceBatch. At emit time,
identify which instructions are FULLY in the window. The emission's
target delta is the cumulative state delta of those complete
instructions, computed per input state.

- Emission covers complete instructions → target is their cumulative
  per-register delta (n_inputs × 32)
- Emission covers 0 complete instructions → masked out of loss

Flatten valid emissions across the batch. Compute pairwise T1
distances (cdist on emissions) and pairwise exec distances (the
register-delta metric above). MSE on sphere.

**Phase 1 is an experimental space, not a committed design.** We
run multiple small experiments with different loss configurations
and pick a committed Phase 1 design based on results.

Base loss for all experiments: **pairwise MSE** between emission
vectors and their target cumulative state deltas, flattened across
the batch. Same proven mechanism as T0→T1. This is the sole loss
for the compressor's vector space.

Experimental variables (gate training signal):

- **Exp A: MSE alone**. Pairwise MSE only. Gate timing is likely
  underdetermined. Observe what the gate naturally does. Is it
  coherent? Random? Degenerate?

- **Exp B: MSE + density** (`T / max(I, 0.1)`). Adds a structural
  bias toward single-instruction emissions. Biased toward our
  assumptions, but the bias matches the architectural intent at
  this level.

- **Exp C: MSE + delta predictor**. Small MLP (2 layers, d_model
  hidden) maps emission → predicted cumulative delta. Per-vector
  quality signal without explicit granularity bias. Biased by
  predictor capacity.

Each experiment is a 1-10K step training run. We compare: gate
firing patterns, loss convergence, pairwise Pearson correlation,
and qualitative emission alignment with instruction boundaries.

**Deferred experiments** (dependent on Exp A-C results):
- Reconstruction loss via a trained decoder. Needs the "several
  valid decodings" grader approach — evaluate whether the decoded
  sequence is execution-equivalent to the original, not whether
  it's identical.
- Round-trip consistency (`v ≈ compress(decode(v))`). Uses the
  compressor as its own verifier. Elegant but requires a decoder
  and the decoder to be differentiable end-to-end.

These deferred experiments are noted as probability-distribution
candidates — possible directions if Exp A-C fail. We don't commit
to them preemptively.

Gate training signal: gates are trained end-to-end through the
emission quality loss. Good gate decisions → better emissions →
lower MSE. No explicit gate supervision — the model discovers
instruction boundaries from the execution signal alone.

Optional gate regularization: penalize very large windows (encourage
timely emission) and very small windows (discourage degenerate
emit-every-token).

No classification heads.

### Training pipeline (experimental):

```bash
# Stage 0: generate sequence data
gen_seq_batches.py --n-batches 10000 --batch-size 256 \
  --max-block-len 3 > short_seqs.bin

# Phase 1 experiments (vary --loss)
train_compressor.py --loss mse < short_seqs.bin             # Exp A
train_compressor.py --loss mse+density < short_seqs.bin     # Exp B
train_compressor.py --loss mse+predictor < short_seqs.bin   # Exp C

# Pick committed Phase 1 design based on results
# Then scale up: longer sequences, more steps
```

Batch size 256 (not 4096) because each sample is ~15-35 tokens.

### Evaluation:
1. Boundary detection: fraction of emissions at instruction
   boundaries
2. Per-emission pairwise correlation (Pearson, Spearman)
3. Context effect: does same instruction produce different emissions
   in different sequential contexts?
4. Equivalence improvement: do ADD-double/SLLI collapse better when
   seen in similar data-flow context?

### Success criteria (Phase 1, emit only, pairwise MSE):
- Per-emission pairwise Pearson r > 0.85 (compressor vector space
  has good geometry, same metric as T0→T1)
- Emit gate firing rate is non-degenerate (fires between
  "force-emit at max capacity" and "force-emit at sequence end")
- Loss converges and doesn't diverge
- Qualitative: inspect when the gate fires. Do emissions align
  with instruction boundaries, or random positions?

Phase 1b success (only if Phase 1 fails):
- Added regularizer (density or predictor) produces sensible gate
  timing without breaking pairwise MSE geometry

Phases 2 and 3 success criteria TBD based on Phase 1 results.

### Risks:
- **Bidi attention homogenization (Lesson 7)**: mitigated by small
  window size and strong execution signal. If it occurs, init from
  pretrained T1Compressor token embeddings.
- **Emit gate degenerate timing**: without rate signal, the gate
  may pick underdetermined or nonsensical moments. Mitigation:
  force-emit rules (end of sequence, max window capacity) guarantee
  some emissions. If the gate's natural timing is bad, add Phase 1b
  regularization.
- **Variable emission counts in batch**: flatten all emissions for
  pairwise comparison; mask appropriately.
- **Window grows too large**: with non-destructive evict, the
  window accumulates tokens. Max capacity limit prevents unbounded
  growth. If capacity is hit often, we learn the window size is
  too small for the task.
- **Phase 2/3 may not work**: progressive unfreezing assumes the
  problem is tractable with each new gate added one at a time.
  If Phase 2 fails, accept may need different training (e.g.,
  REINFORCE) or may need to be replaced with a fixed schedule.

## Stage 2: T1+T2 Feedforward (high-level)

T2Compressor: same gate architecture but over T1 emissions (not
tokens). Input projection replaces token embeddings.

Joint loss: T1 pairwise MSE + T2 pairwise MSE + gate losses.
Monitor gradient norms per level (Lesson 3).

Success: T2 discovers block boundaries (>80% precision), T1 quality
doesn't degrade from joint training.

## Stage 3: Cross-Level Feedback (high-level)

Add cross-attention from T2 state into T1's transformer layers.
Creates recurrence. Monitor gradient norms through feedback path.

Success: T1 emissions improve for long-range dependencies, feedback
loop doesn't diverge.

## Implementation Order

```
0.1: random_basic_block() + execute_sequence()
0.2: SequenceBatch + produce_seq_batch()
0.3: Binary I/O (RVS format)
0.4: gen_seq_batches.py + SeqBatchReader
0.5: Tests
1.1: T1StreamingCompressor architecture (3 gates; fixed accept,
     structural evict-completed, learnable emit; Gumbel-sigmoid)
1.2: Pairwise MSE loss on flattened emissions with per-register
     delta targets
1.3: train_compressor.py with --loss flag (mse, mse+density,
     mse+predictor) + training loop
1.4: Phase 1 experiments: Exp A (mse), Exp B (mse+density),
     Exp C (mse+predictor) on short sequences
1.5: eval_compressor.py: gate firing patterns, emission alignment,
     Pearson correlation, qualitative inspection
1.6: Commit to Phase 1 design based on experiment results, scale up
1.7: Phase 2 design (after Phase 1 committed)
1.8: Phase 3 design (after Phase 2 results)
2.x: T2 stacking (depends on 1.x results)
3.x: Cross-level feedback (depends on 2.x results)
```

## Files deleted or fully rewritten

- `compressor/model.py` — rewritten (T1StreamingCompressor)
- `compressor/train.py` — rewritten (streaming loss + training loop)
- `scripts/train_compressor.py` — rewritten (reads sequence batches)
- `scripts/eval_compressor.py` — rewritten (streaming evaluation)
- `compressor/tests/test_compressor.py` — rewritten
- `compressor/decode.py` — delete (gradient decoder is T1-specific,
  will be rebuilt for streaming when needed)

## Files retained as-is

- `emulator/` — all of it (run, make_ctx, SparseMemory, etc.)
- `tokenizer/` — all of it (encode_sequence, VOCAB, etc.)
- `datagen/datagen.py` — retained (random_instruction, extract_data_val,
  produce_batch for reference; batch I/O still works for legacy data)
- `datagen/__init__.py` — updated with new exports
- `scripts/_batch_util.py` — retained (binary_stdout, validate_batch_header)
- `scripts/gen_batches.py` — retained (can still generate T0-T1 batches)
- Pipeline tools (mux, slice, cat, shuffle, repeat) — retained
- `EXPERIMENT_LOG.md`, design docs — retained

## Verification

Stage 0:
- `pytest datagen/tests/test_seqgen.py` — all pass
- `gen_seq_batches.py --n-batches 5 --batch-size 16` produces valid
  output
- Manual inspection of generated sequences shows data-flow structure

Stage 1:
- `pytest compressor/tests/test_compressor.py` — all pass
- 1K-step smoke test: loss decreases, gates fire non-degenerately
- 10K training run: Pearson r > 0.80, boundary precision > 70%
- Eval shows instruction-boundary-aligned emissions
