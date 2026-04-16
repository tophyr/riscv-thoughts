# Roadmap

## Current: T1 Foundations

### Step 1: Equivalence manifest and coverage

- [x] Enumerate equivalence classes in a manifest file
      (16 classes across 6 structural axes in
      datagen/equivalences.py)
- [x] Per-class manifest: canonical tuples with free variables
      and constraints, contrast tuples
- [x] Eval harness (scripts/eval_equivalences.py): per-class
      intra-cohesion vs inter-separation with PASS/WEAK/FAIL
- [x] Baseline measurement on pre-foundations encoder (3/3/7)
- [x] Injection mechanism: per-batch canonical-tuple injection
      with configurable rate, min_per_class guarantees, and
      per-class boost
- [x] Explicit equivalence loss: dedicated loss term encoding
      all classes' canonicals each step, ~150,000× gradient
      concentration vs batch-MSE. Combined with 5% injection
      reaches 12/0/1 in 100K steps (Exp 25).
- [ ] (Deferred) Smooth-distance / ordering tests: some
      relationships don't fit the binary equivalence/contrast
      frame (e.g., ADDI imm=0 vs imm=1 vs imm=1000 should
      produce roughly monotonic distances). Handle as a
      separate sensitivity harness. sign_test (the only
      remaining FAIL at ratio 0.94) is this category.
- [ ] (Deferred) State-precondition injection for sign_test:
      SLTI imm=0 vs imm=1 only diverges when rs=0, which
      is rare under random sampling. Needs boosted zero-value
      register states, not more training.

### Step 2: Encoder dimensionality

d_out=128 was chosen early and never revisited. The core thesis
is compression, so d_out should be sized deliberately.

- [x] PCA/SVD on trained d_out=128 outputs: 94 dims for 90%
      variance, participation ratio 84.8. ~70% of the 128-d
      space is actively used.
- [ ] MDS on pairwise exec-distance matrix (may not be needed
      given PCA results and the d_out sweep below)
- [ ] d_out sweep {4, 8, 16, 32, 64, 128} with equiv loss +
      injection. d_out=64 run in progress.
- [ ] Measure per-class cohesion and contrast separation at
      each d_out; find where classes start breaking
- [ ] Commit to final d_out

## Next: T1 Piecemeal Assembly

With the manifest and a chosen d_out in hand, assemble T1's
components one at a time.

### Step 3: Encoder (final training)
- [x] RVB single-instruction batch format and pipeline tools
- [x] Format auto-detection (RVB/RVS) in all pipeline tools
- [x] N×N pairwise training loop with nested-log metric,
      dest-type/dest-reg CE heads, and explicit equiv loss
- [x] Pearson > 0.85 on held-out data (achieved 0.91)
- [x] Non-equivalent same-opcode separation (12/13 manifest
      classes PASS via eval harness)
- [ ] Final production training run at chosen d_out

### Step 4: Decoder (reconstruction from frozen encoder)
- [ ] Freeze encoder, train decoder with teacher-forced
      cross-entropy
- [ ] Verify: reconstruction accuracy > 95%
- [ ] Verify: equivalence interpolation still works (midpoint
      decoding test from Exp 21)

### Step 5: Gates (supervised + REINFORCE)
- [ ] Freeze encoder + decoder
- [ ] Train GRU gate controller with REINFORCE
- [ ] Decoder runs every iteration → per-iteration quality signal
- [ ] Verify: >90% of emissions at instruction boundaries

### Step 6: Assembly + fine-tune
- [ ] Unfreeze all components
- [ ] Fine-tune jointly on RVS sequence data
- [ ] Losses: reconstruction (window-size weighted) + pairwise MSE
      + round-trip
- [ ] Verify: assembled system matches step 4 accuracy

## After: T2 Stacking

### Step 7: T2 compressor
- [ ] Same shift-reduce architecture, consuming T1 emission vectors
- [ ] T2 feedback into T1 via cross-attention
- [ ] Fine-tune T1+T2 jointly
- [ ] Verify: T2 discovers block boundaries

## Future

- Longer sequences (scale beyond max_block_len=5)
- T3+ stacking
- Decoder improvements (Gumbel-softmax for differentiable decoding,
  execution-equivalent grading instead of token-exact matching)
