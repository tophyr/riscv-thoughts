# Roadmap

## Current: T1 Piecemeal Assembly

### Step 1: Encoder (N×N pairwise MSE training)
- [x] RVB single-instruction batch format and pipeline tools
- [x] Format auto-detection (RVB/RVS) in all pipeline tools
- [x] N×N pairwise training loop using computed-value execution
      distance (the metric from Exp 3+ that actually works)
- [ ] Train encoder on RVB batches with muxed generation
- [ ] Verify: Pearson > 0.85 on held-out data
- [ ] Verify: immediate sensitivity (ADDI imm=0 vs imm=1 > imm=1000)
- [ ] Verify: non-equivalent same-opcode separation

### Step 2: Decoder (reconstruction from frozen encoder)
- [ ] Freeze encoder, train decoder with teacher-forced cross-entropy
- [ ] Verify: reconstruction accuracy > 95%
- [ ] Verify: equivalence interpolation still works (midpoint
      decoding test from Exp 21)

### Step 3: Gates (supervised + REINFORCE)
- [ ] Freeze encoder + decoder
- [ ] Train GRU gate controller with REINFORCE
- [ ] Decoder runs every iteration → per-iteration quality signal
- [ ] Verify: >90% of emissions at instruction boundaries

### Step 4: Assembly + fine-tune
- [ ] Unfreeze all components
- [ ] Fine-tune jointly on RVS sequence data
- [ ] Losses: reconstruction (window-size weighted) + pairwise MSE
      + round-trip
- [ ] Verify: assembled system matches step 2 accuracy

## Next: T1 Foundations Pass

Once the pieces fit together and the theory is validated end-to-end,
go back and correct the foundational choices that were made before
we had a working training regimen.

### Step 5: Equivalence manifest and coverage

The N×N pairwise MSE only exerts collapse pressure on pairs that
co-occur in a batch. Random sampling under-represents rare
equivalences, and many classes require specific register-state
preconditions to be observable. Fix both.

- [ ] Enumerate equivalence classes in a manifest file
      (commutative operand swap; self-op identities XOR/SUB y,y=0
      and AND/OR y,y=y; zero-reg aliases ADD x,x0,y ≡ MV, SUB x,x0,y
      ≡ NEG, XORI x,y,-1 ≡ NOT; write-to-x0 NOPs across all ops;
      shift-by-0 identities; ADD y,y ≡ SLLI y,1; branch-taken-with-
      equal-operands ≡ JAL x0; signed/unsigned comparison coincidence;
      load-immediate-zero via multiple routes)
- [ ] Per-class manifest: canonical tuples, register-state
      preconditions that make the equivalence observable, contrast
      tuples that must stay separate
- [ ] Eval harness: per-class intra-class cohesion and inter-class
      separation metrics on a trained encoder
- [ ] Baseline: measure current d_out=128 encoder per-class; identify
      which classes already collapse vs which need help
- [ ] Design injection mechanism (per-batch tuple injection vs
      auxiliary equivalence loss vs state-precondition generator
      configs vs hybrid)
- [ ] Retrain with injection; verify per-class cohesion improves
      and contrast pairs stay apart

### Step 6: Encoder dimensionality

d_out=128 was chosen early and never revisited. The core thesis is
compression, so d_out should be sized deliberately as a compression
ratio, not left oversized by inertia.

- [ ] PCA/SVD on trained d_out=128 outputs — measure the
      actually-used subspace dimensionality
- [ ] MDS on pairwise exec-distance matrix — measure intrinsic
      dimension from the target geometry side
- [ ] d_out sweep {4, 8, 16, 32, 64, 128} on a consistent training
      setup with step 5's injected signal
- [ ] Measure per-class cohesion and contrast separation at each
      d_out; plot the collapse order of equivalence classes as
      d_out shrinks (the compression-ratio experiment)
- [ ] Commit to final d_out as the smallest value that preserves
      manifest-defined equivalences and contrasts

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
