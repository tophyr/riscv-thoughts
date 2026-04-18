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
- [x] d_out sweep {16, 32, 64, 128} × {2, 4, 8} layers with
      equiv loss + injection. d_out=64 × 2 layers is the sweet
      spot (13/13 PASS, Pearson 0.906). Depth doesn't break
      capacity ceilings. See Exp 26.
- [x] Per-class cohesion and contrast separation measured at
      each d_out. Pearson ceiling is d_out-dependent (~0.91
      at 64, ~0.89 at 32, ~0.86 at 16), not depth-dependent.
- [x] Decoder eval across all checkpoints (Exp 27): frozen
      encoder + trained decoder peaks at 65% token accuracy.
      Gradient-based inversion peaks at 31%. The encoder
      preserves distances but not identity — reconstruction
      loss is needed during encoder training.
- [ ] Commit to final d_out (d_out=64 is the candidate, pending
      decoder-joint-training validation)

## Next: T1 Piecemeal Assembly

### Step 3: Encoder + Decoder (joint training)
- [x] RVB single-instruction batch format and pipeline tools
- [x] Format auto-detection (RVB/RVS) in all pipeline tools
- [x] N×N pairwise training loop with nested-log metric,
      dest-type/dest-reg CE heads, and explicit equiv loss
- [x] Pearson > 0.85 on held-out data (achieved 0.91 at d_out=64)
- [x] Non-equivalent same-opcode separation (13/13 manifest
      classes PASS at d_out=64)
- [x] Token-match CE joint training tested at w=0.1/0.05/0.01:
      teaches decoder (96% tok_acc) but destroys geometry
      (Pearson 0.66). CE and MSE have fundamental tension
      beyond equivalence conflicts. (Exp 28)
- [x] Round-trip loss (Gumbel-softmax) tested: geometry
      preserved (Pearson 0.90) but decoder produces garbage
      that re-encodes nearby. Too permissive. (Exp 28)
- [x] REINFORCE with execution reward: proof of concept works
      (99% valid instructions, 11% execution-equivalent). The
      principled approach but needs faster emulation for
      multi-sample variance reduction. (Exp 28)
- [x] Batched PyTorch RV32I emulator: all 37 opcodes on GPU
      via torch.where, 1.69ms for B=4096. GPU token parser
      for fully-GPU reward pipeline (28ms total). (Exp 29)
- [x] REINFORCE decoder with K=10 + shaped reward reaches
      23% execution-equivalence on frozen encoder. Convergence
      confirmed but rate limited by frozen encoder geometry.
- [ ] Joint encoder + REINFORCE-decoder training (the encoder
      adapts geometry for decodability via execution-equivalence
      reward — no CE, no equivalence conflict)
- [ ] Final production training run at d_out=64
- [ ] Verify: execution-equivalent decoding rate > 90%
- [ ] Verify: equivalence classes still PASS

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
