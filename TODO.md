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
- [x] Decoder capacity sweep: 8L/d128 plateaus at 85.5% (2.1× params
      too few); 16L/d512 broken by depth-induced bootstrap failure;
      8L/d256 and 8L/d512 reach 95%+ train tok_acc. (Exp 30)
- [x] Generalization confirmed: held-out tok_acc = 72% across all
      decoder sizes — the encoder's T1 geometry sets a hard
      generalization ceiling, decoder capacity only adds memorization
      on top. (Exp 30)
- [x] Equivalence tolerance emerges: 23.4% execution-equivalent rate
      on held-out autoregressive decoding, without explicit
      equivalence training on the decoder. (Exp 30)
- [x] Training infrastructure: WSD schedule, gradient clipping,
      phase-transition checkpoints, micro-batching. (Exp 30)
- [ ] (Deferred) Joint encoder+decoder fine-tuning — would target
      the 72% generalization ceiling, but not necessary for
      thesis validation. Revisit if gate training needs higher
      decoder quality.

### Step 4b: Encoder retraining — magnitude as validity
Triggered by Phase 9 / Exp 33: the prior encoder placed all
compressed windows on the unit sphere with no readable validity
signal. Fix: move T1 into the unit ball and train magnitude as
validity.

- [x] Drop `F.normalize` in the encoder.
- [x] Extend `train_batches` with on-the-fly invalid-window
      augmentation (partial / spanning / multi / bogus). Built
      into the CPU pipeline via the `invalidity` config block.
- [x] Magnitude loss: MSE of `||T1||` against a binary target.
- [x] Pair MSE + destination CE on valid rows only, using
      normalized direction `T1/||T1||` (dest heads) and the
      direction tensor (pair MSE).
- [x] Log `||T1||` per step (valid/invalid means + max). No
      regularization added — magnitudes settled cleanly at
      ~1.0/0.02/1.05.
- [x] Architectural cleanup: dest heads read normalized
      direction (magnitude-invariant classification); decoder
      reads raw T1 (graceful degradation via cross-attention
      magnitude scaling).
- [x] Probe `runs/20260424_190609`: 99.8% magnitude-threshold
      accuracy; valid mean 1.000±0.010, invalid means <0.01.
- [x] Equivalence eval at 50K steps: 11/13 PASS, 1 WEAK
      (commutative_xor at 0.385), 1 FAIL (sign_test, same as
      old encoder; not a regression). Geometry preserved.
- [ ] (Optional) longer training to close commutative_xor.
      Not blocking.

### Step 5: T1 gates — reframed
Original plan: train accept/emit/evict via REINFORCE +
decoder-quality reward. New understanding (post-magnitude-as-
validity): T1 gates are essentially trivial — emit just
thresholds `||T1||`; accept just thresholds it the other way
plus `has_input_remaining`; evict at T1 is purely a structural
state-tracker (genuinely cross-level at T2+ where it has
non-trivial signal from the level above).

- [ ] (For architectural consistency only) Train T1 gate heads
      with structural-rule supervision against the frozen new
      encoder. Will learn near-trivial functions; useful as
      placeholders for the symmetric three-heads-per-level
      architecture.
- [ ] (Decision pending) OR skip learned T1 gates entirely
      and use procedural shift-reduce for T1, defer learned
      gates to T2+ where they have real signal.

### Step 6: Assembly + decoder retrain
- [ ] Decoder retrain against the new ball-encoder. Same
      REINFORCE + execution-equivalence reward as Exp 30, but
      conditioned on raw T1 (graceful degradation property).
- [ ] End-to-end T1 eval: encoder → emit → decoder → token
      reconstruction accuracy on held-out sequences.
- [ ] (Optional) Joint fine-tune across encoder + decoder +
      gates if any of the components plateau.

## Next: T2 Design and Implementation

### Step 7: T2 unit definition (DONE in design)
- [x] Defined: a T2 thought = maximal contiguous subsequence
      of instructions where only the last may be a memory
      access or control-flow change. See WHAT_IS_A_THOUGHT.md
      "Method vs. Cognition" for the framing of why we make
      this choice now.

### Step 8: T2 data pipeline (DONE)
- [x] RVC binary format: per-chunk per-instruction token spans +
      chunk metadata (length, validity, type, reg_delta). See
      `datagen/chunkgen.py`.
- [x] CPU-only chunker pipe tool: `scripts/chunk_t2.py` reads
      RVS, emits RVC. Mux-friendly via existing pipeline tools.
- [x] Invalidity augmentation at the chunker (spanning / multi
      / overlong) parametrizable via config-style weights.

### Step 9: T2 encoder (DONE in design + first run)
- [x] Architecture: bidirectional transformer over the chunk's
      T1 emissions, same shape as T1's window encoder lifted
      one level. d_in=64 (T1's d_out), d_model=256, n_heads=4,
      n_layers=2, d_out=256. Magnitude-as-validity in B^256.
- [x] Aux heads: reg_effect (32-D linear, used for per-register
      pair-MSE), modified_regs (32-D BCE), terminator_type
      (5-class CE). All read normalized direction.
- [x] Compiled forward pass via torch.compile (dynamic=True).

### Step 10: T2 training (FIRST RUN COMPLETE — partial success)
- [x] train_t2() in compressor/train.py with four-loss
      structure: mag + reg_effect (per-register pair-MSE,
      chunked via gradient checkpointing for memory) +
      modified_regs + term_type.
- [x] bf16 autocast in the loss-computation block; halves peak
      memory in the dominant pair-MSE intermediates.
- [x] Pipeline architecture: gen_seq_batches | chunk_t2 |
      train_t2; mux'd 4 parallel gen workers feed the
      single-trainer GPU at 99% utilization on this machine
      alone (no remote-data-gen needed).
- [x] First training run (Exp 41): 5000 steps at target=16K,
      ~4h wall, completed cleanly. `runs/20260427_022927_t2`.
- [x] Validity probe (Exp 42): 99.0% magnitude-threshold
      accuracy. Magnitude-as-validity transferred from T1.
- [x] Equivalence probe (Exp 43): commutative_swap and
      double_to_shl1 collapse cleanly; x0_writes_nop and
      indep_reorder partial; **NON_EQUIV control collapses
      too** (T2 doesn't discriminate operand-level changes
      within a fixed chunk shape).
- [ ] Diagnose / fix `reg_effect_loss` plateau at 0.91. Could
      be loss-weight imbalance, pair-MSE noise floor, or
      pair-sampling sparsity. See Phase 11 conclusions.

### Step 11: T1 ↔ T2 cross-level integration (DEFERRED)
Pending T2 semantic-quality fix (Step 10 follow-up). Once T2
discriminates operand-level changes meaningfully:

- [ ] T2 → T1 evict signal. T2's accept gate firing tells T1
      "I've absorbed this emission, you can evict it." Either
      cross-attention from T2 state, or a separate channel.
- [ ] At this point evict has real, learnable signal at T1 —
      the cross-level absorption acknowledgment. Train T1
      evict head against this.

### Step 10b: T2 training quality follow-ups (NEW)
Targeted at the operand-level discrimination gap surfaced in
Exp 43.

- [ ] Loss-weight rebalancing experiment: keep mag/term/mod at
      1.0, raise reg_effect_weight (e.g., 5x) to prevent the
      easy aux losses from dominating early training.
- [ ] Loss-curriculum experiment: warm up aux losses first,
      then ramp up reg_effect_weight as they saturate.
- [ ] Probe the reg_effect_head output distribution
      (predictions vs targets per register) to distinguish
      "metric noise floor" from "encoder underfitting".
- [ ] More probed input states (n_inputs=8 or 16) for cleaner
      pair-MSE targets.
- [ ] Longer training runs (10K-20K steps) under the rebalanced
      losses.

## Future

- Longer sequences (scale beyond max_block_len=5).
- T3 stacking (sequences of T2 blocks → function-like units).
- Decoder improvements (Gumbel-softmax differentiable decoding;
  execution-equivalent grading instead of token-exact matching).
- Phase-2/Phase-3 experiments (loosened boundary supervision;
  natural language). See WHAT_IS_A_THOUGHT.md "Method vs.
  Cognition" for the framing.
