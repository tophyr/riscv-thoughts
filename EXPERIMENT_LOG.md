# Experiment Log

Chronological record of all experiments with full results and analysis, one
file per phase in [`EXPERIMENT_LOG/`](EXPERIMENT_LOG/). This file is the index.
For theoretical framework, see WHAT_IS_A_THOUGHT.md. For testbed design,
see RISCV_DESIGN.md. For lessons from prior work, see
TRANSFERABLE_LESSONS.md.

**What to read.** Phase 16 is the live record — the register frame and its
open problems. Phases 14 and 15 are the first-read frame's record: the
equivariant skeleton and the faithfulness work, whose dead ends and
measurements are load-bearing and whose numbers are not comparable to phase
16's. Everything before Phase 14 is superseded design, kept because the dead
ends are why the current architecture is shaped the way it is, and re-running
them is the most common way to waste a week here. Read an older phase when
you are about to revisit the question it settled.

| # | Phase | What it established |
|---|---|---|
| 1 | [T0→T1 ALU compression](EXPERIMENT_LOG/phase-01-t0-t1-alu-compression.md) | First T1: correlation loss over execution-state distances gives smooth, factored geometry with opcode clustering — learned from execution semantics, not hand-designed similarity |
| 2 | [All RV32I instruction types](EXPERIMENT_LOG/phase-02-all-rv32i-instruction-types.md) | Generalized 19 ALU opcodes → all 37 RV32I types, full x0–x31; two-component (data_val + pc_val) execution comparison |
| 3 | [MSE on unit sphere](EXPERIMENT_LOG/phase-03-mse-on-unit-sphere.md) | Replaced Pearson + weighted-equivalence with direct MSE distance matching on S^d |
| 4 | [Structured random input states](EXPERIMENT_LOG/phase-04-structured-random-input-states.md) | Pure random int32 anchors make BEQ never branch and crush SLT/SLTU; structured inputs fix the false SLT/BEQ/LW cluster |
| 5 | [Gradient decoder](EXPERIMENT_LOG/phase-05-gradient-decoder.md) | T1 geometry is smooth enough to invert by gradient search; a decoding failure indicts the encoder geometry, not the search |
| 6 | [Sequential context](EXPERIMENT_LOG/phase-06-sequential-context.md) | Whether a preceding instruction improves the T1 vector — the pre-streaming check on window/evict policy |
| 7 | [Streaming compressor with decoder](EXPERIMENT_LOG/phase-07-streaming-compressor-with-decoder.md) | Emit gate + autoregressive decoder trained jointly; decoder reconstruction as the primary signal |
| 8 | [Shift-reduce gate training](EXPERIMENT_LOG/phase-08-shift-reduce-gate-training.md) | The causal accept/emit/evict parser. Gumbel-sigmoid gives zero gradient through Python control flow; gates need Bernoulli + REINFORCE |
| 9 | [Gate training on a frozen encoder](EXPERIMENT_LOG/phase-09-gate-training-frozen-encoder.md) | Plateaued ~70% across eight variants — the encoder structurally lacked a validity feature. **Probe before you tune.** Motivated Phase 10 |
| 10 | [Magnitude-as-validity](EXPERIMENT_LOG/phase-10-magnitude-as-validity.md) | T1 moves into the unit *ball*: magnitude = validity, direction = semantics. No `F.normalize` on the output |
| 11 | [T2 first training run](EXPERIMENT_LOG/phase-11-t2-first-training-run.md) | First T2 over a sequence of frozen-T1 vectors; the chunker becomes a CPU-side data-pipe tool |
| 12 | [T2 loss-set convergence](EXPERIMENT_LOG/phase-12-t2-loss-set-convergence.md) | Many head designs and output geometries converge back to one open-space vector with behavioral losses; value-prediction alone never builds an operator axis |
| 13 | [Pairwise behavioral geometry](EXPERIMENT_LOG/phase-13-pairwise-behavioral-geometry.md) | Built and **set aside** the pairwise/twins-as-positives line. A rename is a non-abelian permutation, so no additive vector can represent it (the torsion theorem) — confirmed empirically at the random baseline |
| 14 | [Rename-equivariant T1/T2](EXPERIMENT_LOG/phase-14-rename-equivariant-t1-t2.md) | **The encoder's architecture.** A register-indexed state machine makes rename equivariance exact by construction; T1 is T2 at seq_len=1. Trained global value numbering (GVN): commutative swaps sat ~28× closer than distinct chunks, on a probe not comparable to later ones |
| 15 | [T2 faithfulness](EXPERIMENT_LOG/phase-15-t2-faithfulness.md) | The first-read frame's record (inputs numbered by textual first-read order); superseded by phase 16. A vp head trained at one fixed set of anchor states stores answers, so anchors are drawn afresh per chunk and vp is scored at held-out anchors. There T1 is exact and T2 at len=2, with a twelve-layer residual vp head, is 99.9993% after 10M steps, one seed. The frame gives reordered independent instructions and dead code different targets from their equivalents, which is why phase 16 replaces it |
| 16 | [The register frame](EXPERIMENT_LOG/phase-16-register-frame.md) | **IN PROGRESS — the live record.** Leaves identified by structural colour and value prediction read from the essence make rename, reordering and dead code collapse by construction, with T1 exact and T2 len=2 at 99.83%. Collapse beyond that is trained by a behavioral distance term over inputs placed by canonical role; the frozen T1 `runs/t1_distance_cosbal_500k` is exact at held-out anchors, and a T2 at len=2 trained on partner pairs drawing every rewrite collapses every equivalence class tried at vp 99.79%, one seed. InfoNCE alone flattens the grading; a collapse lapses when its pairs leave the training mix, and reaches no class the mix does not draw. The distance target is rebuilt on the register-file map and partner pairs are also found by search and proved: the record T2 collapses every class the probes build at vp 99.80%, and a family whose pairs are held out of training does not collapse |
