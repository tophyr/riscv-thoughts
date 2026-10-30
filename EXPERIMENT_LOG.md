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
| 14 | [Rename-equivariant T1/T2](EXPERIMENT_LOG/phase-14-rename-equivariant-t1-t2.md) | **The current architecture.** Register-indexed state machine, equivariance exact by construction; T1 = T2 at seq_len=1; canonical vp fixed; trained GVN collapse (commutativity ~28× vs distinct) |
| 15 | [T2 faithfulness](EXPERIMENT_LOG/phase-15-t2-faithfulness.md) | **IN PROGRESS.** A vp head trained at one fixed set of 8 anchor states stores answers at them, so anchors are drawn afresh per chunk with structure, labels come from fixed witness states, and vp is scored at held-out anchors. T1 is exact there (wiring 100%, vp 99.9999%, with read pulses computed from both operands and a three-layer vp head). Handed a perfect description of a two-instruction chunk, plain ReLU readouts of the vp head's shape are not exact and a twelve-layer residual readout with LayerNorm nearly is; with that head T2 at len=2 is 99.9993% exact after 10M training steps — 36 of 4.8M held-out output values wrong, counting every output register of 38,400 chunks at 64 anchor states — on one seed. Every write and every dead read counts in the targets, so no equivalence is imposed on the binding. Of the equivalences, only a commutative swap of fresh operands collapses, by construction; the first-read frame gives reordered independent instructions and dead code different targets from their equivalents, and computed folds do not collapse. The frame is replaced in phase 16 |
| 16 | [The register frame](EXPERIMENT_LOG/phase-16-register-frame.md) | **IN PROGRESS.** Leaves identified by structural colour, the essence pooled over written slots, value prediction read from the essence through a content-hash pointer per output. T1 exact at held-out anchors (0 of 2.46M); rename, reordering, dead code and fresh commutative swaps collapse to exactly zero by construction; T2 at len=2 99.83% after 1M steps from the essence alone, with composed slots carrying the residual and the essence spanning the first-read frame's range; nothing the op-cell computes collapses, and the essence separates such equivalents by spelling; register-relabelled copies of a chunk duplicate it once the evaluator's input order follows the encoder's colouring; at T1 a behavioral distance term (log-log of output differences over known-bits bounds) collapses equivalent spellings and grades the rest, where InfoNCE alone flattens them, and the immediate head is deleted |
