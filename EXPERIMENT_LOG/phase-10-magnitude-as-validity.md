# Phase 10: Magnitude-as-Validity Encoder Retrain

Phase 9 found that the frozen T1 does not carry instruction completeness, so
no gate trained on it can find instruction boundaries. This phase retrains T1
in the unit ball B^d (no `F.normalize` on the encoder output): magnitude
encodes validity, direction encodes semantics. The encoder is trained with a
direction loss (pair MSE + destination CE on valid rows only, computed on the
normalized direction) and a magnitude loss (MSE of ‖T1‖ against a binary
target — 1 for a valid single instruction, 0 for a partial, spanning,
multi-instruction or bogus window).

## Architecture

**Encoder.** The output is `proj(last_repr)` with no normalization. The
maximum window is 32 tokens: a single instruction is ≤9 tokens and a
multi-instruction window ≤27.

**Consumer split.** Two policies for what reads T1:

- **Classification heads (destination type, destination register) read the
  direction T1 / ‖T1‖.** They classify what the instruction *is*, which does
  not depend on confidence. Reading raw T1 gives a linear classifier a gradient
  incentive to inflate magnitude and makes its behavior on streaming inputs
  depend on magnitude.
- **The decoder reads raw T1.** Its conditioning strength scales with
  magnitude: at ‖T1‖ ≈ 1 it reconstructs confidently; at ‖T1‖ ≈ 0 the
  cross-attention keys and values collapse to ~0, the residual stream runs as
  an unconditioned language model, and the output is high-entropy samples
  from the marginal distribution. Degradation on invalid input is continuous,
  with no threshold.

## Invalid windows

Each batch carries a per-row valid mask. Invalid windows come from four
generators:
- **partial**: `encode(A)[j:k]`, excluding the full instruction.
- **spanning**: `encode(A)[j:] + encode(B)[:k]`, excluding full A + full B.
- **multi**: 2-3 complete instructions concatenated.
- **bogus**: random tokens from the non-special vocabulary.

In this phase 20% of rows are invalid, split partial 0.4 / spanning 0.3 /
multi 0.2 / bogus 0.1. Execution-state targets are zeroed on invalid rows and
the trainer masks them out.

## Experiment 34: 50K-step encoder retrain

d_model=128, n_heads=4, n_layers=2, d_out=64, max window 32. Loss = magnitude
(weight 1.0) + pair MSE + destination-type CE + destination-register CE +
0.05·equivalence. No reconstruction loss (decoder not trained). Invalidity rate
0.2 with the split above. Batch 4096, 50K steps, LR 3e-4 cosine. ~1h15m at
~85 ms/step on an RTX 4090 at 69% GPU utilization (CPU-bound on the data
pipeline).

**Convergence** (mean ‖T1‖, valid / invalid):
- step 100:    0.96 / 0.84 (the destination heads still pull on magnitude
  before they saturate)
- step 200:    1.26 / 0.65
- step 20000:  1.00 / 0.02 (steady state)
- step 50000:  1.00 / 0.01

**Probe** on 10K held-out windows (2K per class):

| class    | mean ‖T1‖ | std    |
|----------|-----------|--------|
| valid    | 1.000     | 0.010  |
| partial  | 0.007     | 0.009  |
| spanning | 0.006     | 0.004  |
| multi    | 0.004     | 0.002  |
| bogus    | 0.007     | 0.039  |

Magnitude threshold (‖T1‖ > 0.5 ⇒ valid): **99.8%** accuracy. A linear probe
on raw T1 reaches only 88.6%: the validity signal is radial, and a linear
classifier cannot read a norm.

**Equivalence eval at 50K:** 11/13 PASS, 1 WEAK (commutative_xor at 0.385,
just over the 0.3 threshold), 1 FAIL (sign_test at 0.826 — the Exp 26 failure
mode, which needs state-precondition injection rather than more training). The
preceding T1 encoder, trained on valid instructions only, reached 13/13 PASS at 100K
steps, so the semantic geometry holds while gaining the validity signal.

## Cost of the reconstruction loss

| config                                    | ms/step (steady) |
|-------------------------------------------|------------------|
| no reconstruction, no invalid rows        | 159              |
| reconstruction, 20% invalid rows (smoke)  | 550              |

The 3.5× is the reconstruction REINFORCE loop (10 sampled decoder forwards and
80 batch executions per step). The ball-versus-sphere change itself is not
measurable. Experiment 34 ran without reconstruction, fed by a chunked pipeline
generating on a remote CPU host and streaming lz4-compressed batches over the
network.

## Consequence for the gates

Once magnitude carries validity, **the gates have nothing semantic to learn at
T1**: emit thresholds ‖T1‖, accept thresholds it the other way (plus
has-input-remaining), and evict at T1 reduces to "has this token already been
emitted" — bookkeeping. Evict becomes a real cross-level signal only at T2+
("has T(n+1) absorbed this emission yet"). T1 gate training is therefore near
trivial, and the next phase moves to T2.

## T2 unit definition

A T2 thought is a **register-state transformation block**: a maximal
contiguous run of instructions where only the LAST may be a memory access
(load/store) or a control-flow change (branch/jump); all earlier instructions
are register ALU ops. Ending at memory ops keeps a thought bounded — only the
finite register state evolves inside it; ending at control flow is the
standard basic-block boundary. Examples: `[ADD, SUB, SLT, JAL]`, `[LW]`,
`[ADDI, ADD, SW]` are each one T2 thought.

This is a unit chosen to benchmark the recursive shift-reduce machinery at
level 2, not a claim about cognition (WHAT_IS_A_THOUGHT.md, "Method vs.
Cognition": prescribe boundaries and verify the algorithm first, weaken
supervision second, apply to natural language last).

## Transferable lessons

- **Magnitude-as-validity serves consumers asymmetrically.** Classifiers want
  magnitude-invariant inputs (semantics do not depend on confidence);
  generators want magnitude-scaled inputs (graceful degradation through
  cross-attention scaling). One output vector serves both with different
  normalization at the consumer.
- **Clean training metrics do not justify a design.** An encoder whose
  classification heads read raw T1 also converges to clean 0/1 magnitudes,
  because the heads saturate and stop pulling on magnitude. The consumer split
  is justified by behavior at intermediate magnitudes (streaming inference, T2
  composition), which training metrics do not exercise.
- **Prescribed unit definitions are appropriate during method validation.**
  Discomfort with prescribing what a thought is belongs to the
  natural-language stage, not to the choices that exercise the method.

---
