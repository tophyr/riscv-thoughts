# Phase 10: Magnitude-as-Validity Encoder Retrain

Implements the design decisions from Phase 9: T1 lives in the
unit ball B^d (no `F.normalize` on encoder output). Magnitude
encodes validity/confidence; direction encodes semantics.
Encoder trained with both a direction loss (existing pair MSE
+ dest CE on valid rows only, using normalized direction) and
a magnitude loss (MSE of `||T1||` against a binary target —
1 for valid single instructions, 0 for partial / spanning /
multi / bogus invalid windows).

## Architecture changes

**Encoder.** Drop `F.normalize` in `_encode_fn` and
`encode_soft`. Encoder outputs `proj(last_repr)` raw. Default
`max_window` reduced from 64 to 32 (the prior default was
oversized — single instructions are ≤9 tokens, multi-instr
windows ≤27).

**Consumer split.** Two distinct policies for what reads from
T1:

- **Classification heads (dest_type, dest_reg) read direction
  (T1 / ||T1||).** They classify what the instruction *is*;
  magnitude shouldn't enter that. Reading raw T1 gave heads a
  gradient incentive to inflate magnitude (linear classifiers
  prefer larger inputs) and produced magnitude-dependent
  behavior on streaming inputs at inference time. Direction is
  semantically the right input.
- **Decoder (when recon enabled) reads raw T1.** Its
  conditioning strength should scale with magnitude — at
  ||T1|| ≈ 1 it reconstructs confidently; at ||T1|| ≈ 0
  cross-attention's K/V projections collapse to ~0, the
  residual stream runs as an unconditioned language model, and
  the decoder produces high-entropy output sampled from the
  marginal distribution. No threshold needed for "invalid"
  handling — degradation is continuous and automatic.

## Pipeline changes

**RVB v2.** Bumped format version from 1 to 2; added
`valid_mask: (B,) bool` as the 7th field. V1 files are no
longer readable (regenerate; cheap).

**Invalid-window generators** (datagen/invalidity.py):
- `partial`: `encode(A)[j:k]` excluding the full-instruction
  case.
- `spanning`: `encode(A)[j:] + encode(B)[:k]` excluding
  full-A + full-B.
- `multi`: 2-3 complete instructions concatenated.
- `bogus`: random tokens drawn from the non-special
  vocabulary.

Configured per-batch via the existing config-dict mechanism
under `invalidity` (mirrors `equivalences`):

    "invalidity": {
      "rate": 0.2,
      "types": { "partial": 0.4, "spanning": 0.3,
                 "multi": 0.2, "bogus": 0.1 }
    }

`produce_instruction_batch` zeros out execution-state fields
(data_vals, pc_vals, dest_types, dest_regs) on invalid rows;
the training loop is responsible for filtering via valid_mask.

## Experiment 34: 50K-step encoder retrain

Configuration: d_model=128, n_heads=4, n_layers=2, d_out=64,
max_window=32. Loss = mag_loss (w=1.0) + mse_loss + dt_loss +
dr_loss + 0.05·eq_loss. No recon (decoder not trained in this
run). Invalidity rate 0.2 with the default type weights.
Batch size 4096. 50K steps at LR 3e-4 with cosine decay.
Training time ~1h15m at ~85ms/step (CUDA, RTX 4090, 69% GPU
utilization — CPU-bound on the data pipeline).

**Convergence trajectory** (||T1|| means; valid / invalid):
- step 100:    0.96 / 0.84 (drift up; dest heads still
  pulling on magnitude before they saturate)
- step 200:    1.26 / 0.65
- step 20000:  1.00 / 0.02 (steady state reached)
- step 50000:  1.00 / 0.01 (final)

**Final probe** against 10K held-out windows (2K per class,
five classes: valid / partial / spanning / multi / bogus):

| class    | mean ||T1|| | std    |
|----------|-------------|--------|
| valid    | 1.000       | 0.010  |
| partial  | 0.007       | 0.009  |
| spanning | 0.006       | 0.004  |
| multi    | 0.004       | 0.002  |
| bogus    | 0.007       | 0.039  |

Magnitude-threshold accuracy (||T1|| > 0.5 ⇒ valid):
**99.8%.** Linear probe on raw T1 reaches only 88.6% — the
validity signal is purely radial, and a linear-over-direction
classifier can't read it. The magnitude-only check is the
clean signal.

**Equivalence eval at 50K:** 11/13 PASS, 1 WEAK
(commutative_xor at 0.385 — just over the 0.3 threshold),
1 FAIL (sign_test at 0.826 — same known failure mode as
Exp 26, requires state-precondition injection rather than
more training). Comparable to the prior encoder which
reached 13/13 PASS at 100K. The geometry recovered without
regression while gaining the validity signal.

## Comparative perf

A/B run before launch:

| config                          | ms/step (steady) |
|---------------------------------|------------------|
| recon=0, invalidity=0           | 159              |
| recon=1, invalidity=0.2 (smoke) | 550              |

The 3.5× difference is the recon REINFORCE loop (K=10 sample
decoder forwards + 80 batch_execute calls per step). The
ball-vs-sphere change itself isn't measurable.

For this validity-only run, recon was off. With the chunked
data pipeline (CPU-side gen on remote + lz4+nc transport)
and no decoder loop, the 50K run completed in ~1h15m.

## Architectural inference

Once magnitude carries validity, **the gates have nothing
semantic to learn at T1.** The previous gate-training plan
(Step 5: REINFORCE accept/evict, supervised emit on parser
ground-truth) becomes a near-trivial exercise:
- emit just thresholds ||T1||
- accept thresholds it the other way (plus has-input-remaining)
- evict at T1 is purely a structural state-tracker that has
  no semantic content to extract from (T1, gru_h)

Evict is genuinely cross-level: at T1 with no T2, "should
this token be evicted" reduces to "has it already been
emitted," which is bookkeeping. At T2+ it becomes "has T(n+1)
absorbed this emission yet" — that's a real cross-level
signal worth learning. T1 evict supervision would just be
memorizing a structural rule.

Implication for the roadmap: Step 5 (T1 gates) is either a
near-trivial training exercise for architectural consistency
or skippable entirely. The research frontier moves to T2.

## T2 design crystallized

A T2 thought is a **register-state transformation block**: a
maximal contiguous subsequence of instructions where only the
LAST may be a memory access (load/store) or control-flow
change (branch/jump). All preceding instructions are pure
register ALU ops.

Termination at memory ops keeps T2 thoughts bounded — only
register state evolves inside a thought, which is finite.
Termination at control flow is the standard basic-block
boundary. Examples:
- `[ADD, SUB, SLT, JAL]` — one T2 thought (3 ALU + jump).
- `[LW]` alone — one T2 thought.
- `[ADDI, ADD, SW]` — one T2 thought.

**Phase framing.** This is a chosen unit definition for
benchmarking the recursive shift-reduce machinery at level 2,
not a declaration about cognition. See WHAT_IS_A_THOUGHT.md
"Method vs. Cognition" for the Phase 1/2/3 framing. Phase 1
prescribes boundaries and verifies the algorithm; Phase 2
weakens supervision and observes emergence; Phase 3 applies
to natural language. The order matters — you can't do Phase
3 without first verifying the algorithm works.

## Transferable lessons from Phase 10

- **Magnitude-as-validity transfers across consumers
  asymmetrically.** Classifiers want magnitude-invariant
  inputs (semantic properties don't depend on confidence);
  generators want magnitude-scaled inputs (graceful
  degradation via cross-attention K/V scaling). The same
  output vector serves both with different normalization
  policies at the consumer side.
- **A test that "just works" can be the failure mode.** The
  v1 retrain converged to clean 0/1 magnitudes despite the
  dest-head magnitude-pulling pressure — the heads saturated
  and stopped pushing. We almost concluded the consumer
  split wasn't needed. The deeper architectural reason
  (inference-time behavior on intermediate magnitudes,
  T2-composition coherence) made it the right change anyway.
- **Heavy-handed unit definitions are appropriate during
  method validation.** The discomfort of "I'm prescribing
  what a thought is" belongs to the natural-language stage,
  not to the prescribed-boundary design choices that exercise
  the method (WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").

---
