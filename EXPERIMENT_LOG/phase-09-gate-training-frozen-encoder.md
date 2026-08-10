# Phase 9: Gate Training on Frozen Encoder/Decoder

Picking up Step 3 of the piecemeal plan (Phase 8 conclusions):
supervise the three shift-reduce gates against a frozen encoder
(Exp 28 / 21) and a frozen decoder (Exp 30, 8L/d256/64m @ 96% peak
tok_acc). The task looked trivial on paper — every emit point is
syntactically marked by the current window forming a complete
RV32I instruction, every accept should fire until that point,
every evict should fire after. Supervision is just parser output.
In practice, gate training plateaued at ~70% emit-transition
accuracy across eight mechanism variants. Diagnosing why led to
two findings that invalidate the Step 3 approach as currently
architected.

## Experiment 31: Supervised + REINFORCE gate training (8 variants)

Architecture unchanged from Exp 23: GRU gate controller reading T1
candidates causally, three independent 2-layer MLP heads for
accept/emit/evict. Training loop vectorized (see the
`_shift_reduce_step` / `torch.compile` refactor that landed with
this phase) for ~10× throughput.

**Supervision sources:**
- `batch_is_complete_instruction` (added for this phase): strict
  parser-side completeness check, stricter than
  `batch_parse_tokens`.
- Per-iteration decoder reconstruction loss on the current window
  (as in Exp 23), used as REINFORCE reward.

**Variants tried:**

| v | Mechanism | Outcome |
|---|-----------|---------|
| 1 | Trajectory-level REINFORCE on reconstruction loss | Never-emit collapse — reward=0 beats any negative trajectory reward, so the policy that never fires emit is a strict global optimum |
| 2 | Added exploration-encouraging bias init (emit_head bias=0.0 so ~50% emit at step 0) | Collapse deferred but identical steady state |
| 3 | Threshold-based reward: `(thresh − loss) × emit_sampled` so correct emits are positive and missed emits are negative | Policy moves but noisy; ~50% emit rate, uniform across iterations — emit gate doesn't learn boundaries |
| 4 | Added per-step advantage normalization after REINFORCE magnitude was ~5-10× the BCE term | Scale issue resolved, but still no boundary learning |
| 5 | Supervised BCE on emit against `batch_is_complete_instruction`; REINFORCE only on accept/evict | Emit tracks supervision to ~70%, accept/evict remain noisy |
| 6 | Transition-based reward: r(t) = valid(t) AND NOT valid(t−1), so only fresh-valid transitions count | Cleaner signal, same ~70% plateau |
| 7 | Discounted reward-to-go (γ=0.9) for accept/evict | No improvement |
| 8 | Larger batch (B=128) + longer run (10K steps) | ~70% plateau persists |

**Steady state across variants (v5-v8):** emit transition accuracy
~60-75%, emissions per sequence in the right ballpark (0.8-1.2× the
instruction count), but emits distributed noisily across the
window rather than locked to boundaries.

## Experiment 32: Parser-signal bug discovered in `batch_is_complete_instruction`

Debugging why training plateaus uncovered that the parser-side
completeness check itself was buggy. The check used fixed expected
lengths per opcode type (R=4, I/LOAD/STORE/JALR=6, BRANCH/LUI/AUIPC=7,
JAL=8), but the tokenizer emits a variable-length NEG token for
negative immediates. A negative-imm I-type instruction tokenizes
to 7 tokens, not 6 — and was therefore labeled *incomplete*.

A synthetic probe of 2000 randomly-generated valid instructions
showed:

| Tokenized length | Marked complete |
|------------------|-----------------|
| 4 (R-type) | 100% ✓ |
| 6 (pos-imm I/LOAD/STORE/JALR) | 100% ✓ |
| 7 (neg-imm I/LOAD/STORE/JALR, or pos-imm BRANCH/LUI/AUIPC) | 45% |
| 8 (pos-imm JAL or neg-imm BRANCH) | 16% |
| 9 (neg-imm JAL) | 0% |

Only ~71% of truly-complete windows were labeled complete. The
emit-gate BCE target was "don't emit at valid boundaries with
negative immediates." This is a strict lower bound on what gate
training could have achieved regardless of mechanism — the
supervision itself was inconsistent.

Fix: detect the NEG token at the opcode-specific imm-start
position and adjust expected length by +1 when present. Tested.
See the "Add batch_is_complete_instruction with NEG-aware length
check" commit.

## Experiment 33: T1 linear probe — encoder is the bottleneck

With the parser fixed, the next question was whether a clean
supervision signal would unblock gate training or whether the
problem was deeper. A linear probe of T1 for "is this a complete
instruction" answers this directly.

**Setup.** Frozen encoder (runs/20260416_153534, d_out=64, 2 layers).
Generate 10K windows in five balanced classes:

- A: complete single instruction
- B: partial (prefix of one instruction, length 1..len−1)
- C: spanning (full instruction + partial next)
- D: two complete instructions concatenated
- E: bogus (random tokens)

Encode each with the frozen encoder, train a logistic-regression
probe on the 64-dim T1 vector to predict `is_complete` (post-fix
labels).

**Results:**

| Probe | Val acc | Majority baseline |
|-------|---------|-------------------|
| Linear (T1 → logit) | **81.5%** | 79.4% |
| MLP (T1 → 128 → logit) | 90.3% | 79.4% |

Linear probe beats the majority baseline by 2.1 points. On class
A (true positives) it achieves only **22.2% accuracy** — it
predicts "not complete" for the majority of actually-complete
windows. The non-linear probe extracts ~10 points of additional
signal but still falls short of the clean supervision ceiling.

**Interpretation.** The frozen encoder was trained on single
complete RV32I instructions via pairwise MSE on register-delta
distance (Exp 19-30). Completeness was never an axis of
supervision. It simply did not learn to separate
complete-instruction windows from partial, spanning, multi-
instruction, or bogus windows in the direction of any particular
linear direction on S^127.

Because the emit head reads T1 linearly (2-layer MLP with the
first layer reading T1 directly), **no amount of gate training
on this encoder can recover boundary behavior from a
representation that does not carry the signal.**

## Decisions

1. **Pause gate training.** Returning to it would require either
   an architecturally different gate head (reading raw tokens —
   duplicating encoder work) or a new encoder. The latter is the
   right answer.

2. **Retrain the encoder with validity awareness.** Augment pair-
   training data with invalid windows (partial / spanning / multi-
   instr / bogus) drawn at parity with the valid N×N batch, and
   supervise validity explicitly.

3. **Represent validity as T1 magnitude in the unit ball B^128,
   not direction on S^127.** Drop the final `F.normalize` in the
   encoder. Valid thoughts live near the sphere (||T1||≈1);
   invalid windows collapse toward the origin (||T1||≈0). The
   origin becomes a structurally singular "not-a-thought" point.
   Validity is then a linearly-readable feature of T1 itself,
   and the emit gate can trivially extract it.

4. **Do not renormalize composed thoughts downstream.**
   Magnitude carries compositional-coherence signal — a failed
   composition (sum of incompatible thoughts) naturally lands
   near origin. Renormalization erases exactly that signal.
   Leave the geometry honest, let the decoder see the confidence
   in its conditioning.

5. **Keep the binary target for initial training.** Complete = 1,
   everything else = 0 — even windows that contain a complete
   instruction plus partial next (`[ADD,r4,r2,r7,BEQ,r4]`) are
   labeled invalid. Whether the geometry develops useful
   intermediate-magnitude behavior organically is a later
   question (possibly relevant for T2 composition).

The abandoned gate-training loop (`train_gates()`) has been
removed from the codebase. The shift-reduce architecture itself
— encoder, GRU controller, three gate heads, decoder, the
vectorized `_shift_reduce_step` — is retained. Gate training
resumes once the encoder carries validity in T1.

**Transferable lessons:**
- Probe before you tune. Eight variants of mechanism weren't
  going to beat a representation that structurally lacks the
  target feature. A 30-line linear probe would have revealed
  the bottleneck before the first training run.
- Verify supervision signals against ground truth. The NEG
  length bug in `batch_is_complete_instruction` silently
  corrupted training for every run in Exp 31.
- "Trained on single valid instructions" is not the same as
  "recognizes valid instructions." The encoder learned the
  semantic task it was given; it did not learn the
  discrimination task we wanted it to perform downstream.

---
