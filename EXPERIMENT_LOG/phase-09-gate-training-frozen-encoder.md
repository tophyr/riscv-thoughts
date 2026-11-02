# Phase 9: Gate Training on Frozen Encoder/Decoder

Phase 8 left the three shift-reduce gates (accept, emit, evict) of the
streaming parser to be trained against a frozen encoder (Exp 28 / 21) and a
frozen decoder (Exp 30: 8 layers, d256, 64-dim memory, 96% peak token
accuracy). Supervision is available from a parser: an emit point is where the
current token window forms one complete RV32I instruction, accept should fire
until that point and evict after it. Gate training plateaued at ~70%
emit-transition accuracy across eight mechanism variants. Two causes were
found: only ~71% of complete windows were labelled complete, and the frozen T1 does
not carry instruction completeness as a readable feature.

## Experiment 31: Supervised + REINFORCE gate training (8 variants)

The gate controller is the Exp 23 architecture: a GRU reading T1 candidates
causally, with three independent 2-layer MLP heads for accept, emit and evict.
The training step was vectorized and compiled for ~10× throughput.

Supervision came from two sources: a parser-side check of whether the window
is one complete instruction (the emit target), and the decoder's
reconstruction loss on the current window (the REINFORCE reward, as in Exp
23). Every variant below used the completeness check with the labelling defect
described in Experiment 32.

| v | Mechanism | Outcome |
|---|-----------|---------|
| 1 | Trajectory-level REINFORCE on reconstruction loss | Never-emit collapse — reward=0 beats any negative trajectory reward, so the policy that never fires emit is a strict global optimum |
| 2 | Emit-head bias initialized to 0.0 (~50% emit at step 0) to encourage exploration | Collapse deferred, identical steady state |
| 3 | Threshold reward `(thresh − loss) × emit_sampled`, so correct emits are positive and missed emits negative | Policy moves but noisily; ~50% emit rate, uniform across iterations — the emit gate does not learn boundaries |
| 4 | Per-step advantage normalization (the REINFORCE term was ~5-10× the BCE term) | Scale balanced, still no boundary learning |
| 5 | Supervised BCE on emit against the completeness check; REINFORCE only on accept/evict | Emit tracks supervision to ~70%; accept/evict noisy |
| 6 | Transition reward: r(t) = valid(t) AND NOT valid(t−1), counting only fresh-valid transitions | Cleaner signal, same ~70% plateau |
| 7 | Discounted reward-to-go (γ=0.9) for accept/evict | No improvement |
| 8 | Batch 128, 10K steps | ~70% plateau persists |

Steady state across v5–v8: emit-transition accuracy ~60-75%, emissions per
sequence 0.8-1.2× the instruction count, but emits spread noisily across the
window rather than locked to instruction boundaries.

## Experiment 32: The completeness labels were wrong for negative immediates

The completeness check used a fixed expected token length per opcode type
(R=4, I/LOAD/STORE/JALR=6, BRANCH/LUI/AUIPC=7, JAL=8). The tokenizer emits an
extra NEG token for a negative immediate, so a negative-immediate I-type
instruction is 7 tokens and was labelled incomplete.

Over 2000 randomly generated valid instructions:

| Tokenized length | Marked complete |
|------------------|-----------------|
| 4 (R-type) | 100% |
| 6 (pos-imm I/LOAD/STORE/JALR) | 100% |
| 7 (neg-imm I/LOAD/STORE/JALR, or pos-imm BRANCH/LUI/AUIPC) | 45% |
| 8 (pos-imm JAL or neg-imm BRANCH) | 16% |
| 9 (neg-imm JAL) | 0% |

Only ~71% of truly complete windows were labelled complete, so the emit target
in Experiment 31 taught "don't emit at valid boundaries with negative
immediates" — a bound on every variant regardless of mechanism. Detecting the
NEG token at the opcode-specific immediate position and adding 1 to the
expected length corrects the labels.

## Experiment 33: T1 linear probe — the encoder is the bottleneck

With correct labels, a probe asks whether the frozen T1 carries completeness at
all.

**Setup.** Frozen encoder (d_out=64, 2 layers). 10K windows in five balanced
classes:

- A: complete single instruction
- B: partial (prefix of one instruction, length 1..len−1)
- C: spanning (full instruction + partial next)
- D: two complete instructions concatenated
- E: bogus (random tokens)

A logistic-regression probe on the 64-dim T1 vector predicts `is_complete`.

| Probe | Val acc | Majority baseline |
|-------|---------|-------------------|
| Linear (T1 → logit) | **81.5%** | 79.4% |
| MLP (T1 → 128 → logit) | 90.3% | 79.4% |

The linear probe beats the majority baseline by 2.1 points, and on class A
(the complete windows) scores only **22.2%** — it calls most complete windows
incomplete. The MLP probe extracts ~10 points more and still falls short.

**Interpretation.** The encoder was trained on single complete RV32I
instructions via pairwise MSE on register-delta distance (Exp 19-30).
Completeness was never supervised, and the encoder does not separate complete
windows from partial, spanning, multi-instruction or bogus ones along any
linear direction. Because the emit head reads T1 through an MLP whose first
layer is linear in T1, **no gate training on this encoder can recover boundary
behavior from a representation that does not carry the signal.**

## Decisions

1. **Gate training stops.** Resuming it would need either a gate head reading
   raw tokens (duplicating encoder work) or an encoder that carries validity;
   the second is chosen.

2. **Retrain the encoder with validity awareness.** Augment pair-training data
   with invalid windows (partial / spanning / multi-instruction / bogus) at
   parity with the valid batch, and supervise validity explicitly.

3. **Represent validity as T1 magnitude in the unit ball, not as a direction
   on the unit sphere.** Drop the final `F.normalize` in the encoder. Valid
   thoughts sit near the sphere (‖T1‖≈1); invalid windows collapse toward the
   origin (‖T1‖≈0), which becomes a structurally singular "not-a-thought"
   point. Validity is then a feature of T1 a gate can read directly.

4. **Do not renormalize composed thoughts downstream.** Magnitude carries
   compositional coherence — a failed composition (a sum of incompatible
   thoughts) lands near the origin, and renormalizing erases exactly that
   signal. The decoder sees the confidence in its conditioning.

5. **Binary target.** Complete = 1, everything else = 0 — including windows
   holding a complete instruction plus part of the next
   (`[ADD,r4,r2,r7,BEQ,r4]`). Whether useful intermediate magnitudes emerge is
   left open.

The shift-reduce architecture itself — encoder, GRU controller, three gate
heads, decoder — is kept; only the gate-training loop is removed.

**Transferable lessons:**
- **Probe before you tune.** Eight mechanism variants could not beat a
  representation that lacks the target feature; a linear probe exposes that
  before the first training run.
- **Verify supervision signals against ground truth.** The NEG-length defect
  corrupted every run in Experiment 31 without an error.
- "Trained on single valid instructions" is not "recognizes valid
  instructions." The encoder learned the semantic task it was given, not the
  discrimination a downstream consumer needs.

---
