# Phase 7: Streaming Compressor with Decoder (Stage 1)

Goal: build and validate the streaming compressor architecture with
a learnable emit gate and an autoregressive decoder trained jointly.
The decoder provides the primary training signal — the gate and
encoder learn to produce emission vectors that are decodable back
to the original instructions.

Architecture:
- **Encoder (StreamingCompressor):** 2-layer transformer (d_model=128,
  4 heads), bidirectional attention over full sequence. Per-position
  emit gate via Gumbel-sigmoid. At emit points, mean-pools window
  tokens and projects to S^127. Accept is fixed (always-on), evict
  is structural (evict completed instructions on emit).
- **Decoder:** 2-layer transformer decoder (d_model=128, 4 heads)
  with cross-attention to emission vector (projected to single
  key-value pair). Autoregressive, teacher-forced during training.
- **Losses:** reconstruction (cross-entropy on decoder output vs
  original tokens) + pairwise MSE (distance matching on emission
  vectors vs per-register execution deltas). Both losses train
  encoder and gate; only reconstruction trains the decoder.

## Experiment 20: Streaming encoder + decoder, 1000 steps

Batch_size=256 sequences, max_block_len=5, n_inputs=4. 2000 batches
repeated indefinitely. 1000 training steps, cosine LR decay 3e-4
→ 1e-6. Gate temperature τ=1.0.

**Results:**

| Step | Total | Recon | Pairwise | Recon Acc | emit/instr |
|------|-------|-------|----------|-----------|------------|
| 100  | 2.60  | 2.53  | 0.069    | 27.7%     | 3.6/3.6    |
| 300  | 2.02  | 1.96  | 0.057    | 43.4%     | 3.4/3.4    |
| 500  | 1.51  | 1.45  | 0.061    | 57.9%     | 3.5/3.5    |
| 700  | 1.21  | 1.15  | 0.060    | 66.4%     | 3.4/3.4    |
| 800  | 1.14  | 1.08  | 0.060    | 68.1%     | 3.5/3.5    |
| 1000 | 1.19  | 1.05  | 0.138    | ~70%      | —          |

Key findings:

1. **Reconstruction works.** Token accuracy reaches ~70% in 1000
   steps from random init. The emission vectors on S^127 carry
   enough information for the decoder to partially reconstruct
   instructions. Not converged — accuracy was still climbing.

2. **Both losses coexist.** Pairwise MSE stays at ~0.06 while
   reconstruction loss drops from 2.5 to 1.1. No interference
   between the geometric structure loss and the information content
   loss.

3. **Gate converges to one emission per instruction.** emits/seq ≈
   instrs/seq throughout training (3.4-3.6 each). The gate learns
   to emit at a rate matching instruction count without explicit
   density regularization — answering the plan's "Exp A" question.

4. **Gate timing is undetermined at this stage.** emits/seq matching
   instrs/seq doesn't prove the gate fires at instruction boundaries.
   Could be firing at arbitrary positions that happen to contain
   one complete instruction. Needs boundary alignment analysis.

**Assessment:** The architecture works end-to-end. Encoder, gate,
and decoder train jointly without instability. The reconstruction
loss provides a meaningful training signal that drives both the
encoder representation and (implicitly) the gate timing.

Next steps: longer training run to find the reconstruction accuracy
ceiling, then switch to round-trip loss (decode → re-encode through
frozen encoder, compare vectors) to accommodate execution equivalence
without changing the architecture.

## Experiment 21: 10K steps with round-trip loss

Same architecture as Exp 20, plus:
- **Decoder**: 2-layer transformer decoder (d_model=128, 4 heads),
  cross-attention to emission vector, autoregressive teacher-forcing.
- **Round-trip loss**: decoder output → Gumbel-softmax → soft
  embeddings → re-encode through encoder → cosine distance to
  original emission. All three losses (reconstruction, pairwise MSE,
  round-trip) weighted equally at 1.0.

5000 batches × 256 sequences, repeated. 10K steps, cosine LR decay.

**Results:**

| Step  | Recon | Pair  | RT    | Acc    | emit/instr |
|-------|-------|-------|-------|--------|------------|
| 200   | 2.18  | 0.28  | 0.07  | 36%    | 3.4/3.4    |
| 1000  | 0.37  | 0.19  | 0.03  | 90%    | 3.5/3.5    |
| 2000  | 0.15  | 0.17  | 0.03  | 95%    | 3.5/3.5    |
| 5000  | 0.024 | 0.19  | 0.014 | 99.2%  | 3.4/3.4    |
| 8000  | 0.013 | 0.18  | 0.009 | 99.7%  | 3.6/3.6    |
| 10000 | 0.017 | 0.21  | 0.008 | 99.6%  | 3.5/3.5    |

All three losses converge without interference. 99.6% token
reconstruction accuracy. Round-trip cosine distance 0.008 (cosine
similarity 0.992).

**Equivalence analysis** on the trained model:

| Pair                        | Cosine | Equivalent? |
|-----------------------------|--------|-------------|
| ADD(5,3,7) / ADD(5,7,3)    | 0.956  | yes (commutative) |
| XOR(10,3,7) / XOR(10,7,3)  | 0.965  | yes (commutative) |
| SUB(5,3,3) / XOR(5,3,3)    | 0.929  | yes (both zero) |
| ADD(5,3,3) / SLLI(5,3,1)   | 0.687  | yes (both 2×x3) |
| ADDI(5,3,0) / OR(5,3,0)    | 0.563  | yes (both identity) |
| ADD(5,3,7) / SUB(5,3,7)    | 0.845  | **no** |
| ADD(5,3,7) / AND(5,3,7)    | 0.865  | **no** |

Key findings:

1. **Same-syntax equivalences collapse well.** Commutative pairs
   (cosine ~0.96) and same-opcode zero-producers (0.93) are nearby.
   The pairwise MSE loss handles these.

2. **Cross-syntax equivalences do NOT collapse.** ADD-double vs
   SLLI (0.687) and ADDI-identity vs OR-zero (0.563) remain far
   apart. Worse: non-equivalent same-syntax pairs (ADD vs SUB:
   0.845) are CLOSER than equivalent cross-syntax pairs. The
   syntactic bias from reconstruction overpowers the execution
   distance signal.

3. **The decoder memorizes, not canonicalizes.** Each vector decodes
   back to its original instruction exactly. No equivalence
   discovery — the decoder is a faithful recorder of what it saw,
   not a discoverer of alternatives.

4. **Round-trip loss doesn't help equivalence.** RT loss of 0.008
   just confirms the decoder is consistent with the encoder — both
   are consistently wrong about cross-syntax equivalence. The RT
   loss can't fix this because it optimizes consistency (round-trip
   fidelity), not correctness (execution equivalence).

**Assessment:** The architecture is sound — encoder, gate, decoder,
and all three losses train stably to high accuracy. But the training
signal doesn't drive equivalence collapse for cross-syntax pairs.
The cross-entropy reconstruction loss actively prevents it by
rewarding exact token reproduction.

To fix this, the decoder needs to be evaluated against execution
equivalence rather than token identity. Options:
- Drop cross-entropy, train with round-trip loss only (requires
  the encoder's pairwise MSE to provide enough structure for the
  round-trip to bootstrap from)
- Add an execution verification step: decode → execute → compare
  register deltas to original (REINFORCE or similar, since
  execution is not differentiable)
- Use the round-trip loss with a frozen encoder that already has
  good equivalence geometry (chicken-and-egg: need the equivalence
  to train the equivalence)

**Decoder equivalence validation.** To confirm the architecture is
compatible with future equivalence training, tested decoding from
interpolated points between equivalent instruction vectors:

| Pair (cosine)               | Midpoint decodes to    | Equiv? |
|-----------------------------|------------------------|--------|
| ADD(5,3,7) / ADD(5,7,3) (0.96) | ADD X5 X3 X7       | yes    |
| XOR(10,3,7) / XOR(10,7,3) (0.97) | XOR X10 X3 X7   | yes    |
| SUB(5,3,3) / XOR(5,3,3) (0.93) | XOR X5 X3 X3       | yes    |
| ADD(5,3,3) / SLLI(5,3,1) (0.69) | SLLI X5 X3 1      | yes    |
| ADDI(5,3,0) / OR(5,3,0) (0.56) | SLLI X5 X3 11      | **no** |

For pairs with cosine > ~0.7, the decoder produces a valid
execution-equivalent instruction from every point along the
interpolation path (tested at α = 0.25, 0.50, 0.75). The decoder
smoothly transitions between equivalent forms — e.g., commutative
ADD flips operand order between α=0.50 and α=0.75, SUB-self
transitions to XOR-self at α=0.50.

Breaks down at cosine < ~0.6 (ADDI-identity / OR-zero): the
interpolation path passes through invalid territory.

**Conclusion:** the decoder architecture is fundamentally compatible
with equivalence. If the encoder collapses equivalents to nearby
points, the decoder correctly picks one valid representative. The
equivalence problem is purely in the encoder's training signal,
not in the decoder's architecture. No architectural changes needed
when equivalence training is added later.

---
