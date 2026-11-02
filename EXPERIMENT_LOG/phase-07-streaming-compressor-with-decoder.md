# Phase 7: Streaming Compressor with Decoder

Goal: build the streaming compressor with a learnable emit gate and an
autoregressive decoder trained jointly, with decoder reconstruction as
the primary training signal: the gate and encoder learn to produce
emission vectors that decode back to the original instructions.

Architecture:
- **Encoder:** 2-layer transformer (d_model=128, 4 heads), bidirectional
  attention over the full sequence. A per-position emit gate via
  Gumbel-sigmoid. At emit points the window's tokens are mean-pooled and
  projected to S^127. Accept is always on; evict removes the completed
  instructions on each emit.
- **Decoder:** 2-layer transformer decoder (d_model=128, 4 heads)
  cross-attending to the emission vector (projected to a single
  key-value pair); autoregressive, teacher-forced in training.
- **Losses:** reconstruction (cross-entropy of decoder output against
  the original tokens) + pairwise MSE (emission-vector distances matched
  to per-register execution-delta distances, Phase 6's metric). Both
  train the encoder and gate; only reconstruction trains the decoder.

## Experiment 20: Encoder + decoder, 1000 steps

Batch 256 sequences, max_block_len=5, 4 input states, 2000 batches
repeated; 1000 steps, cosine LR decay 3e-4 → 1e-6; gate temperature
τ=1.0. The last column is emits per sequence / instructions per
sequence.

| Step | Total | Recon | Pairwise | Recon Acc | emit/instr |
|------|-------|-------|----------|-----------|------------|
| 100  | 2.60  | 2.53  | 0.069    | 27.7%     | 3.6/3.6    |
| 300  | 2.02  | 1.96  | 0.057    | 43.4%     | 3.4/3.4    |
| 500  | 1.51  | 1.45  | 0.061    | 57.9%     | 3.5/3.5    |
| 700  | 1.21  | 1.15  | 0.060    | 66.4%     | 3.4/3.4    |
| 800  | 1.14  | 1.08  | 0.060    | 68.1%     | 3.5/3.5    |
| 1000 | 1.19  | 1.05  | 0.138    | ~70%      | —          |

1. **Reconstruction works.** Token accuracy reached ~70% in 1000 steps
   from random init and was still climbing.
2. **The two losses coexist.** Pairwise MSE held at ~0.06 while
   reconstruction loss fell from 2.5 to 1.1.
3. **The gate emits once per instruction** on average (3.4-3.6 of each)
   without any density regularization. Whether it fires at instruction
   boundaries was not measured; the count alone does not show it.

Encoder, gate and decoder trained jointly without instability.

## Experiment 21: 10K steps with round-trip loss

Experiment 20's architecture plus a **round-trip loss**: decoder output
→ Gumbel-softmax → soft embeddings → re-encoded by the encoder → cosine
distance to the original emission. Reconstruction, pairwise MSE and
round-trip weighted 1.0 each. 5000 batches × 256 sequences, repeated;
10K steps, cosine LR decay.

| Step  | Recon | Pair  | RT    | Acc    | emit/instr |
|-------|-------|-------|-------|--------|------------|
| 200   | 2.18  | 0.28  | 0.07  | 36%    | 3.4/3.4    |
| 1000  | 0.37  | 0.19  | 0.03  | 90%    | 3.5/3.5    |
| 2000  | 0.15  | 0.17  | 0.03  | 95%    | 3.5/3.5    |
| 5000  | 0.024 | 0.19  | 0.014 | 99.2%  | 3.4/3.4    |
| 8000  | 0.013 | 0.18  | 0.009 | 99.7%  | 3.6/3.6    |
| 10000 | 0.017 | 0.21  | 0.008 | 99.6%  | 3.5/3.5    |

All three losses converged together: 99.6% token reconstruction,
round-trip cosine distance 0.008.

**Equivalence on the trained encoder:**

| Pair                        | Cosine | Equivalent? |
|-----------------------------|--------|-------------|
| ADD(5,3,7) / ADD(5,7,3)    | 0.956  | yes (commutative) |
| XOR(10,3,7) / XOR(10,7,3)  | 0.965  | yes (commutative) |
| SUB(5,3,3) / XOR(5,3,3)    | 0.929  | yes (both zero) |
| ADD(5,3,3) / SLLI(5,3,1)   | 0.687  | yes (both 2×x3) |
| ADDI(5,3,0) / OR(5,3,0)    | 0.563  | yes (both identity) |
| ADD(5,3,7) / SUB(5,3,7)    | 0.845  | **no** |
| ADD(5,3,7) / AND(5,3,7)    | 0.865  | **no** |

1. **Same-syntax equivalences collapse** (commutative ~0.96, same-operand
   zero-producers 0.93), driven by the pairwise MSE.
2. **Cross-syntax equivalences do not.** ADD-double vs SLLI (0.687) and
   ADDI-identity vs OR-zero (0.563) are farther apart than the
   non-equivalent ADD vs SUB (0.845). Reconstruction cross-entropy
   rewards exact token reproduction, and that syntactic pull outweighs
   the execution-distance signal.
3. **The decoder records rather than canonicalizes:** each vector
   decodes to its own original instruction.
4. **Round-trip loss does not help equivalence.** It measures
   consistency between decoder and encoder, which are consistently wrong
   about cross-syntax equivalence; it does not measure execution
   equivalence.

**Decoding interpolated points.** Points between the vectors of an
equivalent pair, at α = 0.25, 0.50, 0.75:

| Pair (cosine)               | Midpoint decodes to    | Equiv? |
|-----------------------------|------------------------|--------|
| ADD(5,3,7) / ADD(5,7,3) (0.96) | ADD X5 X3 X7       | yes    |
| XOR(10,3,7) / XOR(10,7,3) (0.97) | XOR X10 X3 X7   | yes    |
| SUB(5,3,3) / XOR(5,3,3) (0.93) | XOR X5 X3 X3       | yes    |
| ADD(5,3,3) / SLLI(5,3,1) (0.69) | SLLI X5 X3 1      | yes    |
| ADDI(5,3,0) / OR(5,3,0) (0.56) | SLLI X5 X3 11      | **no** |

For pairs with cosine above ~0.7 every interpolated point decoded to a
valid equivalent instruction, switching smoothly between forms (the
commutative ADD flips operand order between α=0.50 and α=0.75; SUB-self
becomes XOR-self at α=0.50). Below ~0.6 (ADDI-identity / OR-zero) the
path crosses invalid territory.

## Findings

- The encoder–gate–decoder architecture trains stably to 99.6% token
  reconstruction.
- The decoder is compatible with equivalence: where the encoder places
  equivalents close together, it decodes any point between them to a
  valid representative. The equivalence failure lies in the encoder's
  training signal, which token-match reconstruction pushes toward
  syntax.

Phase 8 replaced the parallel bidirectional encoder with a causal
shift-reduce parser whose gates control information flow.
