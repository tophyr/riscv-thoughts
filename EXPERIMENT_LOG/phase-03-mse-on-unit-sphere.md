# Phase 3: MSE on Unit Sphere

Goal: replace Phase 2's Pearson correlation and weighted equivalence
loss with direct distance matching on the unit hypersphere. Three loss
components: MSE (shape, scale and equivalence), dest_type cross-entropy
and dest_reg cross-entropy.

Architecture:
- T1 output normalized onto the unit sphere S^127.
- MSE loss (T1_dist − target)², target = exec_dist × (2/22), mapping the
  log-scaled exec range [0, 22] onto the sphere's chord range [0, 2].
- The fixed norm prevents collapse, the MSE targets set scale, and the
  classification heads force a minimum spread (classification needs
  distinguishable vectors).

All experiments: d_out=128, batch=4096, lr=3e-4, cosine decay.

## Experiment 12: 10K steps

**Result: loss 4.78 → 0.21, still decreasing.**

- Opcode structure: ALU ops 0.3-0.6 apart; SW 1.4-1.8 from everything;
  BEQ 1.5-1.6 from ALU; SRL/SRA 0.1. Distances span 0-1.8 of the
  maximum 2.0.
- Equivalences small relative to opcode distances: commutative 0.057,
  SUB/XOR zero 0.17, LB/LBU 0.018.
- Register effects: dest differs 1.17, source differs 0.08.
- Pearson 0.84, Spearman 0.69; MSE optimizes absolute distances, not
  correlation.

The SLT/SLTU/LW/BEQ false cluster remained (distances 0.0-0.8): random
inputs never make branch operands equal, and log scaling crushes SLT/SLTU
outputs (0/1). This is an input-state problem, taken up in Phase 4.

## Experiment 13: 100K steps

**Result: loss plateaued at ~0.10 by step 25K.** Pearson 0.95, Spearman
0.66.

Equivalences split by syntax:
- Syntactically similar pairs tightened: SUB/XOR zero 0.03, SLLI/SRLI
  shift-0 0.23, ADD/ADDI identity 0.14.
- Syntactically different pairs loosened: ADD-double/SLLI 0.85,
  commutative 0.21.

Register effects: dest differs 0.76, source differs 0.85, with MSE
pulling against the classification heads.

## Experiment 14: 1M steps

Data from a distributed pipeline (16 local + 128 remote workers).

**Result: loss unchanged from 100K (0.10).** Pearson 0.95, Spearman 0.66.

- Syntactically similar equivalences tightened further: SUB/XOR 0.007,
  SLLI/SRLI 0.026, ADD/ADDI 0.027.
- Syntactically different ones loosened further: ADD-double/SLLI 1.15,
  commutative 0.80, LB/LBU 0.36.
- Register hierarchy inverted: dest differs 0.54, source differs 0.95.
  Source changes have large execution impact and dest changes do not;
  the classification heads keep a residual dest separation.

## Finding: cross-syntax equivalence is starved of training pairs

| Metric | 10K | 100K | 1M |
|--------|-----|------|-----|
| Loss | 0.21 | 0.10 | 0.10 |
| Pearson r | 0.84 | 0.95 | 0.95 |
| ADD-double/SLLI | 0.10 | 0.85 | 1.15 |
| Commutative | 0.06 | 0.21 | 0.80 |
| SUB/XOR zero | 0.17 | 0.03 | 0.007 |

Training past the ~100K loss plateau improved global structure and
syntactically similar equivalences while degrading syntactically
different ones. The cause is the training signal, not capacity.
Register-independent equivalences (SUB x,r,r ≡ XOR x,s,s, both always
zero) occur ~9 pairs per batch and are learned. Register-dependent ones
(ADD x,r,r ≡ SLLI x,r,1, which need matching registers) occur ~0.02
pairs per batch, so nearly every pairwise comparison reinforces opcode
identity. Collapsing them requires presenting such pairs more often or
weighting them more heavily.
