# Phase 3: MSE on Unit Sphere

Goal: replace the Pearson + weighted equiv loss architecture with
direct MSE distance matching on the unit hypersphere. Four loss
components reduced to three: MSE (shape + scale + equivalence),
dest_type CE, dest_reg CE.

Architecture changes:
- F.normalize on T1 output → unit sphere (S^127)
- MSE loss: (T1_dist - target)² where target = exec_dist × (2/22)
- Sphere prevents collapse (fixed norm), MSE provides scale
  (explicit distance targets), CE provides minimum spread
  (classification needs distinguishable vectors)

## Experiment 12: MSE + sphere, 10K steps

d_out=128, batch=4096, lr=3e-4, 10K steps, cosine decay.

**Result: Best structural results yet.**

Loss: 4.78 → 0.21, still decreasing.

Opcode structure:
- ALU cluster: 0.3-0.6 between ops
- SW isolated: 1.4-1.8 from everything
- BEQ far from ALU: 1.5-1.6
- SRL/SRA close: 0.1
- Full sphere usage: distances 0-1.8 out of max 2.0

Equivalences small relative to opcode distances:
- Commutative: 0.057
- SUB/XOR zero: 0.17
- LB/LBU: 0.018

Register effects correct:
- Dest differs: 1.17
- Src differs: 0.08

Correlation: Pearson 0.84, Spearman 0.69. Lower than Pearson-only
runs but MSE optimizes absolute distances, not correlation.

**Known issue:** SLT/SLTU/LW/BEQ false cluster persists (distances
0.0-0.8). This is the data generation problem — random inputs never
trigger branch equality, and SLT/SLTU outputs (0/1) are crushed by
log scaling.

## Experiment 13: MSE + sphere, 100K steps

Same setup, 100K steps.

**Result: Loss plateaued at ~0.10 by step 25K.**

Equivalences split into two groups:
- Syntactically similar pairs improved: SUB/XOR zero 0.03,
  SLLI/SRLI shift-0 0.23, ADD/ADDI identity 0.14
- Syntactically different pairs worsened: ADD-double/SLLI 0.85,
  commutative 0.21

Correlation: Pearson 0.95, Spearman 0.66.

Register effects shifting: dest differs 0.76, src differs 0.85.
MSE pulling back against classification heads.

## Experiment 14: MSE + sphere, 1M steps

Same setup, 1M steps. Distributed training pipeline (16 local +
128 remote workers via lz4+nc).

**Result: Over-training confirmed.** Loss unchanged from 100K
(0.10). 900K additional steps produced zero improvement.

Equivalences:
- Syntactically similar: further improved (SUB/XOR 0.007,
  SLLI/SRLI 0.026, ADD/ADDI 0.027)
- Syntactically different: further degraded (ADD-double/SLLI 1.15,
  commutative 0.80, LB/LBU 0.36)

Register hierarchy flipped: dest differs 0.54, src differs 0.95.
The model learned source changes have large execution impact while
dest changes don't — correct, but classification heads maintain
residual dest separation.

Correlation: Pearson 0.95, Spearman 0.66.

**Root cause analysis:** Cross-syntax equivalence failure is a
training signal problem, not a model capacity problem.

Register-independent equivalences (SUB x,r,r ≡ XOR x,s,s: both
always zero, ~9 pairs/batch) learn well. Register-dependent
equivalences (ADD x,r,r ≡ SLLI x,r,1: same computation but
requires matching registers, ~0.02 pairs/batch) can't get enough
signal. The model clusters by opcode identity because that's what
the overwhelming majority of pairwise comparisons reinforce.

The fix is upstream: either present equivalences more often in
training data, or weight them more heavily in the loss.

## Key Finding: Training Duration Tradeoff

| Metric | 10K | 100K | 1M |
|--------|-----|------|-----|
| Loss | 0.21 | 0.10 | 0.10 |
| Pearson r | 0.84 | 0.95 | 0.95 |
| ADD-double/SLLI | 0.10 | 0.85 | 1.15 |
| Commutative | 0.06 | 0.21 | 0.80 |
| SUB/XOR zero | 0.17 | 0.03 | 0.007 |

More training improves global structure and syntactically-similar
equivalences, but degrades syntactically-different equivalences.
The loss plateau at ~100K suggests the model converges early, then
spends additional training reinforcing opcode-identity patterns.

---
