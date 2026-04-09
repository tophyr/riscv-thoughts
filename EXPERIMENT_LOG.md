# Experiment Log

Chronological record of all experiments with full results and analysis.
For theoretical framework, see WHAT_IS_A_THOUGHT.md. For testbed design,
see RISCV_DESIGN.md. For lessons from prior work, see
TRANSFERABLE_LESSONS.md.

---

## Phase 1: T0→T1 ALU Instruction Compression

Goal: compress a single RV32I ALU instruction's tokens (4-7 tokens) into
a single T1 vector such that pairwise T1 distances are proportional to
execution-state distances. Validate geometric properties: smoothness,
factored structure, opcode grouping, execution equivalence collapse.

Architecture: token embeddings + 2-layer transformer encoder (d_model=128,
4 heads) + mean pooling + linear projection → T1 vector. Jointly trained
end-to-end (no frozen encoder, no pre-training).

### Experiment 1: Ranking loss, full register-state L1 distance

d_out=32, batch=128, lr=1e-3, 5000 steps. Loss: softplus-based ranking
loss over all triples per batch — penalizes violations of execution
distance ordering.

Execution distance: L1 across all 32 registers, averaged over 32 random
input states.

**Result: Loss collapsed to 0.0 by step 400.**

The softplus − ln(2) formulation with clamp(min=0) had zero gradient
as soon as ordering was barely correct. No incentive for separation
margins. The model satisfied the loss trivially without learning
meaningful geometry.

**Lesson**: A loss that only tests correct ordering (binary pass/fail
per triple) cannot drive proportional distance learning. Need continuous
gradient toward better separation, not just correct ordering.

### Experiment 2: Correlation loss, full register-state L1 distance

d_out=32, batch=128, lr=1e-3, 5000 steps. Loss: 1 − Pearson correlation
between T1 pairwise distances and execution distances.

**Result: Loss plateaued at ~0.08-0.10 (r ≈ 0.90-0.92).**

Geometry analysis revealed the model learned almost entirely about
destination register identity:

| Comparison                  | T1 distance |
|-----------------------------|-------------|
| Same dest, imm differs by 1 | 0.05       |
| Same dest, imm differs by 2000 | 0.14    |
| Different dest, same computation | 14.84  |

The execution distance metric was the problem. L1 across 32 registers
produces ~10^9-scale differences when destination registers differ (the
non-written register retains its random initial value), but only
~10^0-10^3-scale differences for computational changes within the same
register. The model correctly learned what the metric overwhelmingly
measured: destination register identity.

**Lesson**: Compare the *computed value* (output[rd]), not the full
register state. The instruction's effect is one value written to one
location. Diffing 32 registers drowns the computational signal in
register-identity noise.

### Experiment 3: Correlation loss, computed-value distance

d_out=32, batch=128, lr=3e-4, 5000 steps. Execution distance changed to:

    mean_over_inputs(log(1 + |computed_val_A - computed_val_B|))
      + 16 * (rd_A != rd_B)

C=16 ≈ log(1 + 8.9M): dest register mismatch is as significant as a
computed value difference of ~8.9 million.

**Result: Loss 0.016, r = 0.985.**

Dramatic improvement across all metrics:

| Comparison                         | T1 distance |
|------------------------------------|-------------|
| ADDI x5,x0,10 vs ADDI x5,x0,11   | 0.04        |
| ADDI x5,x0,10 vs ADDI x5,x0,2000 | 0.74        |
| ADD x5,x3,x7 vs ADD x5,x3,x8     | 1.09        |
| ADD x5,x3,x7 vs ADD x6,x3,x7     | 4.24        |
| ADD x5,x3,x3 vs SLLI x5,x3,1     | 1.99        |

Immediate scaling monotonic. Register hierarchy correct. Opcode grouping
emerging. Execution equivalence still far from zero (1.99 for ADD-to-self
vs SLLI-by-1) but much better than full-state metric.

**Lesson**: The training signal must directly measure what you care about.
Full register-state L1 measured register identity. Computed-value distance
measures computational similarity.

### Experiment 4: d_out=32, long training (500K steps)

Same as Experiment 3 but batch=1024, 500K steps with cosine LR decay
3e-4 → 1e-6.

**Result: Loss floor at 0.015, unchanged from step 5K.**

However, equivalence collapse improved substantially despite no loss
improvement:

| Equiv pair              | Exp 3 (5K steps) | Exp 4 (500K steps) |
|-------------------------|-------------------|--------------------|
| ADD x5,x3,x3 vs SLLI   | 1.99              | 0.48               |
| ADD x5,x3,x0 vs ADDI   | —                 | 0.19               |

The loss plateau was a capacity wall (d_out=32), not a learning rate
issue. But within that capacity, more training continued improving
fine-grained structure even with zero loss improvement.

### Experiment 5: d_out=128, 50K steps

d_out=128 (same as d_model, removing the projection bottleneck),
batch=1024, 50K steps.

**Result: Loss broke through to 0.007. r = 0.994.**

But equivalence collapse regressed:

| Equiv pair              | d_out=32 (500K) | d_out=128 (50K) |
|-------------------------|-----------------|-----------------|
| ADD x5,x3,x3 vs SLLI   | 0.48            | 1.74            |
| ADD x5,x3,x0 vs ADDI   | 0.19            | 0.42            |

More dimensions gave the model room to keep equivalent instructions
apart while still maintaining excellent correlation. The loss doesn't
specifically penalize nonzero distance for equivalent pairs — it just
wants proportional distances overall. Same dynamic as HATA lesson 6
(overcomplete spaces enable shortcuts), at a milder scale.

### Experiment 6: d_out=128, batch=4096, 100K steps

Scaled up batch size for more pairwise signal. Pipelined training:
24 producer processes handle emulation, GPU computes B² distance
matrix and forward/backward.

**Result: Best overall model. Loss 0.007, r = 0.9946.**

Full geometry analysis:

**Opcode distance matrix** (same dest & sources, x5 = f(x3, x7)):

|       | ADD  | SUB  | XOR  | OR   | AND  | SLL  | SRL  | SRA  | SLT  | SLTU |
|-------|------|------|------|------|------|------|------|------|------|------|
| ADD   |  -   | 1.85 | 1.74 | 1.72 | 1.89 | 1.89 | 1.84 | 1.85 | 2.23 | 2.21 |
| SUB   |      |  -   | 1.10 | 1.80 | 2.09 | 2.00 | 2.10 | 2.10 | 2.19 | 2.18 |
| XOR   |      |      |  -   | 1.81 | 2.08 | 2.09 | 2.11 | 2.13 | 2.19 | 2.15 |
| OR    |      |      |      |  -   | 1.45 | 2.06 | 2.06 | 2.02 | 2.08 | 2.11 |
| AND   |      |      |      |      |  -   | 1.79 | 1.91 | 1.89 | 2.18 | 2.19 |
| SLL   |      |      |      |      |      |  -   | 1.46 | 1.47 | 2.24 | 2.26 |
| SRL   |      |      |      |      |      |      |  -   | 0.44 | 1.46 | 1.49 |
| SRA   |      |      |      |      |      |      |      |  -   | 1.41 | 1.43 |
| SLT   |      |      |      |      |      |      |      |      |  -   | 0.26 |
| SLTU  |      |      |      |      |      |      |      |      |      |  -   |

Notable clusters:
- SLT/SLTU: 0.26 — signed vs unsigned comparison, nearly identical
- SRL/SRA: 0.44 — logical vs arithmetic right shift
- SLL/SRL: 1.46 — left vs right shift
- OR/AND: 1.45 — bitwise pair
- SUB/XOR: 1.10 — both involve magnitude/sign manipulation

**Immediate scaling** (ADDI x5, x0, 10 vs ADDI x5, x0, N):

| N    | T1 distance |
|------|-------------|
| 11   | 0.11        |
| 15   | 0.14        |
| 50   | 0.38        |
| 100  | 0.51        |
| 500  | 0.81        |
| 2000 | 1.18        |

Clean monotonic progression.

**Register effects** (ADD x5, x3, x7 vs ...):

| Comparison          | T1 distance |
|---------------------|-------------|
| Identical           | 0.00        |
| One source differs  | 1.59        |
| Both sources differ | 2.15        |
| Dest differs        | 2.47        |
| All regs differ     | 3.27        |

Perfect ordering.

**Execution equivalence**:

| Pair                                   | T1 distance |
|----------------------------------------|-------------|
| ADD x5,x3,x3 vs SLLI x5,x3,1         | 1.29        |
| ADDI x5,x0,0 vs AND x5,x0,x0         | 0.92        |
| ADD x5,x3,x0 vs ADDI x5,x3,0         | 0.36        |
| OR x5,x3,x0 vs ADDI x5,x3,0          | 0.36        |
| XOR x5,x3,x0 vs ADDI x5,x3,0         | 0.35        |
| SUB x5,x3,x3 vs XOR x5,x3,x3 (both=0)| 0.56       |
| ADD x5,x3,x7 vs ADD x5,x7,x3 (comm.) | 0.80        |

Not zero, but equivalent pairs are consistently closer than any
non-equivalent pair with the same dest (minimum ~1.0).

---

### Key Finding: d_out Controls the Equivalence/Correlation Tradeoff

Comparing d_out=32 (500K steps) to d_out=128 (50K and 100K steps):

| Metric | d_out=32 (500K) | d_out=128 (100K) |
|--------|-----------------|------------------|
| Pearson r | 0.985 | 0.995 |
| ADD x5,x3,x3 vs SLLI | 0.48 | 1.29 |
| ADD x5,x3,x0 vs ADDI | 0.19 | 0.36 |
| ADD commutative | — | 0.80 |

Larger d_out gives better global correlation but WORSE equivalence
collapse. The model has more room to keep equivalent instructions
apart while still maintaining excellent proportional distances. The
correlation loss doesn't specifically penalize nonzero distance for
equivalent pairs — it only measures the global linear relationship.

### Key Finding: Correlation Loss Is Structurally Blind to Equivalence

The correlation loss computes Pearson r over ~8.4M pairwise distances
per batch. One pair with (exec_dist=0, T1_dist=55) is a single outlier
among millions of correctly-proportioned pairs. It barely moves the
Pearson r. The model has essentially zero gradient incentive to collapse
execution-equivalent instructions.

For equivalence collapse to work, equivalent pairs need to appear in
the same batch (probability near zero for specific pairs in 189M
instruction space), AND the loss needs per-pair sensitivity (which
Pearson correlation does not have).

This finding motivates the weighted equivalence loss in Phase 2.

---

## Phase 1 Summary

**What worked**:
- Correlation loss over computed-value distances produces well-structured
  T1 geometry with smooth, monotonic, factored distance relationships.
- The model learns opcode clustering, register sensitivity, and immediate
  scaling from pure execution semantics — no hand-designed similarity.
- Pipelined training with producer processes + GPU distance computation
  enables batch 4096 at ~90ms/step.

**What didn't work**:
- Ranking loss (collapsed to zero immediately).
- Full register-state L1 distance (dest register dominated by 10^6x).
- Large d_out without equivalence-specific loss (equivalence collapse
  regressed relative to smaller d_out).
- Correlation loss alone cannot drive equivalence collapse (structurally
  insensitive to individual pair violations).

---

## Phase 2: All RV32I Instruction Types

Goal: generalize the compressor from 19 ALU opcodes to all 37 RV32I
instruction types (loads, stores, branches, jumps, LUI, AUIPC).
Full 32-register space (x0-x31).

Architecture changes:
- Two-component execution comparison (data_val + pc_val) replaces
  single computed-value metric
- Destination classification heads (dest_type: reg/mem, dest_reg:
  which register) provide structural gradient signal
- SparseMemory for arbitrary 32-bit addressing with random registers
- Random PC per input state (distinguishes AUIPC from LUI)
- Full x0-x31 register space (~189M possible instructions)

### Experiment 7: First all-types run (log-scaled distance)

d_out=128, batch=4096, lr=3e-4, 100K steps. Distance metric:
mean_over_inputs(log(1 + |data_diff|) + log(1 + |pc_diff|)).

**Result: Pearson r=0.998, Spearman r=0.973. Best correlation yet.**

Opcode structure showed real cross-type clustering:

| Pair | Distance |
|------|----------|
| SLT/SLTU | 0.4 |
| SRL/SRA | 19.2 |
| SLT/BEQ | 3.5 |
| BEQ/BLT | 4.6 |
| LUI/AUIPC | 11.9 |
| JAL/JALR | 66.2 |

But two problems emerged:

**Metric artifact: SLT/BEQ/LW clustering.** SLT (outputs 0/1),
BEQ (data output always 0), and LW (loads random bytes) clustered
together at distances 3.5-16.5. These are computationally unrelated
but log-scaling compressed their small output magnitudes to look
similar. After log, "always outputs small numbers" looks the same
regardless of what the instruction actually computes.

**Equivalence collapse still poor.** ADD-to-self vs SLLI-by-1 at
55.97 (vs 1.29 in ALU-only model). Commutativity at 43.30 (vs 0.80).
The classification heads encode opcode identity and argument structure
into the T1 vector, creating large distances between syntactically
different instructions. The correlation loss provides negligible
counterpressure because equivalent pairs rarely co-occur in batches
and Pearson r is insensitive to individual pair violations.

**Root cause of log artifact:** log(1+|diff|) was introduced in
Phase 1 to prevent dest-register identity from dominating the
computed-value metric (billion-scale diffs from wrong-register reads
vs small computational diffs). Phase 2 replaced the single-value
metric with classification heads for dest identity. The log no longer
solved any active problem and was actively causing the SLT/BEQ/LW
clustering artifact.

**Root cause of equivalence failure:** Pearson correlation is a global
measure over ~8.4M pairs per batch. One pair with exec_dist=0 and
T1_dist=55.97 is invisible among millions of correctly-proportioned
pairs. The model has no gradient incentive to collapse equivalences.

**Fixes applied for next run:**
- Remove log scaling: raw |diff| for both data and PC components
- Add weighted equivalence loss: weight = max(range_i, range_j) /
  (1 + exec_dist). High-range instructions that agree get massive
  weight; low-range instructions (SLT, BEQ) that trivially agree
  get negligible weight.

### Experiment 8: Raw diff + attract-only equiv loss

d_out=128, batch=4096, lr=3e-4, 100K steps. Log scaling removed.
Added weighted equiv loss: weight = max(range_i, range_j) / (1 +
exec_dist), loss = weighted mean of T1 distances (pushing toward 0).

**Result: Equivalence collapse solved — but over-collapse.**

| Equiv pair | Exp 7 | Exp 8 |
|------------|-------|-------|
| ADD x5,x3,x3 vs SLLI x5,x3,1 | 55.97 | **0.011** |
| ADD x5,x3,x0 vs ADDI x5,x3,0 | 51.67 | **0.022** |
| ADD commutative | 43.30 | **0.002** |
| SLLI vs SRLI (shift by 0) | 28.84 | **0.002** |
| BEQ vs BGE (always taken) | 4.94 | **0.064** |

Near-perfect collapse for high-range equivalences. The weighted equiv
loss worked exactly as designed.

**But:** the model over-collapsed. All T1 distances shrank to a tiny
range. ADD vs SUB: 5.1 (was 35.8). SRL/SRA/SLT/LW/BEQ/JAL all within
0.1-0.6 of each other despite being computationally unrelated.

Pearson r = 0.982, Spearman r = 0.944. Proportional structure maintained
but at compressed scale — everything crammed into a small region.

**Root cause:** the equiv loss only had attractive force (push T1
distances toward zero). The correlation loss is scale-invariant
(Pearson r doesn't care about absolute magnitude). Nothing pushed
distances APART. The only repulsive force was the classification heads,
which separate by destination but not by computation. Two instructions
writing to the same register with different operations (ADD x5 vs
SLT x5) had no separation signal.

**One outlier:** SUB x5,x3,x3 vs XOR x5,x3,x3 = 4.00. Both always
output 0 → range = 0 → equiv weight = 0. Constant-output equivalences
get no collapse signal from the range-weighted loss.

**Fix applied for next run:** make equiv loss bidirectional. Instead
of pushing T1 distances toward zero, push them toward a scaled version
of exec distances: loss = weight * (T1_dist - scale * exec_dist)².
For equivalent pairs (exec_dist=0), target is 0 (attractive). For
non-equivalent pairs, target is positive (repulsive). Scale factor
computed per-batch from the current weighted T1/exec ratio, detached
from gradients.

### Experiment 9: Bidirectional MSE equiv loss

d_out=128, batch=4096, lr=3e-4, 100K steps. Exec distance: raw
(no log). Equiv loss: MSE between T1 distances and scaled exec
distances, with per-batch scale factor.

**Result: Diverged to billions.** Raw exec distances span 0 to 4
billion. Squared error against these magnitudes is catastrophic.

### Experiment 10: Weighted Pearson correlation (replacing MSE equiv)

Same setup, replaced MSE equiv with weighted Pearson correlation.
Bounded [0, 2], scale-invariant, bidirectional. Weight =
(pair_range + 1) / (1 + exec_dist).

**Result: Pearson r=0.988, but equivalence unchanged.** The weighted
Pearson was nearly identical to the unweighted Pearson because with
log-scaled exec distances in [0, 22], the weight ratio was only
~20x. The weighted correlation added no meaningful signal beyond
the unweighted one.

### Experiment 11: Exponential weighting on log-scaled distance

Changed weight from range/(1+exec) to range*exp(-exec) to restore
billion-to-one focusing ratio on log-scaled distances.

**Result: Pearson r=0.908, Spearman r=0.853. Worse than unweighted.**
The exponential weighting concentrated all weight on ~500 pairs with
exec_dist < 3 out of 8.4M total. Weighted Pearson on this narrow
band had near-zero variance in the exec component, making the
correlation degenerate. Noisy gradients from the weighted term
interfered with the unweighted Pearson's convergence.

**Key insight:** Pearson correlation is fundamentally wrong for
equivalence collapse. Pearson measures linear correlation
(T1 = a*exec + b), and the intercept b absorbs equivalence error.
exec_dist=0 maps to T1_dist=b, which Pearson is perfectly happy
with for any b > 0.

---

## Phase 3: MSE on Unit Sphere

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

### Experiment 12: MSE + sphere, 10K steps

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

### Experiment 13: MSE + sphere, 100K steps

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

### Experiment 14: MSE + sphere, 1M steps

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

### Key Finding: Training Duration Tradeoff

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

## Phase 4: Structured Random Input States

Goal: fix the SLT/BEQ/LW false cluster by making input register
states more interesting. With pure random int32, BEQ never branches
(P(rs1==rs2) ≈ 2^-32) and SLT/SLTU outputs (0/1) are crushed by
log scaling. Structured inputs give these instructions a chance to
show their real behavior.

### Experiment 15: Single equal-pair structured random, 100K steps

Modified random_regs():
- 15% of states: one random register pair set equal
- 10% of states: 2-5 registers set to small values (0-31)
- 5% of states: 1-3 extra zeros

**Result: Correlation improved, SLT/BEQ cluster unchanged.**

| Metric | Exp 13 (pure) | Exp 15 (single-pair) |
|--------|---------------|---------------------|
| Pearson r | 0.946 | 0.958 |
| Spearman r | 0.659 | 0.714 |
| SLT/BEQ | 0.1 | 0.1 |
| SLT/SLTU | 0.0 | 0.0 |

The structured inputs improved overall correlation quality but
didn't break the false cluster. The equal-pair rate was too diffuse:
with 1 random pair made equal per structured state, the chance of
hitting BEQ's specific comparison registers was ~1/465 per state.
Over ~5 structured states per batch, only ~0.01 states where a
typical BEQ actually branches. Essentially zero.

Cross-syntax equivalences (ADD-double/SLLI) were unchanged as
expected — structured inputs only change register VALUES, not which
instructions appear in the batch. This is problem #2 (instruction
pairing), not problem #1 (input states).

### Experiment 16: Grouped equal registers, 100K steps

Changed from single equal pair to grouped equals: 8-15 registers
share the same value in 15% of states. A random BEQ now has ~14%
chance of comparing two equal registers per structured state.

**Result: Spearman further improved, SLT/BEQ cluster still unmoved.**

| Metric | Exp 13 (pure) | Exp 15 (single) | Exp 16 (grouped) |
|--------|---------------|-----------------|-----------------|
| Pearson r | 0.946 | 0.958 | 0.956 |
| Spearman r | 0.659 | 0.714 | 0.730 |
| SLT/BEQ | 0.1 | 0.1 | 0.1 |
| SLT/SLTU | 0.0 | 0.0 | 0.0 |

Despite the more aggressive grouping, ~5 structured states × 14%
chance = ~0.7 states where a typical BEQ branches. Out of 32
total, that's ~2% of the distance computation — not enough to
move the cluster.

Equivalence metrics varied between runs (commutative: 0.21 → 0.67
→ 0.27 across experiments 13/15/16) suggesting high run-to-run
variance for rare-pair metrics. Single runs per configuration
can't reliably measure these.

### Key Finding: Two Orthogonal Problems

**Problem 1 (input states):** BEQ/SLT need registers with
"interesting" relationships (equality, near values). Structured
random helps correlation quality (Spearman +0.07) but the current
rates are too low to break the BEQ cluster. Would need much more
aggressive structured rates, or instruction-aware input generation.

**Problem 2 (instruction pairing):** Register-dependent equivalences
(ADD-double/SLLI, commutative) need matching instruction pairs in
the same batch. With ~189M possible instructions and batches of
4096, specific equivalence pairs appear ~0.02 times per batch.
Register-independent equivalences (SUB/XOR zero, ~9 pairs/batch)
learn fine. This problem is unrelated to input states.

The fixes are independent: structured inputs for problem 1,
equivalence pair injection for problem 2.
