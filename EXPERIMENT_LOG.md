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

### Experiment 17: Focused branch batches, 100K steps

### Experiment 17: Focused branch batches, 100K steps

Introduced produce_focused_batch(): batches of 256 instructions
(half branches, half non-branches) with equality-rich register
states (15-25 of 31 registers sharing a value in every state).
Mixed into the training stream via a second muxer at weight 0.5
(~10% of branch instructions from focused context).

Also changed weighted mux mode to enforce ratio strictly — blocks
when the chosen input is empty instead of falling back.

**Result: BEQ branch offset scaling learned for the first time.**

| BEQ offset | Before (all runs) | With focused |
|------------|-------------------|-------------|
| off=10 | 0.001-0.005 | 0.024 |
| off=32 | 0.001-0.008 | 0.048 |
| off=128 | 0.001-0.012 | 0.094 |
| off=1024 | 0.003-0.015 | 0.182 |

Clean monotonic progression — the model learned that different
branch offsets produce different PC values. This was flat noise in
every previous run.

SLT/BEQ cluster slightly improved (0.2 vs 0.1 in previous runs).
SLT/SLTU unchanged at 0.0. Correlation slightly lower (Pearson
0.934 vs 0.956) — the focused batches may add noise to the global
distance metric due to different batch size and instruction
distribution.

Loss trajectory showed oscillation: best 0.026 at 50K, final
0.107. Focused batches have different loss characteristics from
normal batches, causing the loss to spike when a focused batch
is processed.

### Phase 4 Summary

Structured random inputs improved correlation quality (Spearman
0.659 → 0.730) and focused batches enabled branch offset learning.
However, the two core problems remain partially unsolved:

1. SLT/BEQ cluster: improved slightly but not broken. Would need
   even more aggressive focused rates or a different approach.
2. Cross-syntax equivalences: unchanged. Requires instruction
   pairing, not input state changes.

Both problems are fundamentally **context-free ambiguities** that
dissolve with sequential context. A streaming compressor (T1→T2)
would see BEQ preceded by a loop counter increment and know it's
a loop terminator. The single-instruction T1 compressor cannot
resolve these without increasingly complex data engineering that
may not justify the effort.

**Decision: move to multilevel streaming architecture.** The T0→T1
single-instruction compressor provides a good-enough representation
(correct opcode clustering, register sensitivity, immediate scaling,
branch offset learning) that can serve as initialization or
reference for the streaming encoder. The remaining T0→T1 problems
are best solved at T1→T2 where context is available.

---

## Phase 5: Gradient Decoder (Geometry Validation)

Goal: validate that the T1 space is smooth enough for gradient-based
search to invert the compression. Optimize continuous embeddings
through the frozen compressor, snap to discrete tokens, verify via
execution.

### Experiment 18: Gradient decoder on Exp 16 model

Using the best T0→T1 model (Exp 16, grouped structured, Spearman
0.730). Gradient search: Adam on continuous embeddings in the
model's embedding space, 500 steps per restart, multiple restarts
per target, all token sequence lengths 4-8 tried.

**Results:**

| Target | Best decode | Distance | Execution |
|--------|-----------|----------|-----------|
| ADD x5,x3,x7 | ADD x5,x7,x3 | 0.40 | EQUIV (commutativity discovered!) |
| SLLI x5,x3,1 | SLLI x5,x3,1 | 0.79 | EQUIV (exact recovery) |
| ADDI x5,x3,0 | SRL x3,x3,x3 | 1.44 | not equiv |
| SUB x5,x3,x3 | BGEU x27,x5,7 | 0.75 | not equiv (false cluster) |
| BEQ x3,x7,16 | (none valid) | — | — |

**Key findings:**

1. **Commutativity discovered by search.** The gradient decoder found
   ADD x5,x7,x3 (the commutative equivalent) as the closest valid
   instruction to ADD x5,x3,x7's T1 point. The T1 space correctly
   places commutative variants nearby, and gradient search navigates
   there.

2. **Exact recovery works for some instructions.** SLLI x5,x3,1 was
   recovered exactly. The T1 space is smooth enough near this point
   for gradient search + discrete snapping to land on the right
   instruction.

3. **False cluster causes wrong decodes.** SUB x5,x3,x3 (always
   zero) decoded to BGEU (a branch) — confirming the SLT/BEQ/branch
   false cluster pulls zero-producing instructions toward branches.

4. **Some regions are not gradient-navigable.** ADDI identity and
   BEQ couldn't be decoded. The T1 landscape near these points is
   either too rough for gradient search or the discrete snapping
   step loses too much.

5. **Snap distance is significant.** Even successful decodes have
   distance 0.4-0.8 after snapping. The continuous optimum is close
   to the target but the nearest discrete token sequence is further.
   This is expected — discrete tokens are sparse points on S^127.

**Assessment:** The T1 space has smooth, navigable geometry in the
regions where the model learned well (ALU ops, shifts). It's rough
or degenerate in the regions with known problems (false clusters,
branch behavior). The gradient decoder validates both the theory
(search-based inversion works in principle) and the specific model's
limitations.

---

## Phase 6: Sequential Context (Pre-Streaming Validation)

Goal: before building the full streaming compressor with gates,
test whether sequential context actually improves the T1 vector
space. If seeing a preceding instruction helps predict the current
instruction's execution effect, that validates retaining context
in the streaming compressor's window and informs the evict policy.

Architecture: same as Phase 1 (2-layer transformer, d_model=128,
4 heads, S^127 output). The only change is the input: instead of
one instruction's tokens, the model sees a *window* of K consecutive
instructions' tokens. The training target is always the LAST
instruction's single-step register delta — so window_size=1 and
window_size=2 produce vectors for the same set of instructions,
enabling direct comparison.

Execution distance metric: per-register state deltas, averaged
over input states and registers.
```
delta_i(s, r) = regs_after_i(s, r) - regs_before_i(s, r)
exec_dist(i, j) = mean_s(mean_r(log1p(|delta_i(s,r) - delta_j(s,r)|)))
```

Training data: structured basic blocks with data-flow dependencies
(random_basic_block), executed on 4 random input states per sequence.
5000 batches × 256 sequences, repeated indefinitely. 5000 training
steps with cosine LR decay.

### Experiment 19: Fixed-window context comparison

Trained two identical models:
- **window_size=1**: sees one instruction (baseline)
- **window_size=2**: sees the preceding instruction + the target

Same corpus, same architecture, same loss, same targets.

**Results:**

| Model | Pearson | Spearman | Final Loss |
|-------|---------|----------|------------|
| window_size=1 | 0.834 | 0.836 | 0.157 |
| window_size=2 | **0.875** | **0.849** | 0.125 |

Context improves Pearson from 0.834 to 0.875 (+0.041). The
training loss is ~20% lower. Seeing one preceding instruction
carries information that helps predict the current instruction's
execution effect.

**Follow-up: context vector analysis.** To test whether the
improvement comes from data-flow dependency detection or something
more subtle, embedded a fixed target instruction (ADD x5, x3, x7)
with 200 different predecessors: 100 that write to a source register
(x3 or x7, creating data dependency) and 100 that write elsewhere
(no dependency). Measured cosine similarity structure.

| Comparison | ADD | SLLI | BEQ | SUB-self | LW |
|------------|-----|------|-----|----------|-----|
| dep ↔ dep | 0.935 | 0.953 | — | 0.924 | 0.949 |
| nodep ↔ nodep | 0.930 | 0.946 | 1.000 | 0.939 | 0.941 |
| dep ↔ nodep | 0.931 | 0.948 | — | 0.920 | 0.944 |

Key findings:

1. **No combinatorial explosion.** All vectors for the same target
   instruction cluster tightly (cosine sim > 0.92) regardless of
   predecessor. Context causes a small continuous modulation, not a
   partition into distinct regions. S^127 is NOT being asked to
   accommodate 3.6×10^16 two-instruction combinations.

2. **Dep/nodep separation is weak.** The model doesn't cleanly
   separate "predecessor wrote to my source register" from
   "predecessor wrote elsewhere." Cross-group similarity is
   nearly identical to within-group.

3. **BEQ produces identical vectors (cosine 1.000) regardless
   of predecessor.** Since BEQ doesn't write a register, its
   register delta is always zero, and the model correctly learns
   that context is irrelevant for this target.

4. **SUB-self shows the strongest context effect** — the largest
   gap between dep (0.924) and nodep (0.939), and dep-vs-nodep
   (0.920). When the predecessor writes to x5, the "before"
   snapshot for SUB x5,x5,x5 changes, altering the delta even
   though the result is always zero.

5. **Effective dimensionality**: ~54 dimensions capture 90% of
   variance across 200 predecessor variants. The context modulation
   lives in a low-dimensional subspace of S^127.

**Assessment:** Sequential context provides a modest but real
improvement to T1 vector quality. The improvement is NOT from
detecting data-flow dependencies (dep/nodep separation is weak).
It's more likely from the model using predecessor tokens to better
encode the computational type — a form of implicit type context
that helps the pairwise metric even without explicit dependency
detection.

This has two implications for the streaming compressor:
- **Retain recent context in the window.** The evict policy should
  keep at least the most recent completed instruction rather than
  aggressively evicting everything.
- **Don't over-invest in dependency detection.** The model extracts
  value from context without explicitly detecting dependencies.
  The gate doesn't need to be smart about which instructions are
  "relevant" — proximity is sufficient.

---

## Phase 7: Streaming Compressor with Decoder (Stage 1)

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

### Experiment 20: Streaming encoder + decoder, 1000 steps

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

### Experiment 21: 10K steps with round-trip loss

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

## Phase 8: Shift-Reduce Architecture (Gate Training)

Goal: replace the parallel bidirectional encoder with a genuine
streaming shift-reduce parser where gates control information flow
causally. The accept gate admits tokens one at a time, the emit
gate decides when to output a T1 vector, and the evict gate removes
stale tokens from the window.

### Architecture

**Window encoder:** bidirectional transformer over current window
contents. Produces a T1 candidate on S^127 from the window state.
Runs at every iteration (not just at emit points).

**Gate controller:** GRU processes T1 candidates sequentially,
maintaining hidden state across iterations. Three independent gate
heads (2-layer MLPs):
- Accept: sees (next token embedding, T1 candidate, GRU hidden)
- Emit: sees (T1 candidate, GRU hidden)
- Evict: sees (T1 candidate, GRU hidden)

**Decoder:** autoregressive transformer decoder conditioned on T1
vector. Evaluates T1 candidate against current window tokens at
every iteration (not just at emit points), providing per-iteration
reconstruction quality signal.

**Key design decisions:**
- Gates read from the T1 candidate (the compression output), not
  the raw token stream. The emit gate asks "is this vector good
  enough to output?" by judging the compressed representation.
- Tokens arrive strictly in order; the window is always contiguous.
  Accept cannot skip tokens. Evict removes the oldest token (FIFO).
- The GRU provides temporal context ("how has the T1 candidate
  been evolving?") without requiring future information.

### Experiment 22: Gumbel-sigmoid gates (2000 steps)

First attempt: gates use Gumbel-sigmoid with straight-through
estimation. The gate decisions control Python if-statements in
the shift-reduce loop.

**Result: Gates learned nothing.** Accept 17-18%, emit 75-76%,
evict 12% — identical from step 100 to step 2000. The decoder
improved (loss 3.2 → 2.1, accuracy 23% → 34%) by learning to
reconstruct from random windows.

**Root cause:** Converting Gumbel-sigmoid output to Python booleans
(`.item() > 0.5`) and using them in `if` statements breaks the
computation graph. The gate logits receive zero gradient. The
straight-through trick requires the hard decision to multiply a
continuous value, not control program flow.

### Experiment 23: REINFORCE gates with per-iteration decoder (500 steps)

Replaced Gumbel-sigmoid with Bernoulli sampling + REINFORCE.
The decoder evaluates the T1 candidate at every iteration (not
just emit points), providing per-iteration reconstruction quality
as the reward signal.

**Gate training signal:**
- High recon quality + emit=no → penalize (should have emitted)
- Low recon quality + emit=yes → penalize (emitted garbage)
- High recon quality + emit=yes → reward (good emission)
- Low recon quality + emit=no → reward (correctly waited)

Per-iteration advantage = reconstruction loss - moving average
baseline. Separate optimizer for gate parameters (REINFORCE LR)
vs encoder+decoder parameters (normal backprop LR).

**Result: Gates respond but don't learn boundaries.**

| Metric     | Step 25 | Step 250 | Step 500 |
|------------|---------|----------|----------|
| Recon loss | 51.6    | 35.9     | 37.8     |
| Accuracy   | 20.9%   | 28.5%    | 28.7%    |
| Accept     | 87%     | 89%      | 88%      |
| Emit       | 10%     | 11%      | 6%       |
| Evict      | 12%     | 14%      | 14%      |
| Emissions  | 100/3.5 | 95/3.4   | 98/3.7   |

REINFORCE successfully moves the gates (accept jumped to 88%,
emit dropped to 6-10%). But the emit gate learns "fire rarely"
uniformly — not "fire at instruction boundaries." The decoder
reaches only ~29% accuracy, not enough to provide contrast
between "good emit point" (complete instruction, low loss) and
"bad emit point" (partial instruction, high loss).

**Fundamental problem: chicken-and-egg.** The gates need a
competent decoder to know where to emit. The decoder needs
good windows to learn reconstruction. The gates need good
windows to provide good training data. Joint training from
scratch can't bootstrap all three simultaneously.

### Conclusions: Piecemeal Assembly for T1

The joint training experiments (Exp 20-23) conclusively show
that training all components together from scratch doesn't work
for T1. The gate training problem is not that the architecture
is wrong — the shift-reduce parser with REINFORCE is correct —
but that each component needs the others to already work before
it can learn.

However, training components separately IS tractable:
- Encoder on fixed windows → 99.6% decodable (Exp 21)
- Decoder conditioned on those vectors → 99.6% accuracy (Exp 21)
- Gates supervised on instruction boundaries → straightforward

**Decision: assemble T1 from separately-trained components.**

1. **Encoder:** train on fixed windows (already done, Exp 21)
2. **Decoder:** train conditioned on encoder vectors (already done)
3. **Gates:** supervised training on instruction boundaries
4. **Assembly:** combine into the shift-reduce architecture
5. **Fine-tune:** joint training with T2 feedback when T2 exists

This is not a retreat from the streaming architecture. The
shift-reduce parser IS the final architecture. The gates ARE
learned. The only change is training order: components are
pre-trained separately and assembled, then fine-tuned jointly
when cross-level feedback (T2) provides signal that requires
joint optimization.

The philosophy — "gates should discover boundaries themselves"
— is worth pursuing at T2 and above, where thought boundaries
are NOT syntactically obvious and must be inferred from semantic
content. At T1, instruction boundaries are syntactically marked
(opcode tokens), so supervising them is pragmatic, not a
compromise. The architecture supports either approach.

**What the joint training DID accomplish:** the unified experiments
were not wasted. They forced us to design the correct streaming
architecture — the shift-reduce parser with causal gate controller,
bidirectional window encoder, per-iteration decoder evaluation,
and REINFORCE gate training. These architectural decisions would
not have been reached by training components in isolation. The
architecture is correct; only the training order needs to change.
Specifically:
- Gumbel-sigmoid in Python control flow gives zero gradient
  (Exp 22). REINFORCE is the right gate training signal.
- The decoder must evaluate at every iteration, not just emit
  points, so the emit gate gets signal for missed opportunities.
- The gate controller must be causal (GRU), reading from the T1
  candidate (the compression output) rather than raw tokens.
- Gates fire independently per iteration; evict removes one
  token (FIFO) per firing.
- Window-size-weighted reconstruction provides the right cost
  signal for all three gates without explicit density targets.
