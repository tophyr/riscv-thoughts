# Phase 2: All RV32I Instruction Types

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

## Experiment 7: First all-types run (log-scaled distance)

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

## Experiment 8: Raw diff + attract-only equiv loss

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

## Experiment 9: Bidirectional MSE equiv loss

d_out=128, batch=4096, lr=3e-4, 100K steps. Exec distance: raw
(no log). Equiv loss: MSE between T1 distances and scaled exec
distances, with per-batch scale factor.

**Result: Diverged to billions.** Raw exec distances span 0 to 4
billion. Squared error against these magnitudes is catastrophic.

## Experiment 10: Weighted Pearson correlation (replacing MSE equiv)

Same setup, replaced MSE equiv with weighted Pearson correlation.
Bounded [0, 2], scale-invariant, bidirectional. Weight =
(pair_range + 1) / (1 + exec_dist).

**Result: Pearson r=0.988, but equivalence unchanged.** The weighted
Pearson was nearly identical to the unweighted Pearson because with
log-scaled exec distances in [0, 22], the weight ratio was only
~20x. The weighted correlation added no meaningful signal beyond
the unweighted one.

## Experiment 11: Exponential weighting on log-scaled distance

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
