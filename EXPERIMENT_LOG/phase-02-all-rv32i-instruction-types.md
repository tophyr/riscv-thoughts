# Phase 2: All RV32I Instruction Types

Goal: generalize the Phase 1 compressor from 19 ALU opcodes to all 37
RV32I instruction types (loads, stores, branches, jumps, LUI, AUIPC)
over the full x0-x31 register space (~189M possible instructions), and
add a per-pair equivalence signal that Pearson correlation lacks.

Architecture:
- Two-component execution comparison: the data value written
  (data_val) and the next PC (pc_val).
- Destination classification heads (dest_type: register or memory;
  dest_reg: which register) for structural gradient.
- A sparse memory model for arbitrary 32-bit addressing from random
  register values.
- A random PC per input state, which distinguishes AUIPC from LUI.

All experiments: d_out=128, batch=4096, lr=3e-4, 100K steps.

## Experiment 7: Log-scaled distance, correlation loss

Distance: mean_over_inputs(log(1 + |data_diff|) + log(1 + |pc_diff|)).

**Result: Pearson r=0.998, Spearman r=0.973.** Cross-type clustering:

| Pair | Distance |
|------|----------|
| SLT/SLTU | 0.4 |
| SRL/SRA | 19.2 |
| SLT/BEQ | 3.5 |
| BEQ/BLT | 4.6 |
| LUI/AUIPC | 11.9 |
| JAL/JALR | 66.2 |

Two defects:

**SLT/BEQ/LW false cluster.** SLT (outputs 0/1), BEQ (data output always
0) and LW (loads random bytes) clustered at 3.5-16.5 though they compute
unrelated things. Log scaling compresses small output magnitudes, so
"always outputs small numbers" looks alike regardless of the computation.
Phase 1 used the log to keep destination identity from dominating the
distance; with destination identity carried by the classification heads,
the log served no purpose here and caused this cluster.

**No equivalence collapse.** ADD-to-self vs SLLI-by-1 sat at 55.97
(1.29 in the Phase 1 ALU-only model) and commutative pairs at 43.30
(0.80). The classification heads encode opcode identity and argument
structure, separating syntactically different instructions, and Pearson
r gives no counterpressure (Phase 1's finding on correlation loss).

## Experiment 8: Raw distance + attract-only equivalence loss

Log scaling removed: raw |diff| for data and PC. Added a weighted
equivalence loss, weight = max(range_i, range_j) / (1 + exec_dist),
loss = weighted mean of T1 distances (pulling toward 0). High-range
instructions that agree get large weight; low-range instructions (SLT,
BEQ) that agree trivially get negligible weight.

**Result: high-range equivalences collapsed, and so did everything
else.**

| Equiv pair | Exp 7 | Exp 8 |
|------------|-------|-------|
| ADD x5,x3,x3 vs SLLI x5,x3,1 | 55.97 | **0.011** |
| ADD x5,x3,x0 vs ADDI x5,x3,0 | 51.67 | **0.022** |
| ADD commutative | 43.30 | **0.002** |
| SLLI vs SRLI (shift by 0) | 28.84 | **0.002** |
| BEQ vs BGE (always taken) | 4.94 | **0.064** |

All T1 distances shrank into a small range: ADD vs SUB 5.1 (35.8 in
Exp 7); SRL/SRA/SLT/LW/BEQ/JAL all within 0.1-0.6 of each other. Pearson
r = 0.982, Spearman r = 0.944 — proportional structure at a compressed
scale. The equivalence loss only attracts and Pearson r is
scale-invariant, so nothing pushed distances apart; the classification
heads separate by destination, not computation, so ADD x5 vs SLT x5 had
no separation signal.

Constant-output equivalences get no signal from range weighting: SUB
x5,x3,x3 vs XOR x5,x3,x3 (both always 0, range 0, weight 0) stayed at
4.00.

## Experiment 9: Bidirectional MSE equivalence loss

Raw exec distance. Equivalence loss = weight × (T1_dist − scale ×
exec_dist)², with the scale computed per batch from the weighted
T1/exec ratio and detached: target 0 for equivalent pairs, positive for
the rest.

**Result: diverged to billions.** Raw exec distances span 0 to 4
billion; squared error at those magnitudes is unusable.

## Experiment 10: Weighted Pearson correlation

Log-scaled exec distance (range [0, 22]). The MSE equivalence term
replaced by a weighted Pearson correlation — bounded in [0, 2],
scale-invariant, bidirectional — with weight = (pair_range + 1) /
(1 + exec_dist).

**Result: Pearson r=0.988, equivalence unchanged.** On distances in
[0, 22] the weight ratio is only ~20×, so the weighted correlation
matched the unweighted one.

## Experiment 11: Exponential weighting on log-scaled distance

Weight = range × exp(−exec), restoring a billion-to-one focusing ratio.

**Result: Pearson r=0.908, Spearman r=0.853, worse than unweighted.**
All weight landed on ~500 of 8.4M pairs (exec_dist < 3). On that narrow
band the exec component has near-zero variance, the weighted correlation
is degenerate, and its noisy gradient interfered with the unweighted
term.

## Finding: Pearson correlation cannot express equivalence collapse

Pearson measures fit to T1 = a·exec + b; the intercept b absorbs any
equivalence error. exec_dist=0 maps to T1_dist=b, which Pearson accepts
for any b > 0. Phase 3 replaced correlation with direct distance
matching.
