# Phase 1: T0→T1 ALU Instruction Compression

Goal: compress a single RV32I ALU instruction's tokens (4-7 tokens) into
a single T1 vector such that pairwise T1 distances are proportional to
execution-state distances, and check the resulting geometry for
smoothness, factored structure, opcode grouping and collapse of
execution-equivalent instructions.

Architecture: token embeddings + 2-layer transformer encoder (d_model=128,
4 heads) + mean pooling + linear projection → T1 vector, trained
end-to-end from scratch.

## Experiment 1: Ranking loss, full register-state L1 distance

d_out=32, batch=128, lr=1e-3, 5000 steps. Loss: softplus-based ranking
loss over all triples per batch, penalizing violations of execution
distance ordering. Execution distance: L1 across all 32 registers,
averaged over 32 random input states.

**Result: loss reached 0.0 by step 400 without learning geometry.** The
softplus − ln(2) formulation with clamp(min=0) has zero gradient as soon
as an ordering is barely correct, so there is no incentive for separation
margins. A loss that only tests correct ordering per triple cannot drive
proportional distances; it needs continuous gradient toward separation.

## Experiment 2: Correlation loss, full register-state L1 distance

d_out=32, batch=128, lr=1e-3, 5000 steps. Loss: 1 − Pearson correlation
between T1 pairwise distances and execution distances.

**Result: loss plateaued at ~0.08-0.10 (r ≈ 0.90-0.92), and the geometry
encoded almost only destination register identity:**

| Comparison                  | T1 distance |
|-----------------------------|-------------|
| Same dest, imm differs by 1 | 0.05       |
| Same dest, imm differs by 2000 | 0.14    |
| Different dest, same computation | 14.84  |

L1 across 32 registers produces ~10^9-scale differences when destination
registers differ (the non-written register keeps its random initial
value), against ~10^0-10^3 for computational changes within one register.
The metric measured register identity, and the model learned it. The
instruction's effect is one computed value written to one location, so
the distance must compare that value (output[rd]), not the full register
state.

## Experiment 3: Correlation loss, computed-value distance

d_out=32, batch=128, lr=3e-4, 5000 steps. Execution distance:

    mean_over_inputs(log(1 + |computed_val_A - computed_val_B|))
      + 16 * (rd_A != rd_B)

C=16 ≈ log(1 + 8.9M): a destination register mismatch counts as much as
a computed-value difference of ~8.9 million.

**Result: loss 0.016, r = 0.985.**

| Comparison                         | T1 distance |
|------------------------------------|-------------|
| ADDI x5,x0,10 vs ADDI x5,x0,11   | 0.04        |
| ADDI x5,x0,10 vs ADDI x5,x0,2000 | 0.74        |
| ADD x5,x3,x7 vs ADD x5,x3,x8     | 1.09        |
| ADD x5,x3,x7 vs ADD x6,x3,x7     | 4.24        |
| ADD x5,x3,x3 vs SLLI x5,x3,1     | 1.99        |

Immediate scaling was monotonic, the register hierarchy correct, and
opcode grouping present. The equivalent pair ADD-to-self vs SLLI-by-1
stayed far from zero (1.99).

## Experiment 4: d_out=32, 500K steps

Experiment 3's setup at batch=1024, 500K steps, cosine LR decay
3e-4 → 1e-6.

**Result: the loss floor stayed at 0.015 from step 5K on, while
equivalence collapse kept improving:**

| Equiv pair              | Exp 3 (5K steps) | Exp 4 (500K steps) |
|-------------------------|-------------------|--------------------|
| ADD x5,x3,x3 vs SLLI   | 1.99              | 0.48               |
| ADD x5,x3,x0 vs ADDI   | —                 | 0.19               |

The plateau was a capacity wall at d_out=32, not a learning-rate
problem; within that capacity, training refined fine-grained structure
with no change in loss.

## Experiment 5: d_out=128, 50K steps

d_out=128 (equal to d_model, removing the projection bottleneck),
batch=1024, 50K steps.

**Result: loss 0.007, r = 0.994, with weaker equivalence collapse than
Experiment 4** (table under the d_out finding below).

## Experiment 6: d_out=128, batch=4096, 100K steps

Training was pipelined: 24 producer processes ran emulation while the
GPU computed the B² distance matrix and forward/backward, at ~90ms/step.

**Result: loss 0.007, r = 0.9946.**

**Opcode distance matrix** (same dest and sources, x5 = f(x3, x7)):

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

Closest pairs: SLT/SLTU 0.26 (signed vs unsigned comparison), SRL/SRA
0.44 (logical vs arithmetic right shift), SUB/XOR 1.10, OR/AND 1.45,
SLL/SRL 1.46.

**Immediate scaling** (ADDI x5, x0, 10 vs ADDI x5, x0, N), monotonic:

| N    | T1 distance |
|------|-------------|
| 11   | 0.11        |
| 15   | 0.14        |
| 50   | 0.38        |
| 100  | 0.51        |
| 500  | 0.81        |
| 2000 | 1.18        |

**Register effects** (ADD x5, x3, x7 vs ...), correctly ordered:

| Comparison          | T1 distance |
|---------------------|-------------|
| Identical           | 0.00        |
| One source differs  | 1.59        |
| Both sources differ | 2.15        |
| Dest differs        | 2.47        |
| All regs differ     | 3.27        |

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

Equivalent pairs were not at zero, but were consistently closer than any
non-equivalent pair with the same dest (minimum ~1.0).

## Finding: d_out trades equivalence collapse against correlation

| Metric | d_out=32 (500K) | d_out=128 (50K) | d_out=128 (100K) |
|--------|-----------------|-----------------|------------------|
| Pearson r | 0.985 | 0.994 | 0.995 |
| ADD x5,x3,x3 vs SLLI | 0.48 | 1.74 | 1.29 |
| ADD x5,x3,x0 vs ADDI | 0.19 | 0.42 | 0.36 |
| ADD commutative | — | — | 0.80 |

Larger d_out gave better global correlation and worse equivalence
collapse: more dimensions leave room to keep equivalent instructions
apart while keeping distances proportional. This is the overcomplete
shortcut of TRANSFERABLE_LESSONS.md lesson 6, at a milder scale.

## Finding: correlation loss is structurally blind to equivalence

Pearson r over ~8.4M pairwise distances per batch barely moves for one
pair with (exec_dist=0, T1_dist=55); the model has essentially no
gradient toward collapsing execution-equivalent instructions. Collapse
needs equivalent pairs in the same batch (probability near zero for a
specific pair in the 189M-instruction space) and a loss with per-pair
sensitivity, which Pearson correlation lacks.

## Summary

Established:
- Correlation loss over computed-value distances gives smooth, monotonic,
  factored T1 geometry: opcode clustering, register sensitivity and
  immediate scaling, learned from execution semantics with no
  hand-designed similarity.

Dead ends:
- Ranking loss over triples (zero gradient once ordering holds).
- Full register-state L1 distance (destination identity dominates by
  ~10^6×).
- Large d_out without an equivalence-specific loss (worse collapse).
- Correlation loss alone for equivalence collapse.

Phase 2 took up the missing per-pair equivalence signal while extending
the instruction set to all of RV32I.
