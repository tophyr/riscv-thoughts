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

**Open questions for Phase 2**:
- Execution equivalence collapse requires either equivalence injection
  into batches, a dedicated equivalence loss term, or multi-task training.
- Extending beyond ALU requires a distance metric that handles memory
  and control-flow effects — the register-array approach cannot represent
  stores or branches. A structured effect representation (destination +
  value expression) is needed.
- Multi-task training with shared T1 space (opcode head, dest head,
  source head, immediate head, plus execution equivalence) is the
  proposed next architecture.
