# Phase 4: Structured Random Input States

Goal: fix the SLT/BEQ/LW false cluster by making input register
states more interesting. With pure random int32, BEQ never branches
(P(rs1==rs2) ≈ 2^-32) and SLT/SLTU outputs (0/1) are crushed by
log scaling. Structured inputs give these instructions a chance to
show their real behavior.

## Experiment 15: Single equal-pair structured random, 100K steps

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

## Experiment 16: Grouped equal registers, 100K steps

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

## Key Finding: Two Orthogonal Problems

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

## Experiment 17: Focused branch batches, 100K steps

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

## Phase 4 Summary

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
