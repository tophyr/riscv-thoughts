# Phase 4: Structured Random Input States

Goal: break Phase 3's SLT/BEQ/LW false cluster by making input register
states structured. Under pure random int32, BEQ never branches
(P(rs1==rs2) ≈ 2^-32) and SLT/SLTU outputs (0/1) are crushed by log
scaling. All experiments: 100K steps, Phase 3's MSE-on-sphere setup.

## Experiment 15: Single equal pair

Input-state generator:
- 15% of states: one random register pair set equal
- 10% of states: 2-5 registers set to small values (0-31)
- 5% of states: 1-3 extra zeros

**Result: correlation improved, false cluster unchanged.**

| Metric | Exp 13 (pure random) | Exp 15 (single pair) |
|--------|---------------|---------------------|
| Pearson r | 0.946 | 0.958 |
| Spearman r | 0.659 | 0.714 |
| SLT/BEQ | 0.1 | 0.1 |
| SLT/SLTU | 0.0 | 0.0 |

One equal pair per structured state hits a given BEQ's two registers
with probability ~1/465; over ~5 structured states per batch, a typical
BEQ branches in ~0.01 states.

## Experiment 16: Grouped equal registers

8-15 registers share one value in 15% of states, giving a random BEQ a
~14% chance per structured state of comparing equal registers.

**Result: Spearman improved further, false cluster unchanged.**

| Metric | Exp 13 (pure random) | Exp 15 (single pair) | Exp 16 (grouped) |
|--------|---------------|-----------------|-----------------|
| Pearson r | 0.946 | 0.958 | 0.956 |
| Spearman r | 0.659 | 0.714 | 0.730 |
| SLT/BEQ | 0.1 | 0.1 | 0.1 |
| SLT/SLTU | 0.0 | 0.0 | 0.0 |

~5 structured states × 14% ≈ 0.7 states of 32 in which a typical BEQ
branches — ~2% of the distance computation.

Rare-pair equivalence metrics are dominated by run-to-run variance: the
commutative distance read 0.21, 0.67 and 0.27 across Experiments 13, 15
and 16. A single run per configuration cannot rank them.

## Experiment 17: Focused branch batches

Batches of 256 instructions (half branches, half non-branches) whose
register states are equality-rich (15-25 of 31 registers sharing a value
in every state), mixed into the training stream at weight 0.5 (~10% of
branch instructions seen in focused context). The mixer enforced the
ratio strictly, blocking on an empty input rather than falling back.

**Result: branch offset scaling was learned.**

| BEQ offset | Exp 13-16 | Exp 17 |
|------------|-------------------|-------------|
| off=10 | 0.001-0.005 | 0.024 |
| off=32 | 0.001-0.008 | 0.048 |
| off=128 | 0.001-0.012 | 0.094 |
| off=1024 | 0.003-0.015 | 0.182 |

Distances grow monotonically with offset; without focused batches they
were flat noise. SLT/BEQ moved to 0.2 (0.1 in Exp 13-16); SLT/SLTU stayed
at 0.0. Pearson fell to 0.934 (0.956 in Exp 16), plausibly from the
focused batches' different size and instruction mix. The loss oscillated
(best 0.026 at 50K, final 0.107), spiking on focused batches.

## Findings

Two independent problems:

1. **Input states.** BEQ and SLT need registers with equal or nearby
   values. Structured inputs raised Spearman from 0.659 to 0.730, and
   focused batches taught branch offsets, but no rate tried broke the
   SLT/BEQ cluster.
2. **Instruction pairing.** Register-dependent equivalences
   (ADD-double/SLLI, commutative) need the matching instruction pair in
   the same batch, ~0.02 times per batch of 4096 drawn from ~189M
   instructions; register-independent ones (SUB/XOR zero, ~9 pairs per
   batch) are learned. Input states do not affect this.

Both are ambiguities of a single instruction without context: a BEQ
preceded by a loop-counter increment is a loop terminator. The project
moved to a multilevel streaming architecture rather than further data
engineering for the single-instruction compressor, treating the Exp 16
encoder (correct opcode clustering, register sensitivity, immediate
scaling) as good enough to validate the geometry (Phase 5) and test
sequential context (Phase 6).
