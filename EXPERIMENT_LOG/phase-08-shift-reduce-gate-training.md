# Phase 8: Shift-Reduce Architecture (Gate Training)

Goal: replace the parallel bidirectional encoder with a genuine
streaming shift-reduce parser where gates control information flow
causally. The accept gate admits tokens one at a time, the emit
gate decides when to output a T1 vector, and the evict gate removes
stale tokens from the window.

## Architecture

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

## Experiment 22: Gumbel-sigmoid gates (2000 steps)

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

## Experiment 23: REINFORCE gates with per-iteration decoder (500 steps)

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

## Conclusions: Piecemeal Assembly for T1

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

## Experiment 24: Equivalence manifest, injection, and targeted boost

Built the T1 Foundations Pass: a template-based equivalence
manifest, per-class eval harness, and tuple-injection mechanism
in the batch pipeline. Ran six progressive training configurations
to understand what drives equivalence collapse and what blocks it.

**Architecture additions:**

- `datagen/equivalences.py` — manifest of 16 classes covering six
  structural axes: operand-order commutativity, self-operand
  identity, zero-register alias, zero/identity immediate,
  cross-opcode aliasing, and condition-forcing (for branches).
  Each class is a *rule* with free variables (rd, rs, rs1,
  rs2, imm, shamt) and optional constraints. The sampler
  instantiates concrete instructions per binding.
- `scripts/eval_equivalences.py` — per-class intra-cohesion vs
  inter-separation on a trained encoder. Reports ratio = intra /
  inter with PASS (<0.3), WEAK (<0.7), FAIL thresholds.
- `datagen/batchgen.py` — `produce_instruction_batch` injects
  canonical tuples at a configurable rate via an `equivalences`
  sub-dict in the distribution config. Three-pass: guaranteed
  `min_per_class` tuples, per-class `boost` counts, random fill.
- `datagen/instrgen.py` — `_DEST_REGS` now includes x0.
  Write-to-x0 NOPs are a real semantic class; the old exclusion
  was carried forward from sequence training where x0-writes
  "wasted" sequence slots, an inapplicable rationale for
  single-instruction batches.
- `compressor/train.py` — nested-log metric
  `log1p(log1p(|data_diff|+|pc_diff|))` replaces single `log1p`.
  Sharpens low-end distance differentiation so near-equivalence
  contrasts (e.g., ADDI imm=0 vs imm=1, raw diff=1) get more
  gradient.
- Probe instrumentation in `train_batches`: every log step reports
  ADD-ADD, canonical (ADD rs,rs vs SLLI rs,1), SLLI-SLLI, and
  commutative ADD swap distances to watch geometry evolve across
  training.

**Baseline (previous 250K run, no injection, no dest heads,
constant LR bug):** 3 PASS / 3 WEAK / 7 FAIL.

**Progressive configurations:**

| config                                  | steps | PASS | WEAK | FAIL |
|-----------------------------------------|-------|------|------|------|
| 1% injection                            | 10K   | 4    | 2    | 7    |
| 10% + min_per_class=4                   | 10K   | 5    | 4    | 4    |
| + boost shl1=60                         | 10K   | 6    | 1    | 6    |
| + nested log                            | 10K   | 6    | 3    | 4    |
| 250K no boost                           | 250K  | 5    | 4    | 4    |
| 250K + 20% + boost commutatives + shl1  | 250K  | **8**| **2**| **3**|

**Final 250K + boost per-class results:**

```
commutative_add           WEAK (0.38)
commutative_and           PASS (0.22)
commutative_or            PASS (0.27)
commutative_xor           FAIL (0.74)
self_op_zero              PASS (0.004)
self_op_identity          PASS (0.05)
mv_via_zero_reg           FAIL (0.77)
double_is_shl1            PASS (0.12)
write_zero_to_rd          PASS (0.04)
x0_write_nop              PASS (0.002)
branch_always_taken       WEAK (0.35)
branch_never_taken        PASS (0.004)
sign_test                 FAIL (1.60)
```

**Key findings:**

1. **Injection works, but density matters.** 1% (~10 canonical
   tuples/batch) is too thin to overcome random-operand gradient.
   10% + guaranteed 4 tuples/class is adequate for most classes.
   Cross-opcode and high-specificity contrasts need 20% + per-class
   boost (30 tuples/batch for commutatives and double_is_shl1).

2. **Loss plateau vs geometry evolution.** Total loss plateaued
   at 0.156 by step 50K and stayed flat for 200K more steps. But
   geometric probes kept improving dramatically: canon (ADD rs,rs
   vs SLLI rs,1) went from 0.67 at step 50K to 0.13 at step 250K.
   The optimizer wanders the equal-loss manifold, biased by
   persistent injection gradients, producing geometric deformation
   that total loss doesn't measure. **Loss-based convergence is
   uninformative for the actual objective.**

3. **Opcode clustering dissolves with training.** Early training
   (~step 10K) shows tight opcode clusters (ADD-ADD = 0.03). By
   step 50K clusters are gone (ADD-ADD = 1.0). Good for
   cross-opcode equivalences (the ADD-SLLI bridge can form), but
   bad for commutativity: operand order starts mattering to the
   encoder, so swapped pairs no longer collapse for free.
   Commutative boost is required to re-collapse them at long
   training. The short-training "commutative PASS" was a cluster
   illusion, not real commutativity learning.

4. **Cross-opcode aliases collapse with long training.**
   `double_is_shl1` (ADD rs,rs ≡ SLLI rs,1) went from FAIL (1.87
   at 10K) to PASS (0.12 at 250K + boost). The opcode-similarity
   prior is a *learned bias*, not a hard constraint — it dissolves
   under sustained injection pressure at sufficient training
   length. The earlier claim that opcode clustering was
   structurally unbridgeable was wrong.

5. **Persistent failures are metric-bound.** `sign_test`,
   `mv_via_zero_reg`, `commutative_xor` all have contrasts that
   differ only at rare register-value conditions (rs=0) or have
   tiny raw-diff gaps that nested log compresses further. Boost
   doesn't help; either state-preconditions or a different metric
   is needed. Deferred to the smooth-distance experiment.

**Implication for next step:**

The loss-plateau phenomenon argues for moving from implicit
injection-via-batch-MSE to an **explicit equivalence loss term**.
The main MSE has no direct gradient on equivalence collapse — it
only provides gradient through batch-level pair inclusion. A
dedicated equivalence loss (encode canonical tuples per class per
step, target pairwise distance = 0) would provide strong, direct
gradient and should greatly shorten required training. This is
Option B of the earlier injection-vs-loss decision, now supported
by the plateau evidence.

## Experiment 25: Explicit equivalence loss

Adds a dedicated loss term that encodes each class's canonical
templates (under a fresh binding per step) and minimizes their
pairwise squared distances directly (target = 0). This is Option
B, picked up after the Exp 24 plateau analysis showed the main
MSE's per-equivalence-pair gradient is ~0.012% of total gradient
share — effectively invisible, even at high injection rates.

**Mechanism:**
- `compressor/train.equivalence_loss`: for every manifest class
  with >=2 canonical templates, sample a binding, materialize,
  batch all classes' templates into one padded tensor, encode
  once (shared forward), then slice per-class and compute
  `mean((d_ij)² for i<j)` per class. Average over classes.
- Added `--equiv-weight` CLI arg to train_compressor.
- Batched-across-classes encode matters for speed: the first
  implementation had 14 separate forward passes per step and
  doubled step time (49ms → 106ms). Batching brought it back to
  51ms.

**Why it's structurally different from heavy injection:**
Injected pairs have to share the batch's ~8.4M-pair gradient
budget with random pairs. Even at extreme injection rates, the
fraction of *true within-tuple equivalence* pairs in the batch
caps out at ~0.1% of pairs. The explicit loss operates on a
separate set of ~75 encoded instructions with its own gradient
budget, so per-equivalence-pair gradient is ~150,000× stronger
for the same weight.

**Progressive runs:**

| config                              | steps | PASS | WEAK | FAIL |
|-------------------------------------|-------|------|------|------|
| equiv only, w=0.5 (unbatched)       |  (partial, killed — over-collapse) |
| equiv only, w=0.05 (batched)        | 10K   | 5    | 5    | 3    |
| equiv + 5% injection, w=0.05        | 10K   | 7    | 1    | 5    |
| equiv + 5% injection, w=0.05 (100K) | 100K  | **12** | **0** | **1** |

**Final 100K combined per-class:**

```
commutative_add           PASS (0.018)
commutative_and           PASS (0.022)
commutative_or            PASS (0.023)
commutative_xor           PASS (0.018)
self_op_zero              PASS (0.012)
self_op_identity          PASS (0.016)
mv_via_zero_reg           PASS (0.029)
double_is_shl1            PASS (0.016)
write_zero_to_rd          PASS (0.083)
x0_write_nop              PASS (0.003)
branch_always_taken       PASS (0.018)
branch_never_taken        PASS (0.005)
sign_test                 FAIL (0.940)
```

12/13 testable classes PASS in 70 minutes — significantly better
than the 250K + boost run's 8/2/3 in 2.5 hours.

**"Teaching to the test" sanity checks on the final checkpoint:**

- Pearson on *out-of-manifest* random pairs: 0.9129 (vs 0.9268 for
  the 250K + boost reference). The equiv loss cost ~0.014 of
  global correlation — noticeable but small, and still well above
  the 0.85 target. The global geometry was not warped.
- Spearman 0.6436 (vs 0.6604). Similar slight drop; fine-grained
  rank ordering noise is comparable to the previous best run.
- PCA dimensionality: 94 dims for 90% variance, participation
  ratio 84.8 (vs 100 / 90.75 for the reference). Both runs use
  ~70% of the 128-d space actively. No excessive collapse.
- Top 10 singular values are essentially identical between the
  two runs (55.5/56.5, 21.9/21.9, 15.8/15.7, …). This is strong
  evidence both runs converged to geometrically comparable
  attractors — not different local minima.

**Findings:**

1. **Injection and explicit loss are complementary, not
   substitutes.** Explicit loss alone (no injection) gave 5/5/3
   — it learned the abstract rules but regressed on classes that
   benefited from natural random batch co-occurrence (self_op_zero,
   write_zero_to_rd). Adding 5% injection back restored the "many
   varied specific instances" signal those classes need. Combined
   is dramatically better than either alone.

2. **Gradient concentration beats gradient volume.** The 100K
   combined run reached tighter PASS ratios than 250K + boost
   (e.g., commutative_add: 0.018 vs 0.38, commutative_xor: 0.018
   vs 0.74) in less than half the wall time. The explicit loss's
   ~150,000× gradient concentration per equivalence pair is the
   mechanism.

3. **Only sign_test still fails, and the ratio is now 0.94** (vs
   1.60 in the 250K + boost run). This class's contrast —
   SLTI rd,rs,0 vs SLTI rd,rs,1 — only diverges when rs=0, which
   is rare under random-state sampling. Even ~150,000× gradient
   concentration can't overcome the fact that canonical and
   contrast produce identical outputs for nearly all inputs.
   Needs state-preconditions (boost the rare distinguishing
   register values) or a different metric, not more training.

**Cost comparison (wall-clock to 100K-step result quality):**
- Injection-only (20% + boosts), 250K: 150 minutes → 8/2/3
- Combined (5% + equiv w=0.05), 100K: 70 minutes → 12/0/1

**Implication for next step:**

The manifest now collapses reliably with the combined approach.
Remaining question: is the geometry consistent across seeds, or
are we finding comparable-but-not-identical attractors on each
run? The out-of-manifest Pearson and singular-value agreement
between runs suggest consistency, but a true seed-robustness
study would require 3+ runs with varied `--seed`. Deferred in
favor of moving on to the `d_out` compression sweep, which is
the actual purpose of this foundations pass.

## Experiment 26: d_out compression sweep + depth

Sweep d_out ∈ {128, 64, 32, 16} × n_layers ∈ {2, 4, 8}, all at
100K steps with combined training (5% injection + equiv w=0.05).
Goal: find the smallest d_out that preserves equivalence collapse
and global geometry.

**Results — equivalence eval + Pearson (out-of-manifest random):**

| d_out | layers | PASS | sign_test | Pearson |
|-------|--------|------|-----------|---------|
| 128   | 2      | 12/13 | FAIL (0.94) | 0.913 |
| **64**| **2**  | **13/13** | **PASS (0.29)** | **0.906** |
| 32    | 2      | 12/13 | WEAK (0.55) | 0.893 |
| 32    | 4      | 12/13 | FAIL (0.71) | 0.895 |
| 32    | 8      | 12/13 | FAIL (1.65) | 0.895 |
| 16    | 2      | 12/13 | FAIL (0.82) | 0.862 |
| 16    | 4      | 12/13 | FAIL (0.92) | 0.862 |
| 16    | 8      | 12/13 | FAIL (0.89) | 0.863 |

**Key findings:**

1. **d_out=64 is the sweet spot.** Only configuration where all
   13 testable classes PASS. Pearson barely dips (0.906 vs 0.913
   at d_out=128). PCA shows 52 of 64 dims carry 90% variance
   (81% space efficiency, up from 73% at d_out=128).

2. **sign_test has a non-monotonic compression response.** FAIL
   at d_out=128 (too much slack → encoder is sloppy), PASS at 64
   (forced compression helps), WEAK at 32, FAIL at 16. The sweet
   spot is d_out=64 where compression forces distinctions the
   encoder wouldn't make at higher dimensionality.

3. **Pearson ceiling is d_out-dependent, not depth-dependent.**
   At d_out=32: Pearson ≈ 0.894 regardless of 2/4/8 layers.
   At d_out=16: Pearson ≈ 0.862 regardless of 2/4/8 layers.
   More depth cannot break through a capacity ceiling. The
   limitation is how many independent distance relationships
   can be represented in d_out dimensions on the unit sphere.

4. **Depth tightens equivalences but doesn't expand capacity.**
   More layers → tighter PASS ratios (commutative_add goes
   from 0.009 to 0.003 at d_out=32 with 2→8 layers). The
   encoder organizes its limited space more efficiently. But
   Pearson stays flat.

5. **PCA space efficiency increases with compression.** d_out=128
   uses 73%, d_out=64 uses 81%, d_out=32 uses 84%, d_out=16
   uses 88%. Smaller spaces are used more efficiently — less
   wasted dimensionality.

**Conclusion:** d_out=64 with 2 layers is the optimal
configuration. RV32I instruction semantics live on a
~50-60 dimensional manifold. d_out=128 is wasteful; d_out=32
works but loses sign_test; d_out=16 hits a real capacity wall
(Pearson drops to 0.86).

## Experiment 27: Decoder evaluation (reconstruction from frozen encoder)

Tested whether the pairwise-MSE-trained T1 vectors are
information-preserving enough for reconstruction. Two methods
across all 8 encoder checkpoints.

**Method 1: Trained decoder (5K steps, teacher-forced CE).**
Freeze encoder, train a fresh decoder (d_model=128, 2 layers)
to reconstruct token sequences from T1 vectors. Same corpus
across all checkpoints for fair comparison.

| d_out | layers | tok_acc | instr_acc |
|-------|--------|---------|-----------|
| 128   | 2      | 64.9%   | 2.3%      |
| 64    | 2      | 63.9%   | 0.0%      |
| 32    | 2      | 60.0%   | 0.0%      |
| 32    | 4      | 57.1%   | 0.8%      |
| 32    | 8      | 55.5%   | 0.4%      |
| 16    | 2      | 54.9%   | 0.4%      |
| 16    | 4      | 55.4%   | 0.8%      |
| 16    | 8      | 51.7%   | 0.0%      |

All converged (loss plateaued by step 3K). No configuration
exceeded 65% token accuracy or 3% instruction accuracy.

**Method 2: Gradient-based encoder inversion.** Optimize soft
embeddings through encode_soft to match target T1 vector, snap
to nearest token embeddings. Tested on d_out=128 at multiple
learning rates and iteration counts.

| n_iters | lr    | final_loss | tok_acc |
|---------|-------|------------|---------|
| 500     | 0.1   | 0.003      | 31.2%   |
| 2000    | 0.01  | 0.0003     | 23.5%   |
| 2000    | 0.001 | 0.023      | 15.6%   |

Tighter convergence (lower loss) produced WORSE token accuracy.
The optimizer finds preimages that encode correctly but don't
correspond to real token embeddings.

**Diagnosis: distance-preserving ≠ information-preserving.**

The encoder was trained for three objectives: pairwise MSE,
equivalence collapse, and destination classification. None
require the T1 vector to preserve enough information for full
instruction reconstruction. The geometry encodes "how this
instruction relates to others" but not "which instruction this
is."

The ~65% token accuracy is consistent with recovering what IS
in the vector: opcode (well-encoded via equiv loss and MSE)
and dest_reg (explicit CE head) ≈ 2 correct tokens out of 4-7.
Source registers and immediates are poorly encoded (commutativity
collapses operand order, immediate insensitivity persists).

The gradient decode method performs worse (31%) because the
encoder mapping is many-to-one: the equivalence loss explicitly
collapses distinct instructions to the same T1 region. Gradient
optimization finds A preimage but not THE original.

**Implication:**

The encoder needs reconstruction CE as a training loss alongside
MSE + equiv + dest CE. The decoder provides gradient that says
"also be information-preserving." This avoids the Exp 22-23
blind-leading-blind problem because three of four encoder loss
terms (MSE, equiv, dest CE) work independently of the decoder.
The decoder's initial random gradient adds noise but no
systematic error; as the encoder's geometry stabilizes (by
~step 1K from MSE alone), the decoder quickly converges and
its feedback becomes clean signal.

Recommended setup: cold-start joint encoder+decoder training
with recon_weight ≈ 0.1 to prevent the decoder's large initial
loss (~4.5 vs MSE ~0.3) from dominating early gradient.

## Experiment 28: Joint encoder+decoder reconstruction training

Tested three approaches to making the T1 encoder decodable:
token-match CE, round-trip loss, and REINFORCE with execution
reward. Each illuminated a different aspect of the problem.

**Approach 1: Token-match CE at various weights.**

Joint encoder+decoder, cold start, d_out=64.

| weight | tok_acc | Pearson | equiv PASS |
|--------|---------|---------|------------|
| 0.10   | 96%     | 0.660   | 3/13       |
| 0.05   | 85%     | 0.667   | 3/13       |
| 0.01   | ~75%*   | —       | —          |

(*killed mid-run)

Token-match CE teaches the decoder effectively (96% at w=0.1)
but destroys encoder geometry regardless of weight. Pearson
drops from 0.91 to 0.66 even at w=0.05. The cause is NOT
equivalence conflicts (those affect ~0.1% of gradient) but a
general tension: CE wants the encoder to preserve token identity,
MSE wants it to preserve pairwise distances. These partially
conflict because instructions with similar exec_distance need
to be equidistant (MSE) but still individually distinguishable
(CE).

**Approach 2: Round-trip loss (Gumbel-softmax).**

Decode → Gumbel-softmax → soft embeddings → encode_soft →
compare T1 to original. Equivalence-tolerant by design.

| tau schedule     | tok_acc | Pearson | equiv PASS |
|------------------|---------|---------|------------|
| slow (1→0.1/100K)| 6%      | 0.903   | 12/13      |
| fast (1→0.01/25K)| 6%      | 0.903   | 12/13      |

Geometry perfectly preserved (Pearson 0.903 ≈ encoder-only
0.906). But the decoder learns nothing useful: 6% token
accuracy at all settings. The decoder exploits Gumbel-softmax
by producing continuous blends of token embeddings that
re-encode to the target T1 without corresponding to any real
tokens. Even at tau=0.01 (near-discrete), the decoder commits
to specific tokens that form invalid/nonsensical instruction
sequences whose re-encoding happens to be close.

Diagnosis: round-trip loss is too permissive — it accepts any
token sequence (including garbage) that re-encodes nearby.

**Approach 3: REINFORCE with execution-equivalence reward.**

Decoder samples discrete tokens, decoded instructions are
executed via the CPU emulator, compared to original. Reward =
fraction of matching execution outputs across random inputs.

50-step proof of concept (B=256):
```
valid 99%   equiv 11%   tok_acc 0%   reward 0.22
```

The decoder learned valid instruction syntax (99% valid) from
pure execution reward — no token-match supervision. It produces
genuinely different instructions that sometimes compute the same
thing (11% execution-equivalent). Token accuracy is 0% because
the decoder is NOT memorizing tokens; it's discovering functional
equivalents. This is exactly the thesis: decode to any
functionally-equivalent instruction, not the original tokens.

However, REINFORCE at 1000 steps did not converge — high
variance caused the equiv rate to oscillate between 4-12%
without trending upward. Classic REINFORCE credit assignment
problem: sequence-level reward doesn't identify which token
was responsible.

**Performance profiling (B=4096, 16-core parallel CPU):**

```
encoder forward (frozen, compiled):  19ms
decoder forward + sample:            15ms
CPU reward (parallel, 16 workers):  164ms
REINFORCE backward + optim:          37ms
                              total: 243ms/step
```

Validated by replacing CPU reward with fake random reward:
step time dropped from 243ms to 69ms (delta = 174ms, matching
the 164ms measurement). CPU emulation is genuinely the
bottleneck.

Key optimization finding: the initial ~600ms reward time was
caused by 8192 individual GPU→CPU synchronization points
(per-instruction `.item()` and `.tolist()` calls). Bulk
`.cpu().numpy()` transfer reduced this to ~164ms.

**Conclusions:**

1. Token-match CE: decoder works, geometry breaks. Even very
   low weights (0.01) damage Pearson because CE and MSE have a
   fundamental (not just equivalence-related) tension.

2. Round-trip loss: geometry preserved, decoder useless. The
   Gumbel pathway allows "blending" cheats that circumvent
   discrete token commitment.

3. REINFORCE with execution reward: the principled solution.
   Naturally equivalence-tolerant, teaches valid instruction
   syntax, discovers functional equivalents. Needs faster
   execution (GPU emulator) and variance reduction (multiple
   samples per instruction) to converge.

**Status:** REINFORCE decoder training is a work in progress.
The mechanism works (valid instructions produced, equivalences
discovered) but convergence requires a faster emulator to
support multi-sample variance reduction. Next step: build a
batched PyTorch RV32I emulator to bring the 164ms CPU reward
down to ~5ms on GPU, enabling K=10-50 samples per instruction
at the same wall-clock cost.

## Experiment 29: GPU batch emulator + REINFORCE convergence

Built a batched PyTorch RV32I emulator that executes all 37
opcodes in parallel via `torch.where` cascades. Includes a GPU
token parser that converts (B, T) token tensors directly to
instruction fields — no CPU parsing loops.

**GPU emulator performance:**

| Component | Time (B=4096) |
|-----------|---------------|
| batch_execute | 1.69ms |
| batch_parse_tokens | 2.09ms |
| Full reward (parse+execute×4+compare) | 28ms |

Correctness: 100% match against CPU emulator on all non-LOAD
instructions (3248/3248 tested). LOADs use hash-based
deterministic memory (different from CPU's rng-based
SparseMemory, but internally consistent for REINFORCE
reward comparison). STOREs return truncated stored value.

**Optimization journey:**

| Setup | reward_ms | total ms/step |
|-------|-----------|---------------|
| CPU serial (B=256) | 70ms | 110ms |
| CPU parallel 16-core (B=4096) | 164ms | 243ms |
| GPU exec + CPU parse | 130ms | 227ms |
| Fully GPU (parse + exec) | 28ms | 78ms |

Key bottleneck discoveries:
- 8192 individual GPU→CPU `.item()`/`.tolist()` syncs cost
  ~500ms. Bulk `.cpu().numpy()` transfer fixed it.
- CPU token parsing (try_decode_tokens × 8192) cost ~130ms
  even after GPU execution was fast. GPU token parser
  eliminated it.

**REINFORCE convergence with shaped reward:**

| Config | equiv | tok_acc | reward |
|--------|-------|---------|--------|
| Binary reward, K=1 | 9% (flat) | 0% | 0.16 |
| Binary reward, K=10 | 16% (flat) | 0% | 0.25 |
| Shaped reward, K=10 | **23%** | **18%** | **0.56** |

Shaped reward decomposes the binary all-or-nothing signal into
four per-field components: 0.25 × (opcode match) + 0.25 ×
(dest_reg match) + 0.25 × (data_val match fraction) + 0.25 ×
(pc match fraction). This gives the decoder richer credit
assignment and enables convergence that binary reward can't.

The 23% execution-equivalence rate was achieved on a FROZEN
encoder that was never trained for decodability (Exp 27 showed
it only supports 65% token accuracy even with supervised CE).
Joint encoder+REINFORCE-decoder training — where the encoder
adapts to be decodable via execution-equivalence signal, not
token-match CE — is the natural next step and should
significantly improve this rate.

**Implications:**

1. The GPU emulator makes REINFORCE practical at training
   speed (~78ms/step at B=4096), not just feasible.
2. Shaped reward is essential — binary reward plateaus at
   9-16% regardless of sample count.
3. The decoder CAN learn to produce execution-equivalent
   instructions from pure execution reward, even from an
   encoder not designed for decodability.
4. The remaining gap (23% → 90%+) likely requires joint
   training so the encoder geometry supports decoding.

## Experiment 30: Decoder capacity sweep and generalization

The 85.5% plateau of the 8L/d128/64m decoder (Exp 27-28) was
confirmed as a capacity limit, not an encoder information
limit: 300K additional steps at constant LR=3e-4 oscillated
between 84.7-85.5% without breaking through. Zhang et al.
(2017) predicts a memorization floor of params ≥ data points;
at 2.68M params vs 4.5M token predictions (819K instructions ×
~5.5 positions), the ratio was 0.60× — undersized.

**Scaling attempt 1: 16L/d512/64m (69.5M params, 5.1× target).**
Total optimization failure. After 1.6M steps at B=512 (equiv
to 200K steps at B=4096), tok_acc remained stuck at ~17% while
loss dropped from 2.72 to 2.29. Post-mortem diagnostics on the
trained model:

| Test | Baseline | Shuffled vecs | Zero vecs | Random vecs |
|------|----------|---------------|-----------|-------------|
| tok_acc | 16.9% | 3.3% | 14.0% | 4.8% |

The model WAS using emission vectors (shuffling destroyed
accuracy) but the cross-attention pattern was broken: layer 0
had 99.5%-of-max attention entropy (completely uniform), layer
15 had 16.7% (sharp attention). Only the last ~1-2 layers had
learned meaningful selection, and they were working with
intermediate representations from 14 emission-blind layers.

Mechanism: 16 layers without warmup made the gradient path too
long for early-layer cross-attention to bootstrap. Self-attention
and FFN paths (which don't need cross-attention to reduce loss)
learned a position-dependent unigram model first; by the time
late-layer cross-attention could have contributed, the trap was
set. Loss dropped (model learned token frequency) without
accuracy improvement (top-1 never shifted off the marginal mode).

**Scaling attempt 2: 8L/d256/64m (9.55M params, 2.1× target).**
Clean success. At 200K steps at B=512 peaked at 86.7% —
breaking the d128 ceiling for the first time. Full run at 800K
steps at B=1024 (equiv to 200K at B=4096) reached **95.5% peak
tok_acc, 70.4% instr_acc**. No warmup needed at 8 layers —
cross-attention bootstrapped from step 0.

Checkpoint was not saved (forgot `--save-decoder`) — trajectory
recoverable but specific weights lost.

**Scaling attempt 3: 8L/d512/64m (35.9M params, 8.4× target).**
WSD schedule: 25K sine warmup → 625K stable at 3e-4 → 150K
cosine decay. Added gradient clipping at max_norm=1.0 after a
no-clipping attempt loss-exploded at step 30K (loss 0.93 →
1.90 → 2.69, accuracy crashed to 17%). With clipping, survived
one mid-training dip (92.6% → 82.9% at step 230K) and recovered
cleanly. Peaked at 96.2% tok_acc, 78.4% instr_acc at step 700K
during decay.

**Generalization test (held-out instructions, seed=99999):**

| Model | Train tok_acc | Held-out tok_acc | Delta |
|-------|---------------|------------------|-------|
| d256 @ 200K | 82.3% | 71.5% | -10.7% |
| d512 @ 800K | 86.5% | 71.9% | -14.6% |

Held-out accuracy is essentially identical across models
despite 3.7× capacity and 4× data exposure. **The encoder's
T1 geometry carries ~72% tok_acc worth of decodable
instruction-identity signal.** Everything above 72% on
training is pure memorization. More decoder capacity buys
more memorization, not more generalization.

**Equivalence tolerance (autoregressive, held-out, d512):**

| Category | Count | % |
|----------|-------|---|
| Exact match | 1,436 | 7.0% |
| Execution-equivalent (not exact) | 3,352 | **16.4%** |
| Valid but not equivalent | 14,868 | 72.6% |
| Invalid | 824 | 4.0% |
| **Correct (exact + equiv)** | 4,788 | **23.4%** |

The decoder emerges equivalence tolerance without explicit
equivalence training. 16.4% of held-out autoregressive
decodings are execution-equivalent-but-not-identical — the
encoder's equivalence-collapsing geometry (Exp 24-26) is
visible in the decoder's output distribution. Only 4.0%
invalid means the decoder learned RV32I grammar cleanly.

**Implications for the plan:**

1. CE decoder training is validated. The technique reaches
   95%+ train tok_acc given adequate capacity.
2. Generalization is bounded by the encoder. The 72% held-out
   ceiling is consistent across model sizes and training
   budgets — pushing it requires encoder changes, not decoder
   changes.
3. Equivalence tolerance emerges from the encoder geometry
   alone without REINFORCE or explicit equivalence loss on
   the decoder.
4. The decoder is "good enough" to provide reward signal for
   gate training — the 23% execution-equivalent / 4% invalid
   distribution gives clear contrast between "good T1
   emission" and "garbage T1".

**Training infrastructure improvements:**
- Cached + autocast + fused Adam: 72ms/step at d512 B=1024
- Warmup-Stable-Decay (WSD) schedule with configurable
  transitions and phase-endpoint checkpoints
- Gradient clipping: essential for d≥512 to prevent loss
  explosions from occasional bad-batch gradients
- Micro-batching: re-chunks cached batches post-cache for
  VRAM control independent of pre-generated batch size

---
