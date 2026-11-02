# Phase 8: Shift-Reduce Architecture (Gate Training)

Goal: replace Phase 7's parallel bidirectional encoder with a streaming
shift-reduce parser whose gates control information flow causally. The
accept gate admits tokens one at a time, the emit gate decides when to
output a T1 vector, and the evict gate removes stale tokens from the
window. Joint training of this parser failed (Experiments 22-23), so the
rest of the phase trained its components separately, starting with the
T1 encoder's equivalence geometry (Experiments 24-26) and its
decodability (Experiments 27-30).

## Architecture

- **Window encoder:** a bidirectional transformer over the current window
  contents, producing a T1 candidate on S^127 at every iteration.
- **Gate controller:** a GRU reading the T1 candidates in sequence, with
  three independent 2-layer MLP gate heads. Accept sees (next token
  embedding, T1 candidate, GRU hidden state); emit and evict see (T1
  candidate, GRU hidden state).
- **Decoder:** an autoregressive transformer decoder conditioned on the
  T1 vector, scoring the T1 candidate against the window's tokens at
  every iteration, which gives a per-iteration reconstruction signal.

Gates read the compressed representation, not the raw tokens: the emit
gate judges whether the vector is good enough to output. Tokens arrive
strictly in order and the window is contiguous; accept cannot skip
tokens, and evict removes the oldest token (FIFO). The GRU supplies
temporal context without future information.

## Experiment 22: Gumbel-sigmoid gates (2000 steps)

Gates sampled with Gumbel-sigmoid and straight-through estimation, their
decisions driving Python `if` statements in the shift-reduce loop.

**Result: the gates learned nothing.** Accept 17-18%, emit 75-76%, evict
12%, identical from step 100 to 2000, while the decoder improved (loss
3.2 → 2.1, accuracy 23% → 34%) by reconstructing from random windows.
Converting the gate output to a Python boolean (`.item() > 0.5`) breaks
the computation graph, so the gate logits receive zero gradient.
Straight-through requires the hard decision to multiply a continuous
value, not to control program flow.

## Experiment 23: REINFORCE gates with per-iteration decoder (500 steps)

Bernoulli gate sampling trained by REINFORCE, rewarded by the decoder's
per-iteration reconstruction quality: emitting a well-reconstructed
window or waiting on a poorly reconstructed one is rewarded; the
opposite two are penalized. Advantage = reconstruction loss minus a
moving-average baseline. Gate parameters had their own optimizer and
learning rate, separate from the encoder and decoder.

| Metric     | Step 25 | Step 250 | Step 500 |
|------------|---------|----------|----------|
| Recon loss | 51.6    | 35.9     | 37.8     |
| Accuracy   | 20.9%   | 28.5%    | 28.7%    |
| Accept     | 87%     | 89%      | 88%      |
| Emit       | 10%     | 11%      | 6%       |
| Evict      | 12%     | 14%      | 14%      |
| Emissions  | 100/3.5 | 95/3.4   | 98/3.7   |

(Emissions: emits per sequence / instructions per sequence.)

**Result: the gates moved but did not learn boundaries.** The emit gate
learned to fire rarely everywhere, not at instruction boundaries. At
~29% decoder accuracy there is no contrast between a good emit point
(complete instruction, low loss) and a bad one (partial instruction,
high loss).

## Finding: joint training from scratch cannot bootstrap T1

The gates need a competent decoder to know where to emit; the decoder
needs good windows to learn reconstruction; good windows need trained
gates. Trained separately, each part is tractable: an encoder on fixed
windows is 99.6% decodable and a decoder conditioned on it reaches 99.6%
(Experiment 21), and gate supervision from instruction boundaries is
parser output. At T1, instruction boundaries are syntactically marked by
opcode tokens, so supervising them costs nothing; gates that discover
boundaries matter at T2 and above, where boundaries follow meaning.

Design facts the joint experiments settled:
- Gumbel-sigmoid gating Python control flow gives zero gradient;
  REINFORCE trains the gates.
- The decoder must score every iteration, not only emit points, so the
  emit gate is penalized for missed opportunities.
- The gate controller is causal (a GRU) and reads the T1 candidate rather
  than raw tokens.
- Gates fire independently per iteration; evict removes one token per
  firing.
- Window-size-weighted reconstruction gives a cost signal for all three
  gates without explicit density targets.

## Experiment 24: Equivalence manifest and tuple injection

**Equivalence manifest.** 16 classes over six structural axes:
operand-order commutativity, self-operand identity, zero-register alias,
zero/identity immediate, cross-opcode aliasing, and condition-forcing
for branches. Each class is a rule with free variables (rd, rs, rs1,
rs2, imm, shamt) and optional constraints, instantiated per binding into
a tuple of concrete equivalent instructions. 13 classes were testable.

**Evaluation.** Per class, ratio = intra-class distance / inter-class
distance on the trained encoder: PASS < 0.3, WEAK < 0.7, otherwise FAIL.

**Injection.** Instruction batches carried canonical tuples at a set
rate, in three passes: a guaranteed `min_per_class` tuples per class,
per-class `boost` counts, then random fill.

**Other changes in this setup:**
- Destination registers include x0: a write to x0 is a NOP, a real
  semantic class.
- Distance metric `log1p(log1p(|data_diff| + |pc_diff|))` ("nested
  log"), sharpening low-end differences so near-equivalent contrasts
  (ADDI imm=0 vs imm=1, raw diff 1) get more gradient.
- Geometric probes at every log step: ADD-ADD, ADD rs,rs vs SLLI rs,1
  ("canon"), SLLI-SLLI, and commutative ADD swap distances.

Reference point: a 250K-step run with no injection, no destination
heads, and a learning rate stuck constant by a schedule bug scored
3 PASS / 3 WEAK / 7 FAIL.

| config                                  | steps | PASS | WEAK | FAIL |
|-----------------------------------------|-------|------|------|------|
| 1% injection                            | 10K   | 4    | 2    | 7    |
| 10% + min_per_class=4                   | 10K   | 5    | 4    | 4    |
| + boost shl1=60                         | 10K   | 6    | 1    | 6    |
| + nested log                            | 10K   | 6    | 3    | 4    |
| 250K no boost                           | 250K  | 5    | 4    | 4    |
| 250K + 20% + boost commutatives + shl1  | 250K  | **8**| **2**| **3**|

Per class, 250K steps + 20% injection + boost:

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

**Findings:**

1. **Injection density matters.** 1% (~10 canonical tuples per batch) is
   too thin to overcome random-operand gradient; 10% + 4 guaranteed
   tuples per class suffices for most classes; cross-opcode and
   high-specificity contrasts need 20% + per-class boost (30 tuples per
   batch for the commutatives and double_is_shl1).

2. **Loss does not track the objective.** Total loss plateaued at 0.156
   by step 50K and stayed flat for 200K more steps, while canon fell from
   0.67 at step 50K to 0.13 at 250K. The optimizer wanders the equal-loss
   manifold, biased by the persistent injection gradient. Loss-based
   convergence says nothing about equivalence collapse.

3. **Opcode clusters dissolve with training.** At ~10K steps ADD-ADD =
   0.03; by 50K, 1.0. This lets cross-opcode bridges form but makes
   operand order matter, so commutative pairs stop collapsing for free:
   the commutative PASS at short training is a by-product of tight
   opcode clusters, and long training needs commutative boost.

4. **Cross-opcode aliases collapse with long training.** double_is_shl1
   (ADD rs,rs ≡ SLLI rs,1) went from FAIL (1.87 at 10K) to PASS (0.12 at
   250K + boost). Opcode similarity is a learned bias that dissolves
   under sustained injection, not a structural barrier.

5. **The persistent failures are metric-bound.** sign_test,
   mv_via_zero_reg and commutative_xor have contrasts that differ only at
   rare register values (rs=0) or by tiny raw differences that the nested
   log compresses further. Boost does not help.

## Experiment 25: Explicit equivalence loss

In Experiment 24's batch MSE, the gradient share of any one equivalent
pair is ~0.012% of the total. The explicit loss gives equivalences their
own gradient budget: each step, every class with ≥2 canonical templates
is instantiated under a fresh binding, all classes' templates are
encoded in one padded forward pass, and each class contributes
`mean(d_ij² for i<j)` (target distance 0); the term averages over
classes, scaled by `equiv_weight`. Encoding all classes in one pass kept
the step at 51ms (49ms without the term; 106ms with one pass per class).

Injected pairs share the batch's ~8.4M-pair gradient budget with random
pairs, and even at extreme injection rates true within-tuple pairs cap
at ~0.1% of pairs. The explicit term works on ~75 encoded instructions
of its own, ~150,000× more gradient per equivalent pair at equal weight.

| config                              | steps | PASS | WEAK | FAIL |
|-------------------------------------|-------|------|------|------|
| equiv only, w=0.5                   |  (stopped early: over-collapse) |
| equiv only, w=0.05                  | 10K   | 5    | 5    | 3    |
| equiv + 5% injection, w=0.05        | 10K   | 7    | 1    | 5    |
| equiv + 5% injection, w=0.05 (100K) | 100K  | **12** | **0** | **1** |

Per class, 100K steps, equiv + 5% injection:

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

Wall clock: 70 minutes for 12/0/1, against 150 minutes for Experiment
24's 250K-step 8/2/3.

**Checks that the explicit term did not warp global geometry**, against
the 250K-step injection-only run:
- Pearson on random pairs outside the manifest: 0.9129 vs 0.9268.
- Spearman: 0.6436 vs 0.6604.
- PCA: 94 dims for 90% variance, participation ratio 84.8 (vs 100 and
  90.75); both use ~70% of the 128-d space.
- Top 10 singular values essentially identical (55.5/56.5, 21.9/21.9,
  15.8/15.7, …): both runs reached geometrically comparable attractors.
  Each configuration was run on one seed.

**Findings:**

1. **Injection and the explicit loss are complementary.** The explicit
   loss alone (5/5/3) learned the abstract rules but lost classes that
   rely on many varied random instances (self_op_zero,
   write_zero_to_rd); 5% injection restores them.

2. **Gradient concentration beats gradient volume.** The combined 100K
   run reached tighter ratios than 250K + boost (commutative_add 0.018
   vs 0.38, commutative_xor 0.018 vs 0.74) in under half the wall time.

3. **sign_test is metric-bound, not training-bound** (0.94; 1.60 under
   injection only). Its contrast — SLTI rd,rs,0 vs SLTI rd,rs,1 — differs
   only when rs=0, rare under random states, so canonical and contrast
   agree on nearly every input. It needs input states that hit the
   distinguishing value, or a different metric.

## Experiment 26: d_out compression sweep and depth

d_out ∈ {128, 64, 32, 16} × n_layers ∈ {2, 4, 8}, each 100K steps with
equiv + 5% injection, w=0.05. Pearson on random pairs outside the
manifest.

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

**Findings:**

1. **d_out=64, 2 layers is the only configuration passing all 13
   classes**, at Pearson 0.906 (0.913 at d_out=128); 52 of 64 dims carry
   90% of the variance.

2. **sign_test responds non-monotonically to compression:** FAIL at 128,
   PASS at 64, WEAK at 32, FAIL at 16. At 64 the compression forces a
   distinction the encoder leaves unmade with more room.

3. **The Pearson ceiling depends on d_out, not depth:** ≈ 0.894 at
   d_out=32 and ≈ 0.862 at d_out=16 for 2, 4 and 8 layers alike.

4. **Depth tightens equivalences without adding capacity.**
   commutative_add at d_out=32 went from 0.009 to 0.003 from 2 to 8
   layers, with Pearson flat.

5. **Smaller spaces are used more fully:** the fraction of dims needed
   for 90% variance is 73% at d_out=128, 81% at 64, 84% at 32, 88% at 16.

Under this loss and corpus, RV32I single-instruction semantics fit a
~50-60-dimensional manifold: d_out=128 is slack, d_out=32 loses
sign_test, d_out=16 hits a capacity wall (Pearson 0.86). d_out=64 with
2 layers is the encoder the phase carried forward.

## Experiment 27: Reconstruction from the frozen encoder

Whether the pairwise-MSE-trained T1 vectors preserve enough information
to reconstruct the instruction, on all eight Experiment 26 encoders.

**Trained decoder.** Encoder frozen; a fresh decoder (d_model=128,
2 layers) trained 5K steps with teacher-forced cross-entropy, same
corpus for every encoder. All converged by step 3K.

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

**Gradient inversion.** Soft embeddings optimized through the encoder to
match the target T1 vector, then snapped to the nearest token
embeddings; d_out=128.

| n_iters | lr    | final_loss | tok_acc |
|---------|-------|------------|---------|
| 500     | 0.1   | 0.003      | 31.2%   |
| 2000    | 0.01  | 0.0003     | 23.5%   |
| 2000    | 0.001 | 0.023      | 15.6%   |

Tighter convergence gave lower token accuracy: the optimizer finds
preimages that encode correctly but are not real token embeddings.

**Finding: distance-preserving is not information-preserving.** Pairwise
MSE, equivalence collapse and destination classification never require
the vector to identify the instruction; the geometry encodes how an
instruction relates to others, not which instruction it is. ~65% token
accuracy matches recovering the opcode and destination register (≈ 2 of
4-7 tokens); source registers and immediates are poorly encoded
(commutativity collapses operand order, immediates stay weakly
resolved). Gradient inversion does worse because the equivalence loss
makes the encoder many-to-one: it finds a preimage, not the original.

## Experiment 28: Making the encoder decodable

Three ways to add a decodability signal to the d_out=64 encoder, trained
jointly from a cold start. Unlike Experiments 22-23, three of the four
encoder loss terms (MSE, equivalence, destination cross-entropy) do not
depend on the decoder, so the encoder's geometry forms regardless of
early decoder noise.

**Token-match cross-entropy.**

| weight | tok_acc | Pearson | equiv PASS |
|--------|---------|---------|------------|
| 0.10   | 96%     | 0.660   | 3/13       |
| 0.05   | 85%     | 0.667   | 3/13       |
| 0.01   | ~75% (stopped early) | —       | —          |

The decoder learns (96% at w=0.10), and encoder geometry collapses:
Pearson 0.91 → 0.66 at both measured weights. Equivalence conflicts are
~0.1% of the gradient and not the cause; the tension is general — MSE
wants instructions with similar execution distance to be equidistant,
cross-entropy wants each individually distinguishable.

**Round-trip loss (Gumbel-softmax).** Decode → Gumbel-softmax → soft
embeddings → re-encode → compare to the original T1. Tolerant of
equivalence by design.

| tau schedule     | tok_acc | Pearson | equiv PASS |
|------------------|---------|---------|------------|
| slow (1→0.1/100K)| 6%      | 0.903   | 12/13      |
| fast (1→0.01/25K)| 6%      | 0.903   | 12/13      |

Geometry was preserved (0.903 vs 0.906 encoder-only) and the decoder
learned nothing (6%). It produced continuous blends of token embeddings
that re-encode to the target without being real tokens; even at
tau=0.01 its discrete commitments were invalid sequences whose
re-encoding happened to land close. The loss accepts any sequence that
re-encodes nearby.

**REINFORCE with execution reward.** The decoder samples discrete tokens;
the decoded instructions are executed on the CPU emulator; reward =
fraction of matching execution outputs over random inputs. After 50
steps at B=256:

```
valid 99%   equiv 11%   tok_acc 0%   reward 0.22
```

From execution reward alone, with no token supervision, the decoder
learned valid syntax (99%) and produced different instructions that
sometimes compute the same thing (11%) — decoding to any functional
equivalent rather than the original tokens. Over 1000 steps it did not
converge: the equivalent rate oscillated between 4-12%, sequence-level
reward not identifying which token was responsible.

**Cost at B=4096, 16 CPU workers for reward:**

```
encoder forward (frozen, compiled):  19ms
decoder forward + sample:            15ms
CPU reward (parallel, 16 workers):  164ms
REINFORCE backward + optim:          37ms
                              total: 243ms/step
```

A fake random reward cut the step to 69ms, confirming CPU emulation as
the bottleneck. Reward computation with 8192 per-instruction GPU→CPU
syncs (`.item()`, `.tolist()`) took ~600ms; one bulk transfer brought it
to ~164ms.

**Findings:** token-match cross-entropy trains the decoder and breaks the
geometry; round-trip loss keeps the geometry and trains nothing; REINFORCE
with execution reward is equivalence-tolerant and learns syntax, and
needs faster execution for variance reduction by multiple samples per
instruction.

## Experiment 29: GPU batch emulator and REINFORCE convergence

A batched PyTorch RV32I emulator executing all 37 opcodes in parallel
via `torch.where` cascades, with a GPU token parser from (B, T) token
tensors to instruction fields.

| Component | Time (B=4096) |
|-----------|---------------|
| batch_execute | 1.69ms |
| batch_parse_tokens | 2.09ms |
| Full reward (parse+execute×4+compare) | 28ms |

It matched the CPU emulator on 100% of non-LOAD instructions
(3248/3248). LOADs read a hash-based deterministic memory, internally
consistent for reward comparison but different from the CPU emulator's;
STOREs return the truncated stored value.

| Reward implementation | reward_ms | total ms/step |
|-------|-----------|---------------|
| CPU serial (B=256) | 70ms | 110ms |
| CPU parallel 16-core (B=4096) | 164ms | 243ms |
| GPU execute + CPU parse | 130ms | 227ms |
| GPU parse + execute | 28ms | 78ms |

**REINFORCE decoder on a frozen encoder:**

| Config | equiv | tok_acc | reward |
|--------|-------|---------|--------|
| Binary reward, K=1 | 9% (flat) | 0% | 0.16 |
| Binary reward, K=10 | 16% (flat) | 0% | 0.25 |
| Shaped reward, K=10 | **23%** | **18%** | **0.56** |

K is samples per instruction. Shaped reward = 0.25 × (opcode match) +
0.25 × (dest_reg match) + 0.25 × (fraction of data values matching) +
0.25 × (fraction of PCs matching).

**Findings:**
1. The GPU emulator makes REINFORCE run at training speed (~78ms/step at
   B=4096).
2. Binary reward plateaus at 9-16% regardless of sample count; per-field
   shaped reward gives the credit assignment that converges.
3. The decoder learns to produce execution-equivalent instructions (23%)
   from execution reward alone, on an encoder never trained for
   decodability (65% token accuracy under supervised cross-entropy,
   Experiment 27).

## Experiment 30: Decoder capacity and generalization

An 8-layer, d_model=128 decoder on the d_out=64 encoder plateaued at
85.5% token accuracy; 300K further steps at constant LR=3e-4 oscillated
between 84.7-85.5%. At 2.68M parameters against 4.5M token predictions
(819K instructions × ~5.5 positions) it is 0.60× the memorization floor
of parameters ≥ data points (Zhang et al., 2017), so the plateau is a
decoder capacity limit. Larger decoders:

**16 layers, d_model=512 (69.5M params, 5.1× the floor): optimization
failure.** After 1.6M steps at B=512 (≈ 200K at B=4096), token accuracy
stayed at ~17% while loss fell from 2.72 to 2.29.

| Test | Unmodified vecs | Shuffled vecs | Zero vecs | Random vecs |
|------|----------|---------------|-----------|-------------|
| tok_acc | 16.9% | 3.3% | 14.0% | 4.8% |

The model used the emission vectors (shuffling them destroyed accuracy),
but cross-attention was broken: layer 0's attention entropy was 99.5% of
maximum (uniform), layer 15's 16.7%. Only the last one or two layers
selected anything, working from 14 layers of emission-blind
representations. Without warmup the gradient path was too long for
early-layer cross-attention to bootstrap; self-attention and FFN paths
learned a position-dependent unigram model first, lowering loss without
moving top-1 predictions off the marginal mode.

**8 layers, d_model=256 (9.55M params, 2.1× the floor): trained
cleanly**, cross-attention bootstrapping from step 0 without warmup.
86.7% at 200K steps at B=512; 95.5% peak token accuracy, 70.4%
instruction accuracy at 800K steps at B=1024 (≈ 200K at B=4096).

**8 layers, d_model=512 (35.9M params, 8.4× the floor).**
Warmup-stable-decay schedule: 25K sine warmup, 625K stable at 3e-4, 150K
cosine decay. Gradient clipping at max_norm=1.0 is required at this
width: unclipped, the loss exploded at step 30K (0.93 → 1.90 → 2.69,
accuracy 17%). Clipped, it recovered from one dip (92.6% → 82.9% at
step 230K) and peaked at 96.2% token accuracy, 78.4% instruction
accuracy at step 700K.

**Generalization (held-out instructions, seed 99999):**

| Model | Train tok_acc | Held-out tok_acc | Delta |
|-------|---------------|------------------|-------|
| d256 @ 200K | 82.3% | 71.5% | -10.7% |
| d512 @ 800K | 86.5% | 71.9% | -14.6% |

Held-out accuracy is the same despite 3.7× the capacity and 4× the data
exposure. **This encoder's T1 vectors carry ~72% token accuracy of
decodable instruction identity;** training accuracy above that is
memorization, and decoder capacity buys memorization, not
generalization.

**Equivalence tolerance (d512, autoregressive, held-out):**

| Category | Count | % |
|----------|-------|---|
| Exact match | 1,436 | 7.0% |
| Execution-equivalent (not exact) | 3,352 | **16.4%** |
| Valid but not equivalent | 14,868 | 72.6% |
| Invalid | 824 | 4.0% |
| **Correct (exact + equiv)** | 4,788 | **23.4%** |

With no equivalence training on the decoder, 16.4% of held-out decodings
are execution-equivalent but not identical — the encoder's equivalence
collapse (Experiments 24-26) shows in the decoder's output. 4.0% invalid
means the decoder learned RV32I grammar.

**Findings:**
1. Cross-entropy decoder training reaches 95%+ training token accuracy
   given enough capacity.
2. Generalization is bounded by the encoder: the ~72% held-out ceiling
   holds across decoder sizes and budgets.
3. Equivalence tolerance in decoding comes from the encoder's geometry
   alone.

## Handed to Phase 9

Assemble T1 from separately trained components, in steps:

1. Encoder: trained on fixed windows rather than through the gates.
2. Decoder: trained on the frozen encoder's vectors.
3. Gates: supervised on instruction boundaries against both.
4. Assembly into the shift-reduce parser.
5. Joint fine-tuning once a T2 provides cross-level signal.

The decoder's output distribution (23% correct, 4% invalid on
held-out instructions) separates a good T1 emission from garbage, which
is the contrast gate training lacked in Experiment 23.
