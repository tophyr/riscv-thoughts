# RISC-V Thought Compression Testbed

## Purpose

Use RISC-V instruction sequences as a domain with exact, computable
semantics to validate the theoretical framework described in
WHAT_IS_A_THOUGHT.md. The key advantage over natural language: semantic
equivalence is provable via execution, and semantic distance is
measurable via machine-state comparison. This eliminates the probe
coverage problem that made natural language training intractable.

This is not a binary analysis tool. It is a testbed for studying
whether learned compression produces vector spaces with smooth,
factored, interpolable geometry — and whether search-based decoding
can invert the compression. RISC-V is chosen because it admits exact
measurement of the properties the theory predicts.

---

## Why RISC-V

**Small, regular ISA.** RV32I base integer instruction set has 40
instructions. Fixed 32-bit instruction encoding. No variable-length
complexity, no legacy cruft, no implicit flags register. The vocabulary
is tiny compared to x86 (~1500 instructions) or even ARM.

**Exact semantics.** Every instruction's effect on machine state
(32 registers + memory + PC) is fully specified. Two instruction
sequences are semantically equivalent if and only if they produce
identical machine states for all valid inputs. This is testable by
execution on concrete inputs and provable via symbolic execution for
bounded sequences.

**Computable distance.** The "semantic distance" between two instruction
sequences is the difference in their output machine states. This can be
measured as: number of registers with different values, Hamming distance
across the register file, L1/L2 distance on register values treated as
vectors, or memory diff size. This gives a continuous, exact distance
metric — no probes, no approximation, no coverage gaps.

**Free training data.** Generate random instruction sequences, execute
them, record machine states. Generate equivalent sequences via known
transformations. The supply of training pairs with exact equivalence
labels and exact distance measurements is unlimited.

**Natural hierarchy.** Tokens → instructions → basic blocks (sequences
between branches) → functions → programs. Each level composes the level
below with well-defined semantics. The hierarchy is inherent in the
computational structure, not imposed by the architecture. Critically,
the same compression-and-search framework applies at each level
transition (T0→T1, T1→T2, etc.) — the mechanism is uniform, only
the content changes.

**Structural validity is cheaply testable.** Unlike natural language
(where words are open-class and new ones can be coined freely), RV32I
instructions are a closed, well-defined set. Any proposed T0 token
sequence can be cheaply validated: is this a legal instruction? Does it
have a valid opcode, valid register numbers, and an in-range immediate?
This gives a free structural filter on top of the compressor's semantic
scoring during search-based decoding.

---

## Tokenization Strategy

### Option A: Structural tokenization

Each instruction becomes a fixed set of tokens:

```
[OPCODE] [RD] [RS1] [RS2/IMM]
```

Example: `add x5, x3, x7` → `[ADD] [X5] [X3] [X7]`
Example: `addi x5, x3, 42` → `[ADDI] [X5] [X3] [IMM_42]`

Vocabulary:
- 40 opcode tokens (RV32I base)
- 32 register tokens (x0-x31)
- Immediate value tokens (strategy TBD — direct encoding, bucketed,
  or digit-level tokenization)
- Separator/formatting tokens

Total vocabulary: ~100-200 tokens depending on immediate encoding.

### Option B: Raw byte tokenization

Each 32-bit instruction as 4 byte tokens (256 vocabulary). Simpler,
more general, but loses structural information that Option A provides
for free. The model must learn instruction boundaries and field
structure from data.

### Recommendation

Option A. The structural tokenization gives the model instruction
boundaries, field types, and opcode identity for free. There is no
reason to make the model learn what the ISA specification already
defines. Immediate value encoding is the main design choice — digit-
level tokenization (each decimal/hex digit as a separate token) is
likely best for enabling compositional understanding of numerical
values.

---

## Data Generation

### Equivalence pairs (paraphrases)

Generate instruction sequence S. Apply known semantic-preserving
transformations to produce S'. Execute both, verify identical output
states.

Transformations with provable equivalence:
- **Register renaming**: swap all uses of x5 with x7 (and vice versa),
  adjusting all reads and writes. Output semantics identical modulo
  register naming.
- **Instruction reordering**: reorder independent instructions (no
  data dependency between them). Output identical.
- **Strength reduction**: `slli x5, x3, 1` ↔ `add x5, x3, x3`
  (shift left by 1 = multiply by 2 = add to self).
- **Identity insertion/removal**: insert `add x0, x0, x0` (no-op,
  since x0 is hardwired to zero). Remove without semantic change.
- **Immediate decomposition**: `addi x5, x0, 1000` ↔
  `addi x5, x0, 500; addi x5, x5, 500`.
- **Load-store reordering**: reorder loads/stores to non-overlapping
  memory addresses.

Each transformation type produces a different kind of "paraphrase" —
analogous to cross-lingual translation in the natural language setting.
The compressor should map all equivalent sequences to the same point.

### Near-equivalence pairs (controlled distance)

Generate S. Modify one instruction to produce S' with a slightly
different output state. The machine-state difference is the exact
semantic distance.

Examples:
- Change `addi x5, x0, 42` to `addi x5, x0, 43` — one register
  differs by 1.
- Change `add x5, x3, x7` to `sub x5, x3, x7` — one register has
  a different value (magnitude depends on inputs).
- Insert an additional `addi x10, x0, 1` — one register that was
  zero is now 1.

These provide the graded distance signal for training smoothness. The
exact machine-state distance is the ground truth label.

### Contradiction pairs (semantically opposite)

Harder to define for computation than for natural language. Possible
analogues:
- Same inputs, different outputs (different computation entirely).
- Same structure but operations inverted (add ↔ sub, sll ↔ srl).
- These should be far apart in the compressed space.

### Data pipeline

```
1. Generate random valid RV32I instruction sequences (length 5-50)
2. Execute on random input states → record output states
3. Apply transformations → equivalent sequences
4. Execute equivalents → verify identical output states
5. Apply controlled modifications → near-equivalent sequences
6. Execute modifications → record exact state distances
7. Package: (sequence, output_state, equivalences, near_equivalences,
   state_distances)
```

All execution can use a simple RISC-V emulator (riscv-emu, spike, or
a minimal custom emulator for RV32I). No hardware required.

---

## Compressor Architecture

### Bottom-up approach: T0→T1 first

Start at the bottom of the hierarchy:

**T0 → T1: Single-instruction compression.** A single instruction's
tokens (4-7 of them, depending on type) compress to one T1 vector.
This is the smallest unit of "thought" — not a thought itself, but
the atomic building block. Two ADD instructions with different
parameters should be nearby in T1 space but clearly distinguishable.
The T1 space should exhibit the geometric properties the theory
predicts: smoothness, factored structure, interpolation coherence.

**T1 → T2: Sequence compression.** Once T1 works, sequences of T1
instruction vectors compress to T2 block-level representations. This
is where data flow, register dependencies, and computational structure
emerge. T2 is the first level that corresponds to what the theory
calls a "thought" — a compressed representation of a computation.

The same compression-and-search mechanism applies at both levels. The
same decoder architecture that lowers T1→T0 should lower T2→T1. The
only difference at the T0 boundary is that the output must discretize
into a finite token vocabulary.

### Joint encoder-compressor (T0→T1)

Train the encoder and compressor jointly, end-to-end. The compressor's
loss (execution equivalence + distance preservation) directly optimizes
the encoder to produce representations useful for semantic compression.

The encoder is a small transformer that processes an instruction's
tokens and produces contextual representations. The compressor pools
these into a single T1 vector. Both are trained together.

Size: small. RV32I vocabulary is ~90 tokens. Single instructions are
4-7 tokens. A 2-4 layer transformer with d_model=128-256 should be
sufficient for T0→T1. Trainable on a 4090 in minutes.

### Compressor pooling

Takes the encoder's token representations and produces a single
compressed point. Architecture candidates:

- **Cross-attention pooling**: a single learned query slot attends
  over the token representations. Start here.
- **Mean pooling + projection**: simpler baseline. Mean of token
  representations projected to output space.

Output dimensionality: start small (32-128). For T0→T1, the
information content of a single instruction is bounded — opcode
(~6 bits), up to 3 registers (~5 bits each), and an immediate
(~12 bits). The effective information is ~30 bits, so d_out in the
range of 32-64 should be sufficient. Err on the side of too small
rather than too large — overcomplete spaces enable shortcut solutions
(see TRANSFERABLE_LESSONS.md §6).

---

## Training

### Loss terms

**Evolved through experimentation** (see EXPERIMENT_LOG.md for the
full progression). The current best approach:

**MSE distance matching on the unit sphere.** Normalize compressed
vectors to the unit hypersphere (S^(d-1)). MSE between pairwise T1
distances and scaled execution distances. The sphere prevents collapse
(fixed norms), MSE provides scale (explicit nonzero distance targets),
and the combination handles both equivalence (target=0 for equivalent
pairs) and separation (target>0 for different pairs) in one loss.

Execution distance: log1p(log1p(|data_diff| + |pc_diff|)) averaged
over random input states. Nested log compresses the dynamic range
(0 to 2^32) into ~0 to 3.14 while spreading the low end more
uniformly than single log1p, so near-equivalence contrasts (e.g.,
ADDI imm=0 vs imm=1) get meaningful gradient.

**Destination classification heads.** CrossEntropy on dest_type
(register vs memory) and dest_reg (which register). Provides
structural gradient signal that the scalar exec_distance metric
is blind to (exec_distance doesn't distinguish which register
was written, only the computed value).

**Explicit equivalence loss.** Per-step, encodes canonical tuples
from the equivalence manifest (datagen/equivalences.py) and
minimizes within-class pairwise squared distances (target=0).
Combined with low-rate injection of manifest tuples into the
main batch, this reaches 12/13 manifest classes PASS in 100K
steps.

**What didn't work:** Pearson correlation (scale-invariant, allows
nonzero intercept — equivalences don't collapse), weighted Pearson
(degenerate near-zero-variance statistics in the equivalence region),
attract-only losses without sphere (global collapse), ranking loss
(collapsed to zero immediately). See EXPERIMENT_LOG.md experiments
1-11 for details.

### Training data volume

Effectively unlimited. Each training step can generate fresh random
sequences, execute them, and apply transformations on the fly. No
fixed dataset — the generator IS the dataset.

### Compute requirements

The encoder-compressor is tiny (2-4 layers, d_model=128-256, vocab ~90).
The RISC-V emulator is fast (millions of
instructions per second). The entire training loop — generate, encode,
compress, compute loss, backprop — should be dominated by the
transformer forward/backward passes, not by data generation. RTX 4090
should handle this easily.

---

## Evaluation Plan

### 1. Equivalence collapse (basic correctness)

Do provably equivalent sequences compress to the same point?
- Register renaming equivalences
- Instruction reordering equivalences
- Strength reduction equivalences

Metric: cosine similarity between equivalence pair compressed points.
Target: > 0.99.

### 2. Distance preservation (smoothness)

Is compressed distance monotonically related to execution distance?
- Generate sequences with controlled state distances (1 register
  different, 2 registers, 5 registers, etc.)
- Measure compressed distance vs state distance

Metric: Spearman rank correlation between compressed distance and
execution distance. Plot the relationship. Target: strong monotonic
correlation with smooth curve, no discontinuities.

This is the primary geometric measurement and the core publishable
result.

### 3. Factored structure

Are independent aspects of computation encoded on independent axes?
- Generate pairs that differ only in one register's value
- Measure the direction of T(S) - T(S') for different registers
- Directions for independent registers should be approximately
  orthogonal

Metric: cosine similarity between difference vectors for independent
register modifications. Target: near zero (orthogonal).

### 4. Interpolation coherence

Do interpolated compressed points correspond to semantically
intermediate computations?
- Take S1 (adds 10 to x5) and S2 (adds 20 to x5)
- Interpolate: T_mid = 0.5 * T(S1) + 0.5 * T(S2)
- Find nearest training point to T_mid
- Check if its execution result is intermediate (adds ~15 to x5)

This is the hardest test and the most novel measurement. Success here
demonstrates that the compressed space has meaningful geometry between
known points, not just at known points.

### 5. Search-based decoding (thinking)

Given a compressed point, can search-based decoding find an
execution-equivalent sequence?
- Compress sequence S to T(S)
- Run search: a conditioned proposer generates candidates, the
  compressor scores by proximity to T(S)
- Execute the best candidate sequence
- Compare output state to S's output state

Metric: fraction of trials where the decoded sequence is execution-
equivalent (identical output state). Secondary metric: mean state
distance when not exactly equivalent.

This tests the full theory: deterministic compression, search-based
inversion, free verification.

### 6. Cross-transformation generalization

Does the compressor correctly handle transformation types not seen
during training?
- Train on register renaming and reordering only
- Evaluate on strength reduction equivalences (held out)
- If the compressor maps unseen equivalence types to the same point,
  it has learned execution semantics, not transformation patterns.

---

## What This Proves (and Doesn't)

### Proves

- Learned compression can produce vector spaces with smooth, factored,
  interpolable geometry.
- Semantic equivalence can be defined via deterministic compression and
  verified for free.
- Search-based decoding can invert the compression to recover
  semantically equivalent sequences.
- The hierarchy of compression levels provides coarse-to-fine search
  guidance.

### Doesn't prove

- That the same framework works for natural language (where semantics
  are fuzzy, equivalence is approximate, and distance is subjective).
- That the geometric properties scale to large programs or complex
  instruction sequences.
- That the compressor generalizes to unseen computational patterns
  beyond the training distribution.

The RISC-V testbed validates the theoretical framework on a domain
with exact measurement. Natural language is the motivating target, and
success on RISC-V provides evidence and methodology — but not proof —
that the framework transfers.

---

## Publication Framing

"We demonstrate that learned compression of instruction sequences
produces vector spaces with smooth, factored geometry where execution-
state distance is preserved, provable equivalence classes are correctly
collapsed, and search-based decompression recovers semantically
equivalent programs. We propose this as a framework for semantic
compression more broadly, with natural language as the motivating
long-term target."

The contribution is:
1. The theoretical framework (compression defines meaning, thinking
   is search, verification is free)
2. A domain with exact measurement enabling rigorous geometric
   evaluation
3. Empirical demonstration of smooth, factored, interpolable geometry
   in the compressed space
4. Demonstration of search-based verified decompression

The RISC-V domain gives exact measurements that natural language
cannot. This makes the geometric claims rigorous rather than
approximate.

---

## Prior Work and Differentiation

The binary code similarity detection (BCSD) literature is large.
Key differentiations:

- **BCSD asks "are these similar?"** We ask "does the compressed space
  have smooth, factored geometry?" Different question entirely.
- **BCSD uses binary equivalence labels.** We use continuous execution-
  state distance as a graded training signal. Nobody does this.
- **BCSD evaluates on AUC/recall for similarity tasks.** We evaluate
  on interpolation coherence, factored structure, distance preservation,
  and search-based verified inversion. Nobody measures these.
- **Ex2Vec (2025) is closest.** They pre-train by simulating execution,
  learning instruction effects on register states. But their goal is
  better similarity detection, not geometric property investigation or
  search-based decoding.

The framing must be clear: this is not a code analysis tool. It is an
investigation of whether deterministic compression can produce meaning-
spaces with properties that enable verified search-based reconstruction.
RISC-V is the measurement instrument, not the application.
