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

**Natural hierarchy.** Tokens → instructions → register-state
transformation blocks → functions → programs. Each level composes
the level below with well-defined semantics. The hierarchy is
inherent in the computational structure, not imposed by the
architecture. Critically, the same compression-and-search framework
applies at each level transition (T0→T1, T1→T2, etc.) — the
mechanism is uniform, only the content changes.

The level-by-level units we test against:
- **T1 thought** = one complete RV32I instruction (single
  syntactic unit on a token stream).
- **T2 thought** = one register-state transformation block: a
  maximal contiguous subsequence of instructions where only the
  LAST may be a memory access (load/store) or control-flow change
  (branch/jump). Termination at memory ops keeps T2 bounded —
  inside a thought, only register state evolves, which is finite.
  Termination at control flow is the standard basic-block boundary.
- **T3 thought** = function-like unit (sequence of T2 blocks).
  TBD when we get there.

These are chosen unit definitions for benchmarking the recursive
shift-reduce machinery, not declarations about cognition. See
WHAT_IS_A_THOUGHT.md "Method vs. Cognition" for the framing.

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
instruction vectors compress to T2 block-level representations
(register-state transformation blocks; see Premise above). Inside
a T2 thought, only register state evolves — the terminating
instruction commits the block's effect to memory or transfers
control. This is where data flow, register dependencies, and
computational structure emerge. T2 is the first level that
corresponds to what the theory calls a "thought" — a compressed
representation of a computation.

The same compression-and-search mechanism applies at both levels. The
same decoder architecture that lowers T1→T0 should lower T2→T1. The
only difference at the T0 boundary is that the output must discretize
into a finite token vocabulary.

### Encoder design

The encoder is the **rename-equivariant register-state machine** described in
the next section — one shared core for every tier, trained end-to-end with the
behavioral loss set, no separate token-transformer + pooling stage. Output
dimensionality follows the information content: a single T1 instruction is
bounded (opcode ~6 bits, up to 3 registers ~5 bits each, immediate ~12 bits ≈
~30 bits), so T1's `d_out` is small (64); T2 carries multi-instruction binding
plus essence and needs more (512). Err toward too small rather than too large —
overcomplete spaces enable shortcut solutions (see TRANSFERABLE_LESSONS.md
lesson 6).

---

## Rename-Equivariant Compressor

The settled design. It is the architecture validated in EXPERIMENT_LOG.md
Phase 14; this section is the durable rationale for *why* it is shaped the
way it is.

### The nuisance variable is register naming

A tier T_n maps a unit of computation to a vector whose geometry encodes what
the computation *does*, so behaviorally-equivalent programs land in the same
place and the *operator* is the macro organizer of the space. For RISC-V
chunks the nuisance variable is **register naming**: `add r5,r3,r4` and
`add r9,r1,r2` compute the same function with different labels. The
representation must treat the *operation* as the essence and the *register
binding* (which physical register plays which role) as a secondary,
recoverable coordinate — not the other way around.

### The symmetry, and why a rename cannot be a vector addition

A register rename is a permutation π ∈ S₃₂ acting on register names. We want
the encoder to be **equivariant**: `enc(π·C) = ρ(π)·enc(C)` for a
representation ρ that is a group homomorphism (`ρ(π∘σ) = ρ(π)·ρ(σ)`). The
operator-essence part should be **invariant** (an add is an add); the binding
part should **transform predictably** with π (king − man + woman ≈ queen, but
for register roles).

A rename **cannot** be modeled as adding a per-rename vector. If
`ρ(π) = "add v_π"`, the homomorphism law forces `v_{π∘σ} = v_π + v_σ`. For
any transposition t, `t∘t = identity` ⇒ `0 = v_id = 2·v_t` ⇒ `v_t = 0`, and
since transpositions generate S₃₂, **every** v_π = 0. `(ℝ^d, +)` is
torsion-free, but permutations are built from finite-order elements; a
homomorphism into it kills all torsion → kills everything. This is not a
training difficulty — it is a theorem, and it was confirmed empirically: an
additive "consistent displacement" objective gave held-out-π displacement
consistency at the random baseline (`R ≈ 0.09`; EXPERIMENT_LOG.md Phase 13).

The right structure is an **orthogonal / permutation representation**:
`ρ(π) = R_π`, a permutation matrix acting on **register-indexed**
coordinates. O(d) is full of finite-order elements (a swap needs `R² = I`, a
3-cycle `R³ = I` — both exist), so the torsion problem vanishes; it is
norm-preserving (renames must not change validity, which rides on magnitude);
it is exact and identical across all inputs; and it never enumerates the 32!
permutations (a homomorphism is fixed by where it sends ~31 generators).

### The essence/binding split is emergent, not imposed

Do not hand-partition the vector into "these dims = operator, those =
binding." Once S₃₂ acts on a register-indexed space, the space decomposes
canonically into isotypic components: the **invariant** (symmetric) subspace
and its equivariant complement. Operator-essence = the rename-invariant
(symmetric) component — *what* the model routes there is learned; *that* it
exists is forced by the symmetry. Binding = the equivariant per-register
deviations that π permutes. The one structural commitment is that the
representation carries a **register axis** (the entities the symmetry
permutes) — categorically like a GNN's node axis or a CNN's spatial axis.
Exact by-construction equivariance *requires* it; you cannot make an
unstructured flat vector equivariant by construction.

### SSA makes the split structural

The chunk's SSA dataflow DAG already isolates register identity. A physical
register name appears in exactly two places — the **input leaves** (live-in
registers) and the **output bindings** (which register holds each final
value); everything in between references operands by SSA id, the
rename-invariant canonical form. So a rename touches *only the boundary*; the
interior is bit-identical. The invariant essence (interior computation) and
the equivariant binding (boundary register-assignment) are SSA's own
structure, surfaced — not a split we invent.

### The register-state machine

Realize the encoder as a register-indexed state machine over the dataflow:

- State `S ∈ ℝ^{n_regs × d}` — the running essence bound to each register.
- Seed live-in registers with **distinct, anonymous leaf tags** (below).
- For each instruction in order: **read** the operand slots' states, apply a
  shared **op-cell** to produce the result essence, **write** it to the
  destination slot (functional, SSA-style — one-hot read/write, no in-place
  scatter, so the step captures cleanly into a compiled CUDA graph).
- Chunk representation = the final state `S`. Essence = a
  permutation-invariant **attention pool** over the register axis (not a plain
  mean, which would drag in the random leaf-tags of untouched slots); binding
  = the per-slot structure, read back by the binding heads.

This is equivariant by construction: renaming registers permutes which slots
are read and written, so `S` permutes and the symmetric-pool essence is
invariant — *exactly*, with no training. Equivariance is exact precisely
because there are **no per-register learned parameters**: the only
register-indexed input is the anonymous tag, and reads/writes are pure one-hot
tensor ops.

Two channel groups per slot, both equivariant: **value channels** (set on
write, preserved on read — the dataflow state the op-cell threads, which makes
multi-hop value-numbering correct) and **event channels** (accumulate
read/write events with order timestamps, so `live_in`/`in_slot`/`out_slot`
are recoverable from the emitted object; a pure value-state records only
writes). The read pulse is content-dependent, so a behaviorally-irrelevant
read (`AND rs1,x0`) is distinguishable from a relevant one.

**Distinct anonymous leaf tags (subtle but essential).** Inputs must be
*distinguishable* (so `add r2,r3` ≠ `add r2,r2`) but carry *no intrinsic
identity* — a per-register learned embedding `f(r)` would have to be constant
to be equivariant, destroying distinctness. Resolution = the random-node-
feature trick: give each live-in register a random tag, sampled per chunk,
that permutes with π. The network learns to use tags only as "these two
inputs differ," not "this is register 5"; structurally-identical inputs
collapse, distinct ones stay apart.

### The tier contract — never re-embed tokens

**T_{n+1} consumes only the output of T_n, never raw tokens.** Every tier
emits an `(essence, binding)` object: an essence vector (rename-invariant,
"what this unit computes") and binding references (which registers, in which
roles, as equivariant indices into the register axis).

- **T1** (one instruction): the state machine with `seq_len=1`, its per-step
  content the embedded **opcode + immediate tokens** (registers stripped to
  wiring). Essence = the operation; binding = the register filling each role.
- **T2** (chunk): the *same* machine over K steps, its per-step content the
  **frozen T1 essence** for that instruction — *not* a re-embedded opcode. The
  op-cell takes `[T1_essence_i, operand_states]`; the wiring (which registers
  each instruction reads/writes) comes from **T1's predicted binding**, so the
  experiment genuinely tests whether T1's emission is sufficient for the next
  tier. If a tier re-derives its inputs from raw instructions, it has bypassed
  the tier below and the hierarchy is broken. ("T1 essence ≈ a function of
  opcode" does **not** license "so T2 can embed opcodes itself.")

T1 is therefore literally T2 with `seq_len=1`: the only legitimate difference
between the tiers is the ingestion front-end and the hparams.

### GVN: the invariant essence is the value-number

The rename-invariant essence *is* a value-number — two computations producing
the same value get the same essence; different values stay apart.
Rename-equivalence collapses **structurally** (free, from the equivariant
core). Other behavioral equivalences — commutative operand swaps (`a+b==b+a`),
reordering of independent instructions, dead code, algebraic identities — are
*not* structural and are **trained in** by the behavioral targets (the
canonical `out_regs` value-prediction supervises the rename-invariant I/O
function, so equivalent chunks receive identical targets). Distinct behaviors
must stay apart — the essence is behavioral, not a syntactic hash.

### Open problems (additive over the equivariance skeleton)

These are real work, each additive over a stable "register-indexed state; the
32 GP registers are the permutable slots" skeleton:

- **PC / branches.** PC is currently read from the invariant essence
  (`pc_writes`), with `n_regs = 32` (GP only). The clean extension is a
  non-renameable PC slot (renaming permutes only the 32 GP slots).
- **Memory (loads/stores).** Modelled as another non-renameable slot;
  structurally clean, but whether the essence can capture value-dependent
  memory dataflow / aliasing is genuinely unproven — the real expressiveness
  risk. Currently out of scope (`precompute_chunk` raises on memory ops).
- **Immediates.** Rename-invariant content; enrich T1's essence, orthogonal
  to equivariance. Low risk.
- **Validity / parse.** The equivariant intake needs a *valid parse*, so
  validity moves from a learned magnitude toward a parse-level gate. How an
  equivariant T1 integrates with the streaming shift-reduce parser (which
  proposes possibly-invalid windows) is unresolved.
- **Capacity.** The permutation realization is an exact faithful
  representation of S_n, so non-abelianness is handled — but whether a given
  `d` has room for rich binding + essence is empirical.

---

## Training

### Loss terms

**Evolved through extensive experimentation** (see EXPERIMENT_LOG.md, esp.
Phases 9–14). The settled approach is one **behavioral loss set shared by
T1 and T2** (`compressor.train.binding_losses`) — T1 is the
single-instruction special case. No pairwise distance-matching, no explicit
equivalence loss, no syntactic CE heads: equivalence is **emergent**.

**Magnitude / validity loss.** Soft `(‖v‖ − is_valid)²`: ‖v‖→1 for a
valid thought (one complete instruction at T1; a well-formed chunk at T2),
→0 otherwise. Vectors live in open space (no `F.normalize` on the output);
magnitude carries validity, direction carries semantics. A *hard*
normalize onto the unit sphere breaks `out_slot` — train TOWARD the
surface with this soft loss, don't project ONTO it.

**Behavioral binding losses** (the binding heads read the equivariant
per-slot state `T`, one score per register slot; `pc_writes` reads the
invariant essence):
- `live_in` / `live_out`: BCE on which registers are read / written.
- `pc_writes`: BCE on whether the chunk writes PC.
- `in_slot` / `out_slot`: read/write ORDER, as ListMLE (Plackett–Luce)
  over per-register scalar score heads (argsort to decode; duplicate-free).

**Value-prediction (`value_predict`).** Per-anchor, per-output-slot MSE:
from `(essence, canonical input-slot values)` predict each output register's
**canonical** value (the `out_regs` baseline — inputs relocated to canonical
first-read positions, so the target depends only on operand structure, not
register names), on a fixed set of sampled anchor input states. Feeding the
head canonical inputs to match the canonical target is load-bearing: the
earlier bug fed *raw* register values against a *canonical* target, making
the task mathematically inconsistent. vp is a **diagnostic probe** of whether
the essence carries the operator's I/O function — it sits at an intrinsic
floor (regressing wrapped/bitwise ops from log-compressed scalars is hard for
any MLP) and its absolute value is not the objective. The operator-clustering
that the flat encoder could never get from vp comes instead **for free from
the equivariant core** (rename-invariant essence = the value-number) plus the
canonical vp target collapsing the non-structural equivalences.

**Emergent equivalence.** With the behavioral targets alone, provably
equivalent instructions/chunks collapse without any equivalence loss:
rename-twins collapse *structurally* (the equivariant core), and non-rename
behavioral equivalents (commutativity, reorder, dead code) collapse from the
canonical vp target. Measured as GVN collapse split by pair type
(EXPERIMENT_LOG.md Phase 14). A design constraint this surfaces: a
rename-*invariant* behavioral target (e.g. a GVN-bijection distance) is the
WRONG objective here — it would weld apart the binding distinctions the
equivariant representation deliberately keeps recoverable.

Dead-end objectives explored on the way here (collapse losses, pairwise
distance-matching, additive king−queen equivariance, unit-sphere
normalization, syntactic CE heads) are recorded in EXPERIMENT_LOG.md; the
behavioral loss set above superseded them all.

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
