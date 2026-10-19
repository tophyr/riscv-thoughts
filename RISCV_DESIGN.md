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

## Tokenization

**Structural, one token per instruction field** (`tokenizer/`): an opcode
token, register tokens, and an immediate spelled as hex digits — `add x5,x3,x7`
→ `[ADD] [X5] [X3] [X7]`, and `addi x5,x3,42` → `[ADDI] [X5] [X3] [H0] [H2]
[HA]`, with a `NEG` prefix for a negative literal. Field widths are per
instruction type, so the longest instruction is a J-type at nine tokens.

Structural beats raw bytes (four byte tokens per instruction) because it hands
the model instruction boundaries, field types and opcode identity for free,
and there is no reason to make it learn what the ISA specification already
defines. Immediates are digit-level rather than one token per value so that
numerically close literals share structure; a value-per-token vocabulary would
make every immediate an unrelated symbol.

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

**Exception — the two reordering transformations are raw-execution
equivalences, not essence-level ones.** Reordering independent instructions
(and non-overlapping loads/stores) leaves the final register file identical
*with register identity intact*, but it permutes the canonical first-read
input order and the write order, so the two chunks are different ordered
identity-free I/O functions and receive **different** vp targets. Their
equivalence is recovered at the `(essence, binding)` level, not in the essence
(see "GVN: the invariant essence is the value-number"). Do not use them as
positive pairs for an essence-collapse objective or metric.

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

Generation, SSA analysis, anchor execution and the canonical `out_regs` oracle
all run in `datagen`, which streams self-describing RVT batches to a trainer on
stdout. Execution uses this project's own RV32I emulator (`emulator/`); no
hardware and no external simulator. The commands, flags, wire format and
multinode operation are the `datagen` skill.

---

## Compressor Architecture

### The tiers, bottom-up

**T0 → T1** compresses one instruction's tokens to one vector. It is the
atomic building block rather than a thought itself: two `ADD`s with different
operands should be near each other and still distinguishable.

**T1 → T2** compresses a sequence of T1 vectors to one block vector. The block
is a register-state transformation block — a maximal run of instructions where
only the last may touch memory or control flow (see "Why RISC-V" for why the
unit is defined there) — so inside a T2 thought only register state evolves,
and the terminating instruction commits its effect. T2 is the first level
corresponding to what the theory calls a thought: a compressed representation
of a computation, where data flow and register dependencies are what must be
captured.

The same compression-and-search mechanism applies at both levels, and the same
decoder architecture that lowers T1→T0 should lower T2→T1. The only difference
at the T0 boundary is that the output must discretize into a finite token
vocabulary.

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

The settled design. It is the architecture validated in EXPERIMENT_LOG
("Rename-equivariant T1/T2"); this section is the durable rationale for *why*
it is shaped the way it is.

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
consistency at the random baseline (`R ≈ 0.09`; EXPERIMENT_LOG, "Pairwise
behavioral geometry").

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

**The pool's per-head value width is a capacity ceiling, separate from the
essence width.** The pool runs `k` attention heads over the 32 slots and
concatenates their outputs, so each head contributes `d_value / k` of the
essence. If all heads share one value projection, a slot reaches the essence
only through that single `d_value / k`-wide image (plus its `k` attention
logits) — the essence can be 256 wide while its capacity for any one slot is 16.
Each head therefore gets its **own** slice of a `d_value`-wide projection, as in
standard multi-head attention, which makes a slot's image in the essence
`d_value` rather than `d_value / k`. This also decouples two things that would
otherwise trade against each other: heads exist to content-address *different*
written slots, and under a shared projection adding heads to separate more slots
narrowed what each one could carry.

The ceiling binds on whichever field carries the most entropy. Operation
identity is a choice among ten opcodes and survives a narrow channel; a 12-bit
immediate does not, which is why a shared projection showed up as an I-type-only
deficit.

**A slot describes a function, it does not hold a value.** Nothing in the
encoder ever sees a register's contents: the per-step content is the
instruction's opcode and immediate, and the wiring says which slots are read and
written. So a slot's state is a symbolic description of the computation that
produced it, and every consumer — the pool, the binding heads, the
value-prediction head — reads a description rather than a number. The
value-prediction head is therefore an *evaluator*: it is handed the chunk's
input values separately and asked to apply the described function to them.

**The op-cell appends a literal to the description it reads rather than folding
it in, and width is not what it is short of.** When a slot's operand is another
instruction's result, the description it builds has to combine two operations
that each carry a 12-bit literal. Both literals survive — each is recoverable
from the emitted essence at 99.08% or better — while their SUM is not, at 84.48%,
so the chain is described as two separate offsets instead of one. An evaluator
handed the sum predicts the chunk's output at 99.95%; handed the two addends
instead it reaches 79.80%, because decoding a literal out of a continuous code
and carry-propagating it are operations that do not compose in one readout pass.

Doubling `d_value` from 256 to 512 improves those cells once; doubling again to
1024 leaves the worst of them unchanged to three decimal places. What the
description is short of is the fold, not room. That makes folding a chain's
literals an encoder-side property worth having for its own sake, and not only
for the equivalence collapse that motivates it (see "GVN" below): it is what
would let a description be evaluated in one step.

This is equivariant by construction: renaming registers permutes which slots
are read and written, so `S` permutes and the symmetric-pool essence is
invariant — *exactly*, with no training. Equivariance is exact precisely
because there are **no per-register learned parameters**: the only
register-indexed input is each input's canonical read position, and reads/writes
are pure one-hot tensor ops. Anything added to the pool's input must likewise be
register-indexed, or the invariance stops being exact.

Two channel groups per slot, both equivariant: **value channels** (set on
write, preserved on read — the dataflow state the op-cell threads, which makes
multi-hop value-numbering correct) and **event channels** (accumulate
read/write events with order timestamps, so `live_in`/`in_slot`/`out_slot`
are recoverable from the emitted object; a pure value-state records only
writes). The read pulse is content-dependent, so a behaviorally-irrelevant
read (`AND rs1,x0`) is distinguishable from a relevant one.

**Read-order must be consumed conditionally, and only the op-cell can do that.**
Operand binding needs each operand's position in the canonical (first-read)
frame. The essence pool cannot supply it: the pool is a symmetric sum that never
sees `content`, so any channel fed to it is used unconditionally — and a
commutative operand swap changes read-order, so an unconditional consumer stops
being blind to a distinction it must be blind to. The op-cell sees the opcode
and both operands in one activation, so it can use position for `SUB`/`SLT`/`SRL`
and symmetrize it for `ADD`/`AND`/`OR`/`XOR`. This is why the fix belongs at the
op-cell and why every pool channel is a recorded dead end.

**Leaf identity for input registers (subtle but essential).** The state machine
has to seed each register slot before the first instruction reads it, and that
seed faces two opposing requirements. Inputs must be **distinguishable**, so that
`add r2,r3` does not collapse onto `add r2,r2`. But they must carry **no
intrinsic identity**, because a per-register learned embedding `f(r)` would have
to be constant to be equivariant, which destroys distinctness. A seed satisfying
both must therefore distinguish inputs *from each other* without naming any of
them.

A register's **canonical read rank** satisfies both: its position in the
chunk's first-read order, counting only registers read before they are written,
written into the slot on first read. Two distinct inputs receive ranks 1 and 2; a
register read twice receives one rank twice, which is correct because it is one
input; so distinguishability is preserved exactly. Rank is assigned by read
*order*, not by register name, so a rename permutes it along with the slots and
equivariance remains exact — the torsion argument above does not apply, because
rank is not a function of the register. Rank is also the coordinate system the
value-prediction target is already expressed in (see "Training"), which is why it
doubles as the signal that binds a chunk's operands to their canonical positions.

**There is a third requirement, and it is what selects rank: the seed must be
DETERMINISTIC.** The obvious alternative — the random-node-feature trick from
graph learning, giving each live-in register a tag sampled fresh per chunk —
satisfies distinguishability and carries no register identity, but a tag is
redrawn on every forward pass, so the same instruction compresses to a slightly
different point each time it is seen (max |Δ| 1.8e-2, minimum cosine 0.9992 over
four draws of one trained encoder). That contradicts the framework's
foundational claim: a compressor is a deterministic function, and "two token
sequences mean the same thing iff they compress to the same point" is the
equivalence criterion built on it (WHAT_IS_A_THOUGHT.md). It also leaves
invariance to the seed as a *learned* property the architecture does not
enforce, and therefore one a change can silently regress. A rank carries nothing
to be invariant to, so the property holds by construction and the same
measurement reads bit-exactly zero — the same move as building the rename
symmetry into the representation rather than training a flat vector to respect
it, applied one level down to the leaf symbols.

The cost is a shift in *which* invariances come free. A state that records no
read-order cannot move under a commutative operand swap, since nothing in it
distinguishes `a+b` from `b+a` — that invariance is structural. A state that
records read-order can represent the difference. Whether the invariance then
still holds structurally — a swap moves the ranks *between* the two registers,
leaving the pool's multiset of leaf vectors unchanged — or has to be supplied per
opcode by training is UNSETTLED. The exchange is accepted
regardless, because a read-order-free state earns its tighter equivalence
collapse by also being unable to separate chunks that genuinely compute
different functions — the worse failure for a representation whose purpose is to
be a lossless behavioral description of its chunk.

**The other two settings follow from what each consumer can see.** Read positions
are counted as ranks over registers read before written, which *is* the canonical
position; the alternative encoding (`2(k+1)+role`, indexed by step and operand
slot) coincides with it only when every read is fresh and no operand is x0, and
so mis-states the position on ~1.6% of single instructions and on any repeated
read. And the pool — a symmetric sum, structurally blind to which register was
written first — is additionally fed each slot's **write step**, because the
essence must encode an ordered function ("output 1 is this, output 2 is that").
That channel is register-indexed, so invariance stays exact, and a commutative
swap changes neither which slot was written nor when, so equivalence collapse is
not put at risk. Read-order must *not* reach the pool for exactly that reason
(see "Read-order must be consumed conditionally").

**None of the three is a configuration option.** They are the only values any
tier uses, and the code carries no branch for an alternative.

### The tier contract — never re-embed tokens

**T_{n+1} consumes only the output of T_n, never raw tokens.** Every tier
emits an `(essence, binding)` object: an essence vector (rename-invariant,
"what this unit computes") and binding references (which registers, in which
roles, as equivariant indices into the register axis).

A tier's emission is therefore a **pair, not a single vector** — there is no
concatenation step, and `encode_state` returns the two separately. Across the
tier boundary the essence crosses as a float vector and the binding as integer
slot indices (`argmax`/top-k of the score heads). The essence is the *thought*;
the binding is the *interface*. WHAT_IS_A_THOUGHT.md ("The Thought and Its
Interface") carries the conceptual consequences — thought arithmetic is defined
on the essence only, the hierarchy propagates a vector space plus a graph,
expression inverts a pair, and the binding is what makes *composing* two
thoughts determinate rather than ambiguous. Note the division that follows: a
unit's INTERNAL wiring is absorbed into its essence (a chunk whose second
instruction consumes the first's output has that in the vector), while the
binding carries only its EXTERNAL interface — what it reads from outside and
what escapes to outside.

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

**The lower tier's loss set IS the contract, and it is under-specified.** A tier
is frozen before the next one trains, so whatever its own losses shaped into its
emission is exactly what the next tier receives — `value_predict` defines the
essence's content, the binding losses define the interface, and the next tier
gets no vote. Two consequences follow, and both are load-bearing:

- The binding losses carry weight 0.1 against `value_predict`'s 1.0, yet the
  binding is what the next tier routes its wiring on. The interface is optimised
  at a tenth of the priority of the content it carries.
- Satisfying the contract does not determine downstream usability. Six T1s each
  scoring 100.0000% on their own single-instruction value prediction yielded T2s
  spanning 0.059 points of exact match — an order of magnitude beyond T2-level
  run-to-run spread — with their binding accuracies ranked ANTI-correlated to
  their T2s (EXPERIMENT_LOG, "T2 faithfulness", finding 25).

So the only real test of a tier's emission is training the tier above it. A
lower-tier scorecard verifies that tier is intact; it cannot rank candidates.

### GVN: the invariant essence is the value-number

**Global value numbering (GVN)** is the compiler analysis that assigns every
computation in a program a *value number*, so that two computations receive the
same number exactly when they are guaranteed to produce the same value. It is
the standard name for the property this encoder has to have, and "GVN collapse"
throughout these documents means the degree to which equivalent computations
land on one point.

The rename-invariant essence *is* a value-number — two computations producing
the same value get the same essence; different values stay apart.
Rename-equivalence collapses **structurally** (free, from the equivariant
core). Other behavioral equivalences — commutative operand swaps (`a+b==b+a`),
dead code, algebraic identities — are *not* structural, and the targets do not
impose them: a chunk with dead code, an identity or a cancellation is labelled
by what it reads and writes, not by the shorter chunk it is equivalent to.
Collapsing those is the essence's job, pursued as equivalence rather than as
correctness. Distinct behaviors must stay apart — the essence is behavioral, not a
syntactic hash.

**Reordering independent instructions is NOT an essence-level equivalence.**
The canonical frame assigns input positions by FIRST-READ order, so reordering
permutes them: `[ADD x5,x1,x2 ; SUB x6,x3,x4]` canonicalizes to "out_A =
in1+in2, out_B = in3−in4" while its swap canonicalizes to "out_A = in1−in2,
out_B = in3+in4". Verified against the oracle: `in_slot` `[1,2,3,4]` → `[3,4,1,2]`,
`out_slot` `[5,6]` → `[6,5]`, and the per-output-slot vp targets **differ**
(they agree only on raw execution, with register identity intact). So the
supervision actively separates them, correctly: as ordered identity-free I/O
functions they *are* different functions, and their equivalence is recovered
only at the `(essence, binding)` level — the essence says what is computed in
what order, the binding says which register each result lands in. See
WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface". A consequence worth
noting: the essence is **not** a canonical form for the dataflow DAG — the same
DAG under a different linearization gets a different essence.

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

**Evolved through extensive experimentation** (EXPERIMENT_LOG, from "Gate
training on a frozen encoder" through "Rename-equivariant T1/T2"). The settled approach is one **behavioral loss set shared by
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

**Value-prediction (`value_predict`).** Per-anchor, per-output-slot per-bit BCE
(plus a bounded smooth-magnitude term): from `(essence, canonical input-slot
values)` predict each output register's **canonical** value (the `out_regs`
baseline — inputs relocated to canonical first-read positions, so the target
depends only on operand structure, not register names), at anchor input states
drawn afresh for every source chunk (see the end of this section for why one
fixed set fails). Feeding the head canonical inputs to match the
canonical target is load-bearing: the earlier bug fed *raw* register values
against a *canonical* target, making the task mathematically inconsistent.

**vp is the primary objective.** The head is fed raw value *bits*, not
log-compressed scalars. What it measures is the essence composed with the
head's ability to EVALUATE it — a slot describes a function rather than holding
a value (above) — so a description the essence carries perfectly can still score
badly when applying it needs carry propagation. Driving vp to zero is therefore
a demand on the pair, and faithfulness, the property a future T3 actually
consumes, is the weaker half of it (METRICS.md states what it certifies and
how to measure the halves apart).

The head reads the **unit-normalized** essence (`F.normalize(vecs)`,
`t2_value_predict_loss`): magnitude is reserved for validity, so if vp's gradient
reached ‖essence‖ then "produces register output" would start proxying for
"is valid" — only register-writers get vp shaping. The operator identity vp needs
lives in the direction.

**Every output slot is predicted independently, in one pass, and that is
deliberate.** The head emits all `MAX_OUTPUT_SLOTS` slots' values at once from
one shared hidden layer, so a slot whose value depends on another slot's gets no
access to it — a chunk computing `r4 = x+1042; r7 = r4+4` must produce `x+1046`
for `r7` without reusing the `x+1042` it just produced for `r4`.

Chaining them is architecturally possible — the output slots have a defined SSA
write order and the dataflow is a DAG, so the iteration would be over the output
structure rather than over the essence, which has no sequence axis. It is
rejected on the TIER CONTRACT. T_(n+1) receives only what T_n emitted, so a
composite that exists solely as something an iterative head can reconstruct is
not available to the next tier, which then faces the identical problem one level
up. vp's justification is that it proxies what the next tier can recover, so a
chained head would raise the score while leaving the representation as
deficient for the consumer that matters. The composite has to be in the essence,
not in the readout's control flow.

The operator-clustering that the flat encoder could never get from vp comes
**for free from the equivariant core** (rename-invariant essence = the
value-number), and the canonical vp target is what collapses the
*non-structural* equivalences. GVN collapse and vp are both load-bearing — a vp
gain that craters the GVN rows is a regression, not a win.

**The anchor states are drawn afresh for every source chunk, with structure,
and they decide no labels.** With one fixed set the head is fed the same few
input values for every row, so it can store each instruction shape's answers at
those points instead of evaluating the described function — and it does: heads
exact at 8 fixed anchors score ~5% (T2) and 21% (T1) at anchors they never saw
(EXPERIMENT_LOG, "T2 faithfulness", finding 32). A fresh draw per source chunk —
its twins share it, since a rename must not change the targets — makes storing
impossible, so a correct prediction can only come from evaluating. What trains
the evaluator is how many distinct input values a step sees: one shared set per
batch needs 64 anchors to match 8 per row, at 2.7x the step time (finding 33).

Uniform 32-bit values are poor witnesses: a comparison against a 12-bit
immediate is constant over them, two registers almost never hold equal values,
and a sum almost never carries through all 32 bits. Each register is therefore
drawn from a mixture of zeros, small and mid-range values, boundary values,
powers of two and uniform values, and some registers copy a lower-numbered one
exactly, to within a few units, or negated — behavioral input k is read from
register k+1, so the copies are what make two inputs equal, adjacent or
opposite (`draw_anchor_states`).

The anchors decide no structural label. Which reads an instruction's outputs
depend on is decided once per instruction form at a fixed set of witness values
— every pair of 0, 1, -1 and one generic value, which agrees with several
thousand rich states on every ALU form. A chunk's inputs are those reads in
first-read order, counting only registers not yet written — the order T2's read
ranks number them in, given T1's wiring — including the reads of an instruction
whose result is overwritten: they are shown to the encoder, so leaving them out
would shift every later input off its rank. Every register the chunk writes is
an output, including one rewritten with the value it held. A label that
depended on the draw would flip from row to row (`SLTIU x4,x5,40` reads x5 only where some
anchor puts a value below 40 there), and the binding targets with it. The cost
is that a value cancelled across instructions keeps its input:
`ADDI x6,x5,1 ; SUB x7,x6,x5` makes x7 the constant 1 and still lists x5, and
recognising that is the essence's job. Likewise `ADD x5,x5,x0` and
`ADDI x5,x5,1 ; ADDI x5,x5,-1` read and write x5, and
`ADD x5,x6,x7 ; ADDI x5,x8,1` has inputs x6, x7 and x8: calling the first a
no-op, or dropping the dead instruction's inputs from the second, would make
the targets impose an equivalence (identity equals the empty chunk; dead code
equals its absence) that is the essence's to learn.

**Emergent equivalence.** With the behavioral targets alone, provably
equivalent instructions/chunks collapse without any equivalence loss:
rename-twins collapse *structurally* (the equivariant core), and a commutative
operand swap leaves the canonical vp target unchanged. Dead code and identities
are deliberately not given shared targets (above). Measured as GVN collapse split by pair type (EXPERIMENT_LOG,
"Rename-equivariant T1/T2") — note the measurement covers commutative operand
swaps only
(`_commute_variant`), so treat the other types as expected-but-unmeasured.
Instruction reordering is deliberately absent: it is not an essence-level
equivalence at all (see the GVN section above). A design constraint this surfaces: a
rename-*invariant* behavioral target (e.g. a GVN-bijection distance) is the
WRONG objective here — it would weld apart the binding distinctions the
equivariant representation deliberately keeps recoverable.

**Supervise only behaviorally-live signals.** A dead value has no target, so
giving it one asks for a gradient nothing can satisfy and fights the collapse
the essence is supposed to perform. The existing instance is the `rd==x0`
immediate mask: `ADDI x0,rs,k` is the same no-op for every `k`, so its immediate
is not supervised.

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
- Commutative operand swaps
- Strength reduction equivalences

Metric: cosine similarity between equivalence pair compressed points.
Target: > 0.99.

*Instruction reordering does NOT belong on this list* — it is equivalent only
with register identity attached, so the essence is expected (and supervised) to
separate the two orderings. Scoring it here would penalize correct behavior.

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
- Train on register renaming and commutative swaps only
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
