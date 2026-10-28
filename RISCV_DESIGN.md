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

**The two reordering transformations are essence-level equivalences that the
current supervision does not honour.** Reordering independent instructions
(and non-overlapping loads/stores) leaves the function from input registers to
output registers unchanged, so the essence must not move. The value-prediction
target is built in a frame that numbers inputs by first read and outputs by
final write (see "Training"), and reordering permutes both, so the two chunks
receive **different** vp targets and the supervision separates them. Which
register fills which role is the chunk's interface and belongs in the binding
(WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface"); the frame puts the
textual order of two independent instructions into the essence's coordinate
system instead (see "GVN: the invariant essence is the value-number").

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
- Seed each input register with its **structural colour** in the chunk's
  dataflow graph (below) — distinct, anonymous, deterministic, and free of the
  order the instructions are written in.
- For each instruction in order: **read** the operand slots' states, apply a
  shared **op-cell** to produce the result essence, **write** it to the
  destination slot (functional, SSA-style — one-hot read/write, no in-place
  scatter, so the step captures cleanly into a compiled CUDA graph).
- Chunk representation = the final state `S`. Essence = a
  permutation-invariant **attention pool** over the WRITTEN slots — the
  chunk's multiset of output descriptions, nothing about where its inputs
  sit or in what order anything happened; binding = the per-slot structure,
  read back by the binding heads.

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
register-indexed input is each leaf's colour, itself a function of the wiring
alone, and reads/writes are pure one-hot tensor ops. Anything added to the
pool's input must likewise be register-indexed, or the invariance stops being
exact.

Two channel groups per slot, both equivariant: **value channels** (set on
write, preserved on read — the dataflow state the op-cell threads, which makes
multi-hop value-numbering correct) and **event channels** (accumulate
read/write events with order timestamps, so `live_in`/`in_slot`/`out_slot`
are recoverable from the emitted object; a pure value-state records only
writes). The read pulse is content-dependent, so a behaviorally-irrelevant
read (`AND rs1,x0`) is distinguishable from a relevant one.

**Operand order is consumed by the op-cell and nowhere else.** The op-cell sees
the opcode (as content) and both operands in one activation, so it can use the
roles for `SUB`/`SLT`/`SRL` and symmetrize them for `ADD`/`AND`/`OR`/`XOR` —
every commutativity is left to it, and value prediction alone leaves it
incompletely learned (finding 36). The pool is a symmetric sum that never
sees `content`, so any channel fed to it is used unconditionally; no order of
any kind reaches it.

**Leaf identity for input registers (subtle but essential).** The state machine
has to seed each register slot before the first instruction reads it, and that
seed faces four requirements. Inputs must be **distinguishable** where the
register function distinguishes them, so that `add r2,r3` does not collapse
onto `add r2,r2`. They must carry **no intrinsic identity**, because a
per-register learned embedding `f(r)` would have to be constant to be
equivariant, which destroys distinctness. The seed must be **deterministic**:
the random-node-feature trick from graph learning — a tag sampled fresh per
chunk — satisfies the first two but is redrawn on every forward pass, so the
same instruction compresses to a slightly different point each time it is seen
(max |Δ| 1.8e-2, minimum cosine 0.9992 over four draws of one trained encoder),
which contradicts the framework's foundational claim that a compressor is a
deterministic function (WHAT_IS_A_THOUGHT.md) and leaves invariance to the seed
a *learned* property a change can silently regress. And the seed must be **free
of the linearization**: a seed taken from textual first-read order makes two
chunks that compute one function in a different instruction order seed their
leaves differently and part (see "GVN").

A leaf's **structural colour** meets all four: a role-aware Weisfeiler–Lehman
colouring of the chunk's dataflow graph, computed from the wiring the lower
tier emits — dead instructions pruned by a backward pass from the final
writers, every operation's two read roles ordered — and realised in the
encoder as a fixed number of rounds of message passing with fixed random
matrices (`RegisterStateMachine._prepass`), the colouring `datagen.signature`
computes with strings. Two leaves of one colour are symmetric: exchanging
them cannot change any output. A leaf read in both roles of one operation
carries a different colour from one read once, so `x+x` and `x+y` stay apart;
a leaf shared by two instructions differs from one read by either alone; the
two leaves of `ADD x5,x1,x2` carry different colours too, since the encoder is
not told that `ADD` commutes, and collapsing the swap is the op-cell's to
learn. The colour is placed through the same one-hots as every other
register-indexed quantity, so a rename permutes it with the slots and
equivariance is exact; it is a function of the graph alone, so a reordering
of independent instructions or a dead instruction's insertion leaves every
colour bit-identical, and the essences of such pairs are bit-identical on an
untrained model (`probes/frame_scorecard.py`). The colouring is scaffolding —
see "What is built in for good, and what is built in for now".

**None of this is a configuration option.** The colour seed and the pool over
written slots are the only values any tier uses, and the code carries no
branch for an alternative.

### The tier contract — never re-embed tokens

**T_{n+1} consumes only the output of T_n, never raw tokens.** Every tier
emits an `(essence, binding)` object: an essence vector (rename-invariant,
"what this unit computes") and binding references (which registers, in which
roles, as equivariant indices into the register axis). The binding is a graph —
which external register feeds which operation role and which receives which
result; the ordered input and output lists the heads emit today are that graph
flattened along the first-read frame's linearization (see "GVN").

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
What collapses **structurally** (free, from the core): a register rename (the
equivariant state), and everything that leaves the dataflow graph unchanged —
reordering independent instructions, inserting a dead instruction, an
x0-write beside live instructions — because the leaf colours and the written
descriptions are functions of the graph alone and the pool reads written
slots only. The essence IS a canonical form of the live dataflow graph, with
every operand role ordered. What is *not* structural and the targets do not
impose: every commutative swap (the op-cell receives its operands in the
other order and is told nothing about which operations commute), and every
algebraic equivalence — identities, cancellations, a fold of two literals
into one. A chunk carrying those is labelled by what it reads and
writes, not by the shorter chunk it is equivalent to; collapsing them is the
essence's job, pursued as equivalence rather than as correctness. Distinct
behaviors must stay apart — the essence is behavioral, not a syntactic hash.

**The frame the essence is graded in is the register frame.** Which register
feeds which role and which result lands where is the chunk's interface and
belongs in the binding (WHAT_IS_A_THOUGHT.md, "The Thought and Its
Interface"), so no input or output is numbered by textual order anywhere in
the supervision: value prediction is evaluated per written slot, handing the
evaluator that slot's description and its inputs' values and colours gathered
by register (see "Training"), and the targets are the raw register values. A
frame numbered by first-read and final-write order was used before this one,
knowingly, to establish the architecture's capacity to represent vp at all
before the equivalence criterion was imposed on it; it graded the essence on a
linearization and separated reordered, renumbered, dead-code-bearing and
same-value-writing equivalents (EXPERIMENT_LOG, "T2 faithfulness", finding
36); the code carries none of it.

### What is built in for good, and what is built in for now

The principle is: build in what cannot be learned, learn the rest. Two things
are built in for good. The **register symmetry** — a rename is a non-abelian
permutation and no flat vector can be trained to respect it (above), so the
state is register-indexed and reads and writes are one-hot operations. And the
**dataflow graph itself** — which instruction reads which slot in which role —
because it is the lower tier's emitted binding, and consuming it as given is
the tier contract.

Two things are built in for now, as exact structural rules: the **liveness
pass** that prunes dead instructions before the leaves are coloured, and the
**colouring** that gives each leaf its identity. Neither knows anything about
any operation: every operation's two read roles are ordered, and the encoder
is told nothing about which operations commute — a commutative swap is the
op-cell's to learn, and only a pressure toward fewer distinctions (the
collapse objective) can teach it; value prediction alone does not, since it
is satisfied by any description the evaluator can read. Both rules have a
learned form, and together it is one mechanism: a short message pass over the
same dataflow graph, from the final writers back to the leaves, each message
built from the reader's content and its role. Liveness is then the direction
of the pass — a dead instruction is never reached, so it contributes to no
leaf. The colour is the leaf's learned embedding; Weisfeiler–Lehman colouring
is what such a pass computes, with the aggregation learned instead of hashed.
The state machine is the bottom-up half of the same pass; the learned
pre-pass is its top-down half, and with it nothing in the encoder depends on
the order the instructions were written in.

The exact rules go first so that a failure of the frame cannot be confused
with a pass that has not learned liveness or symmetry yet. They are a known
simplification taken on purpose — the same tactic as establishing value
prediction before equivalence collapse — and their removal is owed: once the
collapse scorecard holds with them, the learned pass replaces them, is checked
against them, and they are deleted.

### Open problems (additive over the equivariance skeleton)

These are real work, each additive over a stable "register-indexed state; the
32 GP registers are the permutable slots" skeleton:

- **PC / branches.** PC is not an independent output of a chunk: two chunks
  with identical externally visible register inputs and outputs are equivalent
  whatever their instruction count or final PC, so straight-line advance never
  enters the essence. What a taken branch contributes is open. `pc_writes` is
  read from the invariant essence, with `n_regs = 32` (GP only); if a PC effect
  is ever represented, the clean extension is a non-renameable PC slot
  (renaming permutes only the 32 GP slots).
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

**Value-prediction (`value_predict`), per written slot.** Per-anchor, per-output-
slot per-bit BCE (plus a bounded smooth-magnitude term): for each register the
chunk writes, from that slot's **description** — its value channels in the
emitted state — and, for each external input the output depends on, the
input's value at its own register (as 32 bits + a smooth scalar) and its
colour, predict the register's raw value after the chunk (`vp_slot_loss`).
The inputs are listed per output in colour order by the oracle
(`datagen.signature.output_inputs`, shipped as `out_inputs`); the evaluator
gathers values and colours by register through one-hots, so nothing it sees
is a register number or a position in the text, and a renamed twin presents
the same description with the same values. Anchor states are drawn afresh for
every source chunk (see the end of this section for why one fixed set fails).

**vp is the primary objective.** The evaluator is fed raw value *bits*, not
log-compressed scalars. What it measures is a slot's description composed with
the evaluator's ability to EVALUATE it — a slot describes a function rather
than holding a value (above) — so a description carried perfectly can still
score badly when applying it needs carry propagation. Driving vp to zero is
therefore a demand on the pair, and faithfulness, the property a future T3
actually consumes, is the weaker half of it (METRICS.md states what it
certifies and how to measure the halves apart). The essence receives no vp
gradient directly: it is the pool over the same descriptions vp evaluates, so
it collapses exactly when those descriptions are canonical, and its magnitude
is shaped only by the validity loss.

**Every output slot is predicted from its own description, and that is
deliberate.** A slot whose value depends on another slot's gets no access to the
other's prediction — a chunk computing `r4 = x+1042; r7 = r4+4` must produce
`x+1046` for `r7` from `r7`'s own description, which the op-cell built from
`r4`'s.

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
value-number), and the per-slot target leaves every description free to be
canonical. GVN collapse and vp are both load-bearing — a vp gain that craters
the GVN rows is a regression, not a win.

**The anchor states are drawn afresh for every source chunk, with structure,
placed on the chunk's own input registers, and they decide no labels.** With
one fixed set the evaluator is fed the same few input values for every row, so
it can store each instruction shape's answers at those points instead of
evaluating the described function — and it does: heads exact at 8 fixed
anchors score ~5% (T2) and 21% (T1) at anchors they never saw (EXPERIMENT_LOG,
"T2 faithfulness", finding 32). A fresh draw per source chunk makes storing
impossible, so a correct prediction can only come from evaluating. What trains
the evaluator is how many distinct input values a step sees: one shared set per
batch needs 64 anchors to match 8 per row, at 2.7x the step time (finding 33).

Uniform 32-bit values are poor witnesses: a comparison against a 12-bit
immediate is constant over them, two registers almost never hold equal values,
and a sum almost never carries through all 32 bits. Each register of a draw is
therefore taken from a mixture of zeros, small and mid-range values, boundary
values, powers of two and uniform values, and some positions copy a
lower-numbered one exactly, to within a few units, or negated
(`draw_anchor_states`). The draw's position i+1 is then placed on the chunk's
i-th input register (`precompute_chunk`), so the copies are what make two of a
chunk's inputs equal, adjacent or opposite; the chunk executes on that
register file as it is, and a renamed twin gets the same draw placed on its
own registers.

The anchors decide no structural label. Which reads an instruction's outputs
depend on is decided once per instruction form at a fixed set of witness values
— every pair of 0, 1, -1 and one generic value, which agrees with several
thousand rich states on every ALU form. A label that depended on the draw would
flip from row to row (`SLTIU x4,x5,40` reads x5 only where some anchor puts a
value below 40 there), and the binding targets with it. The binding counts
every such read, a dead instruction's included, and every register the chunk
writes, including one rewritten with the value it held: `ADD x5,x5,x0` and
`ADDI x5,x5,1 ; ADDI x5,x5,-1` read and write x5, and `ADD x5,x6,x7 ; ADDI
x5,x8,1` has inputs x6, x7 and x8. The binding is the interface and says what
was read; the essence is what was computed, and there the dead instruction
leaves no trace (its write is overwritten, its reads are pruned from the
colouring), so dead code collapses by construction. What stays the essence's
to learn is every equivalence that needs the op-cell to compute: a value
cancelled across instructions keeps its input (`ADDI x6,x5,1 ; SUB x7,x6,x5`
makes x7 the constant 1 and still describes a subtraction), and a same-value
write is a description of the identity that only the op-cell can make equal to
nothing.

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

*Instruction reordering belongs on this list* — two orderings of independent
instructions compute one function of their registers. The first-read canonical
frame currently supervises the essence to separate them (see "GVN"), so the row
measures a defect of the frame until the frame changes.

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
