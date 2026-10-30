# RISC-V Thought Compression Testbed

An experimental ML research project investigating whether learned
compression can produce vector spaces with smooth, factored, interpolable
geometry — and whether search-based decoding can invert the compression.

## Reading order

1. This file — the map and the current status.
2. [WHAT_IS_A_THOUGHT.md](WHAT_IS_A_THOUGHT.md) — the theory the whole project
   is an attempt to test.
3. [RISCV_DESIGN.md](RISCV_DESIGN.md) — the testbed and the architecture, with
   the durable rationale for why it is shaped this way.
4. [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) — an index over one file per phase in
   `EXPERIMENT_LOG/`. Read the index, then in full the phase it marks as the
   live record: the register frame and its open problems. Earlier phases are
   superseded design kept for their dead ends — open one when you are about to
   revisit the question it settled.
5. [METRICS.md](METRICS.md) — what each measurement means, what it certifies,
   how far it moves when nothing changed, and the conditions a number from it is
   only true under. Read before quoting any figure.
6. [TODO.md](TODO.md) — the mission, the milestones, and the open tasks.

[DECODER_STRATEGY.md](DECODER_STRATEGY.md) and
[STREAMING_COMPRESSOR.md](STREAMING_COMPRESSOR.md) describe deferred, unbuilt
layers — read their status banners and skip the rest unless you are working on
them. [TRANSFERABLE_LESSONS.md](TRANSFERABLE_LESSONS.md) is background from the
predecessor project. [probes/README.md](probes/README.md) indexes the frozen-
checkpoint diagnostics. The data-generation pipeline and the commands for
running a training cell are documented in the `datagen` skill
(`.claude/skills/datagen/SKILL.md`); its silent-corruption invariants live in
`CLAUDE.md`. Read both before touching data generation.

## The requirement, and how far the code meets it

WHAT_IS_A_THOUGHT.md, RISCV_DESIGN.md and METRICS.md state what a tier's output
must be; TODO.md holds the distance still to cover. The requirement: two chunks
whose externally visible register inputs and outputs are identical compress to
the same essence — whatever their instruction order, instruction count, final
PC, or dead and same-value writes (PC is not an output) — and the binding is a
graph from external registers through operation roles to results.

The code grades the essence in the **register frame**: value prediction reads
the essence, recovering each written output's value from the essence, a
pointer naming which of the chunk's outputs is asked about, and the values of
the registers the output depends on, with no input or output numbered by the
order the instructions are written in or by register number; each input
register is identified by its structural colour in the chunk's dataflow
graph, with every operand role ordered — the encoder is told nothing about
which operations commute; and the essence is the pool over the written slots
only (`compressor/model.py`, `compressor/train.py`, `datagen/signature.py`). Reordering independent
instructions, inserting dead code and adding an x0-write therefore leave the
essence bit-identical by construction. What the code does not yet do: the
binding is still emitted as two ordered lists rather than the graph; the
colouring is an exact structural rule, built in as scaffolding and to be
replaced by a learned pass (RISCV_DESIGN.md, "What is built in for good, and
what is built in for now"); and the equivalences that need the op-cell to
compute — every commutative swap, identity spellings, cancellations, folds of
two literals — are not collapsed.

A frame numbered by textual first-read and final-write order preceded this
one, taken knowingly so that the first question — can the architecture
represent a chunk's output values exactly at all? — could be answered before
the equivalence criterion was imposed. EXPERIMENT_LOG.md records both.

## The Idea

LLMs are excellent summarizers — to a point. At 2000 words they
capture nuance; at 100, mostly the headlines; at 10, a title. The
detail loss is taken as inherent to compression. Yet the same models
demonstrate `king − man + woman ≈ queen`: a vector standing in for
a word both preserves its meaning and supports geometric operations.
What if whole-document compression worked that way — where
`Romeo and Juliet − Renaissance Verona + 1950s New York` pointed
you at *West Side Story*, and the result was a vector you could
decompress back into actual text?

A "thought" is exactly such a compressed representation — a point in
a continuous vector space produced by a deterministic compression
function over tokens. Two token sequences mean the same thing if and
only if they compress to the same point. Thinking becomes akin to
transformations of thought points: composition, differencing, and
recombination in the compressed space.

Thought expression is search: given a thought, find the token sequence
that compresses nearest to it. Expression verifiability falls out of
this for free, because it is simply compressing the expression and
differencing it against the original thought.

See [WHAT_IS_A_THOUGHT.md](WHAT_IS_A_THOUGHT.md) for the full theoretical
framework.

## Why RISC-V

Natural language has fuzzy semantics — you can't prove two sentences mean
the same thing. English's *"I didn't say you should kill him"* can wildly 
change meanings by simply emphasizing each different word. RISC-V instruction 
sequences, by comparison, have exactly-computable semantics: two sequences are 
equivalent if and only if they produce identical machine states. Semantic 
distance is measurable via state comparison. This gives us ground-truth labels 
that natural language cannot.

RISC-V is the measurement instrument, not the application. Success here
provides evidence and methodology — but not proof — that the framework
transfers to natural language.

See [RISCV_DESIGN.md](RISCV_DESIGN.md) for the testbed design.

## Current Status

**The work in progress is T2, and its purpose is to prove the recursive tiering
architecture.** Whether a hierarchy of thought-compressors is possible at all
turns on one question: is a tier's emitted representation sufficient for the
next tier to build on? T1 (one instruction) is a foundation, not evidence. T2
(a chunk of instructions, built only from T1's output) is the first tier that
can answer it.

**The architecture under test.** A tier is a register-indexed state machine
(one `Compressor` class, `compressor/model.py`): 32 register slots start empty,
then per instruction it reads its operand slots, applies a shared op-cell, and
writes the destination slot, emitting the final register-indexed state. Reads
and writes are pure one-hot operations and no parameter is indexed by register,
so renaming registers **permutes the slots exactly** — equivariance holds by
construction, not by training. A slot acquires its identity on first read, from
the register's structural colour in the chunk's dataflow graph, which is what
distinguishes one input from another without naming either and without regard
to the order the instructions are written in; because that colour is a
property of the graph rather than of a random draw, a tier's output is a
deterministic function of its input, which is what the equivalence criterion
below requires. The essence is a permutation-invariant pool over the written
slots of that state; the per-slot remainder is the binding. T1 is literally T2 with
`seq_len=1`; the only tier-specific piece is what supplies content — T1 embeds
one instruction's tokens, T2 is fed a frozen T1's per-instruction essence.
See RISCV_DESIGN.md for why.

**What has been shown.** Equivariance survives training — measured on every
trained model, a rename permutes the emitted state to within zero error, so the
construction above is not degraded by the losses acting on it. And T2 derives
each instruction's wiring from the frozen T1's own binding predictions rather
than from decoded tokens, so the tier contract is actually exercised rather than
assumed: nothing reaches T2 except what T1 emitted.

**What is open.** Two properties, and the framework needs both.

*Faithfulness.* Whether what T2 emits is a lossless behavioral description of
its chunk, measured by whether each output register's value can be recovered
from the essence and scored discretely rather than as a loss value. The score
is the essence composed with a readout that has to EVALUATE it — nothing in
the encoder ever sees a register value, so the essence describes functions
rather than holding numbers. The input states are drawn
afresh for every chunk, with structure, and the score is taken at a fixed set
no model trains on: a readout trained on one fixed set stores answers at it
instead of evaluating. **At one instruction it holds exactly**: T1's emitted
wiring matches the oracle on every instruction and its essence is evaluated
exactly at 2.46 million unseen input values. **At two instructions
it holds to 99.8276%** — 8,333 failures in 4.83 million scored values, every
output register of 38,400 held-out chunks at 64 input states — after 1M
training steps, one seed, a twelve-layer residual readout; outputs that depend
on three inputs carry the residual at 90.5%. Beyond two instructions it is
untested.

*Equivalence.* Two instruction sequences that compute the same function must
compress to the same point — that is the framework's definition of meaning, and
it is required for T2 to count as done. Everything that leaves the chunk's live
dataflow graph unchanged does, exactly and by construction: a register rename,
reordered independent instructions, an inserted dead instruction. Everything
that needs the op-cell to compute does not emerge by itself — a commutative
swap among them — since value prediction alone is satisfied by any essence it
can read. The pressure is a pairwise behavioral distance: every two chunks'
essences are held as far apart as their outputs differ at shared inputs, so
equivalent spellings meet and near-misses stay near. At one instruction it
collapses every spelling group tried — copies, zero, doubling, the sign test,
constants — to within 0.11 of a point while unrelated instructions keep their
spread, at a cost of 108 wrong values of 2.46 million, on comparisons whose
two immediates differ by one. At two instructions it is not built yet: the T2
models measured describe their chunks as written, two chunks that fold two
immediates into the same total landing further apart than two that compute
different totals.

Beyond both lies T3, the first tier that must *compose* bindings rather than
merely emit them.

**Infrastructure.** The frozen-T1 encode, predicted-binding routing, T2 state
machine and losses fold into one compiled CUDA-graph step at a fixed-padded
instruction count (verified bit-identical to the ragged path). Training is a
streaming pipeline — `datagen | train_t2_encoder` — where CPU workers generate
and analyze chunks and the trainer only does forward/backward; on the GPU box
those workers are offloaded to a second machine so they do not contend with the
trainer. There is one RVT wire format and no version field: each batch is
self-describing (magic + dtype signature + 5-field header), so a schema change
is immediate and global. The `datagen` skill covers the pipeline, its flags, and
how to judge whether the GPU is actually saturated.

**GPU batch emulator.** All RV32I opcodes executable in parallel via
`torch.where`, ~1.7ms for B=4096.

See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for detailed results and
[TODO.md](TODO.md) for the roadmap.

## Ancestry

This project builds on [HATA](https://github.com/tophyr/hata), an earlier
attempt at thought compression using natural language (Llama 3.2 1B). HATA
produced 16 experiments' worth of lessons about training dynamics,
architectural constraints, and evaluation pitfalls — all of which informed
the RISC-V approach.

See [TRANSFERABLE_LESSONS.md](TRANSFERABLE_LESSONS.md) for what carried
over.

## Structure

```
emulator/       RV32I emulator + GPU batch emulator (all opcodes via
                torch.where)
tokenizer/      Structural tokenizer
datagen/        RVT batch format + chunk analysis: SSA, anchor
                execution, the structural signature (signature.py),
                behavioral binding targets, equivalence manifest.
compressor/     Equivariant Compressor (T1/T2), decoder, shared training
                loop (train.py), held-out eval (eval.py)
scripts/        Training and evaluation entry points
probes/         Frozen-checkpoint diagnostics (see probes/README.md)
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

Training is a streaming pipeline — `datagen` generates and analyzes chunks on
CPU workers and writes RVT batches to stdout; a trainer reads them on stdin:

```bash
python -m datagen <corpus flags> | python scripts/train_t2_encoder.py <model flags>
```

**The canonical commands for T1 and T2, with the flags that are actually
current, are in the `datagen` skill** (`.claude/skills/datagen/SKILL.md`).
They are not repeated here: two copies drift, and the ones that matter carry
invariants (the vp head must read the anchor states shipped in each batch, or
the targets are silently wrong — `CLAUDE.md`). `scripts/run_t2.sh`
runs one scored training cell end to end and is the normal entry point.

There is no decoder in the tree. Decode-back is a non-goal of the current work;
[DECODER_STRATEGY.md](DECODER_STRATEGY.md) carries the design and points at the
tag holding one discarded attempt.


