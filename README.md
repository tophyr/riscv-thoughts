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
   `EXPERIMENT_LOG/`. Read the index, then the two entries it marks as the live
   record in full: the current architecture, and the open faithfulness problem.
   Earlier entries are superseded design kept for their dead ends — open one
   when you are about to revisit the question it settled.
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
the register's position in the chunk's read order, which is what distinguishes
one input from another without naming either; because that position is a
property of the instruction rather than of a random draw, a tier's output is a
deterministic function of its input, which is what the equivalence criterion
below requires. The essence is a permutation-invariant pool over
that state; the per-slot remainder is the binding. T1 is literally T2 with
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

*Faithfulness.* T2's essence is not yet a lossless behavioral description of its
chunk, measured by whether the chunk's output register values can be recovered
from the essence and scored discretely rather than as a loss value. It is
lossless at one instruction and within a fraction of a percent at two; four and
eight are open, as is everything outside the R/I-type ALU subset the work
currently runs on.

*Equivalence.* Two instruction sequences that compute the same function must
compress to the same point — that is the framework's definition of meaning, and
it is required for T2 to count as done. Sequences related by a register rename
do, exactly. Sequences related by a *computed* equivalence do not: two chunks
that fold two immediates into the same total land as far apart as two chunks
computing different functions. The training objective is not pairwise, so
nothing pushes equivalent chunks together; supplying that pressure is an
outstanding design decision.

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
                execution, canonical out_regs oracle, behavioral binding
                targets, equivalence manifest.
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
invariants (the anchor seed and state count must match between generator and
trainer, or the targets are silently wrong — `CLAUDE.md`). `scripts/run_t2.sh`
runs one scored training cell end to end and is the normal entry point.

There is no decoder in the tree. Decode-back is a non-goal of the current work;
[DECODER_STRATEGY.md](DECODER_STRATEGY.md) carries the design and points at the
tag holding one discarded attempt.


