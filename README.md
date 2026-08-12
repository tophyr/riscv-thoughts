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
   `EXPERIMENT_LOG/`. Read the index, then **Phases 14 and 15 in full**; earlier
   phases are superseded design kept for their dead ends — open one when you are
   about to revisit the question it settled.
5. [TODO.md](TODO.md) — the live mission, the open tasks, and the dead ends.

[DECODER_STRATEGY.md](DECODER_STRATEGY.md) and
[STREAMING_COMPRESSOR.md](STREAMING_COMPRESSOR.md) describe deferred, unbuilt
layers — read their status banners and skip the rest unless you are working on
them. [TRANSFERABLE_LESSONS.md](TRANSFERABLE_LESSONS.md) is background from the
predecessor project. The data-generation pipeline is documented in the `datagen`
skill (`.claude/skills/datagen/SKILL.md`); its silent-corruption invariants live
in `CLAUDE.md`. Read both before touching data generation.

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
(one `Compressor` class, `compressor/model.py`): it seeds 32 register slots,
then per instruction reads its operand slots, applies a shared op-cell, and
writes the destination slot, emitting the final register-indexed state. Reads
and writes are pure one-hot operations and no parameter is indexed by register,
so renaming registers **permutes the slots exactly** — equivariance holds by
construction, not by training. The essence is a permutation-invariant pool over
that state; the per-slot remainder is the binding. T1 is literally T2 with
`seq_len=1`; the only tier-specific piece is what supplies content — T1 embeds
one instruction's tokens, T2 is fed a frozen T1's per-instruction essence.
See RISCV_DESIGN.md for why.

**What has been shown.** Equivariance is exact on trained models. Both tiers
train on the same behavioral losses with **no equivalence loss** — chunks that
compute the same function collapse together from behavioral supervision alone,
which is the property a value-numbering representation has to have. And T2
derives each instruction's wiring from the frozen T1's own binding predictions
rather than from decoded tokens, so the tier contract is actually exercised
rather than assumed: nothing reaches T2 except what T1 emitted.

**What is open.** T2's essence is not yet a *lossless* behavioral description of
its chunk. That is the current mission (TODO.md), measured by whether a chunk's
output register values can be recovered from the essence, and scored discretely
rather than as a loss value. It is lossless at one instruction and close at two;
four and eight are open, as is everything outside the R/I-type ALU subset the
work currently runs on. Beyond that lies T3, the first tier that must *compose*
bindings rather than merely emit them.

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
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train T1 (single-instruction encoder). datagen generates on all CPU
# cores; --count bounds the run; the trainer reads until stdin EOF
# (--n-steps only shapes the LR schedule).
python -m datagen --rule single --twins 2 --batch-size 128 \
    --n-states 8 --anchor-seed 0 --config configs/instr_default.json \
    --count 250000 \
  | python scripts/train_encoder.py --d-model 128 --n-heads 4 --n-layers 2 \
    --d-out 64 --max-window 32 --n-steps 250000 --out-dir t1_run

# Train T2 (chunk encoder on a frozen T1); --count bounds the run.
# NOTE: on the GPU box, local datagen workers contend with the trainer's
# per-step H2D prep and starve the GPU. For a real run, offload generation to
# another machine with multinode_gen.sh (same flags via env vars). See the
# `datagen` skill for what limits throughput once that is done.
#   RULE="branch+cap=4" CONFIG=configs/instr_t2.json ./multinode_gen.sh 2000000 \
#     | python scripts/train_t2_encoder.py --t1-model runs/t1_run/encoder.pt ...
python -m datagen --rule "branch+cap=4" --twins 3 --batch-size 128 \
    --n-states 8 --anchor-seed 0 --config configs/instr_t2.json \
    --count 2000000 \
  | python scripts/train_t2_encoder.py --t1-model runs/t1_run/encoder.pt \
    --d-out 512 --d-event 16 --route binding \
    --n-steps 2000000 --anchor-seed 0 --n-anchor-states 8 --out-dir t2_run

# Train a decoder on a frozen encoder — CURRENTLY BROKEN, do not use.
# train_decoder predates the equivariant refactor: compressor/train.py:1632
# calls a 2-arg `encoder.encode(tok, pad)`, but the current encode() also
# needs the wiring (in0, in1, out) and tags, so it raises TypeError at step 1.
# Decode-back is a non-goal of the current phase; see DECODER_STRATEGY.md.
#   python -m datagen --rule single --twins 0 --batch-size 128 \
#       --config configs/instr_default.json --count 5000 \
#     | python scripts/train_decoder.py --t1-model runs/t1_run/encoder.pt \
#       --d-model 128 --n-heads 4 --n-layers 2 --n-steps 5000 --out-dir dec_run

# Test
python -m pytest
```
