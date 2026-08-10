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

**Rename-equivariant T1/T2 — one Compressor, validated as a real tier
recursion.** The encoder is a register-indexed state machine (one
`Compressor` class, `compressor/model.py`): it seeds 32 register slots with
per-chunk anonymous tags, then per instruction reads operand slots → applies
a shared op-cell → writes the destination slot, and emits the final
register-indexed state. Because the only register-indexed input is the
anonymous tag and reads/writes are pure one-hot ops, a register rename
**permutes the slots exactly** — equivariance is by construction, not trained.
Essence (operator axis) is a permutation-invariant pool over the register
axis; binding is the per-slot remainder. T1 is literally T2 with `seq_len=1`:
the only tier-specific piece is the content provider — T1 embeds an
instruction's opcode+immediate tokens (registers stripped to wiring); T2 is
fed a **frozen T1's per-instruction essence** (the tier contract — never a
re-embedded opcode). See RISCV_DESIGN.md for the theory.

**Shared loss set; equivalence is emergent.** Both tiers train on the same
behavioral losses (`compressor.train.binding_losses`): magnitude-as-validity
(soft `(‖v‖−is_valid)²`), `live_in`/`live_out`, `pc_writes`,
`in_slot`/`out_slot` (read/write ORDER via ListMLE score heads), and a
canonical `value_predict` (predict canonical output-register values from
canonical input values on sampled anchor states). There is **no equivalence
loss** — equivalent chunks collapse from the behavioral targets alone.

**T1 — validated** (`runs/t1_equiv_d128_canonvp_250k`, d128/n4/l2/d_out64,
250k, `--rule single`): equivariance `0.00e+00` (exact, on the trained
model), tag-invariance `1.0000`; binding live_in 99.9% / live_out 100% / pc
100% / in_slot 100% / out_slot 100% / dup 0. Magnitude carries validity,
direction carries semantics (open space — train *toward* the unit surface
with the soft loss, never `F.normalize` onto it).

**T2 — tier recursion validated** (`runs/t2_equiv_d512_cap4_2M_principled`,
d_out=512, `branch+cap=4`, 2M, routing off T1's *predicted* binding). T2
derives each instruction's wiring from the frozen T1's own binding
predictions (`argmax`/top-k of its in/out-slot scores), **not** decoded
tokens — so the run actually tests the claim the whole hierarchy rests on:
*is T1's emitted representation sufficient for the next tier?* Held-out:
live_in 99.7% / live_out 100% / pc 98.4% / in_slot 91.5% / out_slot 99.9% /
dup 0 (in_slot's 91.5% vs 93.8% on ground-truth token wiring is the honest
cost of routing off *predicted*, imperfect binding). Equivariance composes
exactly over a chunk (`0.00e+00`).

**Equivalence collapse (GVN) is the validated headline.** Essence distance
split by pair type: rename-twins `0.008` (structural — free from the
equivariant architecture), **behavioral-non-rename (commutativity) `0.024`**
(TRAINED value-numbering — collapses ~28× vs distinct), distinct `0.659`. The
non-rename collapse is the real result: the register-state machine performs
global value numbering, not merely the free rename-twin collapse.

**Phase 15 (open) — ORDER, then operand BINDING.** The essence is a sum over the
32 register slots; that sum is what makes it rename-invariant, but it is also
order-blind, while the essence must encode *"output 1 is this, output 2 is
that"*. Feeding the pool one scalar per slot — the write step (`--pool-extra
write_time`) — took value-prediction loss **0.3525 → 0.0396** at 2-instruction
chunks (**0.0198** with 4× training) and decode-back **72.9% → 91.5%** per
instruction at 8, while *improving* equivalence collapse.

What remained was operand binding when two instructions share a source register
— 21.2% exact 32-bit match on that subset. No channel fed to the *pool* can fix
it: the pool never sees `content`, so it must use read-order unconditionally,
and a commutative swap changes read-order. The **op-cell** can, since it sees the
opcode and both operands in one activation and can symmetrize per opcode. Giving
it each operand's canonical read position (`--op-cell-rank`) takes that subset to
**100.0% exact** and aggregate vp to 0.0184 at 250k, beating the baseline's 1M.
It is not adopted yet: it regresses tag-invariance 25× (0.0010 → 0.0255), which
drags commutative collapse with it. Equivariance is untouched. See EXPERIMENT_LOG
Phase 15.

`vp` is a real objective with no intrinsic floor: single-instruction vp is
exactly `0.000`, and the head is fed raw value bits, not log-compressed scalars.
Driving it to zero is the current mission (TODO.md), scored discretely as a 100%
exact 32-bit match rate rather than as a loss value. Decode-back is a non-goal of
this phase.

**Infrastructure.** The frozen-T1 encode + predicted-binding routing + T2
state machine + losses fold into ONE compiled CUDA-graph step at a
fixed-padded instruction count (verified bit-identical to the ragged path):
**T2 at 8–9 ms/step @ ~90%+ util**. Streaming pipeline:
`datagen (all cores, --count) | train_t2_encoder` — CPU generates/analyzes
chunks, the trainer only does forward/backward. Judge saturation by **GPU
utilisation**, using ms/step against the same config's known-good figure as a
floor check — ms/step is architecture-dependent and says nothing about how long
the GPU spent waiting. (Ignore transient/periodic dips in the util reading;
those are sampling artifacts. Sustained low util is the thing to fix, and to fix
*before* letting a long run ride.) There is one RVT
wire format and no version field — each batch is self-describing (magic +
dtype signature + 5-field header), so a schema change is immediate and global.

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
# per-step H2D prep and starve the GPU (~54% util). For a real run, offload
# generation to odin with multinode_gen.sh (same flags via env vars) — it
# restores the ~90% / 9ms-step ideal. See the `datagen` skill.
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
