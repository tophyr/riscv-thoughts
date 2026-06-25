# RISC-V Thought Compression Testbed

An experimental ML research project investigating whether learned
compression can produce vector spaces with smooth, factored, interpolable
geometry — and whether search-based decoding can invert the compression.

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

**The headline is GVN behavioral collapse, not value-prediction.** Essence
distance split by pair type: rename-twins `0.008` (structural — free from the
equivariant architecture), **behavioral-non-rename (commutativity) `0.024`**
(TRAINED value-numbering — collapses ~28× vs distinct), distinct `0.659`. The
non-rename collapse is the real result: the register-state machine performs
global value numbering, and not merely the free rename-twin collapse. `vp` is
a **diagnostic probe** sitting at an intrinsic floor (regressing wrapped /
bitwise ops from log-compressed scalar operands is hard for any MLP) — do not
read its absolute value as the objective; GVN collapse is the property that
matters.

**Infrastructure.** The frozen-T1 encode + predicted-binding routing + T2
state machine + losses fold into ONE compiled CUDA-graph step at a
fixed-padded instruction count (verified bit-identical to the ragged path):
**T2 at 8–9 ms/step / 95–97% GPU util**. Streaming pipeline:
`mux_batches (N× gen_batches) | batch_slice | train_t2_encoder` — CPU
generates/analyzes chunks, the trainer only does forward/backward. RVT wire
format is **v8** (5-field header; the retired raw row-outputs path is gone).

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

# Train T1 (single-instruction encoder). gen_batches streams RVT batches;
# the trainer reads until stdin EOF (--n-steps only shapes the LR schedule).
python scripts/gen_batches.py --rule single --twins 2 --batch-size 128 \
    --n-batches 250000 --n-states 8 --anchor-seed 0 \
    --config configs/instr_default.json \
  | python scripts/train_encoder.py --d-model 128 --n-heads 4 --n-layers 2 \
    --d-out 64 --max-window 32 --n-steps 250000 --out-dir t1_run

# Train T2 (chunk encoder on a frozen T1). Several gen workers via
# mux_batches keep the compiled trainer saturated.
python scripts/mux_batches.py --gen-count 8 --mode fifo --seed 0 -- \
    --rule "branch+cap=4" --twins 3 --batch-size 128 --n-batches 2000000 \
    --n-states 8 --anchor-seed 0 --config configs/instr_t2.json \
  | python scripts/batch_slice.py --count 2000000 \
  | python scripts/train_t2_encoder.py --t1-model runs/t1_run/encoder.pt \
    --d-out 512 --d-event 16 --route binding \
    --n-steps 2000000 --anchor-seed 0 --n-anchor-states 8 --out-dir t2_run

# Train a decoder on a frozen encoder
python scripts/gen_batches.py --rule single --twins 0 --batch-size 128 \
    --n-batches 5000 --config configs/instr_default.json \
  | python scripts/train_decoder.py --t1-model runs/t1_run/encoder.pt \
    --d-model 128 --n-heads 4 --n-layers 2 --n-steps 5000 --out-dir dec_run

# Test
python -m pytest
```
