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

**Unified loss set — T1 is the single-instruction special case of T2.**
Both train on one shared behavioral loss set (`compressor.train.binding_losses`):
magnitude-as-validity (soft `(‖v‖−is_valid)²`), `live_in`/`live_out` (regs
read/written), `pc_writes`, `in_slot`/`out_slot` (read/write ORDER via
ListMLE score heads), and `value_predict` (predict output register values
from inputs on sampled anchor states). **Equivalence is emergent** — there
is no equivalence loss; equivalent instructions/chunks collapse from the
behavioral targets alone. Earlier per-register pair-MSE, syntactic
dest/src CE, explicit equivalence-collapse, and `rename_equiv` are
removed; the `behavioral_distance` (GVN/Hungarian) machinery is a dormant
offline oracle.

**T1 encoder/decoder — validated.** T1 in the unit ball (magnitude =
validity, direction = semantics): ~99.8% magnitude-threshold validity
accuracy; equivalence classes collapse without an equivalence loss.
CE-trained decoder on a frozen encoder reaches 95%+ training tok_acc;
generalization ceiling ~72% held-out, set by encoder geometry, with ~23%
of held-out decodings execution-equivalent to the original.

**T2 compressor — converged, single open-space d=512 vector.** A merged
512-d vector (no shape/binding split) learns the binding losses cleanly
(cap=4: in_slot/out_slot ~99%, dup 0). A 2M-step convergence run
(`runs/t2_d512_cap4_2M`, vp 2.15→0.18) established the **central open
finding**: value-prediction does **not** induce an operator/effect axis.
Holding register binding fixed, flipping the computed function barely
moves the vector (cosine gap ~0.01) while changing the binding moves it
~0.18 — binding dominates the geometry ~16×, and more vp training shrank
the gap rather than growing it. The vector also uses only ~13 of 512
effective dimensions, so **capacity is not the bottleneck — the objective
is.** The indicated fix is an oracle `out_regs` pairwise loss
(rename-sensitive behavioral distance) that forces behavioral separation.
See EXPERIMENT_LOG.md Phase 12.

**Infrastructure.** Full-step `torch.compile` (CUDA graphs) + pinned H2D
saturate the GPU at ~95% for T1, T2, and the decoder; the compiled
train step is shared via `run_train_loop`. Streaming pipeline:
`mux_batches (N× gen_batches) | batch_slice | train_t2_encoder` — CPU
generates/analyzes chunks, the trainer only does forward/backward.

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
                execution, behavioral binding targets, equivalence
                manifest. behavioral_distance (GVN) is a dormant oracle.
compressor/     T1/T2 encoders, decoder, shared training loop (train.py)
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
    --d-model 512 --n-heads 4 --n-layers 4 --d-out 512 --max-chunk-len 32 \
    --n-steps 2000000 --anchor-seed 0 --n-anchor-states 8 --out-dir t2_run

# Train a decoder on a frozen encoder
python scripts/gen_batches.py --rule single --twins 0 --batch-size 128 \
    --n-batches 5000 --config configs/instr_default.json \
  | python scripts/train_decoder.py --t1-model runs/t1_run/encoder.pt \
    --d-model 128 --n-heads 4 --n-layers 2 --n-steps 5000 --out-dir dec_run

# Test
python -m pytest
```
