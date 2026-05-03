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

**T1 encoder — validated.** Transformer compressor trained on all
RV32I instructions (single-instruction T0→T1) with nested-log MSE
loss against execution deltas, plus explicit equivalence-collapse
loss and equivalence-injected batches.

- Pearson r = 0.91 between T1 and execution distances at d_out=64
- 13/13 equivalence manifest classes PASS (cross-opcode aliasing,
  commutativity, self-op identity, zero-reg alias, etc.)
- Optimal dimensionality: d_out=64 (validated vs 32/128/256)

**T1 decoder — validated.** CE-trained transformer decoder on a
frozen encoder reaches 95%+ training tok_acc at 8L/d_model=256 and
larger. Generalization ceiling is ~72% held-out tok_acc, set by
the encoder's geometry rather than decoder capacity. 23.4% of
held-out autoregressive decodings are execution-equivalent to the
original without explicit equivalence training on the decoder —
equivalence tolerance emerges from the encoder's collapsing geometry.

**T1 encoder retraining — done.** Encoder retrained with T1
in the unit ball: magnitude carries validity, direction carries
semantics. Trained with invalid-window augmentation (partial,
spanning, multi-instruction, bogus) and a magnitude loss
against a binary `is_complete` target.

- Probe: 99.8% magnitude-threshold accuracy distinguishing
  valid from invalid windows. Mean ‖T1‖ = 1.000±0.010 for
  valid, <0.01 for invalid.
- Equivalence eval: 11/13 PASS at 50K steps (12/13 baseline
  at 100K). Comparable; would close the gap with longer
  training.

**T1 gates — reframed.** Once magnitude carries validity, the
gates have nothing semantic to learn — emit just thresholds
‖T1‖, accept does the inverse, and evict at T1 is purely a
structural state-tracker (it becomes genuinely cross-level at
T2+ where it has real signal from the layer above). To be
trained for architectural consistency or skipped; not a
research blocker either way.

**T2 compressor — first training complete (partial success).**
T2 trained to completion at target=16K, 5000 steps, ~4h wall.
Magnitude-as-validity transferred cleanly (99.0% threshold
accuracy on a held-out validity probe). However, an equivalence
probe surfaced that T2 underdiscriminates operand-level changes:
chunks that differ only in one source register collapse together
in T2 space. Diagnosis: `reg_effect_loss` (per-register pair-MSE)
plateaued at ~0.91 throughout training while the structural aux
losses saturated to near-zero. Loss-weight rebalancing and longer
training are the immediate follow-ups. See EXPERIMENT_LOG.md
Phase 11 for full analysis.

**Pipeline.** Multi-stage CPU-then-GPU pipe:
`gen_chunk_batches | to_pair_batches | train_compressor`. CPU stages
generate instruction chunks and compute pair distances; the trainer
only does forward passes and gradient updates. Multiple parallel
upstream workers via `mux_batches` saturate the GPU.

**GPU batch emulator.** All 37 RV32I opcodes executable in parallel
via `torch.where`, 1.69ms for B=4096. Enables fully-GPU REINFORCE
reward pipelines.

See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for detailed results and
[TODO.md](TODO.md) for the piecemeal assembly roadmap.

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
emulator/       RV32I emulator + GPU batch emulator (all 37 opcodes
                via torch.where, used for REINFORCE reward signal)
tokenizer/      Structural tokenizer (89-token vocabulary)
datagen/        Training data: single instructions (RVB), sequences
                (RVS), and equivalence manifest (16 classes)
compressor/     T1 encoder + decoder models, training loops
scripts/        Training and evaluation entry points
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train encoder (instr mode: single-instruction batches)
./train.sh 100000          # multi-node helper (see train.sh)
# or locally:
python scripts/gen_instr_batches.py --n-batches 10000 --batch-size 4096 \
  --config configs/instr_default.json \
  | python scripts/train_compressor.py --mode instr --n-steps 10000 \
    --d-out 64 --equiv-weight 0.05

# Pre-generate a decoder evaluation corpus
python scripts/gen_instr_batches.py --n-batches 200 --batch-size 4096 \
  --seed 12345 > /tmp/unique_corpus.bin

# Train a decoder on a frozen encoder
python scripts/train_decoder.py --model runs/<timestamp>/encoder.pt \
  --d-out 64 --n-layers 2 --dec-n-layers 8 --dec-d-model 256 \
  --dec-n-memory 64 --n-steps 800000 --cache --micro-batch 1024 \
  --warmup-steps 25000 --decay-steps 150000 \
  --save-decoder /tmp/decoder.pt < /tmp/unique_corpus.bin

# Test
python -m pytest
```
