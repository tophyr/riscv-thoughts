# RISC-V Thought Compression Testbed

An experimental ML research project investigating whether learned
compression can produce vector spaces with smooth, factored, interpolable
geometry — and whether search-based decoding can invert the compression.

## The Idea

A "thought" is a compressed representation — a point in a continuous
vector space produced by a deterministic compression function over tokens.
Two token sequences mean the same thing if and only if they compress to
the same point. Thinking becomes akin to transformations of thought points:
composition, differencing, and recombination in the compressed space. 

Thought expression is search: given a thought, find the token sequence that
compresses nearest to it. Expression verifiability falls out of this for
free, because it is simply compressing the expression and differencing it 
against the original thought.

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

Phase 1 (T0→T1 single-instruction compression) is complete for ALU
instructions. A small transformer compressor trained with execution-
distance correlation loss produces a T1 vector space with:

- Pearson r = 0.995 between T1 and execution distances
- Clean opcode clustering (SLT/SLTU: 0.26 apart, SRL/SRA: 0.44)
- Monotonic immediate scaling
- Correct register sensitivity hierarchy
- Partial execution equivalence collapse

See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for detailed results.

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
emulator/       RV32I emulator wrapper (TinyFive backend)
tokenizer/      Structural tokenizer (89-token vocabulary)
datagen/        Training data generator (equivalence & mutation pairs)
compressor/     T0→T1 compressor model and training loop
scripts/        Training and evaluation entry points
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train
python scripts/train_compressor.py
python scripts/train_compressor.py --batch-size 4096 --d-out 128

# Evaluate
python scripts/eval_compressor.py runs/<timestamp>

# Test
python -m pytest
```
