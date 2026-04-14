# Streaming Compressor Architecture

## Motivation

The current T1 compressor processes single instructions in isolation.
To compress sequences, a naive approach would train a separate T2 model
on top of frozen T1 vectors, manually splitting token streams at
instruction boundaries. This is wrong for two reasons:

1. **Manual boundary splitting is not ML.** The model should discover
   its own unit boundaries from the data, not have them prescribed.
   Instruction boundaries happen to be syntactically unambiguous in
   RV32I (opcode tokens are strong signals), but the principle matters:
   the architecture should generalize to domains where boundaries are
   not obvious.

2. **Single instructions are just short sequences.** A unified model
   that handles variable-length token streams — from 4 tokens (one
   R-type instruction) to hundreds (a basic block or function) — is
   simpler and more general than a two-model hierarchy with a manual
   splitting step between them.

The streaming compressor replaces the fixed-window T1 model with a
streaming transducer that processes token streams of arbitrary length,
emitting compressed representations when it judges it has accumulated
a coherent unit.

---

## Core Mechanism: Neural Shift-Reduce Parser

The streaming compressor is a shift-reduce parser with learned gates.
It maintains a window (the stack) over its input token stream and
makes three independent decisions each iteration:

**Accept gate (shift):** push the next input token onto the window.
Answers: "will having more information result in a more-emittable
vector?"

**Emit gate (reduce):** compress the window into a vector on S^127
and output it. Answers: "is the compressed representation of this
window complete enough to be decodable?"

**Evict gate:** pop the oldest token from the window (FIFO). Answers:
"does the oldest token in the window still contribute meaningful
information?"

The three gates fire independently each iteration. The window
encoder (bidirectional transformer) runs every iteration, producing
a T1 candidate that all gates read from. A GRU processes the T1
candidate sequence to provide temporal context.

Tokens arrive strictly in order. The accept gate cannot skip tokens.
The window is always a contiguous span. If the accept gate rejects
a token, the system must emit/evict before re-presenting it.

The architecture is recursively applicable: T1 processes a token
stream and emits T1 vectors; T2 processes the T1 vector stream and
emits T2 vectors. Same gate mechanism at every level.

### Implementation

**Components:**
- **Window encoder:** bidirectional transformer (2 layers,
  d_model=128, 4 heads) over window contents. Produces T1
  candidate on S^127.
- **Gate controller:** GRU processes T1 candidates causally.
  Three independent MLP heads. Accept sees (next token embedding,
  T1 candidate, GRU hidden). Emit and evict see (T1 candidate,
  GRU hidden).
- **Decoder:** autoregressive transformer decoder conditioned on
  emission vector via cross-attention. Evaluates T1 candidate at
  every iteration (not just emit points).

**Per-iteration loop:**
```
while input or window:
    t1_candidate = window_encoder(window)
    gru_hidden = gru(t1_candidate, gru_hidden)

    accept = accept_gate(next_token, t1_candidate, gru_hidden)
    emit = emit_gate(t1_candidate, gru_hidden)
    evict = evict_gate(t1_candidate, gru_hidden)

    if accept: window.append(next_token)
    if emit: output(t1_candidate)
    if evict: window.pop(0)  # FIFO oldest
```

**Gate semantics:**
| Accept | Emit | Evict | Meaning |
|--------|------|-------|---------|
| yes | no | no | shift: need more tokens |
| yes | yes | no | shift and reduce |
| no | yes | no | reduce first (lookahead signals new thought) |
| no | no | yes | clean stale context before continuing |
| yes | yes | yes | shift, reduce, clean (full cycle) |

**Gate training:** Bernoulli sampling + REINFORCE with per-iteration
advantage from decoder reconstruction quality. Gumbel-sigmoid does
not work (zero gradient through Python control flow).

---

## Applied to RISC-V

### Level 1: Tokens → Instruction-Level Representations

L1 streams over tokens and emits when it recognizes a complete
computational unit. In RISC-V, this will almost certainly correspond
to individual instructions — the opcode token at the start of each
instruction is an unambiguous structural signal. But the model learns
this from data, not from a hardcoded rule.

Verification: the model should emit once per instruction. If it emits
mid-instruction or spans two instructions in one emission, the execution
comparison signal will be weak (partial instructions don't have
well-defined semantics), naturally training it toward instruction-aligned
boundaries.

### Level 2: Instruction Representations → Block Representations

L2 streams over L1's emissions and emits when it has accumulated a
coherent computational block. In RISC-V, this likely corresponds to
basic blocks (sequences between branches) — but again, the model
discovers this.

A basic block's semantics are fully determined by its input state:
given registers, PC, and memory, the block produces a deterministic
output state. The execution comparison at L2 is the machine state delta
over the entire block.

### Inter-Level Streaming

L1's output stream is L2's input stream. The same three-gate mechanism
operates at every level boundary. The interface is identical at every
depth — no special-casing.

This is where the hierarchy manages its own scale. L1 emits one vector
per instruction. L2 consumes these and emits one vector per block. The
"compression ratio" at each level is not fixed — it's determined by
the data. A 3-instruction block and a 30-instruction block both produce
one L2 emission, but the model decides when.

---

## Attention Within and Between Levels

**Within a level (over the active window):** the compressor attends
over the tokens currently accepted in its window. The gates define
what's in the window; the attention operates over whatever's there.
Bidirectional self-attention is a natural fit — the compressor should
be able to relate any accepted token to any other — but this is a
design hypothesis, not a requirement. The gate structure (accept/evict)
controls the window contents; the attention type within the window is
a separate architectural choice.

**Between levels (cross-level feedback):** L2 feeds its current state
back to L1 as context. This is naturally a cross-attention mechanism —
L1's window tokens attend to L2's state as a separate key/value source.
Cross-attention keeps the levels cleanly separated: L1 reads from L2
but L2's state isn't modified by L1's attention. Again, this is a
design hypothesis — other feedback mechanisms (additive conditioning,
gating modulation) are possible.

The distinction matters architecturally: self-attention within a level
lets the compressor freely relate tokens to each other; cross-attention
between levels provides context from above without muddying the level
boundary.

## Cross-Level Feedback

A purely feedforward design loses global context: once a token exits
L1's active window, L1 cannot relate it to future input.

Mitigation: L2 feeds its current state back into L1 via cross-level
attention. L2 has been attending over multiple L1 emissions — it
carries integrated context across several L1 windows that L1 never
saw simultaneously. The feedback restores this global view to L1
without requiring L1 to maintain an unbounded window.

### Why This Matters for RISC-V

Consider a sequence where x5 is set early and used much later:

```
ADDI x5, x0, 42    ← L1 emits, L1 evicts
... 20 instructions ...
ADD x10, x5, x3    ← L1 processes this, but x5's definition is gone
```

Without feedback, L1 doesn't know that x5 was set to 42. With L2
feedback, L2's representation of the earlier block (which includes
x5's definition) is available to L1 via cross-attention. L1 can
recognize the dependency even though the defining instruction is no
longer in its window.

---

## Execution Comparison for Sequences

The single-instruction model extracts (data_val, pc_val) — one value
written to one register, plus the PC. For sequences, the output is
the full machine state delta:

**State delta:** for each register, compute output[r] - input[r].
Registers that didn't change contribute zero. Log-scale each register's
delta independently. The distance between two sequences is the sum of
per-register log-scaled deltas, plus the PC delta.

This naturally handles:
- Single instructions (one register changes, same as current)
- Basic blocks (multiple registers change, each contributes)
- Equivalent sequences (same delta → same point, regardless of internal
  structure)

Memory effects: for sequences with loads/stores, the state delta
includes memory changes. The SparseMemory approach extends — track
which addresses were written, compare the written values.

---

## RISC-V Advantages Over Natural Language

Several open questions from the HATA streaming thinker design are
easier in RISC-V:

**Emission gate training signal.** In natural language, "when to emit"
is subjective — where does one thought end and another begin? In
RISC-V, instruction boundaries are syntactically unambiguous. The
execution comparison provides direct signal: if the model emits at
the wrong boundary (mid-instruction), the resulting vector won't
predict the machine state delta correctly. The loss naturally trains
correct emission timing.

**Eviction coverage criterion.** In natural language, "has L2
adequately covered this token" is hard to define. In RISC-V, coverage
is measurable: if L2's representation can predict the machine state
delta that includes this instruction's effect, the instruction is
covered.

**Boundary verification.** We can directly verify that L1 discovers
instruction boundaries and L2 discovers block boundaries by inspecting
emission timing. In natural language, the "correct" boundaries are
debatable. In RISC-V, they're checkable.

**Data dependencies are explicit.** Register names in the token stream
directly encode data flow. L1 doesn't need to infer co-reference
from context — x5 in one instruction is the same x5 in another
instruction. This makes cross-level feedback more effective because
the dependency structure is in the tokens, not hidden in semantics.

---

## Data Pipeline

Two binary stream formats, both with full pipeline tool support
(all tools auto-detect format):

**RVS (sequence batches):** instruction sequences with per-
instruction register snapshots. For sequence training and gate
training. Generated by `gen_seq_batches.py`.

**RVB (single-instruction batches):** individual instructions with
computed-value execution outputs and PC values, for N×N pairwise
encoder training. Generated by `gen_instr_batches.py`.

**Pipeline tools:** `batch_slice` (inspect/subset), `batch_cat`
(concatenate), `batch_repeat` (multi-epoch), `batch_shuffle`
(randomize order), `mux_batches` (parallel generation and weighted
mixing). All format-agnostic.

---

## Training

T1 components are trained separately and assembled, then fine-tuned
jointly. Joint training from scratch fails due to circular
dependencies between gates, encoder, and decoder (Exp 22-23).
See TODO.md for the current roadmap and EXPERIMENT_LOG.md for
full experimental results.

**Losses:**
- **Pairwise MSE:** N×N MSE between T1 vector distances and
  computed-value execution distances. Trains encoder geometry.
- **Reconstruction:** teacher-forced cross-entropy from emission
  vector to instruction tokens. Trains decoder; window-size
  weighting incentivizes early emission.
- **Round-trip:** decode → Gumbel-softmax → re-encode → cosine
  distance. Ensures decoder consistency with encoder.
- **REINFORCE:** per-iteration decoder reconstruction quality as
  reward signal for gate decisions. Required because discrete gate
  decisions break the gradient chain (Gumbel-sigmoid gives zero
  gradient through Python control flow).

**Data formats:**
- RVB (single-instruction batches) — N×N pairwise MSE for encoder training
- RVS (instruction sequences) — for gate and joint training

---

## Relationship to LZ77

The streaming thinker is analogous to LZ77 (the basis of gzip, zstd):

| LZ77 | Streaming Compressor |
|---|---|
| Search buffer (recent raw content) | Emitted vectors + L2 feedback |
| Lookahead buffer | Active window |
| Extend lookahead before committing | Accept gate |
| Emit (offset, length) pointer | Emit gate |
| Advance search buffer, evict old | Evict gate |

The key difference: LZ77 is lossless (exact matches). The streaming
compressor is semantically lossy by design — the emit criterion is
"sufficient computational content accumulated," not "longest exact
match found." That criterion must be learned.

---

## Open Questions

### Resolved

1. **Gate architecture.** Resolved: GRU gate controller processes
   T1 candidates causally. Three independent MLP heads read from
   (T1 candidate, GRU hidden). Accept also sees next token embedding.
   Gates use Bernoulli sampling + REINFORCE during training.

2. **Emission granularity.** Resolved for T1: one emission per
   instruction. The window-size-weighted reconstruction loss
   naturally drives this. T2 granularity TBD.

3. **Attention type within the window.** Resolved: bidirectional
   self-attention within the window. The window is small (one
   instruction ≈ 4-8 tokens), so homogenization is not an issue.

4. **Training data.** Resolved: RVS format for sequences, RVB
   format for single-instruction batches. Full pipeline with
   generators, mux, slice, cat, repeat, shuffle.

5. **Gate training signal.** Resolved: per-iteration decoder
   evaluation provides reconstruction quality at every step.
   REINFORCE with per-iteration advantage (loss minus moving
   average baseline). Gumbel-sigmoid doesn't work (zero gradient
   through Python control flow).

### Open

6. **Equivalence collapse.** Cross-syntax equivalences (ADD-double
   ≈ SLLI) don't collapse from pairwise MSE or reconstruction loss.
   Longer training with N×N repulsion may help (Exp 4 showed
   equivalence improving past loss plateau). If not: round-trip
   loss with execution verification, or equivalence pair injection.

7. **Feedback loop stability.** T2 feedback into T1 creates
   recurrence. Untested. The structural argument (eviction ensures
   fresh bottom-up signal) may be sufficient.

8. **Feedback mechanism.** Cross-attention from T2 state is the
   planned approach. Alternatives (additive conditioning, gating
   modulation) exist. Empirical when T2 is built.

9. **Output space per level.** S^127 works for T1. Whether each
   level needs its own dimensionality is empirical.

10. **Memory in state delta.** Per-register deltas work for the
    current instruction set. Memory effects (load/store) are
    captured via shared SparseMemory across instructions in a
    sequence, but not explicitly in the distance metric.
