# Streaming Compressor Architecture

## Status: Next Direction — After T1 Single-Instruction Validation

---

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

## Core Mechanism: Three Learned Gates

The streaming compressor maintains an active window over its input
token stream. At each step it makes three independent learned decisions:

**Include gate:** consume more tokens from the stream into the active
window. Fires when the model predicts it cannot yet represent the
current window's content without more future context.

**Emit gate:** produce a compressed vector from the current active
window and push it to the output stream. Fires when the model judges
that a coherent unit of computational content has been accumulated.

**Evict gate:** drop specific tokens from the active window. Fires
when a token has been sufficiently captured in emitted vectors and
is no longer needed for future emissions.

The three gates operate independently. A token can influence multiple
emissions before eviction (a base register used by several load/store
instructions). A token can be evicted without triggering an emission
(a nop or redundant instruction). A token can persist across multiple
emissions from other positions.

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

## Cross-Level Feedback

A purely feedforward design loses global context: once a token exits
L1's active window, L1 cannot relate it to future input.

Mitigation: L2 feeds its current state back into L1's attention as
additional key/value pairs. L2 has been cross-attending over multiple
L1 emissions — it carries integrated context across several L1 windows
that L1 never saw simultaneously. The feedback restores this global
view to L1 without requiring L1 to maintain an unbounded window.

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

## Training Approach

### Phase 1: Variable-Length Sequences (Current → Next)

Train the existing model architecture on variable-length token
sequences instead of single instructions. Start with sequences of
1-5 instructions, scale up. The model learns to handle any length
through the same mean-pooling mechanism, but applied to longer
sequences.

Execution comparison: full state delta, log-scaled per register.
The loss is the same Pearson correlation between T1 pairwise distances
and execution distances.

This validates that the architecture can handle sequences at all,
without the streaming gates. The model sees the entire sequence at
once (no windowing) and compresses it to one vector.

### Phase 2: Streaming Gates

Replace the "see everything at once" approach with the gated streaming
mechanism. The model processes tokens left-to-right with a bounded
active window, emitting vectors when it judges a unit is complete.

The emit gate's training signal comes from execution: emitted vectors
should predict the machine state delta of the tokens they cover. If
the model emits too early (partial instruction), the prediction is
poor. If it emits too late (spanning two unrelated instructions), the
prediction is noisy.

### Phase 3: Multi-Level Hierarchy

Stack L2 on top of L1. L2 processes L1's emissions and emits
block-level representations. Cross-level feedback connects them.

The execution comparison at L2 is the state delta over the entire
block. The L2 loss is the same correlation loss but over block-level
distances.

---

## Relationship to LZ77

The streaming thinker is analogous to LZ77 (the basis of gzip, zstd):

| LZ77 | Streaming Compressor |
|---|---|
| Search buffer (recent raw content) | Emitted vectors + L2 feedback |
| Lookahead buffer | Active window |
| Extend lookahead before committing | Include gate |
| Emit (offset, length) pointer | Emit gate |
| Advance search buffer, evict old | Evict gate |

The key difference: LZ77 is lossless (exact matches). The streaming
compressor is semantically lossy by design — the emit criterion is
"sufficient computational content accumulated," not "longest exact
match found." That criterion must be learned.

---

## Open Questions

1. **Gate architecture.** How do the three gates read from the active
   window? Attention over the window → scalar gate logit? Or learned
   threshold on some window state summary?

2. **Emission granularity.** Does L1 emit one vector per instruction,
   or could it emit sub-instruction vectors (one for the opcode+dest,
   one for the computation)? The execution signal will determine this
   empirically.

3. **Memory in the state delta.** How to represent memory changes in
   the delta for the execution comparison. Loads read from memory,
   stores write to it. The delta needs to capture both without a
   fixed-size memory diff.

4. **Training schedule.** Can L1 and L2 be trained independently
   (L1 first with stop-gradient, then L2 on frozen L1, then joint
   fine-tuning)? Or does concurrent training work from the start?
   The HATA experiments suggest staged training may be necessary to
   avoid collapse.

5. **Feedback loop stability.** L2 feedback into L1 creates a
   recurrence. The HATA doc's structural argument (eviction constraint
   ensures fresh bottom-up signal) applies here. Whether it's
   sufficient in practice is empirical.
