# Streaming Compressor Architecture

## Status: Active Design — Building on T0→T1 Single-Instruction Results

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

## Training Approach

### What T0→T1 Taught Us

The single-instruction compressor (17 experiments, see EXPERIMENT_LOG.md)
established:

- **S^127 geometry works.** Unit-sphere normalization (F.normalize on
  the output) prevents collapse while allowing MSE distance matching
  to set scale. The sphere constrains norms; MSE provides explicit
  nonzero distance targets for repulsion.

- **MSE > Pearson for distance matching.** Pearson correlation is
  scale-invariant and allows a nonzero intercept (exec_dist=0 doesn't
  force T1_dist=0). MSE directly targets absolute distances.

- **Context-free ambiguities are real and unsolvable at T0→T1.**
  BEQ≈SLT (both produce small outputs, both PC+4 on random inputs)
  and ADD-double≈SLLI (execution-equivalent but syntactically different,
  too rare to co-occur in batches). These dissolve with sequential
  context.

- **Classification heads provide structural scaffolding** (dest_type,
  dest_reg) but over long training they override behavioral signal
  for syntactically-different equivalences. The T1→T2 architecture
  should consider whether structural heads are needed or whether the
  cross-level feedback provides sufficient structure.

- **Pretrained T1 weights may not transfer.** The T0→T1 encoder was
  trained with bidirectional attention + mean pooling + pairwise MSE
  on isolated instructions. The streaming architecture processes tokens
  through gates with cross-level feedback — different enough that
  starting from scratch (with the knowledge gained) may be as effective
  as adapting pretrained weights. Token embeddings are the most likely
  transferable component.

### Joint Training

The T0→T1 experiments assumed a frozen-T1-then-build-T2 approach.
This is wrong. L1's gates need L2's feedback to make good decisions.
L1's output features should optimize for what L2 needs, not just for
pairwise distance between isolated instructions. The levels must train
jointly.

Training signal: the execution state delta over the sequence covered
by each emission. If L1 emits a vector covering tokens 0-6 (one
instruction), that vector should predict the machine state change from
those tokens. L2 emits a vector covering several L1 emissions (a basic
block), and that vector should predict the full block's state delta.

The loss at each level is MSE between the emitted vector's predicted
state delta and the actual state delta. The sphere geometry (normalize
to S^127 or S^(d-1)) applies at each level independently.

### Open: Training Schedule

Whether to train L1+L2 from scratch simultaneously, or to bootstrap
L1 alone first and then introduce L2, is an empirical question. The
HATA experiments (TRANSFERABLE_LESSONS.md) suggest staged training
may be necessary to avoid collapse. But the cross-level feedback
means L1 can't fully converge without L2, so pure staging doesn't
work. A curriculum approach (start with short sequences, scale up
as the model learns) may be more appropriate than a staged approach
(train L1, freeze, train L2).

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

4. **Feedback loop stability.** L2 feedback into L1 creates a
   recurrence. The HATA doc's structural argument (eviction constraint
   ensures fresh bottom-up signal) applies here. Whether it's
   sufficient in practice is empirical.

5. **Attention type within the window.** Bidirectional self-attention
   is the natural choice (accepted tokens should freely attend to each
   other), but TRANSFERABLE_LESSONS.md lesson 7 warns that randomly-
   initialized bidirectional attention homogenizes. The gated window
   is smaller than a full sequence, which may help, but this needs
   empirical validation.

6. **Feedback mechanism.** Cross-attention from L2 state is the
   cleanest design (L1 reads L2's state without modifying it). But
   alternatives exist: additive conditioning on gate logits, learned
   bias injection, concatenating L2 state into the self-attention
   window. The right mechanism depends on how much influence L2 should
   have over L1's internal representations vs just its gate decisions.

7. **Output space per level.** S^127 worked for T0→T1. Each level
   could have its own dimensionality — lower-level emissions might
   need more dimensions (instruction-level detail) while higher levels
   need fewer (more abstract). Or all levels could share a space.
   The WHAT_IS_A_THOUGHT.md theory suggests level-specific
   dimensionality sized to the complexity of relationships at that
   level.

8. **Training data.** T0→T1 used random instructions. T1→T2 needs
   instruction SEQUENCES — real programs or structured synthetic
   sequences. The batch pipeline infrastructure (gen_batches, mux,
   etc.) can be adapted for sequence data, but the generation logic
   changes fundamentally.
