# Roadmap

What is left to do, in order. Nothing here is a history.

- How a number is measured, what it certifies, and how far it moves when
  nothing changed: [METRICS.md](METRICS.md).
- The evidence behind any result named here:
  [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
- The architecture and why it is shaped this way:
  [RISCV_DESIGN.md](RISCV_DESIGN.md).
- Running a training cell or a corpus: the `datagen` skill. Scoring a frozen
  checkpoint: [probes/README.md](probes/README.md).

## Mission

**T2 is done when its essence is a lossless behavioral description of its
instruction chunk AND two chunks that compute the same function land on the same
point.** Both are required; neither implies the other, and a model can score
perfectly on one while failing the other completely.

**Faithfulness — the essence determines the chunk's output values.** Can the
chunk's output register values be recovered from the essence, given the input
values? This is what a future T3 consumes: a next tier can only recover what T2
preserved. `vp` is how it is scored, and it scores the essence composed with the
vp head's ability to EVALUATE it — the encoder never sees a register value, so a
slot holds a symbolic description of the function that produced it and the head
is asked to evaluate that description at anchor input states. Those states have
to be ones the head never trained on: with a fixed set it stores answers instead
of evaluating (METRICS.md). `vp = 0` at held-out anchors is sufficient for
faithfulness and not necessary.

**Equivalence — computed equivalents collapse.** Two chunks are equivalent when
their externally visible register inputs and outputs are identical — whatever
their instruction count, final PC, instruction order, or dead and same-value
writes; PC is not an independent output. The binding is the interface: a graph
from external registers through operation roles to results, not an ordered
list. "Two token sequences mean the
same thing if and only if they compress to the same point" is the framework's
definition of meaning (WHAT_IS_A_THOUGHT.md), so an essence that separates
`ADDI r1,r1,256; ADDI r1,r1,-16` from `ADDI r1,r1,32; ADDI r1,r1,208` is not a
thought vector regardless of its vp.

Where each equivalence class stands. The essence is graded in the register
frame (RISCV_DESIGN.md, "GVN"): every class that leaves the chunk's live
dataflow graph unchanged collapses **by construction**, and the frozen T1
reads exactly zero on them (`probes/frame_scorecard.py`, 2000 pairs per
class); every class that needs the op-cell to compute is the essence's to
learn, and the figures for those are the last ones measured, in the first-read
frame that preceded this one (`separation` = distance between inequivalent
pairs over distance between equivalent ones; larger is better; under ~10x is
noise):

| equivalence class | status |
|---|---|
| register rename | 0 by construction |
| reordering of independent instructions | 0 by construction |
| dead code against its live part | 0 by construction |
| an x0-write beside live instructions | 0 by construction |
| commutative swap, any form | op-cell — the encoder is told nothing about which operations commute; first-read frame: 0.04–0.065 against 0.48 `distinct` |
| one no-op exchanged for another (`ADD xa,xa,x0` vs `ADDI xd,xd,0`) | op-cell; first-read frame: 0.0901, 5.9x |
| `XORI` chains with the same composite mask | op-cell; first-read frame: 0.5121, 0.8x |
| additive chains with the same total offset | op-cell; first-read frame: 0.4374, 1.0x |

The guard on all of this is the opposite row: pairs a transformation did NOT
leave equivalent must stay far apart — the aliasing control reads 0.32 against
0.83 for unrelated instructions at T1. A collapse objective that pulls those
together is a worse failure than loose collapse (METRICS.md).

**Nothing trains the op-cell rows** — that is their common cause. vp cannot
supply the pressure: it is not a pairwise objective, so two distinct
descriptions can both evaluate correctly, and for an instruction that writes
no register it emits no supervised slot at all and therefore no gradient
whatsoever. How well one of them collapses tracks how much the op-cell has to
learn to see it, not whether seeing it needs arithmetic: exchanging one no-op
for another needs no computation on operands and still reached only 5.9x.

## Milestones

1. **[CURRENT] Faithful at len=2, at held-out anchors, in the register
   frame.** T1 holds exactly: the frozen `runs/t1_register_500k` is 100.0000% exact
   on 2,457,600 held-out triples with a three-layer residual vp head, its
   emitted wiring exact on every instruction
   (`probes/frame_scorecard.py`). The T2 over it is at **99.8322%** — 8,113
   failing triples of 4,834,176 — after 1M steps, one seed, binding exact on
   every chunk, with outputs that depend on three inputs (a composed slot over
   disjoint inputs) at 90.3% and everything else above 99.86% (EXPERIMENT_LOG,
   "The register frame", finding 38). In the first-read frame the same
   configuration reached 5,418–5,511 failing at 1M and 36 at 10M (finding
   35); the 10M cell in this frame is the next run, and its 3-input slots are
   where to look.

2. **Computed equivalents collapse at len=2.** Everything that leaves the live
   dataflow graph unchanged collapses by construction (Mission's table); what
   remains is every equivalence the op-cell has to compute. Task 1. **Settle it
   before chunks get longer**: every added instruction multiplies the
   equivalences a chunk participates in, so a len=4 model would learn the same
   wrong geometry at four times the scale.

3. **vp = 0 at len=4, then len=8.** Unmeasured in the register frame. A dead
   instruction is 15% of len=4 and 46% of len=8 chunks (finding 33); its reads
   are in the binding and pruned from the essence at every length.

4. **Beyond the happy path.** Everything above is writers-only (R/I-type ALU,
   valid, non-x0). Then: no-write instructions (task 2), PC/branches, memory,
   longer and looser chunks.

5. **T3.** Sequences of T2 blocks. Prerequisite: decide what a *composed*
   binding is — T3 is the first tier that must combine bindings rather than just
   emit them (WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface"). Settle
   that before building, not during.

## Open tasks

In order. Task 1 is required for the MVP. The rest are queued, not parallel
work.

### 1. Collapse the computed equivalences the op-cell owns (milestone 2 — required for MVP)

Two chunks computing the same function must land on the same point. The
classes the frame makes structural do (Mission's table); the classes that need
the op-cell to compute — a commutative swap whose operand is another
instruction's input or result, identity spellings, cancellations, folds of two
literals — do not, and on the last models measured the failure on the folds
was total rather than partial.

**It is not a T2 deficit — T1 fails the same class.** Single-instruction pairs
sharing an identical binding, so the whole difference is in the essence, sit
within a factor of 2 of pairs computing *different* functions (first-read
frame, finding 23). **At both tiers, an equivalence collapses when the
architecture cannot represent the difference and otherwise does not.** Any fix
has to be designed for T1 and T2 together: they are one class (`compressor/`),
so a mechanism that works at T1 applies at T2 unchanged. A collapse change to
the core therefore remints T1; batch it with any other T1 change where that
costs nothing.

**Nothing in the project covers this class.** Equivalences are enumerated in one
place — the manifest in `datagen/generate.py` — and all 15 of its classes are
single-instruction **rewrites**, one operation expressed another way: every
template in a class is one `InstructionTemplate`, a single instruction, so a
class can only relate one instruction to another. Everything built on that manifest inherits the
limit: the corpus's `equivalences.rate` augmentation, `eval.py`'s
commutative-swap row, and a pairwise equivalence loss the trainer carried until
commit `7797c81`, recoverable from git history. A **fold** — where seeing the
equivalence requires arithmetic on the operands — appears in none of them.

**Start at identity recognition.** It is the best-defined target in either task,
it isolates "untrained" from "cannot compute", and one capability surfaces in
three places:

- collapse: exchanging one no-op for another reaches only 5.9x;
- T1's geometry (first-read frame, finding 23): three spellings of the
  identity (`ADDI x,y,0` against `ORI`/`XORI`/`SLLI x,y,0`) sit 0.2794 apart,
  and `ADD x,y,y` against `SLLI x,y,1` sits 0.5696 apart, both within 2x of
  the 0.6948–0.9675 between instructions computing *different* functions;
- the targets: a same-value self-write (`ADD x5,x5,x0`) is a read and a write,
  so the binding is never told that it is the identity; recognising
  `+x0`, `+0`, `|0`, `^0` and `<<0` as the identity is the essence's job.

**The register frame is built** (RISCV_DESIGN.md, "GVN" and "Leaf identity for
input registers"): value prediction is evaluated per written slot from that
slot's description and its inputs' values and colours gathered by register
(`vp_slot_loss`), the targets are raw register values, the essence is the pool
over the written slots only, and a leaf's identity is its structural colour —
a role-aware Weisfeiler–Lehman colouring of the live dataflow graph with
every operand role ordered, computed in the oracle (`datagen/signature.py`)
and, from T1's emitted wiring, in the encoder (`RegisterStateMachine._prepass`).
The encoder is told nothing about which operations commute: the list of
commutative opcodes reaches only the oracle's labels. The oracle's colouring agrees with
the emulator on 300 random chunks per class at len 2 and 3: rename,
reordering, every commutative swap and dead-code insertion leave the signature
unchanged, and every signature-equal pair found under aliasing, sharing,
opcode and immediate controls is equal on the emulator or a register rename of
the other (`datagen/tests/test_signature.py`). On an untrained encoder of the
same shape, reorder, dead code and an x0-write beside a live instruction give
bit-identical essences.

**Built in now, to be learned later — scaffolding, recorded as such.** The
liveness pass and the colouring are exact structural rules in the encoder;
neither knows anything about any operation. Their learned form is a short
message pass over the same dataflow graph, from the final writers back to the
leaves, each message built from the reader's content and role: liveness is
then the direction of the pass (a dead instruction is never reached) and the
colour is the leaf's learned embedding. The exact rules go first so that a
failure of the frame cannot be confused with a pass that has not yet learned
liveness; once the collapse scorecard holds with them, the
learned pass replaces them, is checked against them, and they are deleted.
This is the same tactic as establishing vp before collapse: a known
simplification taken on purpose, with its removal owed. What stays built in for
good is the graph itself — the wiring T1 emits — and the register symmetry,
which cannot be learned into a flat vector (RISCV_DESIGN.md, "What is built
in for good, and what is built in for now").

**The signature is the scorecard's label, never a target.** Two chunks with
equal signatures must sit at ~0; the classes the signature does not merge —
identity spellings, cancellations, immediate folds — are exactly the fold work
the op-cell owes, and the "Essence or op-cell?" and "What is the objective?"
decisions below are about them. Training the essence toward the signature
would separate those classes: the syntactic-hash failure RISCV_DESIGN.md's
"GVN" section names.

Decisions that remain, in order:

- **What shapes the essence?** Value prediction reads a written slot's
  description in `T`, so the essence gets no vp gradient and is shaped by
  nothing but the validity magnitude and `pc_writes`: on the 1M T2, unrelated
  chunks sit 0.023 apart against 0.67–1.19 in the first-read frame, and the
  aliasing control is inside that (finding 37). Measure first whether the
  essence still carries the written descriptions — a readout fitted on frozen
  essences against the written slots' values — then decide the objective
  that makes it answer for them, since the next tier consumes the essence,
  not `T`.
- **The binding on the wire.** Per output register, its external input
  registers in the colour order the evaluator gathers them in (`out_inputs`
  already ships it); the ordered input and output lists (`in_slot`/`out_slot`
  ListMLE) go away. What the heads predict instead, and how T3 consumes it, is
  open; until then the binding heads and their ordered targets are unchanged.
- **How wide is the class?** On the 1M register-frame T2s: a commutative
  swap whose operand is shared or a result 0.0125 against 0.0281 unrelated
  (finding 38); additive chains with one composite 0.0236 against 0.0072 for
  a different composite (0.3×), `XORI` chains 0.0200 against 0.0049;
  `ADDI r,r,0` against `ADD r,r,x0` 0.0227 against 0.0181 for `ADDI r,r,1`
  (finding 37). Nothing the op-cell has to compute
  collapses, and every one of these distances is at the scale the essence's
  whole range has shrunk to (the decision above).
- **Essence or op-cell?** The measured distances are essence distances, but the
  composite is formed one step earlier, when the second instruction's op-cell
  reads the slot the first wrote. A cell that computed `i1+i2` there would
  collapse the pair before any pooling.
- **What is the objective?** A pairwise term over generated equivalent pairs is
  the obvious candidate, and nothing in the loss is pairwise today — that was a
  deliberate choice, and undoing it is the decision. Commutativity is its
  easiest member: value prediction gives both operand orders one target and
  still leaves them apart, because it is satisfied by any description the
  evaluator can read (finding 36, and the swap rows of finding 37). Injecting manifest
  equivalents into the corpus (`equivalences.rate`) is a recorded failure at
  buying collapse with data alone. The manifest would also need
  multi-instruction classes, which means generating folds rather than rewrites.
- **What does it cost?** Every collapse gain so far has traded against
  discriminativeness, and a term that pulls equivalents together can pull
  inequivalents together too. The guard is `behav-diff` (METRICS.md), which must
  stay large.

### 2. No-write instructions are invisible to the essence

Blocks milestone 4 (~⅓ of the ISA). Content reaches the essence only through a
register write (`value[out] = res` is gated on `out != 0`), so for branches,
stores and x0-writes the op-cell result is computed and discarded. For an
x0-write that is correct — PC is not an output, so the instruction is equivalent
to its absence (below) — but for branches and stores, whose effects are on the
PC and memory, it loses the behavior. Verified:
valid-no-output and invalid windows produce the *identical* essence (cos
1.0000), so validity cannot separate them and all branches collapse to one
point.

Designed, not built: route the op-cell result into a non-renameable **operation
slot** (seeded zero, fixed under rename, like x0) for every active instruction,
so content reaches the essence regardless of register write. Open: whether a
single slot last-write or an accumulation across instructions; and what a taken
branch contributes to the essence. Routing the result must not make an
x0-write no-op visible at all: it is equivalent to its absence (below), so a
slot that carried its content would separate equivalents.

**PC is not an independent output.** Two chunks with identical externally
visible register inputs and outputs are equivalent whatever their instruction
count or final PC. So an x0-write, a same-value write or a dead instruction
beside a live chunk leaves the chunk equivalent to the live chunk alone, and
the distance between them is a collapse failure under task 1, not a missing
target here; no PC-effect vp target is shipped (`out_pcs` stays unshipped). The
one case the decision does not settle by itself is a taken branch, whose only
effect is on the PC; confirm what it contributes when branches enter scope,
before building any target for them.

### 3. Give T1 a validity task

T1's output norm is meant to say whether its token window is a complete, valid
instruction — near 1 if so, near 0 for a partial instruction, a window spanning
an instruction boundary, several instructions run together or random tokens —
with the meaning in the direction (Phase 10, magnitude-as-validity). That is
what a segmenter reading a raw token stream needs, and a gate trained on an
encoder without it plateaued at ~70% (Phase 9). The loss exists
(`(‖essence‖ − valid)²`, `valid_weight` 0.1), but the writers corpus declares
no `invalidity` block, so every row is valid, the target is always 1 and the
loss teaches nothing: the current T1 cannot reject invalid input. Every chunk T2
sees is valid by construction, so this does not block the milestones above; it
is needed before anything leaves the happy path. Invalid windows are generated
by `datagen/invalidity.py` from a config's `invalidity` block.

## Open ideas

Not scheduled and not yet a task; each needs a decision before it becomes one.

### Steer the corpus mix from the training signal, while training runs

A corpus config fixes its composition (opcode mix, `chain.rate`) for the whole
run, which presumes the right proportions are known in advance. They are not:
at a fixed step budget, raising the composed-chunk rate buys depth-2 accuracy by
spending depth-1 exposure, and depth-2 accuracy tracks the absolute number of
composed rows seen rather than their share (EXPERIMENT_LOG, "T2 faithfulness",
finding 30). A static rate is a guess at a trade a controller could manage
continuously: generate less of what the model has mastered, more of what it is
still learning, and raise a cell's rate again if its accuracy slips.

The lever has to be on the DATA side. Reweighting the loss toward hard slots
(`--vp-depth-weight`) failed because it amplifies gradient from the same rare
rows, while generation supplies new ones (finding 17).

Prior work to start from: Teacher–Student Curriculum Learning (Matiisen et al.,
2017) samples tasks by the absolute slope of their learning curves, so a
plateaued task fades and a forgotten one returns; Automated Curriculum Learning
(Graves et al., 2017) treats the mix as a bandit rewarded by learning progress;
DoReMi (Xie et al., 2023) and Online Data Mixing (Albalak et al., 2023) reweight
pretraining domains online; Prioritized Level Replay (Jiang et al., 2021) does
it over a procedural generator's settings, the closest structural match here.

What would have to be settled:

- **Drive on learning progress, not error level.** Some rows cannot be
  satisfied — where T1's emitted wiring disagrees with the tokens (~3% of len=2
  chunks), T2 is graded against a chunk it was not shown — and an error-chasing
  controller would pour data into them indefinitely.
- **Rare cells give noisy readings.** The rarest error bucket at len=2 is about
  one supervised triple per batch, so the controller must be slow and smoothed.
- **The knobs are not the buckets.** Error is measured on sources × immediate ×
  depth (METRICS.md); the generator controls opcode mix and `chain.rate`, and
  nothing controls shared sources directly. It needs a knob-to-bucket mapping or
  new generator knobs.
- **Plumbing.** The pipeline is a one-way filter with generation on odin behind
  a deep prefetch queue, so it needs a back-channel, or per-bucket streams that a
  trainer-side mux selects among (`datagen --mode weighted` biases order, not
  count, so it does not do this as-is).
- **Reproducibility.** A run would no longer be described by one config file; it
  would have to record its mix trajectory. The cosine LR schedule also assumes a
  fixed distribution.

The comparison it has to win is against the best fixed (rate, steps) pair, scored
on the naturally drawn corpus.

### Draw anchor values aimed at the instruction

Anchors are drawn per source chunk, so a draw could look at the chunk it is for.
The residual of a T1 trained at fresh structured anchors is two input patterns
the generic draw reaches only by chance: a value carrying through a long run of
zero bits (six ADDI failures, all on one held-out input, 0x723005C9 with a
negative immediate), and a comparison within a few units of the immediate
(`SLTIU x30,x10,1492` at 1497) — 7 failing triples of 819,200 (EXPERIMENT_LOG,
"T2 faithfulness", finding 33). At two instructions, a twelve-layer residual
readout handed a perfect chunk description fails on the same kind of value: 130
of its 777 misses on a composed-only corpus are one held-out state where a
difference of 0xEFFFFEFF plus an immediate carries from bit 9 to bit 28
(finding 34). Two classes would aim at them: a source value
set to `imm + δ` for small δ, and a uniform value with a random span of bits
forced to all-0 or all-1. The cost is a draw specialised to the opcode set, and
a held-out set that would need the same classes to test them on purpose rather
than by chance. The len=2 model's own 36 misses are not of this kind — three
composed `SUB` chunks each a bit off at six anchors, three composed `ADD`
chunks and 15 single-instruction slots each wrong at one anchor, nine of those
one `SUB` operand pair at one anchor state (finding 35) — so this draw is not
aimed at them.

### A smaller vp head at T2

T2's vp head is twelve residual layers because no plain ReLU head evaluates a
perfectly described two-instruction chunk exactly (METRICS.md; finding 34). The
criterion T1's head was chosen by — the smallest head that still reaches
correct results, so the essence does as much of the work as possible — has not
been applied at T2. Applying it means putting pressure on the encoder to
describe the chunk well enough for a shallower head. The essence's geometry has
tightened with the deep head — mean distance between unrelated chunks 1.19 with
the plain four-layer head, 0.67–0.69 with the residual head at 10M, swap
collapse unchanged (finding 35) — and whether that costs a shallower consumer
is what this would measure.

Depth need not cost parameters: a recurrent head, one block applied repeatedly,
would be another way to shrink it, if an iterated fixed block can do what twelve
distinct ones do. Not tried.

It may also not need shrinking at all. The vp head is the measuring instrument,
not T3. If len=8 chunks turn out to need an absurd head — 64 layers, say — that
says the chunk's function is expensive to evaluate, not that the essence is
poor. What has to be ensured is that the encoder produces an essence the next
tier can USE; it does not have to be ensured that T3 can EXECUTE the chunk. On
that reading the head's size is a cost of scoring, not a property the essence
is graded on.

### Shrink the batch payload, if generation moves off this machine

Per-row anchors make a batch 2.7x larger compressed, ~84 MB/s at T2's step rate
over the link from the generator machine. Local generation keeps up at len=2
(~468 batches/s from 26 workers against T2's ~200), so this waits until it does
not. Shipping only the canonical input columns of the anchors and only the
written slots of `out_regs` would remove most of it.

### Settings argued at 8 fixed anchors

The chain rate, the vp-as-evaluator analysis (findings 21, 26, 29, 30) and the
architecture settings argued from vp were measured at 8 fixed anchor states,
where the head stored answers (finding 32). None has been re-derived at held-out
anchors; the step budget has (1M against 10M, finding 35). Re-check one when a
decision turns on it.

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
