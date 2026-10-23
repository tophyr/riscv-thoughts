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

**Equivalence — computed equivalents collapse.** "Two token sequences mean the
same thing if and only if they compress to the same point" is the framework's
definition of meaning (WHAT_IS_A_THOUGHT.md), so an essence that separates
`ADDI r1,r1,256; ADDI r1,r1,-16` from `ADDI r1,r1,32; ADDI r1,r1,208` is not a
thought vector regardless of its vp.

Where each equivalence class currently stands, on the len=2 model of milestone 1
(`separation` = distance between inequivalent pairs over distance between
equivalent ones; larger is better; under ~10x is noise):

| equivalence class | distance | separation |
|---|---|---|
| register rename | 0.0000 | — (free by construction) |
| commutative operand swap | 0.0270–0.0352 | 36–47x |
| one no-op exchanged for another (`ADD xa,xa,x0` vs `ADDI xd,xd,0`) | 0.0901 | 5.9x |
| `XORI` chains with the same composite mask | 0.5121 | 0.8x |
| additive chains with the same total offset | 0.4374 | 1.0x |

The guard on all of this is the opposite row: pairs a transformation did NOT
leave equivalent must stay far apart, and they do — 0.88 and 0.96 across the
same two draws, against a `distinct` mean of 1.26. A collapse objective that
pulls those together is a worse failure than loose collapse (METRICS.md).

**How well a class collapses tracks how much the architecture has to *learn* to
see it, not whether seeing it needs arithmetic.** Exchanging one no-op for
another needs no computation on operands and still reaches only 5.9x, so "the
encoder cannot compute with operands" does not explain the failure on its own.
**Nothing trains any of it** — that is the common cause. vp cannot supply the
pressure: it is not a pairwise objective, so two distinct points can both
predict the same outputs correctly, and for an instruction that writes no
register it emits no supervised slot at all and therefore no gradient
whatsoever.

## Milestones

1. **[CURRENT] vp = 0 at len=2, at held-out anchors.** 98.72–98.86% exact over
   the frozen `runs/t1_writers_500k_vp3`, with a four-hidden-layer plain T2 vp
   head, two instruction-stream seeds, scored end to end through T1's emitted
   wiring at all 64 held-out anchors (METRICS.md; EXPERIMENT_LOG, "T2
   faithfulness", finding 33). `SUB` carries 45–49% of the failing triples and
   slots composed from the other instruction's result 41–44%. T2's binding is
   exact on every chunk scored.

2. **Computed equivalents collapse at len=2.** Chunks computing the same
   function land as far apart as chunks computing different ones (Mission's
   table). Task 2. **Settle it before chunks get longer**: every added
   instruction multiplies the equivalences a chunk participates in, so a len=4
   model would learn the same wrong geometry at four times the scale.

3. **vp = 0 at len=4, then len=8.** Was 0.244 and 0.461 at the 8 fixed anchors,
   before canonical read ranks. A chunk's input list counts the reads of dead
   instructions, which are 15% of len=4 and 46% of len=8 chunks (finding 33), so
   T2's read ranks match the target's frame at every length.

4. **Beyond the happy path.** Everything above is writers-only (R/I-type ALU,
   valid, non-x0). Then: no-write instructions (task 3), PC/branches, memory,
   longer and looser chunks.

5. **T3.** Sequences of T2 blocks. Prerequisite: decide what a *composed*
   binding is — T3 is the first tier that must combine bindings rather than just
   emit them (WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface"). Settle
   that before building, not during.

## Open tasks

In order. Task 1 gates milestone 1, and task 2 is required for the MVP. The
rest are queued, not parallel work.

### 1. vp = 0 at len=2 over the frozen T1

Anchors are drawn afresh for every source chunk, from a structured mix, and
ride in the batch; structural labels come from fixed witness states and decide
nothing at the anchors; every write counts; scoring is at one fixed held-out set
(RISCV_DESIGN.md; METRICS.md; finding 33). T1 passes both of its losses within
one triple: `runs/t1_writers_500k_vp3`, whose vp head has three hidden layers —
the smallest that reaches correct results — emits the oracle's wiring on
100.0000% of instructions and is 99.9999% exact on vp at held-out anchors (1
failing triple of 819,200, a comparison next to its immediate; Open ideas). Over
it, `scripts/run_t2.sh`'s cell — four-hidden-layer plain vp head, chain rate
0.10, 1M steps — is 98.72–98.86% exact on vp with exact binding (finding 33).

**T2's vp head is free in shape until this milestone passes** (METRICS.md).
Handed a perfect description of a two-instruction chunk, no plain ReLU head of
the vp head's shape is exact, while a twelve-layer residual head with LayerNorm,
after 10M steps, misses 0.0024% of held-out triples on a corpus where every
chunk is composed (finding 34). A smaller head is the requirement after that, not before.

- **A residual vp head with LayerNorm in T2.** `_build_vp_head`
  (`compressor/model.py`) builds a plain Linear+ReLU stack, and `vp_head_live`
  (`compressor/train.py`) assumes one: it splits the first layer's weights by
  hand to skip empty input slots. Both need the residual form, and a checkpoint
  with the plain head must still load. Two instruction-stream seeds over the
  frozen T1, scored at all 64 held-out anchors against the plain four-layer
  cells.
- **What a perfect description still leaves.** The twelve-layer residual head's
  remaining misses are carries rippling through long runs of ones in `SUB`,
  `ADD` and `ADDI` writers, at a few held-out states (finding 34; Open ideas,
  "Draw anchor values aimed at the instruction").
- **A smaller head, once a deep one passes.** Put pressure on the encoder to
  describe the chunk well enough for a smaller head, judged by the criterion
  T1's head was chosen by: the smallest head that still reaches correct
  results, so the essence does as much of the work as possible.
- **One head form at both tiers, once T2's residual head is settled.** T1's vp
  head is the plain three-layer stack. Remint T1 with the residual form at the
  smallest depth that still passes, show that neither tier regressed, and then
  delete the plain form. Batch it with task 2's T1 remint if the timing lines
  up.
- **The batch payload, if generation moves off this machine.** Per-row anchors
  make a batch 2.7x larger compressed, ~84 MB/s at T2's step rate over the link
  from the generator machine. Local generation keeps up at len=2 (~468 batches/s
  from 26 workers against T2's ~200), so this waits until it does not. Shipping
  only the canonical input columns of the anchors and only the written slots of
  `out_regs` would remove most of it.
- **The residual.** With the plain four-layer head over the frozen T1, `SUB` is
  89.7–90.0% exact and carries 45–49% of the failing triples; slots composed
  from the other instruction's result are 78.3–79.3% and carry 41–44%.
- **Everything argued from 8-anchor vp**: the chain rate, the step budget
  (`run_t2.sh` runs 1M; 2M was chosen at 8 anchors), the
  vp-as-evaluator analysis (findings 21, 26, 29, 30), and the architecture
  settings argued from vp.

### 2. Collapse computed equivalences (milestone 2 — required for MVP)

Two chunks computing the same function must land on the same point. They do not,
and the failure is total rather than partial (Mission's table).

**It is not a T2 deficit — T1 fails the same class.** Single-instruction pairs
sharing an identical binding, so the whole difference is in the essence, sit
within a factor of 2 of pairs computing *different* functions, while the
commutative swap — free because the pool is a multiset over register slots —
sits at 117x. **At both tiers, an equivalence collapses when the architecture
cannot represent the difference and otherwise does not.** Any fix has to be
designed for T1 and T2 together: they are one class (`compressor/`), so a
mechanism that works at T1 applies at T2 unchanged. A collapse change to the
core therefore remints T1; batch it with any other T1 change where that costs
nothing.

**Nothing in the project covers this class.** Equivalences are enumerated in one
place — the manifest in `datagen/generate.py` — and all 15 of its classes are
single-instruction **rewrites**, one operation expressed another way
(`template_instr_counts=[1]`). Everything built on that manifest inherits the
limit: the corpus's `equivalences.rate` augmentation, `eval.py`'s
commutative-swap row, and a pairwise equivalence loss the trainer carried until
commit `7797c81`, recoverable from git history. A **fold** — where seeing the
equivalence requires arithmetic on the operands — appears in none of them.

**Start at identity recognition.** It is the best-defined target in either task,
it isolates "untrained" from "cannot compute", and one capability surfaces in
three places:

- collapse: exchanging one no-op for another reaches only 5.9x;
- T1's geometry: three spellings of the identity (`ADDI x,y,0` against
  `ORI`/`XORI`/`SLLI x,y,0`) sit 0.2794 apart, and `ADD x,y,y` against
  `SLLI x,y,1` sits 0.5696 apart, both within 2x of the 0.6948–0.9675 between
  instructions computing *different* functions;
- the targets: a same-value self-write (`ADD x5,x5,x0`) is a read and a write,
  so the binding is never told that it is the identity; recognising
  `+x0`, `+0`, `|0`, `^0` and `<<0` as the identity is the essence's job.

**Dead-code elimination is part of this task.** A dead instruction's reads are
the chunk's inputs (`ADD x5,x6,x7 ; ADDI x5,x8,1` has inputs x6, x7, x8), so
the targets never make a chunk with dead code equal to its live part; making
the essences land together is collapse work, measured as equivalence rather
than as correctness.

Decisions to make before building, in order:

- **How wide is the class?** Both families tested fail. Check whether any
  computed equivalence collapses, and whether the rewrite classes that DO
  collapse do so because they are trained on or because the architecture makes
  them free.
- **Essence or op-cell?** The measured distances are essence distances, but the
  composite is formed one step earlier, when the second instruction's op-cell
  reads the slot the first wrote. A cell that computed `i1+i2` there would
  collapse the pair before any pooling.
- **What is the objective?** A pairwise term over generated equivalent pairs is
  the obvious candidate, and nothing in the loss is pairwise today — that was a
  deliberate choice, and undoing it is the decision. Injecting manifest
  equivalents into the corpus (`equivalences.rate`) is a recorded failure at
  buying collapse with data alone. The manifest would also need
  multi-instruction classes, which means generating folds rather than rewrites.
- **What does it cost?** Every collapse gain so far has traded against
  discriminativeness, and a term that pulls equivalents together can pull
  inequivalents together too. The guard is `behav-diff` (METRICS.md), which must
  stay large.

### 3. No-write instructions are invisible to the essence

Blocks milestone 4 (~⅓ of the ISA). Content reaches the essence only through a
register write (`value[out] = res` is gated on `out != 0`), so for branches,
stores and x0-writes the op-cell result is computed and discarded. Verified:
valid-no-output and invalid windows produce the *identical* essence (cos
1.0000), so validity cannot separate them and all branches collapse to one
point.

Designed, not built: route the op-cell result into a non-renameable **operation
slot** (seeded zero, fixed under rename, like x0) for every active instruction,
so content reaches the essence regardless of register write. Open: whether a
single slot last-write or an accumulation across instructions; whether routing
the result makes equivalent x0-write no-ops spuriously distinct across
immediates; and whether branches also need a real PC-effect vp target (the SSA
oracle computes `SELECT_PC` but does not ship it).

**Decide first whether PC advance is part of the behavior**, because it gates
work in another task. An instruction that writes no register — a branch not
taken, a store, an x0-write — changes nothing but the PC (or memory), and the
oracle gives it no supervised slot, so vp is silent about it while the essence
still moves. Two readings, and nothing currently distinguishes them:

- **PC advance counts.** Then a chunk with such an instruction and one without
  genuinely differ, the essence is right to separate them, and vp is
  *incomplete* — it has no PC target at all. `out_pcs` is computed and only a
  `pc_writes` boolean is shipped, so this would mean widening the RVT batch.
- **PC advance does not count.** Then those chunks are equivalent and their
  distance is a collapse failure, which puts it in task 2.

Deferred deliberately: it is a definition, not an experiment, and it should be
settled before either task 2 builds a collapse objective or this task ships a
PC-effect target — each would bake in an answer.

### 4. Give T1 a validity task

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
than by chance.

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
