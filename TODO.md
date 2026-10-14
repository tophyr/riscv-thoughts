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
is asked to evaluate that description at 8 anchor points. `vp = 0` is therefore
sufficient for faithfulness and not necessary (METRICS.md).

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
predict the same outputs correctly, and for a true no-op it emits no supervised
slot at all and therefore no gradient whatsoever.

## Milestones

1. **[CURRENT] vp = 0 at len=2 — not reached; 99.9364% exact.** Raising the
   density of dataflow-composed chunks in generation and doubling the step
   budget (a chain rate of 0.10 at 2M steps) takes overall exact 32-bit match
   from 99.8270% to **99.9364%** on a naturally drawn corpus — 512 failing
   (row, output-slot, anchor) triples out of 805,408, a 58% error reduction —
   and improves every error bucket over the unmodified corpus simultaneously.
   The deficit was never architectural: the same model at the same width, depth
   and training budget over the same frozen T1 scores 19.52% on a
   two-immediate composed chain at the corpus's natural composition and ~97%
   when such chains are common. What remains is the (rate, steps) choice
   (task 1) and whatever sits below ~99.94%.

2. **Computed equivalents collapse at len=2.** Chunks computing the same
   function land as far apart as chunks computing different ones (Mission's
   table). Task 2. **Settle it before chunks get longer**: every added
   instruction multiplies the equivalences a chunk participates in, so a len=4
   model would learn the same wrong geometry at four times the scale.

3. **vp = 0 at len=4, then len=8.** Was 0.244 and 0.461 before canonical read
   ranks. Those ranks match the vp target's frame on only 83.7% of len=4 and
   52.9% of len=8 chunks — the residual is dead code, which no causal read-time
   counter can know — so the len=2 mechanism may not transfer intact.

4. **Beyond the happy path.** Everything above is writers-only (R/I-type ALU,
   valid, non-x0). Then: no-write instructions (task 5), PC/branches, memory,
   longer and looser chunks. The vp metric itself needs strengthening here too —
   8 fixed anchors are not enough to certify a lossless description on a harder
   corpus (METRICS.md).

5. **T3.** Sequences of T2 blocks. Prerequisite: decide what a *composed*
   binding is — T3 is the first tier that must combine bindings rather than just
   emit them (WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface"). Settle
   that before building, not during.

## Open tasks

Task 1 settles the faithfulness milestone and task 2 is required for the MVP.
The rest are queued, not parallel work.

### 1. Settle the composed-chain density and the step budget together

**The len=2 residual is a corpus-density problem and the fix is measured; what
is left is choosing the operating point.** Chunks where one instruction consumes
another's result are ~4.7% of a randomly drawn corpus. Raising that fraction via
`chain.rate` carries overall exact 32-bit match from 99.8270% to 99.9364% on a
naturally drawn corpus — 512 failing triples against 1044. Evidence, the
per-bucket tables and the arm measurements are EXPERIMENT_LOG finding 30.

**What is being chosen is a (rate, steps) pair, not a rate.** Depth-2 accuracy
tracks the absolute number of composed rows seen, not their proportion; the
proportion only sets how fast they accumulate and how much depth-1 exposure is
surrendered per step. At a fixed 1M steps raising the rate trades depth-1 for
depth-2 and the aggregate turns on bucket size; doubling the steps at a halved
rate removes the trade and improves every bucket at once. The best pair measured
is a rate of 0.10 at 2M steps.

**Three decisions, in order:**

- **Run R=0 at 2M.** Until it exists, "2M steps" and "a rate of 0.10 at 2M
  steps" are not separated. A step budget alone has moved composed-slot
  accuracy substantially before — quadrupling it did so at a narrower register
  width, against the superseded syntactic-frame target, so that figure is not
  comparable to current ones and indicates only a direction (finding 17). If
  the budget alone reaches most of this, the rate is an efficiency knob rather
  than a necessary one and nothing need be adopted.
- **Then pick the pair**, on what the tier is FOR rather than on the headline
  number: a higher rate buys the composed cells, which is what a future T3
  consumes most of, at the cost of the common depth-1 cells and of aggregate
  points. One cell per candidate.
- **Then adopt it** by adding the rate's own config under `configs/` — one
  file per arm, named for the rate, which is what a run copies into its
  `corpus.json` — and remint against it. The rate does not obviously move the
  frozen T1, whose chunks are single instructions so a chain rate is a no-op
  there; confirm rather than assume.

**Replication standard for this task:** two corpus draws at one rate spread
0.021 points, against effects of 0.05–0.11. Any candidate pair needs its own
draw-to-draw null, not one borrowed from another pair (METRICS.md).

### 2. Collapse computed equivalences (milestone 2 — required for MVP)


Two chunks computing the same function must land on the same point. They do not,
and the failure is total rather than partial (Mission's table).

**Collapse is also the only encoder-side lever left on milestone 1**, and the
payoff is measured: the head fails on a two-literal chain because it must
decode a literal out of the essence and carry-propagate in one pass, and handed
the composite instead a readout on the same essence goes 79.80% → 99.95%
(EXPERIMENT_LOG finding 21). An essence that folded `i1,i2` onto `i1+i2` would
hand it that composite.

**It is not a T2 deficit — T1 fails the same class.** Single-instruction pairs
sharing an identical binding, so the whole difference is in the essence, sit
within a factor of 2 of pairs computing *different* functions, while the
commutative swap — free because the pool is a multiset over register slots —
sits at 117x. **At both tiers, an equivalence collapses when the architecture
cannot represent the difference and otherwise does not.** Any fix has to be
designed for T1 and T2 together: they are one class (`compressor/`), so a
mechanism that works at T1 applies at T2 unchanged. It also means task 3's
validity remint should not happen before this is settled, or T1 will be
reminted without it.

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
- T1's binding, and hence T2's vp: every `dest` misprediction measured is an
  instruction the oracle says modifies nothing, i.e. T1 predicting a write for
  `ADD x5,x5,x0`. That is 6.2% of T2's remaining vp error, arriving through
  task 3's tier interface rather than through anything T2 does.

It is not a weighting problem — T1's binding is otherwise accurate against the
oracle. It is a missing capability: knowing that `+x0`, `+0`, `|0`, `^0` and
`<<0` are the identity.

Decisions to make before building, in order:

- **How wide is the class?** Both families tested fail. Check whether any
  computed equivalence collapses, and whether the rewrite classes that DO
  collapse do so because they are trained on or because the architecture makes
  them free.
- **Essence or op-cell?** The measured distances are essence distances, but the
  composite is formed one step earlier, when the second instruction's op-cell
  reads the slot the first wrote. A cell that computed `i1+i2` there would
  collapse the pair before any pooling — and would fix task 1, which an
  essence-level term alone would not.
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

### 3. Give T1 a validity task


The frozen T1 (`runs/t1_writers_250k_rank`) is a diagnostic trained on a clean
corpus (`configs/instr_alu_writers.json`, which declares no `invalidity`
block), so it has no validity task and cannot reject invalid input. That is the
only remaining gap between it and a production T1: it trains against the
corrected oracle, at the core's settled settings, with the `rd==x0` immediate
mask applied (`_t1_immediates` in `compressor/train.py`), and its
single-instruction vp is a 100.0000% exact 32-bit match
(`probes/t1_scorecard.py`). Needed before anything leaves the happy path.

**A candidate T1 can only be judged by training a T2 over it** (~65 min at
len=2, 1M steps).
T1-side metrics do not predict what T2 extracts: six T1s all scoring 100.0000%
on their own single-instruction vp yielded T2s spanning 0.059 points, with their
binding accuracies ranked *anti*-correlated to their T2s. Budget for that before
proposing a T1 sweep, and replicate.

Scrutinise when reminting: **the binding heads are weighted 0.1 against vp's
1.0** — `in_slot_weight`, `live_in_weight`, `live_out_weight` and
`out_slot_weight` are all 0.1 while `value_predict_weight` is 1.0. T2 consumes
the binding as its WIRING, so the tier interface is optimised at a tenth of the
priority of the essence. T1 is nonetheless accurate against the oracle's
binding, so this is a thing to check, not a known deficit.

### 4. The anchor states are unstructured, and branches will need them to be


`make_anchor_states` (`datagen/compare.py:319`) draws pure uniform int32 with
only `x0 = 0`. Phase 4 established that this makes comparison and branch
behaviour degenerate — with random int32, `P(rs1 == rs2) ~ 2^-32`, so BEQ never
branches — and fixed it by structuring the states: occasional equal pairs, small
values, extra zeros. That function still exists with its structuring intact
(`emulator/cpu_emulator.py:344` `random_regs`) but nothing in the live pipeline
calls it; only tests do.

Latent, not active: on the writers-only ALU corpus no branches are generated and
`SLT` still varies across the 8 anchors. But `pos1 == pos2` holds in 0 of the 8,
so the moment branches enter (milestone 4) every BEQ is not-taken at every
supervised point. Re-point `make_anchor_states` at the structured draw, or
restate why pure random is right now — and note it changes every anchor, so it
reminces the frozen T1 and invalidates every vp number measured against the old
states (CLAUDE.md INVARIANT 1: the generator's and trainer's anchor seed and
count must match).

### 5. No-write instructions are invisible to the essence


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

**Decide first whether PC advance is part of the behavior**, because it is not
only a branch question and it gates work in two other tasks. A true register
no-op — `ADD x5,x5,x0`, `ADDI x5,x5,0` — changes nothing but the PC, and the
oracle correctly gives it no supervised slot, so vp is silent about it while the
essence still moves (0.1041). Two readings, and nothing currently distinguishes
them:

- **PC advance counts.** Then a chunk with a no-op and one without genuinely
  differ, the essence is right to separate them, and vp is *incomplete* — it has
  no PC target at all. `out_pcs` is computed and only a `pc_writes` boolean is
  shipped, so this would mean widening the RVT batch.
- **PC advance does not count.** Then those chunks are equivalent and the 0.1041
  is a collapse failure, which puts it in task 2.

Deferred deliberately: it is a definition, not an experiment, and it should be
settled before either task 2 builds a collapse objective or this task ships a
PC-effect target — each would bake in an answer.

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

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
