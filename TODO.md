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
predict the same outputs correctly, and for a true no-op it emits no supervised
slot at all and therefore no gradient whatsoever.

## Milestones

1. **[CURRENT] vp = 0 at len=2, at held-out anchors — unmeasured; blocked on
   task 1.** Every vp figure so far is at the one fixed set of 8 anchor states
   the head trained on, where it stored answers: the len=2 model that is 99.94%
   exact there scores ~5% at anchors it never saw (EXPERIMENT_LOG, "T2
   faithfulness", finding 32). The milestone is scored end to end, through T1's
   emitted wiring (METRICS.md), so besides the remint it needs the three wiring
   defects removed that make ~96% of the 8-point residual unsatisfiable for any
   T2 (finding 31):
   - SLTIU's source is judged non-behavioral at T1 for every immediate, because
     uniform-random anchor states make `rs <u imm` constant (24% of the 8-point
     failures; task 1).
   - The oracle orders a chunk's behavioral inputs by first syntactic read,
     including reads that are non-behavioral at their own instruction and that
     T1 therefore never emits, so T2's operands arrive swapped (54%; most of it
     is SLTIU reads, which task 1 also removes). The rest is removed too if task
     1 builds the frame from per-instruction relevance composed by dataflow, its
     preferred design.
   - T1's binding drops the source of `SUB`/`XOR`/`SLT rd,x0,rs` on every such
     instruction (18%; task 3).

   T2's corpus is `configs/instr_alu_writers_chain010.json` at 2M steps, both
   chosen at the 8 training anchors and to be re-checked after the remint.

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
   valid, non-x0). Then: no-write instructions (task 4), PC/branches, memory,
   longer and looser chunks.

5. **T3.** Sequences of T2 blocks. Prerequisite: decide what a *composed*
   binding is — T3 is the first tier that must combine bindings rather than just
   emit them (WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface"). Settle
   that before building, not during.

## Open tasks

Task 1 comes first: until it lands, no vp figure says whether a function was
learned. Task 2 is required for the MVP, and task 3 carries one of the wiring
fixes milestone 1 needs. The rest are queued, not parallel work.

### 1. Remint T1 and T2 with anchor states redrawn every batch, from a structured draw

**vp has only ever been trained and scored at one fixed set of 8 anchor states,
and the head stored answers at them.** Every row of every batch sees the same 8
register files, and the canonical frame sources a chunk's k-th input from
position k+1 of them, so input 1 takes 8 distinct values in an entire run.
Scored at anchors it never saw, the len=2 T2 that is 99.94% exact falls to ~5%,
and T1's own head from 100.0000% to 21% (EXPERIMENT_LOG, "T2 faithfulness",
finding 32). Every vp number in the project is an 8-point figure.

**The objective can train the function; the fixed set is what prevents it.** A
fresh head trained on FROZEN T1 essences with anchors redrawn per batch reaches
98.98% at anchors it never saw, still climbing at 200k steps, and does at least
as well as one handed the opcode and immediate directly (96.62%). T2's essence
has not had the same test, and is more at risk: T1 also trains a linear
immediate head, while T2's essence is shaped by vp alone.

**Uniform values are also poor witnesses, so the draw needs structure.**
`make_anchor_states` (`datagen/compare.py:319`) draws pure uniform int32. A
comparison against a 12-bit immediate is then constant, so the oracle marks
SLTIU's source non-behavioral and T1 learns to emit no read for it — 24% of the
8-point len=2 failures directly and most of the 54% from the shifted input frame
(finding 31) — and two registers almost never hold equal values, so every BEQ
is not-taken once branches enter (milestone 4; Phase 4). Phase 4's structured
draw — occasional equal pairs, small values, extra zeros — still exists as
`random_regs` (`emulator/cpu_emulator.py:344`), called only by tests.

Decisions to make before building:

- **How the structural labels stay deterministic.** The anchors do two jobs.
  They decide the oracle's structural labels — which inputs and outputs an
  instruction has, hence T1's binding targets and the canonical frame the vp
  target is built in — and they supply the input values vp is evaluated at.
  Labels are a property of the instruction, so they must not depend on which
  anchors a batch happened to draw: `SLTIU x4,x5,40` is "reads x5" only in a
  batch where some anchor puts a value below 40 there, so per-batch anchors used
  for both jobs would flip its binding target between batches.

  **Preferred: decide relevance per instruction, and compose it for chunks by
  dataflow.** For each instruction, decide which of its reads matter by testing
  it against a small FIXED, generic witness set (zeros, small values of both
  signs, equal pairs) — no per-opcode perturbation rules, and cacheable per
  instruction form, since a single instruction is cheap to test at hundreds of
  states. Zero alone witnesses SLTIU at every nonzero immediate: `0 <u k` holds
  and its bit-flip `0xFFFFFFFF <u k` does not. A chunk input is then behavioral
  iff a path of per-instruction-relevant reads connects it to a chunk output —
  static liveness over the SSA graph, with no values and nothing to invert,
  where testing chunk inputs by execution would mean solving for an input that
  pushes an intermediate across a boundary. The vp values come from per-batch
  anchors, which decide no labels. Consequences:
  - Labels are deterministic and independent of any anchor draw, and the
    witness set is never learned from, so it cannot be memorized.
  - The canonical frame is built from the same per-instruction relevance T1's
    binding is trained on, plus the wiring T2 is given, so a correct T1 cannot
    produce the frame-shift class of finding 31; a T1 binding error still can.
  - Cancellation ACROSS instructions (`ADDI x5,x5,1 ; ADDI x5,x5,-1`) no longer
    removes an input: the targets stop imposing that equivalence, and
    recognising it becomes the essence's job (task 2).
  - Whether a same-value write counts as a write is still task 4's open
    definition; read relevance is unaffected by it.

  Alternatives if that does not pan out:
  - decide chunk-level labels by execution against one large, fixed,
    structured witness set, with vp values from per-batch anchors. Keeps
    value-level cancellation in the labels, at the cost of a second anchor set
    and execution over hundreds of states per chunk.
  - keep one set per batch for both jobs, and guarantee that every batch's draw
    contains the witnesses. Simpler plumbing, but it needs a guarantee per
    narrow function (`SLTIU …,1` varies only at 0), inside an 8-state budget
    that also has to exercise full-width carries.
- **The vp draw.** Structured as well, with enough generic 32-bit values to
  exercise full-width carries, where the fresh head's shortfall sits (`ADD`,
  `SUB`).
- **How many per batch.** 8 today; the per-step cost of more is unmeasured.
- **The wire format.** The anchors ride in every RVT batch and the trainer reads
  them from it instead of rebuilding one fixed set from a seed. That is a format
  change and a rewrite of CLAUDE.md's INVARIANT 1.
- **The held-out set.** One fixed anchor set no model ever trains on, from the
  same structured distribution, used by the criterion and every probe
  (`probes/vp_budget.py` already takes an anchor seed).

Then remint T1, then T2 over it, and score both at the held-out set. Every
choice whose evidence was vp needs re-checking against held-out numbers rather
than inheriting: the chain rate and 2M budget, the vp-as-evaluator analysis
(findings 21, 26, 29, 30), and the architecture settings argued from vp. Fold in
task 3's binding fix if it is ready, since that remints T1 too.

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
core therefore remints T1, as do tasks 1 and 3; batch them where that costs
nothing, but no order between them is required.

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
  instruction the oracle says modifies nothing, i.e. T1 reporting a read and a
  write for `ADD x5,x5,x0`. It costs T2 nothing measurable: on the chain-rate
  0.10 len=2 model none of the failing triples at the 8 training anchors sits in
  such a chunk, because the oracle's chunk frame counts that read as well. It is
  an error only under the oracle's current definition of a write, which is open
  (task 4).

If that definition stands, it is a missing capability: knowing that
`+x0`, `+0`, `|0`, `^0` and `<<0` are the identity.

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

### 3. Give T1 a validity task


The frozen T1 (`runs/t1_writers_250k_rank`) is a diagnostic trained on a clean
corpus (`configs/instr_alu_writers.json`, which declares no `invalidity`
block), so it has no validity task and cannot reject invalid input. That and
one binding error (below) are the gaps between it and a production T1: it
trains against the corrected oracle, at the core's settled settings, with the `rd==x0` immediate
mask applied (`_t1_immediates` in `compressor/train.py`), and its
single-instruction vp is a 100.0000% exact 32-bit match at the 8 training
anchors (`probes/t1_scorecard.py`) and 21% at anchors it never saw — a stored
answer, which task 1 replaces. Needed before anything leaves the happy path.

**A candidate T1 can only be judged by training a T2 over it** (~65 min at
len=2, 1M steps).
T1-side metrics do not predict what T2 extracts: six T1s all scoring 100.0000%
on their own single-instruction vp at the 8 training anchors yielded T2s spanning 0.059 points, with their
binding accuracies ranked *anti*-correlated to their T2s. Budget for that before
proposing a T1 sweep, and replicate.

Scrutinise when reminting: **the binding heads are weighted 0.1 against vp's
1.0** — `in_slot_weight`, `live_in_weight`, `live_out_weight` and
`out_slot_weight` are all 0.1 while `value_predict_weight` is 1.0. T2 consumes
the binding as its WIRING, so the tier interface is optimised at a tenth of the
priority of the essence. The binding has one measured deficit against the
oracle: whenever x0 is the FIRST operand of `SUB`, `XOR` or `SLT`, T1 emits no
source at all (0.0% of 400 random instructions per opcode, against 95–100% for
every other operand form measured), although the oracle marks the source
behavioral. That is 18% of T2's len=2 failures at the 8 training anchors
(EXPERIMENT_LOG, "T2 faithfulness", finding 31).

### 4. No-write instructions are invisible to the essence


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
oracle gives it no supervised slot, so vp is silent about it while the
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

**Decide with it whether a same-value write counts as a write.** The oracle
currently says no: a register the chunk READ counts as written only if its final
value differs from its initial one at some anchor (`_actually_modified`,
`datagen/compare.py`), so `ADD x5,x5,x0` has no input and no output. That forces
one answer to a question with two defensible ones, and it has three costs:

- It decides a structural question by comparing values, at 8 anchor points.
- It reaches across instructions: `ADDI x5,x5,1 ; ADDI x5,x5,-1`,
  `AND x5,x5,x5` and any chain folding back to identity also lose their slot and
  their input, so the binding and vp targets demand the computed-equivalence
  recognition that task 2 has not built. At longer chunks the dropped input also
  shifts every later input's canonical position.
- The binding is the interface the next tier wires on, and for an interface the
  safe contract is "may write": reporting a same-value write is harmless to a
  consumer, while omitting a real one is not. Under that contract T1 predicting
  a write for `ADD x5,x5,x0` is correct.

The alternative counts every syntactic write to a read register: the chunk then
maps input x5 to output x5 by identity, its vp target stays self-consistent, and
`ADD x5,x5,x0` stays equivalent to `ADDI x5,x5,0` but not to an empty chunk —
which is the PC-advance question again. Identity collapse would then be the
essence's job (task 2), not something the targets impose on the binding.

- **Accepting both answers is possible, as a way to defer the definition.**
  Dropping a write is not one unscored slot: output slots are positional in
  write order, so later outputs renumber, and the dropped register leaves the
  behavioral inputs, so later inputs shift canonical position. The two answers
  are two canonical frames. The oracle would execute in both (it already
  re-executes when the syntactic and behavioral frames differ), the RVT batch
  would ship both target sets, and vp, T1's binding losses and every scorer
  would take the per-row minimum. That is semantically the "may write"
  contract — an omitted write is accepted only when it is a same-value write.
  Costs: frames multiply as 2^k in the number of same-value writes; "same
  value" is still decided at 8 anchors; and the vp head's input mask, which
  comes from the oracle's input count rather than from T2, would be supplied
  per frame, so the minimum rewards whichever ground-truth count helps — derive
  that mask from T2's own state in the same change. Emitting "may write"
  unconditionally is the simpler destination: one frame, one changed rule.
Behavioral READS are not part of this question and stay: they set the canonical
frame of the vp target (EXPERIMENT_LOG, "T2 faithfulness", finding 14).
Changing the write rule changes the oracle, so it remints T1 and T2 and resets
every vp baseline.

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
