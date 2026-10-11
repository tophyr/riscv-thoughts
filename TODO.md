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

**Faithfulness — vp = 0.** Can the chunk's output register values be recovered
from the essence, given the input values? This is what a future T3 consumes: a
next tier can only recover what T2 preserved.

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

1. **[CURRENT] vp = 0 at len=2 — not reached.** `--d-out 512` at 1M steps over
   the frozen `runs/t1_writers_250k_rank` is at **99.828%** exact 32-bit match
   (`runs/t2_len2_d512_s1000000_dep4_vp1_pool16_pma_pewrite_time_rtrank_seedrank_t1rank_writers`).
   The whole residual is one shape — an immediate that does not survive being
   composed with another instruction's result — which is task 1.

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
   longer and looser chunks. The vp metric itself needs strengthening here too —
   8 fixed anchors are not enough to certify a lossless description on a harder
   corpus (METRICS.md).

5. **T3.** Sequences of T2 blocks. Prerequisite: decide what a *composed*
   binding is — T3 is the first tier that must combine bindings rather than just
   emit them (WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface"). Settle
   that before building, not during.

## Open tasks

Task 1 is the current focus. Task 2 is required for the MVP, overlaps task 1 on
one subset without containing it, and is sequenced after it, before chunks get
longer. The rest are queued, not parallel work.

### 1. An immediate does not survive composition

**The single open cause of the len=2 vp residual.** The error lives where an
immediate meets a **composed operand** — the writing instruction taking its
operand from the other instruction's result rather than from a chunk input.
That cell is 93.01% exact and carries 39.5% of all remaining error; adding a
shared source on top takes it to 81.08%. Each condition **alone** is solved: an
immediate alone 99.909%, a shared source alone 99.980%, composition alone
99.957%, and every population carrying no immediate at all is at or above
99.95%.

What is already known, with the evidence in EXPERIMENT_LOG:

- **The immediate is encoded perfectly and destroyed on consumption.** The
  producer's own slot carries its literal at 99.99%+, so nothing is lost writing
  it; it is lost when that slot is *read* by the second instruction, which is
  the op-cell applied to its own previous output. Not the pool — both slots are
  pooled identically and only the composed one degrades.
- **What decides it is carry propagation, not fusibility.** Bitwise throughout
  (`XORI;XORI`) is exact; a carry-propagating writer is ~15% whatever it
  consumes; a bitwise writer on a carry-produced intermediate is 47.78%. Shifts
  behave like carries. `ADDI;ADDI` is algebraically the easiest of the four and
  scores no better than the pair that genuinely needs two stages, so the model
  does not take the algebraic shortcut.
- **It is a near-miss, not a dropout.** On `ADDI;ADDI`, 96.6% of predictions
  land within one immediate's width of the target and 0.13% drop a literal
  outright.
- **The cell is two sub-cases that trade against each other.** One literal
  decoded and combined with values the head is handed as bits (`ADDI;ADD`) is
  readout-limited and moves ~20 points under two interventions sharing no
  mechanism. Two literals folded into one (`ADDI;ADDI`) is reached by nothing,
  and everything that helps the first drives it further down. The second subset
  is what task 2 covers, and the only one it covers.
- **Ruled out as aggregate levers:** the vp head (refits and fine-tunes on
  frozen essences all score below the trained head; depth 4 trained end-to-end
  leaves the aggregate inside the noise floor); the pool; gradient share; the
  leaf-versus-composed operand scale mismatch; content crowded out of the
  op-cell's first layer; and capacity (`d_out` 512 → 1024 leaves the error cells
  unchanged to three decimals). Also not a description mismatch — restricting
  the cell to rows where T1's wiring matches the tokens, or to rows whose target
  is identical under either canonical frame, leaves it at its own 93.01%.

**Do next: settle whether the residual is essence-side or head-side.** The two
available readings contradict. Against the head: every refit and fine-tune on
frozen essences scores below the trained head. For the head: the producer's slot
is predicted at 99.99% from the same essence in the same forward pass that fails
on the composed slot, so the description is there, and a head trained end-to-end
at depth 4 moves one sub-case 20 points.

The experiment that separates them is to feed the head the intermediate value as
explicit bit features, the way the canonical inputs already are, and retrain.
**It is a diagnostic, not a change to adopt** — handing the head a value the
essence is supposed to carry makes vp stop certifying what milestone 1 claims —
so what it produces is a decision about the acceptance criterion rather than a
fix. `probes/depth_intermediate.py` is the nearest thing already built and found
no gain, but with a fitted readout on the superseded target at `d_out` 256, so
it does not settle it.

Then, if still open:

- **Supervise the intermediate value**, which the SSA oracle computes and the
  RVT batch does not ship. Distinct from feeding it to the head, and lower
  prior: the intermediate slot's own vp is already 99.99%+, so there is nothing
  there to supervise into existence.
- **Re-test two mechanisms refuted only against the superseded target**: a
  direct linear path from an instruction's content to the slot it writes, and
  weighting vp by dataflow depth. Both were measured at `d_out` 256 / 250k
  against a target whose error composition differs materially from the current
  one. Recover either from git history (`--content-skip`, `--vp-depth-weight`).

Score any composed-operand change on the arms of
`probes/vp_compose_control.py`, not on the aggregate or the depth-2 bucket —
both interventions above moved those arms 15–25 points while the aggregate
stayed flat.

### 2. Collapse computed equivalences (milestone 2 — required for MVP)

Two chunks computing the same function must land on the same point. They do not,
and the failure is total rather than partial (Mission's table).

**Collapse would also close milestone 1's `ADDI;ADDI` sub-case**: that vp error
exists because the head is handed two literals and must compute their composite,
and an essence that collapsed `i1,i2` onto `i1+i2` would hand it the composite
instead, reducing the job to the single-literal case it already does at 99.9%.

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
limit: the `--inject-equiv` augmentation, `eval.py`'s commutative-swap row, and
a pairwise equivalence loss the trainer carried until commit `7797c81`,
recoverable from git history. A **fold** — where seeing the equivalence requires
arithmetic on the operands — appears in none of them.

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
  deliberate choice, and undoing it is the decision. `--inject-equiv` is a
  recorded failure at buying collapse with data alone. The manifest would also
  need multi-instruction classes, which means generating folds rather than
  rewrites.
- **What does it cost?** Every collapse gain so far has traded against
  discriminativeness, and a term that pulls equivalents together can pull
  inequivalents together too. The guard is `behav-diff` (METRICS.md), which must
  stay large.

### 3. Give T1 a validity task

The frozen T1 (`runs/t1_writers_250k_rank`) is a diagnostic trained on a clean
corpus (`--inject-invalid 0`), so it has no validity task and cannot reject
invalid input. That is the only remaining gap between it and a production T1: it
trains against the corrected oracle, at the core's settled settings, with the
`rd==x0` immediate mask applied (`compressor/train.py:275`), and its
single-instruction vp is a 100.0000% exact 32-bit match
(`probes/t1_scorecard.py`). Needed before anything leaves the happy path.

**A candidate T1 can only be judged by training a T2 over it** (~2.4h at len=2).
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

### 5. Fix `_behaviorally_equal`

`compressor/eval.py`'s equivalence predicate compares the whole 32-register
canonical file, including input registers that `_canonical_state` merely
relocated, so it reports only 9.2% of genuine commutative equivalences as
equivalent at len=1 — where the vp target is identical for 100% of them. vp
supervises the written slots only, so the comparison should be `out_regs` at
`out_slot_regs`. It currently gates the `distinct` pool, where unrelated chunks
are almost never equivalent, so the effect there is small; the predicate is
wrong wherever else it gets used.

### 6. Overlap the trainer's host-side work with the GPU step

len=2 cells run at ~50% GPU utilisation and the bottleneck is the trainer's
host-side per-step preparation, not generation — diagnosed and measured in the
`datagen` skill. What is missing is the fix: prefetch the next batch's
preparation on a worker thread, or build the dense grid on-device. Worth ~2x
wall-clock on every len=2 run.

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
