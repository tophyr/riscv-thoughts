# Metrics

What each measurement means, what it does and does not certify, how far it
moves when nothing has changed, and the conditions under which a number from it
is true. Every rule here exists because a figure was once quoted without it and
a wrong conclusion followed.

Measured values and the evidence behind them live in EXPERIMENT_LOG.md. This
file is the instrument, not the readings.

---

## vp — value prediction

**What it measures.** Given the chunk's input values, can its output register
values be recovered from the essence? The head reads `(essence, canonical
input-slot values)` and predicts each output register's canonical value. It is
the metric that matters because the essence is what a future T3 consumes: a next
tier can only recover what this one preserved.

**The head's shape is part of what a vp number certifies, so quote it with the
number.** vp scores the essence composed with a readout, and that readout is the
only consumer a T2 essence has — T3 does not exist yet — so its capacity is an
implicit claim about what the next tier will be able to read. `vp = 0` at a
deeper head is a strictly weaker statement than at a shallow one, because
capacity granted to the head is structure the representation is no longer
required to expose, and that compounds at every tier of a hierarchy whose
premise is cheap coarse checks. Both heads are pre-activation residual stacks
with LayerNorm. T1's is the smallest that still reaches correct results: three
hidden layers of 1024. T2's is twelve layers deep, because a readout that
cannot evaluate a perfectly described chunk exactly measures its own limit
rather than the essence's, and at two instructions every plain ReLU readout of
the vp head's shape is such a readout (EXPERIMENT_LOG, "T2 faithfulness",
finding 34). A T2 vp figure is therefore read against the same head on a
perfect description of the chunk, which, after 10M steps, gets 8 of 4.8M
held-out output values wrong (a value is one output register of one chunk at
one anchor state) where T2's essence gets 36 wrong (finding 35). Until a
smaller head is shown to reach the same score, a T2 figure certifies less than
a T1 figure of the same value.

**Score it as exact 32-bit match of the predicted output value, never
per-bit.** The output is discrete, and per-bit flatters a broken subset badly:
96% per-bit has been 74% exact, and 81% per-bit on one probe column was 21%
exact. The loss value itself is a per-bit BCE and therefore asymptotic — it
never reaches 0 however good the model is, so it is a training signal and not an
acceptance criterion.

**What counts as solved.** The loss is asymptotic, so "solved" at a given chunk
length means **100% exact 32-bit match on `probes/vp_budget.py` at anchor
states the model never trained on**, which scores the model's own head over the
training corpus and crosses the three properties
that decide a slot.

**A vp number in a training log is held out in its rows and in its inputs.**
`datagen` generates every batch fresh, draws each source chunk's anchor states
afresh, and the trainer never revisits one, so a training-log vp is measured at
input points the head has not seen. It is still noisy from batch to batch — the
draw decides how hard a batch's values are — and it is not exact match, so
`probes/vp_budget.py` remains the criterion. `scripts/eval.py` exists to score
*binding* and the collapse rows on a fixed corpus.

### What `vp = 0` certifies, and what it does not

A head trained at a fixed set of anchor states can satisfy vp by storing answers
at those points instead of evaluating, and trained heads did: models exact at 8
fixed anchors scored ~5% (T2) and 21% (T1) at states they never saw
(EXPERIMENT_LOG, "T2 faithfulness", finding 32). **So `vp = 0` certifies a
function only at anchor states the model never trained on.** Training draws
fresh anchors for every source chunk and ships them in the batch (CLAUDE.md,
INVARIANT 1), so no input point repeats; scoring uses one fixed held-out set
(`heldout_anchor_states`, a seed no generator uses), which every scorer reaches
by generating its corpus with `datagen --heldout-anchors`. `probes/vp_budget.py`
and `probes/t1_scorecard.py` do so. Its value classes and copy rates are pinned
separately from the training draw's, so tuning the training draw does not move
the criterion; it contains no negated pairs, so it exercises full-width carries
only through its uniform values.

**Near exact, a few of the 64 held-out states carry most of the residual.** On a
corpus where every chunk is composed, one state carried 160 of a twelve-layer
readout's 777 failing triples and six states carried 55%, against 1/64 each
under uniform exposure (finding 34). A residual's size therefore depends on
which states the set holds, and a score over a subset of them can be decided by
one hard value pair; score at all 64.

The essence is not what memorized. It is a fixed function of the instruction —
the encoder never sees a value — so a fresh head trained with anchors redrawn
per batch can test what a frozen essence carries: on the fixed-anchor T1 it
reaches 98.98% at held-out anchors (finding 32).

**It measures the head's arithmetic as well as the essence's content.** The
encoder never sees a register value — its inputs are each instruction's opcode
and immediate and the wiring — so a register slot holds a symbolic description
of the function that produced it and the head must EVALUATE that description at
the anchor points. vp scores the composition.

How much the evaluation costs depends on the head. A T1 trained at fresh
structured anchors is 99.9999% exact at held-out anchors with a three-hidden-
layer head (finding 33), so at one instruction a small head evaluates it. Fed
ground-truth instruction features instead of an essence, no one-hidden-layer
head reaches exact at any width tried, two layers reach 99.36% and four 100.00%
(finding 33). At two instructions, fed ground-truth chunk features, no plain
ReLU head of four to twelve layers is exact, and a twelve-layer residual head
with LayerNorm, after 10M steps, misses 8 of 4.8M held-out triples on the graded
population (finding 34); T2's essence through that head misses 36 (finding 35).
So at T2 the readout's capacity is a live confound in every vp number, and the
perfect-description figure for the same head is the floor a T2 figure is read
against. The measurements that put the cost near zero
at T2 — exact three-term addition, a decoded constant added exactly — were made
at 8 fixed anchors, where a stored answer and an evaluated one cannot be told
apart (finding 32).

Do NOT attribute a residual by reconstructing the instructions and executing
them. That measures instruction reconstruction, and then answers the question
with an emulator — if re-executing the chunk is allowed, the compressed vector
is not being tested at all.

**vp scores T1∘T2, not T2.** T2's operand and destination slots come from T1's
emitted binding (`_t1_predicted_wiring`), never from the tokens, while the
target is built from the tokens' SSA. Where the two disagree, T2 encodes one
chunk and is graded against another, and the row is unsatisfiable however good
T2 is. "T1 has perfect vp" does not license reading the pipeline number as T2's:
vp scores `vp_head` on T1's essence, while the wiring comes from
`in_score_head` / `out_score_head` / `live_in_head`, a different loss at a tenth
of the weight. Re-check the wiring with `scripts/eval.py`'s T1 binding score
("Binding", below) after any T1 change; it compares against the ORACLE's
per-instruction binding, not the tokens', since T1's binding is behavioral and
the tokens are syntactic.
The pipeline number is the acceptance criterion: "solved" means 100% through
T1's emitted wiring, so a defect in T1's binding, the anchor states or the
oracle's frame blocks it as surely as one in T2. Splitting the score by whether
T1's wiring agrees with the oracle attributes a residual; it does not replace
the criterion.

### Reading the error budget

`probes/vp_budget.py` is the first thing to reach for. It crosses three
properties of an output slot, each of which decides whether that slot is
predicted correctly:

- **sources** — `cross-instr` when the chunk's two instructions read a common
  register, `distinct` otherwise. One instruction reading the same register
  twice (`ADD x3,x5,x5`) is a separate and benign case.
- **immediate** — whether any instruction in the chunk carries one. R-type ALU
  operations take two registers; I-type take a register and a 12-bit literal.
- **depth** — the slot's dataflow depth: 1 when the instruction that wrote it
  read only chunk inputs, 2 when one of its operands is another instruction's
  result.

The counting unit throughout is the **(row, output slot, anchor) triple** that
vp supervises. Each bucket is reported both as a rate and as **err mass**, its
share of all failing triples. **A rate and a mass are different facts and
neither implies the other**: a rare cell can be terrible and carry almost none
of the error, a common one excellent and carry most of it.

**Cross the axes; do not pair one-dimensional probes.** Three separate fixes
were each verified on their own axis with the others held at an easy setting,
and the cell where two conditions hold at once appeared in none of them — and
was 90% of the error.

### A difference is only real against a measured floor

**Every sweep carries a null intervention** — the same configuration trained
twice on separately generated corpora — or no row in it is interpretable. The
null must be run **on the arm under test**, because floors do not generalise
between cells: two configurations have differed 20× in their own run-to-run
spread, so a floor borrowed from a quiet cell inflates every significance claim
made against it. Where the quantity is a difference between frozen lower tiers,
the lower tier is what needs replicating.

**Replicating means re-drawing the corpus, not re-initialising.** Both generator
pools default to one seed, so re-running a cell without changing it measures
initialization variance only; the `datagen` skill has the env var that redraws
the stream.

**Rows sharing one setting but differing in another are not replicates.**
Neither are rows trained against different oracles. Pooling them to reach a
usable `n` manufactures confidence, and can count the same quantity as signal in
one comparison and as noise in another.

The floors measured so far are in "Current calibration" at the end of this
file. Inherit them only within the configuration they were measured on.

**An aggregate that does not move is not evidence that nothing happened.** Two
interventions have moved individual constructed arms by 15–25 points while the
aggregate and the whole depth-2 bucket stayed flat. Score a change on the
population it targets.

### Conditions a vp number is only true under

- **The corpus it ran on.** An R-only corpus excludes the `SLTIU rd,rs,-k` shape
  entirely, and reading 8.8× lower error there than on the mixed corpus was
  taken as a property of the model for a whole phase of work. Quote the
  mixed-corpus figure for statements about the model.
- **The canonical frame its target was built in.** Numbers predating the
  behavioral canonical frame are not comparable to current ones: a model is
  scored on the target it trained on, and the two targets differ on 12.5% of
  rows. Compare on the rows where they agree, where the change took 99.7383% →
  99.8347%.
- **The chunk length.** Figures are per-instruction or per-slot and do not carry
  across lengths.

---

## Binding — the wiring the next tier is given

**The requirement at T1:** T1's emitted wiring is the oracle's, for every
instruction. T2 never sees tokens; it routes each instruction's operands and
destination off T1's binding heads (`_t1_predicted_wiring`), so a mis-wired
instruction hands T2 a chunk different from the one it is graded on, and no T2
can recover it (vp scores T1∘T2, above).

**Score it per instruction, exact, as T2 consumes it.** `scripts/eval.py`
(`t1_binding_accuracy`) counts an instruction correct only if the emitted
`(in0, in1, out)` equals the oracle's first two input slots in order and its
first output slot, x0 where absent. **"Solved" is 100%.** It also lists every
opcode-and-operand form below 100% — x0 in either source, aliased sources, a
destination aliasing a source — because the defects found so far each occupy
whole forms at 0% while the aggregate stays near 98.5%.

Do not score the binding per register. A per-register average over 32 slots
reads 1.000 when one instruction in seventy is wired wrong, and a per-slot
order check that ignores how many reads the heads say exist passes an
instruction that is given no input at all. Both read 1.000 on a T1 whose
emitted wiring is 98.5% exact.

**At T2 the binding is a chunk's interface**, the wiring T3 will route on. The
required form is a graph — which external register feeds which operation role
and which receives which result. The current form is two ordered lists: every
register the chunk reads from outside, in first-read order, and every register
it writes, in write order; the order is the linearization the first-read frame
imposes (RISCV_DESIGN, "GVN"), so the score below certifies the current form,
not the required one. `scripts/eval.py` (`t2_exact_binding`)
decodes those lists from T2's heads the way T1's are decoded — the live heads
give the count, the score heads the order — and counts a chunk correct only if
both equal the oracle's. **"Solved" is 100%.** It breaks the score down by chunk
shape: one instruction reading the other's result, two reading a common input,
a write overwritten unread, a read the output ignores, x0 or aliased operands,
a destination that is also a source; and lists the opcode-and-operand forms
carrying the most failing chunks.

Both scores are against the oracle's definitions: every write counts, so a
same-value self-write (`ADD x5,x5,x0`) is scored as a read and a write of x5,
and a dead instruction's reads are inputs.

---

## Equivalence collapse

**The requirement:** two instruction sequences that compute the same function
must land on the same point. A class is scored as the distance between
equivalent essences against the distance between inequivalent ones —
**separation**, larger is better.

`scripts/eval.py` reports four distances between essences, and they are not the
same kind of quantity:

- **rename** — a chunk against the same chunk with its registers relabeled,
  encoding the relabeled copy with its slots permuted to match. Equivariance is
  exact by construction, so this reads ~0 for any model at all. It asserts the
  core is intact; it is not a measure of training quality.
- **behavioral** — a chunk against a variant with a commutative instruction's
  operands swapped, **split** by whether the swap actually preserved the chunk's
  function. Swapping changes first-read order, so if a later instruction reads
  one of the swapped registers the canonical input positions shift and the two
  chunks compute genuinely different functions. About 2% of generated pairs are
  non-equivalent this way, they sit at the `distinct` scale, and they inflate a
  pooled mean several-fold. **Judge collapse on the equivalent subset.** That
  subset is itself three populations (`probes/swap_classes.py`): pairs with
  fresh operands and a third-register destination, invariant by construction
  and reading float noise; pairs whose destination overwrites a swapped source;
  and pairs whose swapped operand already carried a rank. The row's mean is
  mostly the first population's zeros, so a small value does not say the other
  two collapse — the probe reports them separately.
- **behav-diff** — the non-equivalent subset above. It should be LARGE. A model
  that collapses it is failing to distinguish different functions, which is a
  worse failure than loose collapse.
- **behav-mixed** — both pooled. Not interpretable on its own; kept only for
  continuity with older records.
- **distinct** — unrelated chunks. The separation denominator.

**Pass `--gvn-chunk-len` matching the model's chunk length.** `gvn_collapse`
builds its own pairs and defaults to 4, so a len=2 model scored without it is
measured out of distribution — worth an order of magnitude (equivalent-pair
distance 0.0712 at chunk_len=4 against 0.0287 in distribution, same checkpoint).
`scripts/run_t2.sh` passes it.

**`probes/additive_fusion.py` scores a class `eval.py` cannot.** Each chunk is
two instructions of one opcode writing one register, so the essence describes
only the composite, the intermediate having been overwritten. **fusion** is the
same composite reached by a different split of the two immediates; **different**
is a different composite, and is the denominator without which a small fusion
distance would only mean the encoder collapses everything. The probe builds the
chunks directly and takes a distance — no fitted readout anywhere — and asserts
through the emulator that each `fusion` pair really is behaviorally identical
and each `different` pair is not.

### A separation gap under ~10× is no evidence

The denominator is a mean over a few hundred pairs of a near-zero quantity, so
it is tail-dominated. Two corpus draws of one configuration give 46.6× and
41.2×, and two scorers building independent pair samples have disagreed on the
*sign* of smaller gaps. The instability is
concentrated in the denominator, so two draws can report a ratio as 0.8 and 1.0
when they are one number.

Rank configurations on vp exact match instead, against the floor that matches
the comparison ("Current calibration").

**Collapse and vp are both load-bearing.** A vp gain that craters the collapse
rows is a regression, not a win — and the converse: every collapse gain measured
so far has traded against discriminativeness, so `behav-diff` is the guard on
any collapse objective.

---

## Two metrics that are currently inert

**Magnitude as validity** is a health check, not a live objective: the corpus
injects no invalid windows, so every validity target is 1 and vp normalizes the
essence anyway. Essences should sit on the unit sphere, and a drop in the
fraction of essences near norm 1 is a symptom to chase rather than a semantic
failure; no current scorer reports it.

**Reconstruction** (decoding the essence back to instructions) is an explicit
non-goal, and no current scorer measures it. It is per-instruction and compounds
over a chunk, so two lengths are not comparable.

---

## Probe methodology

A probe is a small readout trained on a frozen model to test what a
representation contains. Several conclusions drawn from one had to be withdrawn,
in two recurring ways — **a probe scoring below the model's own trained head is
measuring the probe** (an under-powered or data-starved readout read as missing
information), and **a probe scoring near the majority-class frequency has
learned nothing** (at an 88% base rate, 88% means it answers by default, so the
minority case needs separate treatment). Both are cheap to prevent: calibrate
every probe against a quantity known independently, and check its score against
the base rate before believing it.

**A hard subset is a harder question, so scoring it lower proves nothing on its
own.** What breaks that confound:

- **Hand the readout the thing the hard case has to compute.** Giving it the
  true intermediate value removes "this subset needs two arithmetic steps" as an
  explanation.
- **Prefer a measurement with no fitted readout at all.** Building the inputs
  directly and taking a distance between essences has no capacity, no training
  set and no sample-size asymmetry. The results that have held up are of this
  kind.
- **Asking for a quantity whose difficulty does not vary with the subset is NOT
  sufficient.** One composed instruction's 12-bit immediate returned 28.9%,
  53.2%, 74.7% and 97.0% across four designs differing only in readout depth and
  in whether the fit was per-group or pooled — the spread *is* the probe.
  Neither fitting mode is neutral (pooled underfits a subpopulation needing its
  own decoding function; per-group starves the rare subset of rows), so report
  both, and treat a divergence between them as a fact about the probe.

**To locate a loss, measure one quantity at every stage and find where it
drops.** Recovery of a 12-bit immediate, stage by stage: T1's essence 100.000%
→ T2's written slot 99.3% → the pool's view of that slot 79.2% → T2's essence
76.9%. The last two agreeing ruled out the attention weights and pointed at the
projection beneath them — one measurement refuting three hypotheses.

**To size a suspected bottleneck, refit it for one task and sweep its width.**
The pool's projection as a trainable bottleneck gave 8→77.2%, 16→97.1%,
32→99.3%, 256→99.6% on I-type, separating two explanations one number
conflates: it was too narrow *and* below its own ceiling, because it was
multiplexing everything else.

**A defect whose cost falls on a different row than the one carrying it is
invisible to any split of the corpus.** Splitting by "does this row have the
defect" reported one target bug as harmless for as long as it was measured that
way, because the cost fell on the rows that did NOT carry it. Where a defect can
make two different inputs collide, build both sides of the collision and vary
one thing.

**Diagnose before training.** Probe frozen checkpoints until the diagnosis
yields a *falsifiable* hypothesis, then spend GPU time only to test that
mechanism. One round of probing found in hours what ten training runs had not.

---

## Current calibration

**Everything in this section is a measurement, not a rule, and it is scoped to
the configuration named.** The rules above outlive any model; these numbers do
not. When the configuration changes — a different chunk length, a different
frozen lower tier, a changed oracle — they become unknown rather than
approximate, and the null intervention has to be re-run. Do not inherit a floor
across that boundary.

Scope for all of it: chunks of 2 instructions, `d_out` 512, over the frozen
`runs/t1_writers_250k_rank`, on `configs/instr_alu_writers.json`, scored at the 8
training anchor states — so these are floors on a stored-answer score, and none
carries over to a model trained with anchors redrawn per batch. The runs
behind each row are in EXPERIMENT_LOG, "T2 faithfulness" (findings 25 and 31); this
table is the operative copy, that is the evidence. Neither T1 named in this
section is kept in `runs/`, so these floors stand as recorded and cannot be
re-run.

| quantity | value | measured over |
|---|---|---|
| retraining the T2 on a fresh corpus moves exact match | 0.002–0.009 points | two configurations, each replicated |
| retraining the frozen T1 beneath it moves exact match | 0.023 points (σ) | five same-config T1s |
| the same, on depth-2 exact match | 0.24 points | the same five |
| spread between two configurations' own floors | 20× | the two above |
| one T2 cell against another of the same config, chain rate 0.10 | 0.011 points (σ per cell) | nine cells, `vp_budget` on one shared corpus |
| two corpus draws at 2M steps, chain rate 0.10 | 0.0025 points | two cells, `vp_budget` on one shared corpus |
| two corpus draws at 2M steps, chain rate 0 | 0.0003 points | two cells, `vp_budget` on one shared corpus |

At held-out anchors the floor is larger. Over `runs/t1_writers_250k_ctxpulse`,
len=2, chain rate 0.10, four-layer vp head, 1M steps, scored on 300 batches × 16
held-out anchors:

| quantity | value | measured over |
|---|---|---|
| two instruction-stream seeds of one T2 configuration | 0.04–0.08 points | two configurations, each at seeds 0 and 7 |
| a null intervention's cost | ~65 min | one len=2 cell at 1M steps, ~3.9 ms/step, scoring included |

With the twelve-layer residual vp head — the only form the code builds — over
the plain-headed three-layer T1 of finding 33 (not the frozen
`runs/t1_writers_500k_resvp3`), len=2, chain rate 0.10, 1M steps, every cell
scored on `runs/eval/t2_heldout64_len2.rvt` at all 64 held-out anchors
(EXPERIMENT_LOG, "T2 faithfulness", finding 35):

| quantity | value | measured over |
|---|---|---|
| two instruction-stream seeds of one T2 configuration | 0.002 points (5,511 against 5,418 failing triples of 4.8M) | seeds 0 and 7 |
| the same configuration with a plain four-layer vp head | 0.155 points | seeds 0 and 7 |
| a null intervention's cost | ~8 ms/step, so ~2.3 h at 1M steps | one len=2 cell, 26 local generator workers |

The 10M-step cell over the frozen T1 (36 failing triples) is one seed, so no
floor is measured for it; a claim against it needs its replicate first.

So: unless the specific cell has been replicated, take the floor from the table
whose configuration matches — ~0.01 for a plain-headed cell at 8 anchors, 0.002
for a residual-headed cell at 1M steps — and 0.023 for anything comparing one T1
against another. With a per-cell σ of 0.011, the mean of a two-cell pair moves
by ~0.008, so two such pairs cannot be told apart below ~0.02. No T1
configuration measured is distinguishable from any other.

**The error-budget axes are also len=2-specific.** `sources × immediate ×
depth` crosses the three properties that decide a len=2 slot. Depth cannot
exceed 2 there; a longer chunk admits deeper composition and more ways for two
instructions to share a source, so the axes themselves need revisiting before
the budget means the same thing.
