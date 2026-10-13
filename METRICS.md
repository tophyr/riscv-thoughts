# Metrics

What each measurement means, what it does and does not certify, how far it
moves when nothing has changed, and the conditions under which a number from it
is true. Every rule here exists because a figure was once quoted without it and
a wrong conclusion followed.

Measured values live elsewhere: a model's current score is in TODO.md, and the
evidence behind a result is in EXPERIMENT_LOG.md. This file is the instrument,
not the readings.

---

## vp — value prediction

**What it measures.** Given the chunk's input values, can its output register
values be recovered from the essence? The head reads `(essence, canonical
input-slot values)` and predicts each output register's canonical value. It is
the metric that matters because the essence is what a future T3 consumes: a next
tier can only recover what this one preserved.

**The head's depth is part of the criterion, so quote it with the number.**
vp scores the essence composed with a readout, and that readout is the only
consumer a T2 essence has — T3 does not exist yet — so its capacity is an
implicit claim about what the next tier will be able to read. `vp = 0` at a
deeper head is a strictly weaker statement than at a shallow one, because
capacity granted to the head is structure the representation is no longer
required to expose, and that compounds at every tier of a hierarchy whose
premise is cheap coarse checks. The adopted cell uses one hidden layer of 1024
(`run_t2.sh` VP=1); depth 4 was measured and trades rather than gains.

**Score it as exact 32-bit match of the predicted output value, never
per-bit.** The output is discrete, and per-bit flatters a broken subset badly:
96% per-bit has been 74% exact, and 81% per-bit on one probe column was 21%
exact. The loss value itself is a per-bit BCE and therefore asymptotic — it
never reaches 0 however good the model is, so it is a training signal and not an
acceptance criterion.

**What counts as solved.** The loss is asymptotic, so "solved" at a given chunk
length means **100% exact 32-bit match on `probes/vp_budget.py`**, which scores
the model's own head over the training corpus and crosses the three properties
that decide a slot. `probes/vp_slot_b_split.py` is a second, narrower bar on
operand binding alone: it generates from an R-type-only corpus, so it cannot see
an immediate failure at all — it sat at 99.8% on its overlap column while the
model was recovering I-type outputs at 65.9%, and did not move when that was
fixed. Read it as a binding check beside the budget, not as the criterion.

**A vp number in a training log is already held out.** `datagen` generates every
batch fresh and the trainer never revisits one, so there is no train/test split
to construct and no memorization to discount — quote the training log directly.
`scripts/eval.py` exists to score *binding* and the collapse rows on a fixed
corpus, not to de-bias vp.

### What `vp = 0` certifies, and what it does not

vp is supervised *and* scored against **8 fixed anchor states**
(`make_anchor_states(8, seed=0)`, the same set gen-side, trainer-side, in
`compressor/eval.py` and in every probe), and the head's input features are
those same 8 vectors for every row. So `vp = 0` certifies that the output values
**at 8 fixed input points** are recoverable from the essence. That is weaker
than "the essence encodes the chunk's I/O function": a model could in principle
satisfy it by storing 8×32 bits per output slot without representing operand
binding at all.

Accepted for the current happy-path corpus — it is cheap and it has so far
tracked the mechanisms under investigation. It is not adequate to certify a
lossless behavioral description once the corpus gets harder. Strengthening it
means more anchors, a held-out anchor set the head never trained on, or anchors
resampled per batch — the last requires shipping them in the RVT batch, i.e. a
change to CLAUDE.md's INVARIANT 1.

**It measures the head's arithmetic as well as the essence's content, and the
head's arithmetic is not the limit.** The encoder never sees a register value —
its inputs are each instruction's opcode and immediate and the wiring — so a
register slot holds a symbolic description of the function that produced it and
the head must EVALUATE that description at the anchor points. vp scores the
composition.

What the composition costs is measured, and it is small where the quantity the
head needs is actually present. The head adds three given 32-bit values at
100.0000% and decodes-and-adds one constant at a single instruction at
100.0000%; on a composed chunk its prediction equals the known summands plus
some plausible 12-bit constant on 95–99.6% of triples, so the arithmetic holds
and only the scalar is wrong. Across the composed ADD-family arms its accuracy
tracks a LINEAR readout of the constant it needs, to within 4 points on two of
three arms. So a vp residual on composed operands points at a quantity the
essence does not carry linearly, not at a weak readout — and **the way to
attribute one is to identify the quantity the head needs and measure how
accessible it is**, with `probes/compose_readback.py` for which part of a
composed slot's description is present.

Do NOT attribute a residual by reconstructing the instructions and executing
them. That measures instruction reconstruction, which the project already does
better elsewhere (`probes/t2_reconstruct.py`), and then answers the question
with an emulator — if re-executing the chunk is allowed, the compressed vector
is not being tested at all.

**vp scores T1∘T2, not T2.** T2's operand and destination slots come from T1's
emitted binding (`_t1_predicted_wiring`), never from the tokens, while the
target is built from the tokens' SSA. Where the two disagree, T2 encodes one
chunk and is graded against another, and the row is unsatisfiable however good
T2 is. "T1 has perfect vp" does not license reading the pipeline number as T2's:
vp scores `vp_head` on T1's essence, while the wiring comes from
`in_score_head` / `out_score_head` / `live_in_head`, a different loss at a tenth
of the weight. Re-check the gap with `probes/vp_frame.py` after any T1 change,
and note that probe must compare against the ORACLE's per-instruction binding,
not the tokens' — both it and T1 are behavioral, while the tokens are syntactic.
Separating the two scores properly means building the target from T1's emitted
wiring rather than from the tokens.

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
  pooled mean several-fold. **Judge collapse on the equivalent subset.**
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
41.2×, and `scripts/eval.py` and `probes/t2_faithfulness.py` disagree on the
*sign* of
smaller gaps because they build independent pair samples. The instability is
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
essence anyway. Essences should sit on the unit sphere, and a regression in
`frac<0.9` is a symptom to chase rather than a semantic failure.

**Reconstruction** (decoding the essence back to instructions) is an explicit
non-goal — check it has not regressed with `probes/t2_reconstruct.py` at the
model's own chunk length, nothing more. It is per-instruction and compounds over
a chunk, so two lengths are not comparable.

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
one thing (`probes/vp_drop_control.py` is that shape).

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
`runs/t1_writers_250k_rank`, on `configs/instr_alu_writers.json`. The runs
behind each row are in EXPERIMENT_LOG, "T2 faithfulness" (finding 25); this
table is the operative copy, that is the evidence.

| quantity | value | measured over |
|---|---|---|
| retraining the T2 on a fresh corpus moves exact match | 0.002–0.009 points | two configurations, each replicated |
| retraining the frozen T1 beneath it moves exact match | 0.023 points (σ) | five same-config T1s |
| the same, on depth-2 exact match | 0.24 points | the same five |
| spread between two configurations' own floors | 20× | the two above |
| a null intervention's cost | ~2 h | one len=2 cell at 1M steps, 7 ms/step solo |

So: use ~0.01 for a T2-level claim unless that specific cell has been
replicated, and 0.023 for anything comparing one T1 against another. No T1
configuration measured is distinguishable from any other.

**The error-budget axes are also len=2-specific.** `sources × immediate ×
depth` crosses the three properties that decide a len=2 slot. Depth cannot
exceed 2 there; a longer chunk admits deeper composition and more ways for two
instructions to share a source, so the axes themselves need revisiting before
the budget means the same thing.
