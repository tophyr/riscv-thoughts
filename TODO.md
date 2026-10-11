# Roadmap

Forward-looking only. Findings and their evidence live in EXPERIMENT_LOG.md;
architecture and rationale in RISCV_DESIGN.md. Nothing here is a history.

## Mission

**T2 is done when its essence is a lossless behavioral description of its
instruction chunk AND two chunks that compute the same function land on the same
point.** Both are required; neither implies the other, and a model can score
perfectly on one while failing the other completely.

**Faithfulness — vp = 0.0.** vp measures whether the chunk's output register
values can be predicted from the essence given the input values. It is the
metric that matters because T2's essence is what a future T3 consumes: a next
tier can only recover what T2 preserved.

**Equivalence — computed equivalents collapse.** "Two token sequences mean the
same thing if and only if they compress to the same point" is the framework's
definition of meaning (WHAT_IS_A_THOUGHT.md), so an essence that separates
`ADDI r1,r1,256; ADDI r1,r1,-16` from `ADDI r1,r1,32; ADDI r1,r1,208` is not a
thought vector regardless of its vp. How well a class collapses tracks how much
the architecture has to *learn* to see it, not whether it needs arithmetic —
measured as the distance between equivalent essences against the distance
between inequivalent ones (`separation`, larger is better):

| equivalence class | distance | separation |
|---|---|---|
| register rename — free by construction | 0.0000 | — |
| commutative operand swap — the pool nearly gives it away | 0.0270 | 46.6× |
| one no-op exchanged for another (`ADD xa,xa,x0` vs `ADDI xd,xd,0`) | 0.0901 | 5.9× |
| additive chains with the same total offset | 0.4288 | 0.8× |

Measured on the len=2 model of milestone 1 unless noted; the rename and
commutative rows come from `scripts/eval.py`'s split `behavioral` row (the
pooled row is not interpretable, EXPERIMENT_LOG finding 15), additive chains
from `probes/additive_fusion.py`, and the no-op row from a 600-pair measurement
on an earlier checkpoint. **A separation gap under ~10× is noise** at these pair
counts: independent samples disagree on the sign of differences that size
(finding 25), so the rows are ordered by magnitude, not finely ranked.

**Nothing trains collapse**, which is the common cause: seeing that `+x0` is the
identity needs no arithmetic at all and still only reaches 5.9×, so "the encoder
cannot compute with operands" does not explain the failure on its own.
vp cannot supply the pressure either — it is not a pairwise objective, so two
distinct points can both predict the same outputs correctly, and for a true
no-op it emits no supervised slot at all and therefore no gradient whatsoever.

**The vp number in a training log is already a held-out number.** `datagen`
generates every batch fresh and the trainer never revisits one, so there is no
train/test split to construct and no memorization to discount — quote the
training log directly. (`scripts/eval.py` exists to score *binding* and the GVN
rows on a fixed corpus, not to de-bias vp.)

## Milestones

1. **[CURRENT] vp = 0.0 at len=2 — not reached.** Current best:
   `--d-out 512` at 1M over the frozen `runs/t1_writers_250k_rank`
   (`runs/t2_len2_d512_s1000000_dep4_vp1_pool16_pma_pewrite_time_rtrank_seedrank_t1rank_writers`), on the
   criterion — exact 32-bit match of the predicted output value — **99.828%**.

   **Two noise floors, and the larger one governs any T1 comparison.** Retraining
   the T2 on a fresh corpus moves this 0.002–0.009 points across the two
   configurations where it was replicated; retraining the frozen T1 beneath it
   moves it **0.023 points** (σ over five same-config T1s), and depth-2 by
   **0.24**. Use ~0.01 for a T2-level claim unless that cell has been
   replicated, and 0.023 for anything comparing T1s. No T1 configuration
   measured is distinguishable from any other (EXPERIMENT_LOG finding 25).

   **Numbers from before the vp target was moved into the behavioral canonical
   frame are not comparable to these** (EXPERIMENT_LOG finding 14). A model is
   scored on the target it trained on and the two differ on 12.5% of rows; on
   the rows where they agree, the change took 99.7383% → 99.8347%.

   The gap is one shape: an immediate survives in the essence when its
   instruction is read in isolation, and degrades when that instruction's
   operand is the other instruction's result. The two composed-immediate cells
   are 63% of all remaining error at 1.4% of the triples. Every population
   carrying no immediate is at or above 99.95%, a shared source alone is at
   99.98%, and composition without an immediate is at 99.95%.

   **What fails is carry propagation through a composed operand, not the
   immediate as such.** A bitwise chain at depth 2 is exact; a chain whose
   second operation propagates carries is at ~15% whatever it consumes. Task 1
   has the matrix. Milestone 2 does not close this — it covers only the subset
   where a composite literal exists to fold.

   **Score this with `probes/vp_budget.py`**, which crosses the three properties
   of an output slot that decide whether it is predicted correctly:

   - **sources** — `cross-instr` when the chunk's two instructions read a common
     register, `distinct` otherwise.
   - **immediate** — whether any instruction in the chunk carries one. R-type
     ALU operations take two registers; I-type take a register and a 12-bit
     literal.
   - **depth** — the slot's dataflow depth: 1 when the instruction that wrote it
     read only chunk inputs, 2 when one of its operands is another instruction's
     result.

   The unit is the (row, output slot, anchor) triple that vp supervises, and
   **err mass** is the cell's share of all failing triples. A rate and a mass
   are different facts: a rare cell can be terrible and carry little of the
   error, a common one excellent and carry most of it.

   | sources | immediate | depth | triples | exact | err mass |
   |---|---|---|---|---|---|
   | distinct | yes | **2** | 16.8k | **93.72%** | **58.2%** |
   | distinct | yes | 1 | 790k | 99.930% | 30.4% |
   | **cross-instr** | yes | **2** | 752 | **80.59%** | 8.1% |
   | cross-instr | no | 2 | 1.5k | 97.79% | 1.8% |
   | distinct | no | 1 | 310k | 99.995% | 0.9% |
   | cross-instr | yes | 1 | 37.6k | 99.984% | 0.3% |
   | distinct | no | 2 | 10.0k | 99.950% | 0.3% |
   | cross-instr | no | 1 | 40.4k | 100.0000% | 0.0% |

   Worst individual opcode cells at depth 2: SLTIU 78.8%, SLLI 80.3%, ADDI
   79.5%, ~1k triples each.

   Every probe's numbers describe the corpus it ran on, and the spread is wide
   enough to have hidden a bug for a phase: an R-only corpus excludes the
   `SLTIU rd,rs,-k` shape entirely, and reading 8.8× lower error there than on
   the mixed corpus was taken as a property of the model (EXPERIMENT_LOG finding
   19). Quote the mixed-corpus figure for statements about the model.

2. **Computed equivalents collapse at len=2.** Two chunks that compute
   the same function must land on the same point. They currently land as far
   apart as two chunks computing different functions: additive chains with the
   same total offset sit **0.4288** apart against **0.3588** for chains whose
   totals differ by 1 — separation **0.8x** — while a register rename sits at
   0.0122. XOR chains are 0.7x. Measured directly, with no fitted readout, by
   `probes/additive_fusion.py`.

   **It overlaps milestone 1 but does not contain it.** Collapsing two literals
   onto their composite would turn `ADDI;ADDI` into the single-literal case the
   head already predicts at 99.9%. It does nothing for `XORI;ADDI` or
   `ADDI;XORI`, where no composite literal exists and the difficulty is the
   carry arithmetic itself — and those score 14.05% and 47.78%, no better than
   the fusible pair's 17.07%. Milestone 1 has to be closed on its own terms.

   **Settle it before chunks get longer.** Every added instruction multiplies
   the equivalences a chunk participates in, so a len=4 model would be learning
   the same wrong geometry at four times the scale.

   No loss, data augmentation or metric in the project has ever covered this
   class: the equivalence manifest contains only single-instruction rewrites.
   Task 2 has the evidence and the decisions to make before building.

3. **vp = 0.0 at len=4, then len=8.** Was 0.244 and 0.461 pre-rank.
   Note `rank` matches the vp target's canonical frame on only 83.7% of len=4 and
   52.9% of len=8 chunks (the residual is dead code, which no causal read-time
   counter can know), so the len=2 mechanism may not transfer intact.
4. **Beyond the happy path.** Everything above is writers-only (R/I-type ALU,
   valid, non-x0). Then: no-write instructions, PC/branches, memory, longer and
   looser chunks.

   **Also strengthen the vp metric itself here.** vp is supervised *and* scored
   against **8 fixed anchor states** (`make_anchor_states(8, seed=0)`, identical
   gen-side, trainer-side, in `compressor/eval.py:410` and in every probe), and
   the head's input features are those same 8 vectors for every row
   (`in_vals_i`, `compressor/train.py`). So `vp = 0` certifies "the output values
   at 8 fixed input points are recoverable from the essence", which is weaker
   than "the essence encodes the chunk's I/O function" — in principle a model
   could satisfy it by storing 8×32 bits per output slot without representing
   operand binding at all. That is accepted as the success metric for the happy
   path: it is cheap and it has so far tracked the mechanisms we care about. It
   is *not* good enough to certify a lossless behavioral description once the
   corpus gets harder. Options when we get there: more anchors, a held-out
   anchor set the head never trained on, or anchors resampled per batch (which
   means shipping them in the RVT batch, i.e. a change to INVARIANT 1).

   **vp scores T1∘T2, not T2.** T2's operand and destination slots come from
   T1's emitted binding (`_t1_predicted_wiring`), never from the tokens, while
   the target is built from the tokens' SSA. When T1's binding disagrees with
   the oracle's, T2 encodes one chunk and is graded against another, and the row
   is unsatisfiable however good T2 is. **"T1 has perfect vp" does not license
   reading the pipeline number as T2's performance** — vp scores `vp_head` on
   T1's essence, while the wiring comes from `in_score_head` /
   `out_score_head` / `live_in_head`, a different loss at a tenth of the weight.
   Measured on the current pair, the two are close: T1's binding matches the
   oracle on 96.6% of len=2 chunks and **92.6% of the residual sits on rows with
   no mismatch at all**, so the pipeline figure is a fair proxy for T2 today.
   That is a measurement, not a property — re-check it with
   `probes/vp_frame.py` after any T1 change, and note the probe must compare
   against the ORACLE's binding, since both it and T1 are behavioral while the
   tokens are syntactic. Separating them properly, if it is ever needed, means
   building the target from T1's emitted wiring rather than from the tokens.
5. **T3.** Sequences of T2 blocks. Prerequisite: decide what a *composed*
   binding is — T3 is the first tier that must combine bindings rather than
   just emit them (WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface").
   Settle that before building, not during.

## Open tasks

Task 1 is the current focus. Task 2 is required for the MVP, overlaps task 1 on
one subset without containing it, and is sequenced after, before chunks get
longer. The rest are queued, not parallel work.

### 1. An immediate does not survive composition

**The single open cause of the len=2 vp residual.** The error lives in the cells
where an immediate meets a **composed operand** — the writing instruction taking
its operand from the other instruction's result rather than from a chunk input
(milestone 1's table):

| condition paired with the immediate | exact | err mass |
|---|---|---|
| composed operand (dataflow depth 2) | 93.01% | 39.5% |
| composed operand AND a shared source | 81.08% | 5.4% |

Each condition **alone** is solved: an immediate alone 99.909%, a shared source
alone 99.980%, composition alone 99.957%. What fails is the combination, and the
common factor is the immediate — the highest-entropy field an instruction
carries, and the only one that has to survive being folded into another
instruction's result.

**The shared-source half of this was a broken target, not an encoder limit**, and
is closed: canonical positions were assigned over live syntactic inputs while the
vp head was fed as though they covered only the behavioral ones, so two chunks
with a bit-identical encoder input carried different targets. That cell went
97.11% → 99.23%. See EXPERIMENT_LOG finding 14 for the mechanism and
`probes/vp_drop_control.py` for the construction that isolates it — a
manufactured collision, because no split of the corpus can see this class of
defect.

**Ruled out** (EXPERIMENT_LOG phase-15 finding 17, each against a ±0.75-point
run-to-run noise floor):

- **the vp head** — no refit, fine-tune, or deeper head beats the trained one;
- **the two-step arithmetic chain** — handing a readout the true intermediate
  value changes nothing;
- **the pool** — the gap is already present in the register slot's own state;
- **gradient share** — `--vp-depth-weight` at 8 and 32 leaves the hard cells
  flat and degrades the rest;
- **the operand scale mismatch** — `--value-norm` puts leaf and composed
  operands on one shell. Re-run at the current config on the corrected target:
  aggregate 99.7474% and the cell 92.97%, both inside noise. It is not a null
  though — see the trade-off below;
- **readout capacity as an aggregate lever** — `--vp-head-depth 4` end-to-end
  leaves the aggregate at 99.7587% and the cell at 92.74%, and costs GVN
  separation (16.7× against 23.9%/24.7×);
- **content crowded out of the op-cell's first layer** — `--content-skip` gives
  the content its own path into the slot and lands inside the noise;
- **capacity** — `d_out` 256 → 512 buys a fixed amount, 512 → 1024 buys nothing
  at either budget and leaves the error cells unchanged to three decimals.

**Ruled out as of the behavioral-frame model** (`probes/vp_frame.py`): the
remaining error in this cell is not a description mismatch. Restricting it to
rows where T1's emitted wiring matches the tokens reads 93.43%, and restricting
it to rows whose target is identical under either canonical frame reads 92.63% —
both at the cell's own 93.01%. So neither T1's binding prediction nor the frame
is what is failing here, unlike the shared-source cell.

**Localised: the immediate is encoded perfectly and destroyed on consumption.**
Six arms of one depth-2 shape, only the literals moving
(`probes/vp_compose_control.py`, 24k triples each; "intermediate" is the first
instruction's own slot, "composed" the second's):

| arm | intermediate | composed |
|---|---|---|
| `ADD ; ADD` | 100.0000% | **100.0000%** |
| `ADD ; ADDI` | 100.0000% | 96.375% |
| `ADDI ; ADD` | 99.946% | **68.196%** |
| `ADDI ; ADDI` | 99.875% | **18.917%** |
| `SLT ; ADDI` | 100.0000% | 99.571% |
| `ADDI ; SLT` | 100.0000% | 100.0000% |

Composition is free when no literal is involved. The producer's own slot carries
its literal at 99.99%+ in every arm, so nothing is lost writing it — it is lost
when that slot is *read* by the second instruction. On the corpus the same
asymmetry holds: a literal in the producer carries 40.4% of depth-2 error
against the writer's 30.0%, and a `cmp` producer, whose intermediate is one bit
wide, sits at ~99.7% for every writer (`probes/vp_depth2.py`).

**What the head produces instead is a near-miss, not a dropout.** On
`ADDI;ADDI`, 96.6% of predictions land within one immediate's width of the
target and only 0.13% drop a literal outright. The essence supplies the two
literals separately; the head has to compute the composite and gets it
approximately.

**What decides it is CARRY PROPAGATION, not the immediate and not
fusibility.** All I-type pairs of the same shape, composed slot scored, with the
mean number of output bits that are constant across the 8 anchors beside each —
bits the head gets without consulting the input at all, and the reason the
`ORI`/`ANDI`/`SLLI` columns cannot be read as successes:

| producer ↓ / writer → | `ADDI` | `XORI` | `ORI` | `ANDI` | `SLLI` |
|---|---|---|---|---|---|
| `ADDI` | 17.07% | 47.78% | 93.58% | 91.78% | 12.37% |
| `XORI` | 14.05% | **98.77%** | 99.90% | 99.66% | 15.43% |
| `ORI` | 46.00% | 99.50% | 99.98% | 99.90% | 48.77% |
| `ANDI` | 48.83% | 99.73% | 99.97% | 99.92% | 46.27% |
| `SLLI` | 67.47% | 77.93% | 99.61% | 99.93% | 63.78% |
| *free bits (`ADDI`/`XORI` producer)* | *0.2* | *0.1* | *16.7* | *15.5* | *16.4* |

The four cells at full difficulty — producer and writer both in
{`ADDI`, `XORI`} — are the interpretable ones, and they order by carries, not by
whether a composite literal exists:

- **bitwise throughout** (`XORI;XORI`) — 98.77%, exact.
- **carry-propagating writer** — ~15% whatever it consumes (`ADDI;ADDI` 17.07%,
  `XORI;ADDI` 14.05%). A shift writer behaves the same way, relocating bits.
- **bitwise writer on a carry-produced intermediate** (`ADDI;XORI`) — 47.78%.

**`ADDI;ADDI` is algebraically the easiest of the four** — one addition of the
input with the single literal `i1+i2` — and scores no better than `XORI;ADDI`,
which genuinely needs two stages. The model does not take the algebraic
shortcut; that is the non-collapse of task 2, and it is a component of this
cell rather than the whole of it.

**The contrast that localises it is depth 1, where `ADDI` is 99.9% exact.** Same
operation, same literal, same output width; the only difference is that the
addend arrives as a canonical input, handed to the head as explicit bits,
instead of having to come through the essence. `XORI;XORI` is exact for the same
reason — output bit *k* is `x_k ^ (i1^i2)_k`, so the head never needs the
intermediate at all.

**The cell is two sub-cases with different constraints, and they trade against
each other.** Two interventions sharing no mechanism — a deeper readout, and
RMSNorm on every value entering a slot — produce the same redistribution
(EXPERIMENT_LOG finding 22):

| arm | seed 0 | seed 7 | `--value-norm` | `--vp-head-depth 4` |
|---|---|---|---|---|
| `ADDI;ADD` | 64.71% | 72.50% | **89.16%** | **90.97%** |
| `ADDI;ADDI` | 17.22% | 16.48% | **6.67%** | **2.52%** |

All four columns are on the PRE-`_actually_modified` oracle (finding 24). The
baseline arms are unchanged by that correction — the current checkpoint reads
68.20% and 18.92%, inside the seed spread — so the comparison stands, but a new
intervention has to be scored against the current baseline, not this table.

- **One literal decoded and combined with values the head is handed as bits**
  (`ADDI;ADD`) was **readout-limited**, and moves ~20 points — far outside the
  7.8-point seed spread.
- **Two literals folded into one** (`ADDI;ADDI`) is reached by nothing, and
  every intervention that helps the first case drives it further down. That is
  the subset equivalence collapse covers, and the only one it covers.

So the residual is not "essence versus head": both bind, on different
sub-cases. **An aggregate that does not move is not evidence that nothing
happened** — score any composed-operand change on the arms of
`probes/vp_compose_control.py`, which is where both interventions showed 15-25
point effects while the aggregate and the depth-2 bucket stayed flat.

**Settle first whether the residual is essence-side or head-side.** vp asks the
head to EVALUATE the chunk's function at the anchor points, so a function the
essence describes perfectly can still score badly if evaluating it is hard for a
shallow readout — and the two available readings contradict (EXPERIMENT_LOG
finding 21). Against the head: every refit and fine-tune on frozen essences
scores below the trained head. For the head: the producer's slot is predicted at
99.99% from the same essence in the same forward pass that fails on the composed
slot, so the description is there, and a head trained end-to-end at depth 4
moves one of the two sub-cases 20 points (finding 22).

The experiment that separates them is to feed the head the intermediate value as
explicit bit features, the way the canonical inputs already are, and retrain.
**It is a diagnostic, not a change to adopt** — handing the head a value the
essence is supposed to carry makes vp stop certifying what milestone 1 claims —
so what it produces is a decision about the acceptance criterion rather than a
fix. `probes/depth_intermediate.py` is the nearest thing already built and found
no gain, but with a fitted readout on the syntactic-frame target at `d_out` 256,
so it does not settle it.

**Other untested levers:**

- **Supervising the intermediate value**, which the SSA oracle computes and the
  RVT batch does not ship. Distinct from feeding it to the head: the intermediate
  slot's own vp is already 99.99%+, so there is nothing there to supervise into
  existence.
- **Two refuted mechanisms scored only against the syntactic canonical frame**:
  a direct linear path from an instruction's content to the slot it writes, and
  weighting vp by dataflow depth. Both were measured at `d_out` 256 / 250k
  against a target whose error composition differs materially from the
  behavioral frame's, so neither refutation holds there. Recover either from git
  history (`--content-skip`, `--vp-depth-weight`). Normalizing values into the
  slot — the third of that group, and the one aimed squarely at the op-cell
  reading its own previous output — is refuted at the current configuration too
  (finding 22).

### 2. Collapse computed equivalences (milestone 2 — required for MVP)

Two chunks computing the same function must land on the same point. They do not,
and the failure is total rather than partial. Distances between essences, where
each chunk is two instructions of one opcode writing one register — so the
essence describes only the composite, the intermediate having been overwritten:

- **fusion** — the same composite reached by a different split of the two
  immediates (`ADDI r1,r1,256; ADDI r1,r1,-16` against
  `ADDI r1,r1,32; ADDI r1,r1,208`). Must be ~0.
- **different** — a different composite. The denominator; without it a small
  fusion distance would only mean the encoder collapses everything.
- **separation** — different / fusion. Must be large.

| family | fusion | different | separation |
|---|---|---|---|
| `ADDI/ADDI`, same total offset | 0.4288 | 0.3588 | **0.8x** |
| `XORI/XORI`, same composite mask | 0.5112 | 0.3612 | **0.7x** |

The same chunk with its registers relabelled sits at 0.0122 (median 0.0001), so
the encoder is intact and the measurement is sound.

**Collapse would also close milestone 1's remaining cell** (see task 1): the
composed-immediate vp error exists because the head is handed two literals and
must compute their composite. An essence that collapses `i1,i2` onto `i1+i2`
hands it the composite instead, reducing the job to the single-literal case it
already does at 99.9%.
`probes/additive_fusion.py` builds the chunks directly and takes a distance —
no fitted readout — and asserts through the emulator that each `fusion` pair is
behaviorally identical and each `different` pair is not. Extend it with a new
family by adding a `combine`/`resplit`/`perturb` triple to `FAMILIES`.

**It is not a T2 deficit — T1 fails the same class.** Single-instruction
essences on the frozen `runs/t1_writers_250k_rank`, each pair with an identical
binding (one live input, same register, same destination), so the whole
difference is in the essence (`probes/t1_scorecard.py`, 2000 pairs per row):

| pair | same function | essence L2 |
|---|---|---|
| `ADD x5,x3,x3` vs `SLLI x5,x3,1` | yes | **0.5696** |
| `ADDI x5,x3,0` vs `{ORI,XORI,SLLI} x5,x3,0` | yes | 0.2794 |
| `ADD x5,x3,x4` vs `ADD x5,x4,x3` | yes | **0.0083** |
| `SLLI x5,x3,1` vs `SLLI x5,x3,2` | **no** | 0.6948 |
| `ADD x5,x3,x4` vs `SUB x5,x3,x4` | **no** | 0.9675 |

Pairs computing the same function sit within a factor of 2 of pairs computing
different ones; the commutative swap, which the architecture makes free because
the pool is a multiset over register slots, sits at 117×. Read the ratio and not
the raw distances — those move 3× across checkpoints with no relation to the
settings (EXPERIMENT_LOG finding 23). **At both tiers, an equivalence collapses when the
architecture cannot represent the difference and otherwise does not.** Any fix
has to be designed for T1 and T2 together; they are one class (`compressor/`),
so a mechanism that works at T1 applies at T2 unchanged. It also means the
validity remint of task 3 should not happen before this is settled, or T1 will
be reminted without it.

**Nothing in the project covers this class.** Equivalences are enumerated in
one place — the manifest in `datagen/generate.py` — and all 15 of its classes
are single-instruction **rewrites**, one operation expressed another way
(`template_instr_counts=[1]`). Everything built on that manifest inherits the
limit: the `--inject-equiv` data augmentation, `eval.py`'s `behavioral` GVN row,
and a pairwise equivalence loss that the trainer carried until commit `7797c81`
and which is recoverable from git history. A **fold** — where seeing the
equivalence requires arithmetic on the operands — appears in none of them.
Accordingly, every checkpoint spanning the current architecture fails it, the
oldest worst at 0.5x, with equivalent chunks further apart than inequivalent
ones.

The manifest's coverage is the limit, not the arithmetic: exchanging one no-op
for another is a single-instruction rewrite needing no computation on operands,
is absent from the manifest, and reaches only 5.9x. Any objective built here
should be scored on that class too, because it isolates "untrained" from
"cannot compute".

**Identity recognition is one capability and it surfaces in three places** —
start here, because it is the best-defined target in either task:

- collapse: exchanging one no-op for another reaches only 5.9x (above);
- T1's geometry: three spellings of the identity (`ADDI x,y,0` against
  `ORI`/`XORI`/`SLLI x,y,0`) sit 0.2794 apart and `ADD x,y,y` against
  `SLLI x,y,1` sits 0.5696 apart, both within 2× of the 0.6948–0.9675 between
  instructions computing *different* functions;
- T1's binding, and hence T2's vp: **every** `dest` misprediction measured is an
  instruction the oracle says modifies nothing, i.e. T1 predicting a write for
  `ADD x5,x5,x0`. That is 6.2% of T2's remaining vp error, arriving through
  task 3's tier interface rather than through anything T2 does.

So this spans tasks 2 and 3, and it is not a weighting problem — T1's binding is
otherwise accurate against the oracle. It is a missing capability: knowing that
`+x0`, `+0`, `|0`, `^0` and `<<0` are the identity.

**vp cannot deliver it.** Two chunks computing the same function have the same
vp target, and two distinct points can both predict it correctly. Nothing in the
loss is pairwise — that was a deliberate design choice, and undoing it is the
decision to make before building.

Open questions, in order:

- **How wide is the class?** Both families tested fail. Check whether any
  computed equivalence collapses, and whether the rewrite classes that DO
  collapse (commutative swap, 0.0290 at 24x separation) do so because they are
  trained on or because the architecture makes them free.
- **Does collapse have to happen in the essence, or in the op-cell?** The
  distances above are essence distances, but the composite is formed one step
  earlier, when the second instruction's op-cell reads the slot the first wrote.
  A cell that computed `i1+i2` there would collapse the pair before any pooling
  — and would fix task 1, which the essence-level distance alone would not.
- **What is the objective?** A pairwise term over generated equivalent pairs is
  the obvious candidate. `--inject-equiv` is a recorded failure at buying
  collapse with data alone — it adds correlated rows without
  adding collapse pressure, since nothing ties an injected variant's essence to
  its base. The manifest would also need multi-instruction classes, which
  requires generating folds rather than rewrites.
- **What does it cost?** Every collapse gain so far has traded against
  discriminativeness (findings 11, 16), and a term that pulls equivalents
  together can pull inequivalents together too. `behav-diff` — pairs the
  transformation did NOT leave equivalent — is the guard, and must stay large.

### 3. Give T1 a validity task

The frozen T1 (`runs/t1_writers_250k_rank`) is a diagnostic trained on a clean
corpus (`--inject-invalid 0`), so it has no validity task and cannot reject
invalid input. That is the only remaining gap between it and a production T1:
it trains against the corrected oracle, at the core's settled settings, with the
`rd==x0` immediate mask applied (`compressor/train.py:281`), and its
single-instruction vp is a 100.0000% exact 32-bit match
(`probes/t1_scorecard.py`). Needed before anything leaves the happy path.

**A candidate T1 can only be judged by training a T2 over it** (~2.4h at
len=2). T1-side metrics do not predict what T2 extracts: six T1s all scoring
100.0000% on their own single-instruction vp yielded T2s spanning 0.059 points,
with their binding accuracies ranked *anti*-correlated to their T2s
(EXPERIMENT_LOG finding 25). Budget for that before proposing a T1 sweep, and
replicate — retraining one T1 moves its T2 by 0.023 points on its own.

Scrutinise when reminting:

- **The binding heads are weighted 0.1 against vp's 1.0.** `in_slot_weight`,
  `live_in_weight`, `live_out_weight` and `out_slot_weight` are all 0.1 while
  `value_predict_weight` is 1.0. T2 consumes the binding as its WIRING, so the
  tier interface is optimised at a tenth of the priority of the essence. T1 is
  nonetheless accurate against the oracle's binding — 3.4% of len=2 chunks
  mismatch and 92.6% of T2's residual error is on rows with no mismatch at all —
  so this is a thing to check, not a known deficit.

### 4. No-write instructions are invisible to the essence

Blocks milestone 3 (~⅓ of the ISA). Content reaches the essence only through a
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
oracle already computes `SELECT_PC` but does not ship it).

**Decide first whether PC advance is part of the behavior**, because it is not
only a branch question and it gates work in two other tasks. A true register
no-op — `ADD x5,x5,x0`, `ADDI x5,x5,0` — changes nothing but the PC, and the
oracle correctly gives it no supervised slot, so vp is silent about it while the
essence still moves (0.1041, milestone 2's table). Two readings, and nothing
currently distinguishes them:

- **PC advance counts.** Then a chunk with a no-op and one without genuinely
  differ, the essence is right to separate them, and vp is *incomplete* — it has
  no PC target at all. `out_pcs` is computed and only a `pc_writes` boolean is
  shipped, so this would mean widening the RVT batch.
- **PC advance does not count.** Then those chunks are equivalent and the 0.1041
  is a collapse failure, which puts it in task 2.

Deferred deliberately: it is a definition, not an experiment, and it should be
settled before either task 2 builds a collapse objective or this task ships a
PC-effect target — each would bake in an answer.

### 5. Overlap the trainer's host-side work with the GPU step

The bottleneck is diagnosed and measured — see the "Throughput" bullet under
"How to run" — and what is missing is the fix: prefetch the next batch's
preparation on a worker thread, or build the dense grid on-device. Worth ~2×
wall-clock on every len=2 run.

## Constraints — do not regress

- **Options do not survive on their own.** Once a better way to do something is
  settled, the worse way gets deleted rather than kept behind a flag: ongoing
  branching complexity is a cost we do not pay if we can pay a one-time
  reshaping cost instead. Breaking changes — including reminting the frozen T1 —
  are acceptable **on the one condition that new and old can be compared and
  shown not to have regressed.** The equivariant core therefore has no settings
  and the trainers expose only dimensions and loss weights; a refuted mechanism
  is recoverable from git history, not from a flag.
- **Exact equivariance.** `state_max_err` must stay `0.00e+00`. Anything fed to
  the pool must be register-indexed.
- **Equivalence (GVN) collapse.** The requirement: two instruction sequences that
  compute the same function must land on the same point. `scripts/eval.py` reports
  it as four distances between essences, and they are not the same kind of
  quantity.

  **rename** compares a chunk against the same chunk with its registers
  relabeled, encoding the relabeled copy with its anonymous tags permuted to
  match. Equivariance is exact by construction, so this reads ~0 for any model at
  all; it is an assertion that the core is intact, not a measure of training
  quality. **behavioral** compares a chunk against a variant with a commutative
  instruction's operands swapped. **distinct** is the separation denominator: the
  distance between unrelated chunks.

  `scripts/eval.py` reports the behavioral distance **split** by whether the
  swap actually preserved the chunk's function, and the split is the thing to
  read. Swapping a commutative op's operands changes the chunk's first-read
  order, and if a later instruction reads one of the swapped registers the
  canonical input positions shift and the two chunks compute genuinely different
  functions. About 2% of generated pairs are non-equivalent in this way; they sit
  at the `distinct` scale and can inflate the pooled mean several-fold. Judge
  collapse on the equivalent subset (`behavioral`); the non-equivalent subset
  (`behav-diff`) should be LARGE, and a model that collapses it is failing to
  distinguish different functions. `behav-mixed` pools both and exists only for
  continuity with older records.

  Measured on the len=2 model of milestone 1 at `--gvn-chunk-len 2`, over two
  corpus draws of ~200 pairs each: genuinely-equivalent pairs 0.0270 and 0.0352
  against a `distinct` mean of 1.26, so separation 47× and 36×, with
  non-equivalent pairs held apart at 0.88 and 0.96. The spread between the two
  draws is the measurement's own noise, not a difference between models — see
  the paragraph below before reading any gap of that size as signal.

  **These ratios resolve large differences only.** The denominator is a mean
  over a few hundred pairs of a near-zero quantity, so it is tail-dominated: two
  corpus draws of one configuration give 46.6× and 41.2×, and `scripts/eval.py`
  and `probes/t2_faithfulness.py` disagree on the SIGN of smaller gaps because
  they build independent pair samples. **Treat a gap under ~10× as no
  evidence**, and rank configurations on vp exact match instead, against the
  floor that matches the comparison — ~0.01 between T2s, 0.023 between frozen
  T1s (EXPERIMENT_LOG finding 25). Phase 14's 0.0238 / 28× is not a floor to
  compare against either: it predates the rename-row fix of finding 12, which
  makes every earlier GVN number incomparable.

- **Magnitude-as-validity — a health check, not a live objective.** The writers
  corpus injects no invalid windows, so every row's validity target is 1 and vp
  consumes `F.normalize(vecs)` anyway; nothing we currently run should produce
  an invalid output. Essences should therefore sit on the unit sphere. A
  regression in `frac<0.9` is a symptom worth chasing, not a semantic failure.
  Real validity supervision is deferred to open task 4.
- **Decode-back / reconstruction is a NON-GOAL this phase.** Do not spend
  effort improving it; do check it has not regressed, with
  `probes/t2_reconstruct.py` at the model's own chunk length. The figure is
  per-instruction and scales with that length, so the two recorded points are
  not comparable to each other: 96.7% opcode / 95.8% immediate at len=2 on the
  model of milestone 1, and 91.5% / 95.4% at len=8 on the len=8 model of
  EXPERIMENT_LOG phase-15 finding 5. (If anyone does pick it up later:
  `scripts/train_decoder.py` is stale and calls a pre-equivariant 2-arg
  `encode()`.)
- **Supervise only behaviorally-live signals.** Dead values have no target; the
  `rd==x0` immediate mask is the existing instance.

## Dead ends — do not re-run

No option exists for any mechanism below. A flag name identifies the arm that
produced the number; recover it from git history to re-measure one.

- **Capacity.** Scaling the model's dimensions and steps together leaves
  aggregate vp essentially unchanged at a fixed chunk length. `d_out` 256 → 512
  is the one exception and it does not repeat: 512 → 1024 gives +0.9 points of
  depth-2 exact match at 250k and −0.7 at 1M, both inside the noise floor, with
  GVN separation slipping 42× → 36×, and it leaves the error cells of task 1
  unchanged to three decimal places. `d_out 512` is adopted; the remaining error
  is not capacity-shaped.
- **Weighting vp by dataflow depth** (`--vp-depth-weight`). Raising composed
  slots' share of the vp gradient from 2.5% to 18% (W=8) and 46% (W=32) leaves
  depth-2 flat at both settings and degrades depth-1 monotonically (overall bit
  accuracy 99.834% → 99.751% → 98.839%). Composed slots are not gradient-starved.
- **A content bypass into the slot** (`--content-skip`, a direct linear path
  from an instruction's content to the slot it writes, so the content does not
  share the op-cell's first layer with decoding the operands). Inside the noise
  floor. The composed instruction's content is not blocked on its way into the
  slot — it is fused with the operand once there.
- **Normalizing values into the register slot** (`--value-norm`, RMSNorm on
  every value as it enters a slot, so a seeded leaf and a computed op-cell
  output share one scale). The scale gap is real and measured — a composed
  operand excites the op-cell's operand weights at 0.08–0.14× a leaf's energy —
  but closing it leaves depth-2 slightly worse (I-type 95.49% → 94.79%).
- **Slot-competition pooling** (`pool_mode=compete`, each register slot
  softmax-assigning across the pooling heads so the heads divide the register
  file instead of piling onto the salient few). No vp benefit, and with
  read-order recorded in the slots it made the essence sensitive to the
  per-chunk random leaf tags of the time.
- **Feeding the op-cell each operand's canonical read position as extra
  scalars** (`op_cell_rank`). A redundant second route to a position the
  rank-seeded slots already supply through the operand values the cell reads,
  and enabling both scores worse than seeding alone.
- **Supplying the pool more ordering information than the per-slot write step.**
  Feeding it read-order, or the whole event block, leaves the shared-source
  deficit at ~83% and costs equivalence collapse; read-order plus canonical
  ranks is the worst on vp (0.0434 against the write-step baseline's 0.0405)
  while paying that cost (behavioral 0.0084 → 0.0115, separation 310× → 224×). The pool cannot see `content`, so it must use
  any channel unconditionally; that is the reason, and it does **not** extend to
  content-conditioned consumers. The op-cell is the one consumer this line does
  not cover — see open task 1.
- **Injecting commutative equivalents (`--inject-equiv`) to train the
  invariance.** Makes it worse, not better: on genuinely-equivalent pairs 0.0110
  → 0.0235 and separation 119× → 54×, with vp 0.0082 → 0.0252. Nothing in the
  loss ties an injected variant's essence to its base (there is no pairwise term,
  by design), so it adds correlated rows without adding collapse pressure while
  reducing effective diversity. Partially confounded — the injected rows displace
  others, so that run saw 545 instructions per batch against 768 — but the
  direction is unambiguous.
- **More training as a fix for commutative collapse.** 250k → 1M moves it
  0.0426 → 0.0373, while an encoder seeded from random tags moves 0.0027 →
  0.0023 over the same 4×; the gap is untouched.
- **Treating tag-sensitivity as the cause of the commutative-collapse
  regression.** Removing the random tags entirely takes
  tag-sensitivity to the architectural floor and leaves collapse unchanged (0.0426
  against `op_cell_rank`'s 0.0366 and a 0.0027 baseline). Both are downstream of
  read-order reaching the essence.
- **Treating the behavioral row as contaminated.** Commute pairs share the vp
  target on 100% of len=1 and 97% of len=2 chunks, so the row measures what it
  claims and the regression is not an artifact of it.
- **Probe methodology.** Several conclusions had to be withdrawn. Always
  calibrate a probe against a number known independently, and compare its score
  against the base rate — a probe scoring at the majority-class frequency has
  learned nothing, and one scoring below the model's own trained head is
  measuring itself.

## Known defects


- **`_behaviorally_equal` (`compressor/eval.py`) is over-strict.** It compares the
  whole 32-register canonical file, including input registers that `_canonical_state`
  merely relocated, so it reports only 9.2% of genuine commutative equivalences as
  equivalent at len=1 (the vp target is identical for 100% of them). vp supervises
  the written slots only, so the comparison should be `out_regs` at `out_slot_regs`.
  It gates the `distinct` pool, where unrelated chunks are almost never equivalent,
  so the effect there is small — but the predicate is wrong wherever else it is used.
- **The test suite does not load any real checkpoint**, so it cannot catch a
  state_dict incompatibility: adding an unconditional `rank_emb` passed all 285
  tests and failed at step 0 of the next run against the frozen T1.

## How to run

- Frozen T1 `runs/t1_writers_250k_rank/encoder.pt` (`run_t2.sh`'s default,
  overridable with `T1=`); corpus `configs/instr_alu_writers.json`. Checkpoints
  trained before the core's settings settled do not load.
- Model knobs on `scripts/train_t2_encoder.py` are dimensions only:
  `--d-out`, `--d-event`, `--op-cell-depth`, `--vp-head-depth`,
  `--n-pool-heads`. `scripts/run_t2.sh` exposes each as an env var (`DOUT`,
  `OC`, `VP`, `NPOOL`) and names the run directory after them, so two cells
  that differ in any of them cannot overwrite each other. The adopted
  configuration, which is the script's default: `d_out=512 op_cell_depth=4
  vp_head_depth=1 n_pool_heads=16`.
  **The core's own settings are not knobs** — read positions are canonical
  ranks, slots are seeded from those ranks, and the pool reads each slot's write
  step, identically at both tiers, with no branch for an alternative.
- **Never edit `scripts/run_t2.sh` while a copy of it is running.** Bash reads a
  script incrementally, so an edit shifts the file offsets under the running
  shell. Doing this mid-sweep produced a spurious `syntax error near 'fi'` and
  re-ran the eval-corpus block's redirect outside its `if`, truncating
  `/tmp/t2_eval_2.rvt` to 0 bytes — which surfaces as `binding: … rows=0` in
  every eval afterwards, GVN rows unaffected because they build their own pairs.
  Wait for the sweep, or edit a copy.
- One training cell, generation offloaded to odin:
  `LEN=2 STEPS=250000 MODES=pma NPOOL=16 bash scripts/run_t2.sh`
- **A replicate needs `GEN_SEED`.** Both datagen pools default to `--seed 0`, so
  re-running a cell otherwise redraws the same instruction stream and measures
  only initialization variance. `GEN_SEED=7 … TAG=s7 bash scripts/run_t2.sh`
  draws a different corpus from the same distribution. It is unrelated to
  `ANCHOR_SEED`, which must stay equal to the trainer's `--anchor-seed`
  (INVARIANT 1). At len=2/1M, replicating the T2 alone spreads 0.002-0.009
  points of overall exact match; replicating the frozen T1 beneath it spreads
  0.023 (EXPERIMENT_LOG finding 25).
- **A run's `t2.pt` appears long before the run ends** — the trainer checkpoints
  periodically. Wait on the trainer process exiting, or on `run_t2.sh` logging
  `final:`; scoring the file as soon as it exists reads a partial model.
  — trains, then runs `scripts/eval.py` and the faithfulness probe. `TAG=<name>`
  suffixes the run dir so a re-run of a recorded cell does not overwrite it.
- **Method: probe first, then train.** Diagnose on frozen checkpoints until the
  diagnosis yields a *falsifiable* hypothesis, and spend GPU time only to test
  that specific mechanism. Round 2 of this phase found in hours what Round 1's
  ten training runs did not.
- **Throughput: GPU utilisation is the primary signal**, with ms/step as a
  *floor* check against the same config's known-good figure. ms/step alone is
  too architecture-dependent to compare across configs, and it says nothing
  about how long the GPU spent waiting. A transient or periodic dip in the
  util reading is a sampling artifact, not a bottleneck — sustained low util is
  the thing to act on, and to act on *before* letting a long run ride.

  **len=2 cells run at ~50% util and the bottleneck is the trainer's host-side
  work, not generation.** Measured during a len=2 cell: local trainer 633% CPU
  (6.3 cores) on per-step prep, odin's 98 workers at ~9.8% each (backpressured,
  not saturated), 64 MB/s over the wire. The GPU step is cheap at d256/len=2, so
  the host prep dominates. Adding remote workers does not help — re-measured at
  `REMOTE_GEN=120`, where the GPU sat at ~37% while odin's load average was ~11
  of 128 cores and the link carried 20 MB/s of a ~118 MB/s budget. A cheaper GPU
  step lowers utilisation further, so utilisation falling after an architecture
  change is not by itself evidence of a regression — and the converse holds:
  `--d-out 512` runs at ~60% against d256's ~40% because its GPU step is more
  expensive, not because anything about the pipeline improved. Fix is overlap (task 5);
  costs ~2× wall-clock on every len=2 cell until then. A run comparing against a
  recorded baseline should keep the pipeline as-is so the corpus mix the FIFO mux
  delivers is unchanged.
- Probes (`probes/`, all read frozen checkpoints, none require training):
  `vp_slot_b_split.py` — operation vs operands, disjoint vs shared sources,
    reported per-bit AND as exact 32-bit match (the acceptance criterion; per-bit
    flatters a broken subset — 81% per-bit on the overlap column was 21% exact);
    `opcell_rank_sufficiency.py` — whether slot B's state holds everything but
    the operand→position association, with a no-rank calibration arm;
    `vp_superposition.py` — one output vs two, and asserts slot values are
    unchanged by the other instruction; `vp_imm_share.py` — R-type vs I-type
    share, immediate bit positions, and error crossed with dataflow depth — the
    mixed-corpus counterpart to `vp_slot_b_split`, which is R-only and so cannot
    see an immediate failure at all; `imm_ceiling.py` — recovers one quantity at
    each stage (T1 essence → T2's written slot → the pool's view of that slot →
    T2 essence) to localize WHERE something is lost; `pool_bottleneck.py` —
    refits a bottleneck for a single task and sweeps its width, to size a
    suspected capacity limit before changing the architecture;
  `vp_budget.py` — **the one to reach for first**: the whole error budget from
    the model's own head, crossed over sources x immediate x depth plus opcode
    and bit position, each bucket reported as a rate AND as its share of total
    error. Takes a corpus config, so the same splits can be scored on the
    population a past claim was measured on;
  `vp_depth2.py` — splits the depth-2 error by WHICH of the two instructions
    carries the immediate (producer, writer, both) and by opcode class, which a
    chunk-level "has an immediate" axis cannot show;
  `vp_compose_control.py` — the same question built rather than found: one
    depth-2 shape, six arms, only the literals moving, plus the producer's own
    slot scored beside the consumer's so a literal lost on consumption is
    distinguishable from one never encoded;
  `vp_frame.py` — whether the essence and the vp target describe the SAME chunk.
    Splits vp by how T1's emitted wiring differs from the tokens' (a dropped
    read, an added one, a swap, a wrong destination) and by whether the
    behavioral and syntactic canonical frames coincide. **The reference is the
    ORACLE's per-instruction binding, not the tokens'** — both it and T1 are
    behavioral, and comparing against syntactic token wiring counts T1
    *agreeing* with the oracle as an error (it inflated T1's apparent share of
    the residual from 7% to 37%). Reach for it when a cell resists every
    capacity lever, which is what a mismatch looks like;
  `vp_drop_control.py` — the same question answered causally: the second
    instruction held identical while only the first varies, so a difference
    cannot be explained by the chunks being harder. Use this shape whenever a
    defect can make two different chunks produce the same encoder input — a
    split of the corpus cannot see one, because the cost falls on the rows that
    do NOT carry the defect;
  `additive_fusion.py` — whether two chunks computing the same function land on
    the same point. Builds the chunks directly and takes a distance, with no
    fitted readout, and asserts through the emulator that the pairs really are
    equivalent;
  `t1_scorecard.py` — scores a T1 ALONE, which nothing else does (`eval.py`'s
    GVN rows need a T2): exact 32-bit match of T1's own vp head split by operand
    type, and the distance between single instructions computing the same
    function against ones computing different functions, every pair sharing an
    identical binding so the whole difference is in the essence. **It verifies a
    T1 is intact; it cannot rank T1s** — every candidate scores 100.0000% on the
    vp half, and the geometry half's raw distances move 3× across checkpoints
    with no relation to the settings, so read its ratios and only against a
    checkpoint's own numbers (findings 23, 25). Pass `n_batches=0` to run the
    geometry half alone, which builds its instructions in-process and so does
    not spawn a local `datagen` that would perturb a training run sharing the
    box;
  `depth_composition.py` — whether a vp deficit is in the essence or the head,
    by refitting and fine-tuning heads on frozen essences; `depth_stage.py` —
    the same quantity recovered at each encoder stage (slot → pool view →
    essence), split by dataflow depth; `depth_intermediate.py` — removes the
    "composition is a harder function" confound by handing the readout the true
    intermediate value;
  `t2_reconstruct.py` and `t2_faithfulness.py` — regression checks only.

  **Scoring GVN: pass `--gvn-chunk-len` matching the model's chunk length.**
  `gvn_collapse` builds its own pairs and defaults to 4, so a len=2 model scored
  without it is measured out of distribution — worth an order of magnitude
  (equivalent-pair distance 0.0712 at chunk_len=4 against 0.0287 in
  distribution, same checkpoint). `scripts/run_t2.sh` passes it.

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
