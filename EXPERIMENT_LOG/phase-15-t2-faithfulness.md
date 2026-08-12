# Phase 15 (IN PROGRESS): making the T2 chunk vector faithful to the instructions it encodes

**Status: open. This is working state, not settled record.** Claims here are
corrected in place as they are disproven, so read it as current belief rather
than as a chronology of what was believed when.

**Headline:** two ordering channels — per-slot write step into the pool, and
canonical read position as the input registers' leaf identity — take len=2 vp
from 0.3525 to **0.0082** and shared-source operand binding from 21.2% to 99.8%
exact. Adopted config: `--pool-extra write_time --read-time-mode rank
--seed-mode rank`.

## The question

T2 compresses a short sequence of RISC-V instructions (a "chunk") into a single
vector, the **essence**. Two properties have to hold for the project to work,
and each has a metric:

- **`vp`** — can the chunk's *output register values* be predicted from the
  essence (given the input values)? This is the essence's quality as an input
  to a future T3, which will only ever see what T2 preserved. Reported as a
  loss; 0 is perfect, ln(2)≈0.69 is no better than guessing per bit.
- **reconstruction** — can the essence be decoded back into the original
  *instructions*? This is the "express a thought as tokens" half of the
  framework.

Phase 15 asked whether the essence is good enough at either. The setup is a
deliberately easy corpus (`configs/instr_alu_writers.json`: R/I-type ALU only,
always valid, never writing x0) on a frozen, verified T1
(`runs/t1_writers_250k` — single-instruction vp exactly 0.000). Nothing in this
phase is a T1 problem.

**The starting symptom:** a single instruction is encoded perfectly, and
quality falls off as chunks get longer — vp 0.34 / 0.60 / 0.68 at 2 / 4 / 8
instructions, and reconstruction ~67% per instruction, which compounds to ~4%
for a whole 8-instruction chunk. Decode-back did not work.

## What was explored

Two rounds, with different assumptions.

**Round 1 — assume it is a capacity or training problem.** Swept training
steps (200k–1M), vector width (256 vs 512), op-cell depth, the depth of the
readout head that consumes the essence, the number of pooling heads, and an
alternative pooling mode. Also measured how the essences are distributed in the
vector space.

**Round 2 — stop guessing and localise the loss.** Instead of training
variants, probe frozen models: compare what a readout can recover from the
whole-chunk essence against what it can recover from the encoder's *internal
per-register state*, then split that state channel by channel until the
missing piece is isolated.

## What was discovered

**1. Neither the tier boundary nor the model's overall width was the problem.**
T1's essence carries a single instruction perfectly — its own value prediction
is exact and a fitted readout recovers the 12-bit immediate at 100.000% — so
nothing is lost in what T2 is handed. Scaling T2's dimensions and training steps
barely moved vp either. (Width does matter in one specific place, the pool's
per-slot projection; see finding 16. "More width" as a blunt lever is what
failed, not width as such.)

**2. The information survives inside the encoder and is destroyed on the way
out.** At 2 instructions, a readout recovers 80% of the output bits from the
essence but 98% from the encoder's per-register state. So the encoder computes
the right thing and the final pooling step loses it.

**3. The cause: the pooling step is order-blind, and the essence has to
describe an ordered function.** The essence is built by summing over the 32
register slots. Summing is what makes it invariant to register renaming — a
deliberate and load-bearing property — but it also means the essence receives
an unordered bag of results with nothing marking which register was written
first. It must nonetheless encode *"output 1 is this, output 2 is that"*.

**4. One number per register fixes most of it.** Feeding the pool each slot's
**write step** (`pool_extra=write_time`) supplies the missing ordering. It is
safe on both properties the architecture depends on: it is indexed by register
so renaming permutes it (invariance preserved, verified exactly), and a
commutative operand swap does not change which slot was written or when, so
equivalence-collapse is not put at risk. Most of the gain survives with
single-head pooling; multi-head's separate contribution is geometric, keeping
essences on the unit sphere (0.7% off-sphere vs 27.7% with one head).

| len=2, 250k steps (GVN columns pre-metric-fix, finding 12) | vp | rename dist | commutative dist | separation | off-sphere |
|---|---|---|---|---|---|
| before | 0.3525 | 0.0023 | 0.0097 | 268× | 3.7% |
| after | **0.0396** | **0.0014** | **0.0084** | **337×** | **0.7%** |

Nothing was traded away — the equivalence-collapse metrics improved too. With
4× the training: **vp 0.0198**. The gain holds at every length, shrinking as
chunks grow: 8.9× at len 2, 2.5× at len 4, 1.5× at len 8.

**5. Decode-back improved sharply, without being trained for.** At 8
instructions, reconstruction went from 72.9% to **91.5%** per instruction —
roughly **8% → 49%** for a whole chunk. Ordering was a bottleneck shared by
both metrics. For scale, the entire Round-1 sweep (ten training runs) had moved
this figure 66.9% → 72.5%.

**6. What is left at len=2, and where.** Total error is ~0.77 points of bit
accuracy, and it splits into two independent causes:

- **~65% — instructions that share a source register with each other.** On
  pairs with *disjoint* sources the essence is at 99.999%; on the ~12% of pairs
  that share one it drops to ~83% per-bit, which is **21.2% exact 32-bit match**.
  Operation identity survives (opcode recovered at 99%+); what fails is knowing
  *which* input feeds which operand — recoverable at ~99.8% on disjoint pairs but
  only ~49%/~4% (operand 0 / operand 1) on overlapping ones. **Solved by
  read-order in the value path** (finding 11).
- **~35% — encoding of immediate values.** Present even on single
  instructions, concentrated exactly in bits 0–11, the 12-bit immediate field.
  **Solved by per-head pool value projections** (finding 16).

**7. Composition depth costs the most per slot, but starts too rare to
dominate.** Slots whose operand is another instruction's result rather than a
chunk input carry a 30–60× higher error rate than depth-1 slots, on about 2.4%
of slots — so they are ~15% of the total error while the two larger causes
stand. Once those are fixed this becomes the whole residual, which is where
len=2 now sits.

**8. Two attempts to feed the pool more ordering information both failed, and a
third run tested nothing.** `pool_extra=order` (the read-order channel, in the
`stamp` encoding) and `pool_extra=event` (the entire event block) both left the
shared-register deficit at 83%. The reason appears to be that read-order is
recorded on the register *being read*, while the fact needing it is stored on the
slot *doing the reading* — associating the two requires a comparison across
slots, which a sum cannot perform. Supplying the fact does not supply the ability
to use it.

The third run, `read_time_mode=rank`, was `pool_extra=write_time`
(`runs/t2_len2_d256_s250000_dep4_vp1_pool16_pma_pewrite_time_rtrank_writers/hparams.json`),
so the corrected read-order never reached the pool — it changed only the event
channels, which the essence does not see under `write_time`. It was structurally
incapable of moving essence-side operand recovery and says nothing about whether
a corrected read-order would help.

**`pool_extra=order` with `read_time_mode=rank` — the cell that would say — also
fails, on the corrected `rank` encoding.** It is the worst of the four cells on
vp (0.0434 against the `write_time` baseline's 0.0405) and pays the finding-9
cost: GVN behavioral collapse 0.0084 → 0.0115, separation 310× → 224×. **The
pool is therefore exhausted** — every channel it can be fed is spent, and the
cross-slot-join argument predicts exactly this.

**9. Feeding the pool everything has a real cost.** Handing it the whole event
block improves vp ~17% but **doubles** the distance between behaviourally
equivalent chunks and halves their separation from distinct ones. The event
block carries operand read-order, and a commutative swap changes exactly that
— so the essence stops being blind to a distinction it must be blind to.
`write_time` is the one ordering channel that is free of this.

**10. More training helps `vp` but no longer helps decode-back.** Going 250k →
1M halves vp (0.0404 → 0.0198) while reconstruction stays flat or drifts down.
The two goals have decoupled.

**11. An input register's leaf identity should be its canonical read position,
not a random tag.** Each register slot is seeded before the chunk runs, and that
seed is what distinguishes one input from another. Seeding it with the register's
position in the chunk's first-read order — counting only registers read before
they are written — instead of a per-chunk random tag preserves what the seed is
for (two distinct inputs get positions 1 and 2; a register read twice gets one
position twice) while making it deterministic and carrying no register identity,
so equivariance stays exact.

This is what fixes operand binding, and it has to arrive by this route. The
essence pool is a symmetric sum that never sees an instruction's opcode, so any
ordering channel handed to the pool is applied unconditionally — and swapping a
commutative instruction's operands changes read-order without changing what the
chunk computes, so an unconditional consumer stops being blind to a distinction
it must be blind to. Putting the position in the seed instead means the
per-instruction cell reads it as part of the operand values it already consumes,
and that cell sees the opcode, so it can use position for `SUB`/`SLT`/`SRL` and
symmetrize it for `ADD`/`AND`/`OR`/`XOR`.

Measured on chunks of two instructions that read a common source register — the
subset where the deficit lived — exact 32-bit match of the predicted output rose
from 21.2% to 99.8%, and recovery of which canonical input feeds each operand
from ~4–49% to ~98% (`probes/vp_slot_b_split.py`). A probe established this was
worth trying before any training: given the encoder's per-slot state for the
second output register plus the true read positions, a readout predicts that
register's value at 97.4% exact against 37.4% without them, so the information
was already present and only the association was missing
(`probes/opcell_rank_sufficiency.py`). Feeding the positions to the
per-instruction cell as extra inputs, rather than through the seed, reaches the
same binding result but scores worse elsewhere and is redundant with seeding.

Chunks of 2 instructions, 1M steps, `--gvn-chunk-len 2`, with commute pairs split
by whether the swap preserved the chunk's function (`scripts/eval.py` reports
this split):

| | random-tag seed, no read-order | **canonical-rank seed** | rank seed + injected commutative pairs |
|---|---|---|---|
| vp | 0.0198 | **0.0082** | 0.0252 |
| rename (architecture assertion) | 0.0002 | 0.0005 | 0.0005 |
| tag-invariance | 0.0006 | 0.0005 | 0.0004 |
| commute pairs, same vp target | 0.0021 | 0.0110 | 0.0235 |
| commute pairs, different vp target | 0.0003 | 0.8789 | 0.9144 |
| separation on true equivalences | 579× | 119× | 54× |

**What recording read-order costs.** A state that records no read-order cannot
represent the difference between `a+b` and `b+a`, so invariance to a commutative
swap is structural. A state that records it can, so the same invariance becomes
something the training signal must supply, and it does so imperfectly:
genuinely-equivalent chunks sit 5× further apart. Neither more training (250k →
1M moves it 0.0426 → 0.0373, while the random-tag encoder moves 0.0027 → 0.0023
over the same 4×) nor injecting commutative variants into the corpus closes the
gap — the latter makes it worse. Accepted, for two reasons: the random-tag
encoder maps chunks computing genuinely *different* functions to 0.0003, i.e. it
cannot separate them at all, which is the same information loss this phase
exists to remove; and 119× separation is above the 28× Phase 14 validated the
architecture on.

**12. `gvn_collapse`'s rename row was measuring tag-invariance.** `enc()` drew
fresh random tags per call, so base and twin got independent tags while the
comment above it claimed they were pinned. Since equivariance is exact, that row
could only ever be nonzero because of tags. It now permutes the twin's tags by
the same relabeling (making it an architecture assertion that reads ~0) and
reports tag-invariance as its own row. **Every GVN number measured before this is
not comparable**, including Phase 14's.

**13. `rank` counts registers read BEFORE written.** Counting reads of in-chunk
results consumes a rank and pushes every later input off by one — the `stamp`
bug with dataflow as the trigger. Agreement with the frame the vp target is
built in: 96.6% of len=2 chunks, 83.7% at len=4, 52.9% at len=8. The residual is
dead code, which no causal read-time counter can know, so `rank` is exact on the
shared-source subset and approximate at len=8 by construction.

**14. The canonical-frame caveat is common, and harmless.** Canonical positions
cover all live syntactic inputs while `in_slot_regs` lists only the behavioral
subset, so a dropped input shifts every later slot and the vp head is fed a value
the target never used: 5.9% of len=2 chunks, 14.8% at len=4, 26.6% at len=8, 0%
at len=1. Measured on the best len=2 checkpoint, the shifted rows score 99.09%
per-bit / 96.7% exact against 99.21% / 94.5% for the rest — an input is dropped
precisely when the output does not depend on it.

**15. A pooled commutative-swap distance is not interpretable.** Swapping a
commutative op's operands changes first-read order, and therefore the canonical
function, whenever a later instruction reads one of the swapped registers — 2% of
pairs at len=2. Those sit at the `distinct` scale and inflate a pooled mean
several-fold (0.0102 equivalent-only against 0.0373 pooled, from 6 pairs in 200).
`gvn_collapse` reports the split. The equivalence label behind it compares the
values value-prediction is supervised on — the canonical register file at the
WRITTEN slots — because comparing the whole 32-register file instead calls two
chunks different whenever they merely relocated inputs to different canonical
positions, which is true of 90.8% of genuine commutative equivalences at len=1.

**16. The pool gave every attention head ONE shared value projection, capping
what the essence could carry about any single slot at `d_value / n_pool_heads`.**
The essence is `n_pool_heads` attention-weighted sums over the 32 register
slots. Each head had its own query, but all of them drew from one `pool_v` of
width `d_value / n_pool_heads` — 16 dimensions at d_value=256, n_pool_heads=16.
A slot therefore reached the essence through 16 projected values plus its
`n_pool_heads` attention logits, and no more: **the essence was 256 wide and its
capacity for any one slot was 16.** Standard multi-head attention gives each
head its own value projection; the shared one was a deviation.

Giving each head its own slice of a `d_value`-wide projection raises a slot's
image in the essence from 16 dimensions to 256. It also removes a standing
tension — under the shared projection, adding heads to separate more written
slots narrowed what each could carry, and the two now scale independently.

Measured at len=2/1M against the same config with the shared projection: vp
**0.0080 → 0.0021**; per-bit accuracy of the predicted output value 99.762% →
**99.939%** at len=2 and 99.083% → **99.999%** at len=1; the model's own value
prediction on I-type single instructions 65.9% → **99.954%** exact 32-bit match.
Immediates fall from 61.7% of the len=2 residual to 39.4%, and every remaining
category is now dominated by dataflow depth. The essence also sits better on
the unit sphere (1st-percentile norm 0.924 → 0.977, none below 0.9).

**Single-instruction chunks are lossless again, and the scope condition is part
of the result.** A T2 trained at len=1 with 16 pool heads and per-head
projections scores 100.000% exact 32-bit match on both R-type and I-type, at
`value_pred = 1.3e-06`
(`runs/t2_len1_d256_s250000_dep4_vp1_pool16_pma_pewrite_time_rtrank_seedrank_phv_writers`).
The earlier len=1 result that this restores was measured at **one** pool head
(`runs/t2_len1_d256_s200000_dep1_vp4_writers`, 2.7e-06), where the shared
projection is the full `d_value` and the ceiling described above does not exist
— so it stopped being true when pool heads were added for an unrelated reason,
and nothing detected that. Any future change to `n_pool_heads` or to the pool's
projections invalidates this number and requires re-running it.

The cost is looser equivalence collapse: the distance between commutatively
equivalent chunks rises 0.0102 → 0.0287 against a `distinct` mean of 1.28, so
separation falls 128× → 45× — the same direction as finding 11's tradeoff, a
more discriminative essence collapsing equivalents less, and still above the 28×
Phase 14 validated on. Operand binding is unchanged (`in_slot` 0.979 → 0.980;
the shared-source column of `vp_slot_b_split` 99.806% → 99.765%, within that
probe's noise).

## Decisions taken

- **The pool reads each slot's write step (`pool_extra=write_time`).** NB the
  constructor and CLI default is still `none`, which reproduces the
  pre-ordering-fix model; new runs must pass the flag explicitly.
- **Feeding the pool the full event block is rejected** on the equivalence-
  collapse cost (finding 9), despite its vp gain. Any future fix must keep
  read-order out of the invariant essence.
- **The op-cell and vector width are ruled out** as targets (findings 1, 7).
  Effort moves to how the essence is *assembled*, not how much capacity it has.
- **"Spread the essence over more dimensions" is abandoned as a goal.** The
  ordering fix *reduced* effective dimensionality (26→15) while improving
  reconstruction 19 points, contradicting the Round-1 theory that
  concentration was the pathology.
- **Seed register slots from canonical read position, count read positions as
  ranks over registers read before written, and give the pool each slot's write
  step** (`--seed-mode rank --read-time-mode rank --pool-extra write_time`).
  Feeding the positions to the per-instruction cell as extra inputs
  (`--op-cell-rank`) is a second route to the same binding result and redundant
  with seeding. The 5× commutative-collapse cost is accepted on the evidence in
  finding 11, not waived.
- **Report vp discretely.** `probes/vp_slot_b_split.py` prints exact 32-bit match
  alongside per-bit; 81% per-bit on the overlap column was 21% exact, so per-bit
  alone made a mostly-broken subset look nearly solved.
- **Decode-back is explicitly OUT OF SCOPE for this phase** — the objective is
  vp. Finding 10 says only that if it is picked up later, more discriminative
  training will not deliver it; it would need its own loss. Until then the
  requirement is simply not to regress it (91.5% / 95.4% at len=8).
- **Diagnosis before architecture.** Round 2's probes found in hours what Round
  1's ten training runs did not. Localise with probes on frozen checkpoints
  first; train only to test a specific mechanism.

## Dead ends (do not re-run)

- **Capacity and training budget.** Wider vectors (256 vs 512) and 5× steps
  leave vp essentially unchanged at a given chunk length.
- **`pool_mode=compete`** (slot-competition pooling, an attempt to force the
  pooling heads to divide the register file between them). Tested in both the
  original and the current configuration: no vp benefit, and with ordering
  enabled it makes the essence sensitive to the anonymous per-chunk register
  tags — rename distance 0.0251 against 0.0014 — which it must not be.

## A methodological note

Several conclusions in this phase were drawn from probes — small readouts
trained on a frozen model to test what a representation contains — and several
had to be withdrawn. Two failure modes recurred:

- **A probe scoring below the model's own trained head is measuring the probe,
  not the model.** Under-powered or data-starved probes were repeatedly read as
  evidence of missing information.
- **A probe scoring near the majority-class frequency has learned nothing.** On
  a task where 88% of cases share one answer, an 88% score means the readout is
  answering by default; the minority case must be trained or evaluated
  separately.

Both are cheap to guard against: calibrate every probe against a quantity
already known independently, and check its score against the base rate before
believing it.

Two probe designs paid for themselves repeatedly, and are worth reaching for by
default:

- **To locate a loss, measure the SAME quantity at every stage of the pipeline
  and find where it drops.** Exact recovery of the 12-bit immediate by a fitted
  readout, stage by stage: T1's essence 100.000% → T2's written register slot
  99.3% → the pool's view of that slot 79.2% → T2's essence 76.9%
  (`probes/imm_ceiling.py`). The last two numbers agreeing is what rules out the
  attention weights as the culprit and points at the projection underneath them.
  A single end-to-end number cannot do this, and three separate hypotheses about
  the cause were refuted for free by one measurement.
- **To size a suspected bottleneck, refit it from scratch for one task and sweep
  its width.** Reproducing the pool's projection as a trainable bottleneck of
  varying width (`probes/pool_bottleneck.py`) gave 8→77.2%, 16→97.1%, 32→99.3%,
  256→99.6% exact 32-bit match on I-type, R-type at 100% throughout. That
  separates two explanations one number would have conflated: the deployed
  16-wide projection is too narrow *and* it was performing below even its own
  16-wide ceiling because it was multiplexing everything else. Both are answered
  by widening; neither would have been answered by more supervision, which is
  what the standing hypothesis (wire up T2's unused `imm_head`) proposed.

Both probes are forward passes plus a small fit — minutes, against the hours a
training run costs to test the same hypothesis wrongly.

## What `vp = 0` does and does not certify

vp is supervised *and* scored against **8 fixed anchor states**
(`make_anchor_states(8, seed=0)` — the same set in the generator, the trainer,
`compressor/eval.py:410`, and every probe), and the head's input features are
those same 8 vectors for every row (`in_vals_i`, `compressor/train.py`). So
`vp = 0` certifies that the output values **at 8 fixed input points** are
recoverable from the essence. That is weaker than "the essence encodes the
chunk's I/O function": a model could in principle satisfy it by storing 8×32
bits per output slot without representing operand binding at all.

Accepted as the success metric for the happy path — it is cheap and it has so
far tracked the mechanisms under investigation. It is not adequate to certify a
lossless behavioral description once the corpus gets harder; strengthening it
(more anchors, a held-out anchor set, or per-batch resampled anchors shipped in
the RVT batch) is queued against TODO.md milestone 3.

**Acceptance criterion for "len=2 solved":** vp trains as a per-bit BCE, so the
loss value is asymptotic and never literally 0. The output is discrete, so score
it discretely — **100% exact 32-bit match on the `probes/vp_slot_b_split.py`
subset, including its overlap column.** T1 reaches exactly that on both operand
types, so the bar is demonstrated rather than aspirational.

**That criterion covers operand binding only, and must be read alongside a
mixed-corpus score.** `vp_slot_b_split` generates from `configs/instr_r_only.json`,
so its chunks carry no immediates at all: it was at 99.8% on its overlap column
throughout the period when the model was recovering I-type outputs at 65.9%, and
it did not move when that was fixed. `probes/vp_imm_share.py` scores the mixed
training corpus and splits the error by operand type, which is what makes an
immediate-encoding failure visible. A metric that cannot see a failure is not
evidence of its absence.

## Open threads

1. **Dataflow composition** — the whole remaining len=2 vp error, now that
   ordering, operand binding and immediate encoding are each closed. Output
   slots whose operand is another instruction's result score 98.6% (I-type) and
   99.3% (R-type) per-bit against 99.95%+ for slots reading only chunk inputs.
   This is the case where the op-cell must read back its own output, so the
   cell's output geometry has to be legible to its own input — a demand the
   depth-1 case, where every operand is a freshly seeded leaf, never makes.

2. **Decoder + reconstruction loss** — the standing untried lever for goal 2.
3. **Flag cleanup** — decided in principle: **options do not survive on their
   own.** `read_time_mode` collapses to `rank` only (`stamp` is a known wrong
   encoding — it equals the canonical position only when no register is read
   twice); `seed_mode` collapses to `rank`; `op_cell_rank` is deleted, being a
   redundant second route to the same read positions; `pool_extra` keeps `write_time` and drops the recorded
   dead ends `order`/`event`; `pool_mode` drops `compete`.
   The cost is real and must be paid, not skipped: `stamp` is baked into every
   existing checkpoint including the frozen `runs/t1_writers_250k`, so deleting
   it means reminting T1 and re-running the current-best len=2 config, then
   comparing the new numbers against the recorded ones (vp, the
   `vp_slot_b_split` columns, the GVN rows, `state_max_err`, binding
   accuracies). Breaking changes are acceptable; unverified breaking changes are
   not.
4. **Everything after the Round-1 sweep is len=2 only.** In particular,
   input-ordering recovery sits at chance (0.518) at len=8 and is unexplained.
