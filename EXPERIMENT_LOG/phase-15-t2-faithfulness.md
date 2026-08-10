# Phase 15 (IN PROGRESS): making the T2 chunk vector faithful to the instructions it encodes

**Status: open. This is working state, not settled record.** Claims here are
corrected in place as they are disproven, so read it as current belief rather
than as a chronology of what was believed when.

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

**1. It was not capacity, and it was not the T1→T2 handoff.** Width and steps
barely moved vp. A T2 trained on single-instruction chunks reaches vp 0.000, so
one instruction survives the tier boundary perfectly; the loss appears only
when two or more are combined.

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

| len=2, 250k steps | vp | rename dist | commutative dist | separation | off-sphere |
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
  that share one it drops to ~83%. Operation identity survives (opcode
  recovered at 99%+); what fails is knowing *which* input feeds which operand
  — recoverable at ~99.8% on disjoint pairs but only ~51% on overlapping ones.
- **~35% — encoding of immediate values.** Visible even on single
  instructions, concentrated exactly in bits 0–11, the 12-bit immediate field.
  This shrinks with training rather than being structural.

**7. Composition depth is not the problem, and neither is the op-cell.**
Slots written from other slots' results are worse individually but too rare to
matter (~15% of the error). Running the trained model on single-instruction
inputs gives 99.999% on R-type instructions, so the per-instruction
computation is essentially exact.

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
a corrected read-order would help. The cell that would say — `pool_extra=order`
with `read_time_mode=rank` — is **untried**: the `order` run used `stamp`, which
is off-by-one exactly on shared-source pairs, so the pool was handed a corrupted
read-order on precisely the failing subset.

**9. Feeding the pool everything has a real cost.** Handing it the whole event
block improves vp ~17% but **doubles** the distance between behaviourally
equivalent chunks and halves their separation from distinct ones. The event
block carries operand read-order, and a commutative swap changes exactly that
— so the essence stops being blind to a distinction it must be blind to.
`write_time` is the one ordering channel that is free of this.

**10. More training helps `vp` but no longer helps decode-back.** Going 250k →
1M halves vp (0.0404 → 0.0198) while reconstruction stays flat or drifts down.
The two goals have decoupled.

## Decisions taken

- **`pool_extra=write_time` is the adopted setting for new runs.** NB the
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
subset, including its overlap column.** T1 and T2-at-len=1 already reach exact
zero, so the bar is demonstrated rather than aspirational.

## Open threads

1. **Shared-source-register operand binding** — ~65% of the remaining len=2
   error.

   What the two pool experiments (finding 8) share is that the pool never sees
   `content` (`softmax(pool_k(pool_src) · q)` over slots,
   `compressor/model.py:277-295`), so it must use any channel it receives
   *unconditionally*. That one fact explains both halves of the result — why
   nothing helped (read-order is recorded on the slot being *read*, while the
   fact needing it lives on the slot doing the *reading*; associating them is a
   cross-slot join a sum cannot perform) and why `event` cost equivalence
   collapse (a commutative swap changes read-order, and the pool cannot know to
   ignore that for `ADD` but not for `SUB`).

   **The hypothesis this yields: the consumer has to be content-conditioned.**
   The op-cell is the only consumer that sees the opcode and both operands in
   one activation, and it is the only one never modified — it reads just the
   `value` channels (`model.py:204`). Feeding it each operand's canonical read
   rank makes the association local, and *unlike the pool*, it can use operand
   position for `SUB`/`SLT`/`SRL` while symmetrizing it for the commutative ops.
   Conditional vs unconditional use of read-order is the whole claim, and it is
   falsifiable.

   Test order: first a probe (readout given the frozen per-slot state *plus*
   each operand's true canonical rank — if that does not reach ~100% on the
   overlap subset, the information is gone upstream of the pool and the op-cell
   change cannot help either), then one training run scored on vp **and** the
   GVN rows. The failure mode to watch is finding 9 reappearing inside the value
   channels: commutative distance doubling, separation halving.

   `read_time_mode` must be `rank` for this: `stamp` overshoots whenever a
   repeat read leaves a hole, i.e. it is wrong *exactly on the shared-source
   subset that is failing*, and nothing has yet tested `rank` anywhere the
   essence can see it. The secondary candidate, ranked below the op-cell, is the
   untried `pool_extra=order` + `read_time_mode=rank` cell (finding 8) — the
   cross-slot-join argument predicts it still fails and it carries the finding-9
   equivalence risk, but it has never been run.

2. **Immediate encoding** — ~35% of the remaining error. T2 has a built but
   unused immediate head, skipped because a chunk has no single immediate — but
   it has one per instruction.
3. **Decoder + reconstruction loss** — the standing untried lever for goal 2.
4. **Flag cleanup** — decided in principle: **options do not survive on their
   own.** `read_time_mode` collapses to `rank` only (`stamp` is a known wrong
   encoding — it equals the canonical position only when no register is read
   twice); `pool_extra` keeps `write_time` plus whatever open thread 1 adds, and
   drops the recorded dead ends `order`/`event`; `pool_mode` drops `compete`.
   The cost is real and must be paid, not skipped: `stamp` is baked into every
   existing checkpoint including the frozen `runs/t1_writers_250k`, so deleting
   it means reminting T1 and re-running the current-best len=2 config, then
   comparing the new numbers against the recorded ones (vp, the
   `vp_slot_b_split` columns, the GVN rows, `state_max_err`, binding
   accuracies). Breaking changes are acceptable; unverified breaking changes are
   not.
5. **Everything after the Round-1 sweep is len=2 only.** In particular,
   input-ordering recovery sits at chance (0.518) at len=8 and is unexplained.
