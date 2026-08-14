# Phase 15 (IN PROGRESS): making the T2 chunk vector faithful to the instructions it encodes

**Status: open. This is working state, not settled record.** Claims here are
corrected in place as they are disproven, so read it as current belief rather
than as a chronology of what was believed when.

**Headline.** Four fixes — per-slot write step into the pool, canonical read
position as the input registers' leaf identity, per-head pool value projections,
and building the vp target in the behavioral rather than the syntactic canonical
frame — take len=2 exact 32-bit match of the predicted output value to
**99.744%**, from 99.738% on the rows where the last of those leaves the target
unchanged. Adopted config: `--d-out 512 --pool-extra write_time
--read-time-mode rank --seed-mode rank`; best model
`runs/t2_len2_d512_s1000000_dep4_vp1_pool16_pma_pewrite_time_rtrank_seedrank_behavframe_writers`,
replicated at `…_behavframe_s7_writers`.

**Aggregates do not span the frame change** — a model is scored on the target it
trained on, and the two targets differ on 12.5% of rows. Compare on the rows
where they agree (finding 14).

**What remains is one shape: a composed immediate.** The writing instruction
takes an operand from another instruction's result *and* carries a 12-bit
literal: 93.01% exact, 39.5% of all remaining error at 1.4% of the triples.
Composition alone is at 99.957% and an immediate alone at 99.909%. Separately,
the essence does not collapse computed equivalences at all: two chunks that
compute the same function sit as far apart as two that do not (finding 18).

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
stand. Once those are fixed they become the largest single cause but not the
only one: ~32% of the residual, against ~46% still sitting on R-type depth-1
slots (finding 17).

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
from 21.2% to 99.8%. **That measurement is on an R-only corpus and holds there:
re-scored on the current best model, all-R chunks with a cross-instruction
shared source reach 99.939%. It does not hold once an immediate is present in
the chunk — the same category reads 96.781% there, and that single cell is 47%
of the model's entire remaining error (finding 19).** The rest of this finding
reads on its own terms; only the word "solved" was too broad.

Before the caveat, the numbers as measured: and recovery of which canonical input feeds each operand
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

**14. The vp target has to be built in the BEHAVIORAL canonical frame. Built in
the syntactic one, it is not a function of what the encoder is shown.**

The canonical frame is what makes the target register-name-free. `out_regs` is
not "x9 = x5 + 100" but "x9 = (the k-th input this chunk reads) + 100":
`_canonical_state` re-sources each input from anchor position 1, 2, … in read
order before executing, so two chunks computing the same function agree whatever
registers they name.

Which inputs get numbered decides everything, and two lines of
`precompute_chunk` disagreed about it. Positions were assigned over the **live
syntactic** inputs, while `in_slot_regs` — which decides what the head is fed —
lists only the **behavioral** ones. `t2_value_predict_loss` then feeds input slot
*k* the value at position *k+1*, asserting that behavioral input *k* sits at
position *k+1*. That is false whenever a live-but-non-behavioral input precedes
it, which is **13.1% of len=2 chunks** (7.1% at len=1, 21.4% at len=4, 31.7% at
len=8).

The encoder is never given such a read. T1's binding heads are behavioral, so
`_t1_predicted_wiring` correctly omits it, and T2's own read-rank counter never
sees it. So the target moved on evidence the encoder does not have:

```
SLTIU x7, x5,  -300 ; ADDI x9, x5, 100      target x9 = 588246066
SLTIU x7, x11, -300 ; ADDI x9, x5, 100      target x9 =  47830916

T1-emitted wiring, both:  in0 [0,5]  in1 [0,0]  out [7,9]
essence cosine = 1.000000
```

A 12-bit immediate sign-extends to a huge UNSIGNED value, so `SLTIU rd,rs,-300`
is the constant 1 and its source is read but irrelevant. In the first chunk `x5`
is input #1; in the second, `x11` takes position 1 and `x5` becomes #2. The
encoder cannot tell them apart, so **two identical inputs demanded different
answers** and the head satisfied whichever was commoner — the shifted case, 15×
more frequent, which it can serve by memorizing the fixed anchor constant.

**Why the natural measurement says the opposite.** Splitting rows by whether
their own frame shifted shows the shifted rows scoring BETTER (99.835% against
99.754% exact). That reading is reproducible and useless: the shifted rows are
the ones the model learned, and the cost is paid by the *unshifted* rows that
collide with them. A split on "does this row have the defect" is structurally
blind to a defect whose unit is a **pair** of rows. Manufacturing both sides of
the collision is what exposes it — the second instruction held identical while
only the first varies (`probes/vp_drop_control.py`, 24k triples per arm):

| first instruction | before | after |
|---|---|---|
| `SLTIU rdA,rs,-k` — drops the scored slot's own source | **1.06%** | **100.0000%** |
| `SLTIU rdA,rsB,-k` — drops an unrelated register | 99.99% | 100.0000% |
| `SLTI rdA,rs,+k` — same shape, genuinely depends on `rs` | 99.99% | 100.0000% |
| `ADDI rdA,rs,+k` | 99.71% | 99.9917% |

**The fix** is to spend canonical positions on the behavioral inputs. Finding
which inputs are behavioral needs a frame to perturb within, so
`precompute_chunk` builds the syntactic frame, uses it only to compute
`input_mags`, and re-executes in the behavioral frame (skipped when the two
coincide, which is the common case). `out_regs` is unaffected by how a
non-behavioral input is sourced, so GVN invariance is preserved; the oracle,
`in_slot_regs`, T1's emitted wiring and T2's read ranks then share one frame.

**Measured across the frames.** The two frames give byte-identical targets on
the 87.5% of triples with no live-but-non-behavioral input, and that population
is the only fair comparison — an aggregate over everything compares two
different tasks. On it, at the same config and budget (len=2, `d_out` 512, 1M):

| identical-target rows | syntactic frame | behavioral frame |
|---|---|---|
| all | 99.7383% | **99.8347%** |
| `cross-instr \| imm \| depth 1` | 97.196% | **99.307%** |
| `distinct \| imm \| depth 2` | 92.180% | 92.626% |
| `same-instr \| no-imm \| depth 1` | 99.544% | 99.938% |
| `cross-instr \| no-imm \| depth 1` | 99.980% | 99.980% |

37% less error, better in every cell that moved. The whole-corpus aggregate
barely moves (99.757% on the old target against 99.744%/99.753% on the new,
two seeds) because each model is scored on its own convention — the syntactic-
frame model reads 96.97% against the corrected target. **Do not compare
aggregates across this change**; compare the identical-target rows.

The two seeds also put the run-to-run noise floor at 1M steps at **±0.01 points
of overall exact match**, and within 0.1 points on every cell above 1k triples —
far tighter than the ±0.75 recorded at 250k in finding 17.

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

**17. Only one lever moves the remaining error, and it is width in the value
path — once.** Six configurations trained at len=2/250k, each differing from the
recorded baseline in one knob, scored on the acceptance criterion (exact 32-bit
match of the predicted output value) and split by dataflow depth. A slot is at
*dataflow depth 2* when the instruction that wrote it read another
instruction's result rather than a chunk input; `d_out` is the register-state
value width, which sizes one slot, the op-cell's hidden layer, and the pool's
per-head slice together.

| len=2, 250k | depth-2 exact | depth-1 exact | overall per-bit | GVN behavioral |
|---|---|---|---|---|
| baseline (`d_out` 256) | 73.78% | 99.29% | 99.834% | 0.0353 |
| *the same config re-run* | *74.53%* | *99.31%* | *99.831%* | *0.0352* |
| `--vp-depth-weight 8` | 74.95% | 98.29% | 99.751% | 0.0331 |
| `--vp-depth-weight 32` | 72.83% | 85.05% | 98.839% | 0.0344 |
| `--value-norm` | 72.24% | 99.16% | 99.814% | 0.0318 |
| `--content-skip` | 72.88% | 99.32% | 99.831% | 0.0360 |
| **`--d-out 512`** | **83.31%** | **99.74%** | **99.900%** | 0.0316 |

The italic row is the run-to-run control — the identical configuration trained
again on a freshly generated corpus — and it puts the noise floor at **±0.75
points of depth-2 exact match**. `d_out 512`'s +9.5 is ~13× that; every other
row is inside it or negative. For scale, four times the training budget buys
73.78% → 90.71%.

**Width pays once and then stops.** `d_out 1024` was run at both budgets:
+0.9 points at 250k and −0.7 at 1M, both inside the noise floor, with GVN
separation slipping 42× → 36×. `d_out 512` remains the best configuration. More
tellingly, the error cells of finding 20 do not move at all between 512 and 1024
— the largest of them reads 97.11% and 97.10%. **The remaining error is not
capacity-shaped**, and the 256 → 512 gain bought a specific fixed amount rather
than relieving a general limit.

**What the three refuted levers rule out.** Each targeted a mechanism that a
measurement had identified, and each failure narrows the cause:

- **Composed slots are not gradient-starved.** `vp_depth_weight` weights an
  output slot by `W^(depth-1)`, raising depth-2's share of the vp gradient from
  2.5% to 18% (W=8) and 46% (W=32). Depth-2 does not move at either setting
  while depth-1 degrades monotonically.
- **The op-cell's two operand populations sit at very different scales, and
  that is not the cause.** A leaf operand is one of 32 `rank_emb` vectors,
  ‖·‖ ≈ 10; a composed operand is the cell's own output, ‖·‖ ≈ 2.1, and excites
  the cell's learned operand weights at 0.08–0.14× a leaf's energy.
  RMSNorm on every value entering a slot
  (`--value-norm`) puts both populations on one shell and scores 1.5 points
  below the baseline.
- **The instruction's content is not blocked on its way into the slot.**
  `--content-skip` adds a direct linear path from an instruction's content to
  the slot it writes, so the content need not share the op-cell's first layer
  with decoding the operands. It lands inside the noise floor.

**The decoder is not the constraint either.** Fine-tuning the trained vp head on
the natural slot mix leaves depth-2 at 90.2%; fine-tuning it on depth-2 slots
only drops it to 88.4% while destroying depth-1; fresh heads trained on depth-2
only reach 82.8% (one hidden layer) and 78.8% (four). Every head-side arm lands
below the trained head's 91.1% (`probes/depth_composition.py`). Nor is the
two-step arithmetic the difficulty: a depth-2 value is `f2(f1(a,b),c)`, and
handing a readout the true intermediate `f1(a,b)` changes nothing — 85.9%
against 86.0% at the same fitting budget, while removing the essence collapses
it to 15.8% (`probes/depth_intermediate.py`).

**18. The encoder does not fuse additive chains at all — equivalent chunks are
as far apart as inequivalent ones.** `ADDI r1,r1,256; ADDI r1,r1,-16` and
`ADDI r1,r1,32; ADDI r1,r1,208` compute the same function of r1: only the sum of
the immediates is a property of the function, the split is not. A
value-numbering encoder must map them to one point. On the best len=2 model
(`d_out 512` at 1M), over 2000 such pairs:

| | essence distance |
|---|---|
| **fusion** — same total, different split (MUST be ~0) | **0.4634** |
| **different** — totals differing by 1 (the denominator) | 0.3901 |
| rename — the same chunk relabelled (~0 by construction) | 0.0057 |

Separation **0.8×**. The essence separates two chunks that compute the same
function exactly as much as two that do not, while the rename row confirms the
encoder is intact. `probes/additive_fusion.py` builds the chunks directly and
takes the distance, so there is no fitted readout anywhere in it; it asserts the
hand-built tokens round-trip through the decoder and that the emulator agrees
each `fusion` pair is behaviorally identical and each `different` pair is not.

This is a different equivalence class from the one `scripts/eval.py` scores.
Its `behavioral` row swaps a commutative operation's operands, which leaves the
multiset of operands unchanged; fusing two immediates into one requires
computing with them. Nothing had tested the second class before, and the model
fails it completely.

It is also consistent with the loss. vp is not a pairwise objective: two chunks
computing the same function have the same vp target, but nothing pushes their
essences together, and two distinct points can both predict correctly. The same
reasoning is already recorded for why `--inject-equiv` failed. What is new is
that the gap is total rather than partial — 0.8× separation, not a loose
collapse.

A composed slot keeps the two instructions' immediates separately rather than
folding them into one composite, which is the same fact seen from the other
side: an encoder that fused them could not tell these two chunks apart, and this
one tells them apart as readily as it tells apart chunks computing different
functions.

**19. A category verified on a corpus that excludes immediates says nothing
about the corpus the model trains on.** Finding 11's shared-source fix was
verified on `configs/instr_r_only.json`, whose chunks carry no immediates.
Scoring the same category on the mixed training corpus (`probes/vp_budget.py`,
model's own head) split it in two:

| cross-instruction shared source | exact | share of all error |
|---|---|---|
| chunk is all-R | **99.939%** | 0.8% |
| chunk carries an immediate | **96.781%** | **47.4%** |

A 51× difference in error rate inside one category, same model and corpus. The
whole corpus told the same story: the syntactic-frame model read 99.972% exact
overall on `instr_r_only` against 99.756% on the mixed corpus — 8.8× the error,
same checkpoint.

The gap was not a limit of the encoder. It was the syntactic canonical frame of
finding 14, which lands almost entirely on chunks carrying an immediate:
`SLTIU`/`SLTI` against a 12-bit immediate is the common way an instruction reads
a register it does not depend on, and an all-R corpus has none. In the
behavioral frame the same category reads **99.307%** against 97.196% on rows
whose target is identical either way. The lesson about corpus scope stands
regardless of the cause: the measurement was 51× off because the population was
chosen to exclude the failing case.

Also distinguish the two things called "a shared source", because only one is a
problem: one instruction reading the same register twice (`ADD x3,x5,x5`) is at
99.82%, while two instructions reading a common register is at 98.32%. The first
collapses two operands of one op; the second ties two ops to one canonical input
position.

Both single-axis probes report their own population correctly:
`vp_slot_b_split` measures the shared-source axis on a corpus that holds
immediates out by construction, and `vp_imm_share` measures the immediate axis
without distinguishing shared sources. A statement about the model needs the
axes crossed — see "A methodological note".

**20. Every remaining failure is an immediate combined with something else;
nothing is a problem on its own.** Three properties of an output slot decide
whether it is predicted correctly:

- **sources** — `cross-instr` when the chunk's two instructions read a common
  register, `distinct` otherwise. (One instruction reading the same register
  twice is a separate, benign case; see finding 19.)
- **immediate** — whether any instruction in the chunk carries one. R-type ALU
  operations take two registers; I-type take a register and a 12-bit literal.
- **depth** — the slot's dataflow depth: 1 when the instruction that wrote it
  read only chunk inputs, 2 when one of its operands is another instruction's
  result.

Crossed on the best len=2 model, over the 1.21M (row, output slot, anchor)
triples that vp supervises in a 600-batch sample, with each cell's share of all
failing triples (`probes/vp_budget.py`; the seed-7 replicate agrees within 0.1
points on every cell above 1k triples):

| sources | immediate | depth | triples | exact | err mass |
|---|---|---|---|---|---|
| distinct | yes | **2** | 17.4k | **93.01%** | **39.5%** |
| distinct | yes | 1 | 767k | 99.909% | 22.5% |
| distinct | no | 1 | 290k | 99.879% | 11.4% |
| **cross-instr** | yes | 1 | 42.2k | 99.228% | 10.6% |
| **cross-instr** | yes | **2** | 888 | **81.08%** | 5.4% |
| cross-instr | no | 2 | 1.5k | 98.24% | 0.8% |
| cross-instr | no | 1 | 39.2k | 99.980% | 0.3% |
| distinct | no | 2 | 9.4k | 99.957% | 0.1% |

**Of the two conditions an immediate had to survive, one was a broken target and
one is real.** Being placed in the canonical frame alongside a shared source was
the syntactic-frame bug of finding 14; in the behavioral frame that cell is at
99.23% and carries a quarter of the error it did. Being **composed** — the
writing instruction taking an operand from another instruction's result — is
untouched by it, and is now the open problem: `distinct | immediate | depth 2`
is at 93.01% with correct wiring and both frames agreeing, and carries 39.5% of
all remaining error at 1.4% of the triples.

Composition is not hard as such: all-R composed slots are at 99.957%. Composing
an **immediate-carrying** instruction is, by three orders of magnitude in error
rate. That is also why `d_out` helped in finding 17 — width in the value path is
what a slot needs to carry an immediate *and* the structure it is combined with.

## Decisions taken

- **The pool reads each slot's write step (`pool_extra=write_time`).** NB the
  constructor and CLI default is still `none`, which reproduces the
  pre-ordering-fix model; new runs must pass the flag explicitly.
- **Feeding the pool the full event block is rejected** on the equivalence-
  collapse cost (finding 9), despite its vp gain. Any future fix must keep
  read-order out of the invariant essence.
- **Width is spent as a lever.** Scaling the model as a whole did nothing
  (finding 1); widening the pool's per-head value projection closed the
  single-instruction immediate (finding 16); `d_out` 256 → 512 bought a fixed
  amount and 512 → 1024 buys nothing at either budget, leaving the error cells
  unchanged to three decimal places (finding 17). `d_out 512` is adopted; the
  remaining error is not capacity-shaped and further width is not the answer.
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
  1's ten training runs did not.
- **Every sweep carries a null intervention** — the same configuration trained
  twice on separately generated corpora. Without it no row is interpretable; at
  these run lengths it costs ~35 minutes.
- **Report an error BUDGET, not an accuracy.** A bucket's rate and its share of
  total error are different facts and neither implies the other:
  immediate-carrying instructions at depth 1 are at 99.909% and are 22.5% of the
  error, while composed slots with an immediate are at 93.01% and are 39.5% of
  it. `probes/vp_budget.py` reports both, crossed. Localise with probes on frozen checkpoints
  first; train only to test a specific mechanism.
- **Build the vp target in the behavioral canonical frame** — the same frame
  T1's emitted binding and T2's read ranks are in, so the target is a function
  of what the encoder is shown (finding 14). The syntactic frame is not an
  option to keep behind a flag: it makes two chunks with a bit-identical encoder
  input carry different targets.
- **A defect whose cost falls on a different row than the one carrying it is
  invisible to any split of the corpus.** Splitting by "does this row have the
  defect" said the frame bug was harmless for as long as it was measured that
  way. Where a defect can make two inputs collide, build both sides of the
  collision and vary one thing (`probes/vp_drop_control.py`).

## Dead ends (do not re-run)

- **Capacity.** Scaling the model's dimensions and steps together left
  aggregate vp essentially unchanged at a given chunk length, and `d_out`
  512 → 1024 leaves even the specific error cells unchanged (finding 17). The
  one exception was 256 → 512, which bought a fixed amount and did not repeat.
- **Buying gradient share for the hard slots.** Weighting vp by dataflow depth
  (`--vp-depth-weight`) at 8× and 32× leaves depth-2 flat and costs depth-1
  (finding 17).
- **Normalizing values into the register slot** (`--value-norm`), to put a
  seeded leaf and a computed op-cell output on one scale. The scale gap is real
  and measured, but closing it does not help depth-2 (finding 17).
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

A third failure mode governs every question of the form "why is this *subset*
worse": **a hard subset is a harder question, so scoring it lower proves
nothing on its own.** Two of the confound-breakers work, one does not, and the
fourth item is what says whether any of it is signal:

- **Hand the readout the thing the hard case has to compute.** Giving it the
  true intermediate value removes "this subset needs two arithmetic steps" as an
  explanation. It changed nothing, which is what moved the diagnosis off the
  decoder (finding 17).
- **Prefer a measurement with no fitted readout at all.** Building the inputs
  directly and taking a distance between essences has no capacity, no training
  set and no sample-size asymmetry — findings 18 and 20 both rest on
  measurements of that kind, and they are the two that held up.
- **Asking a readout for a quantity whose difficulty does not vary with the
  subset is NOT sufficient.** An instruction's opcode and immediate are the same
  size of question whether or not its operand was computed, which looks like a
  fair comparison. Asking a readout for one composed
  instruction's 12-bit immediate returned **28.9%, 53.2%, 74.7% and 97.0%**
  across four designs differing only in readout depth, whether the readout was
  fit per group or on all rows pooled, and how many rows the rare group got. The
  spread is the probe. Neither fitting choice is neutral: pooled underfits a
  subpopulation that needs a different decoding function, per-group hands the
  rare subset a fraction of the training rows, and both report the difference as
  a property of the model. A probe of this kind has to report both fitting
  modes, and a divergence between them is a fact about the probe.
- **Run the null intervention.** The same configuration trained twice, on
  separately generated corpora, is the only thing that says which differences in
  a sweep are real. At these run lengths it costs ~35 minutes, and without it
  none of finding 17's rows can be read.

**Cross the axes, do not pair one-dimensional probes.** Findings 11, 16 and 17
each verified a fix on its own axis with the others held at an easy setting, and
the record even names the blind spot of each probe. The cell where two
conditions hold at once appeared in none of them, and it is 90% of the error
(findings 19, 20). `probes/vp_budget.py` crosses the splits and reports each
bucket's rate alongside its share of total error, which is the shape of output
that makes an interaction visible.

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

1. **A composed immediate** — the whole of what is left at len=2, and not
   capacity-shaped: `d_out` 512 → 1024 leaves it unchanged (finding 17), and the
   head, the two-step arithmetic chain, the gradient share, the operand scale
   mismatch and content crowding are each separately refuted. Composition alone
   is at 99.957% and an immediate alone at 99.909%, so what fails is the pair.
   The same question decides milestone 2, since a slot at depth *d* must
   describe *d* instructions' worth of computation and len=4/8 chunks are mostly
   deep.

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
