# Phase 15 (IN PROGRESS): making the T2 chunk vector faithful to the instructions it encodes

**Status: open.** Faithfulness at two instructions is largely settled;
equivalence collapse is not, and nothing beyond two instructions or outside the
R/I-type ALU subset has been attempted. Claims here are corrected in place, so
read them as current belief rather than as a chronology.

**Headline.** Two properties have to hold and each has its own state.

*Faithfulness.* Exact 32-bit match of the predicted output value reaches
**99.9364%** at two instructions — 512 failing (row, output-slot, anchor)
triples out of 805,408 — on `configs/instr_alu_writers.json` at `cap=2` over the
frozen `runs/t1_writers_250k_rank`, at `--d-out 512` with a one-hidden-layer
value-prediction head. Four architecture fixes carried it to 99.83%: the
per-slot write step fed into the essence pool, canonical read position as the
input registers' leaf identity, per-head pool value projections, and two
corrections to the oracle's supervision (findings 14 and 24). The remaining
factor was not architectural at all but the training corpus: chunks where one
instruction consumes another's result are ~4.7% of a randomly drawn corpus and
the hardest shapes a fraction of a percent, and generating more of them
accounts for the rest (finding 30). The core's own settings are fixed on first
principles rather than chosen by measurement (finding 25).

*Equivalence.* Unresolved, and untouched by any of the above. Two chunks that
compute the same function sit as far apart as two that do not (finding 18), and
T1 holds `x+x` and `x<<1` further apart than two different shifts (finding 23).
What collapses at either tier is what the architecture makes free, such as a
commutative operand swap; nothing in the objective asks for the rest.

**Quote the floor that matches the comparison.** Retraining the T2 alone over a
fixed frozen T1 has spread 0.002 and 0.009 points of exact match on the two
configurations where it was replicated, and 0.021 on a third; retraining the
frozen T1 spreads 0.023 (finding 25). Use ~0.01 for a T2-level claim unless the
specific cell has been replicated, and 0.023 for anything comparing T1s.
`probes/vp_budget.py` draws a fresh unseeded corpus per invocation, which moves
its own aggregate by ~0.02 points between runs of one model — score every arm of
a per-bucket comparison against ONE corpus file.

**Aggregates do not span either oracle change** — a model is scored on the
target it trained on. For the canonical-frame change compare on the rows where
the two agree (finding 14); the `_actually_modified` change moved the measured
figure 0.10 points and the learned result not at all (finding 24).

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
(`runs/t1_writers_250k_rank` — single-instruction vp is an exact 32-bit match on
100.0000% of supervised triples, every opcode, by `probes/t1_scorecard.py`).
Nothing in this phase is a T1 problem.

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
exists to remove; and equivalent pairs stay two orders of magnitude closer than
unrelated ones, which is the property the metric exists to assert. (Phase 14's
28× is not a usable floor to compare against: it predates the rename-row fix of
finding 12, which makes every earlier GVN number incomparable.)

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
barely moves (99.757% on the syntactic-frame target against 99.744%/99.753%
on the behavioral one, both measured before finding 24's correction,
two seeds) because each model is scored on its own convention — the syntactic-
frame model reads 96.97% against the corrected target. **Do not compare
aggregates across this change**; compare the identical-target rows.

Those two seeds put the **T2-level** run-to-run spread at 1M steps at 0.009
points of overall exact match, and within 0.1 points on every cell above 1k
triples — far tighter than the ±0.75 recorded at 250k in finding 17. It holds
the frozen T1 fixed, so it is NOT the floor for a comparison between T1s
(finding 25), and it is not a universal T2-level figure either: a second
replicate pair, on the corrected oracle, spread 0.002.

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
more discriminative essence collapsing equivalents less. Operand binding is
unchanged (`in_slot` 0.979 → 0.980;
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
separation slipping 42× → 36× against each other. `d_out 512` remains the best
configuration. More tellingly, the error cells of finding 20 do not move at all
between 512 and 1024 — the largest of them reads 97.11% and 97.10%. **The
remaining error is not capacity-shaped**, and the 256 → 512 gain bought a
specific fixed amount rather than relieving a general limit.

Every figure in this finding is on the syntactic-frame target at `d_out` 256
unless stated, and the behavioral-frame correction (finding 14) changed the
error composition materially. The width conclusion is unaffected — it is a
comparison of two configurations measured together — but the three refuted
levers each deserve one re-test, `--value-norm` most of all: it targets the
op-cell reading its own previous output, which finding 21 now implicates
directly.

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

**Fine-tuning the decoder is not a lever; its CAPACITY is** (finding 22, which
supersedes the reading this paragraph originally carried). The arms below refit
heads on frozen essences, and the two that vary head DEPTH are data-limited —
depth-2 slots are 2.5% of the supervision, so a fresh head sees ~9k examples and
neither reaches the trained head from scratch. They cannot answer the depth
question, and trained end-to-end where the head sees the whole stream, depth 4
moves individual cells by more than 20 points. What the arms do establish is
that fine-tuning the trained head on frozen essences never beats it.
Fine-tuning the trained vp head on
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
| **fusion** — same total, different split (MUST be ~0) | **0.4143** |
| **different** — totals differing by 1 (the denominator) | 0.3727 |
| rename — the same chunk relabelled (~0 by construction) | 0.0008 |

Separation **0.9×**. The essence separates two chunks that compute the same
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

**Arithmetic is not what separates the classes that collapse from the ones that
do not.** Exchanging one true no-op for another (`ADD xa,xa,x0` against
`ADDI xd,xd,0`, with the rest of the chunk held fixed) is a single-instruction
rewrite requiring no computation on operands, and it reaches only **5.9×**
(0.0901 against 0.5322 for a chunk computing a different function, 600 pairs,
register-file equivalence verified through the emulator). The ordering across
classes — rename 0.0008 free by construction, commutative swap 0.0292 at 24×,
no-op exchange 0.0901 at 5.9×, additive fusion 0.4143 at 0.9× — tracks how much
the architecture has to learn to see the equivalence, and **nothing trains any
of it**. vp is doubly unable to here: a true no-op has no supervised output slot
at all, so it emits no gradient about that instruction while still moving the
essence by 0.1041.

Whether that pair is equivalent at all depends on a question the project has not
settled: the two chunks advance PC by the same amount only because both are two
instructions long. If instruction count is part of the behavior, the essence is
right to separate them. `out_pcs` is computed by the oracle and only a
`pc_writes` boolean is shipped, so neither reading is currently testable.

It is also consistent with the loss. vp is not a pairwise objective: two chunks
computing the same function have the same vp target, but nothing pushes their
essences together, and two distinct points can both predict correctly. The same
reasoning is already recorded for why injecting manifest equivalents into the
corpus failed. What is new is that the gap is total rather than partial —
0.9× separation, not a loose collapse.

A composed slot keeps the two instructions' immediates separately rather than
folding them into one composite, which is the same fact seen from the other
side: an encoder that fused them could not tell these two chunks apart, and this
one tells them apart as readily as it tells apart chunks computing different
functions. That is also the source of the composed-immediate vp error, since the
head then has to form the composite itself (finding 21) — but the two are not
the same defect: `XORI` chains are equally uncollapsed and predicted exactly.

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
| distinct | yes | **2** | 16.8k | **93.72%** | **58.2%** |
| distinct | yes | 1 | 790k | 99.930% | 30.4% |
| **cross-instr** | yes | **2** | 752 | **80.59%** | 8.1% |
| cross-instr | no | 2 | 1.5k | 97.79% | 1.8% |
| distinct | no | 1 | 310k | 99.995% | 0.9% |
| **cross-instr** | yes | 1 | 37.6k | 99.984% | 0.3% |
| distinct | no | 2 | 10.0k | 99.950% | 0.3% |
| cross-instr | no | 1 | 40.4k | 100.0000% | 0.0% |

**Of the two conditions an immediate had to survive, one was a broken target and
one is real.** Being placed in the canonical frame alongside a shared source was
the syntactic-frame bug of finding 14; in the behavioral frame that cell is at
99.23% and carries a quarter of the error it did. Being **composed** — the
writing instruction taking an operand from another instruction's result — is
untouched by it, and is now the open problem: `distinct | immediate | depth 2`
is at 93.01% with correct wiring and both frames agreeing, and carries 39.5% of
all remaining error at 1.4% of the triples. Neither T1's binding prediction nor
the canonical frame is what fails there (`probes/vp_frame.py`): restricting the
cell to rows where T1's emitted wiring matches the tokens reads 93.43%, and
restricting it to rows whose target is identical under either frame reads
92.63%, both at the cell's own rate.

Composition is not hard as such: all-R composed slots are at 99.957%. Composing
an **immediate-carrying** instruction is, by three orders of magnitude in error
rate. That is also why `d_out` helped in finding 17 — width in the value path is
what a slot needs to carry an immediate *and* the structure it is combined with.

**21. What fails at depth 2 is that the essence holds two literals and not
their sum, and the vp head cannot decode a literal and add it in one pass.**
Six arms of one depth-2 shape — two instructions, the second reading the first's
result — with only the placement of the 12-bit literals varying, so any
difference between them is the literal and not the composition
(`probes/vp_compose_control.py`, 24k triples per arm). Both output slots are
scored: the *intermediate* is the first instruction's own slot at depth 1, the
*composed* slot the second's at depth 2.

| arm | intermediate | composed |
|---|---|---|
| `ADD ; ADD` | 100.0000% | **100.0000%** |
| `ADD ; ADDI` | 100.0000% | 96.242% |
| `ADDI ; ADD` | 99.913% | **68.175%** |
| `ADDI ; ADDI` | 99.904% | **19.521%** |
| `SLT ; ADDI` | 100.0000% | 99.550% |
| `ADDI ; SLT` | 100.0000% | 100.0000% |

**The encoder never sees a register value.** Its only inputs are each
instruction's opcode and immediate — through the frozen T1's essence — and the
wiring; the anchor states enter nowhere but the vp head's own features. A
register slot therefore does not hold a number, it holds a symbolic description
of the function that produced it, and vp asks the head to EVALUATE that
description at the 8 anchor points. The arms above score the composition of
those two things.

**Both of a composed chain's literals are in the essence; their sum is not.**
On `ADDI rB,rA,i1 ; ADDI rC,rB,i2`, a readout fitted on the frozen essence
recovers i1 at 100.0000% and i2 at 99.08% exact, against a 0.024% base rate for
a 12-bit literal — and recovers `i1+i2` at only 84.48%
(`probes/compose_readback.py`, 24k chunks, 12k fitting steps; every figure in
this paragraph is from that one run). The gap is not pooling: the composed register
slot itself reads 99.85% / 99.96% for the two literals and 91.25% for their sum,
so the op-cell appends the writer's literal to the description it reads rather
than folding it in. The sum is not hiding in phase form either — T1 is trained
to carry an immediate as a rotary code with power-of-2 periods, and in such a
code adding two numbers is adding their angles, but `rotary(i1+i2)` is linearly
recoverable at R² 0.708 against 0.675 for i2 alone, i.e. no better than its less
accessible summand. A fold would make the sum MORE available than either
addend.

**What the head is short of is the sum, and nothing else.** Five readouts of one
architecture and budget on the frozen essence, all targeting the same oracle
value for the composed slot of `ADDI;ADDI` (24k chunks × 8 anchors, held-out
fifth):

| the readout is handed | exact 32-bit |
|---|---|
| essence + the canonical input — the head's own task | 79.80% |
| essence + input + **the composite `i1+i2`** | **99.95%** |
| essence + input + both literals separately | 92.79% |
| essence + input + the intermediate `a+i1` | 79.52% |
| input + both literals, NO essence — the arithmetic ceiling | 95.24% |

Handing over the composite beats handing over both summands and beats the
no-essence ceiling, because it reduces the task to a single two-term addition;
the same arm without the essence at all reads 99.9974%, so the composite makes
the essence redundant for this cell. Handing over the *intermediate* instead
buys nothing, which is why supervising or feeding the intermediate is not the
lever it looks like.

**The cost is the WRITER's literal specifically.** i1 and i2 are identically
distributed and the target is symmetric in them, so a gap between them is a fact
about the essence. Over three readout seeds (per-arm spread ≤0.9 points):

| the readout must decode | mean exact |
|---|---|
| neither literal | 92.785% |
| i1 only — the producer's, which passed through a slot | 90.880% |
| i2 only — the writer's, which arrived in content | 80.148% |
| both | 80.050% |

Decoding the producer's literal costs 1.9 points and decoding the writer's
costs 12.6, and decoding both costs no more than the writer's alone. The
producer's literal is also the more accessible one geometrically: a LINEAR map
on the essence recovers it at 92.0% of bits (rotary R² 0.962) against 37.6%
(R² 0.675) for the writer's. It appears in two register slots — its own and,
transformed, the composed one — where the writer's appears in one.

**Carries are necessary to the failure but not sufficient.** A readout of the
same shape fed CLEAN bits of i1 and i2 is exact on `i1 XOR i2` and `i1 AND i2`
at any depth, and reaches 96.2% on `i1 + i2`; a 32-bit three-term addition of
values handed over as bits is exact (`ADD;ADD`, 100.0000%). So the architecture
adds what it is given and fails on what it must first decode. The corpus orders
the same way: `logic -> logic` depth-2 slots are at 99.38% while `add -> add`
are at 82.4% and `shift -> shift` at 91.0% (`probes/vp_depth2.py`).

**On the corpus, a literal in a depth-2 chain is 56% of all remaining error.**
Crossed by which instruction is I-type, over 804,728 supervised triples of which
1,430 fail:

| producer → writer | triples | exact | share of all error |
|---|---|---|---|
| I-type → I-type | 2,432 | 84.09% | 27.1% |
| I-type → R-type | 6,344 | 94.51% | 24.3% |
| R-type → I-type | 2,864 | 97.73% | 4.5% |
| R-type → R-type | 7,520 | 99.85% | 0.8% |

**The constructed arms sample the hard tail of their bucket, by construction.**
`vp_compose_control.py` draws both literals uniformly over the full 12-bit
range, while the corpus gives shift opcodes a 5-bit shift amount, so the
`add -> add` bucket's 82.4% is carried by narrow literals. Restricted to a
producer literal needing 9–11 bits, the corpus reads 12.5% for that bucket —
the constructed arm's 19.5%, not the bucket aggregate. Quote the arms for a
mechanism and the budget for a share of the error.

**22. Two unrelated interventions land on the same trade: whatever makes a
composed operand easier to decode helps the case needing ONE literal out of the
essence and costs the case needing TWO.** `--vp-head-depth 4` deepens the
readout; `--value-norm` RMSNorms every value entering a register slot so a
seeded leaf and the op-cell's own output sit on one shell. They share no
mechanism. Both were trained end-to-end at the current config on the corrected
target, and scored on the constructed arms with two seeds of the unmodified
config giving the run-to-run floor:

| arm | seed 0 | seed 7 | `--value-norm` | `--vp-head-depth 4` |
|---|---|---|---|---|
| `ADDI;ADD` | 64.71% | 72.50% | **89.16%** | **90.97%** |
| `ADDI;ADDI` | 17.22% | 16.48% | **6.67%** | **2.52%** |
| `ADD;ADDI` | 96.20% | 95.49% | 92.67% | 93.48% |
| `ADD;ADD` | 100.0000% | 100.0000% | 100.0000% | 100.0000% |

Neither moves the aggregate: 99.7474% and 99.7587% against 99.7443%/99.7530%,
all inside the ±0.01-point floor, and `distinct | immediate | depth 2` is
92.97%/92.74% against 93.01%/93.08%. Neither is adopted. Every figure in this
finding is on the pre-`_actually_modified` oracle (finding 24); the baseline
arms are unchanged by that correction, so the comparison stands, but a new
intervention must be scored against the current checkpoint. The deep head also
costs collapse (GVN separation 16.7× against 23.9%/24.7×); `--value-norm` does
not (26.9×).

**Head capacity is a real constraint for one of the two sub-cases**, which the
frozen-essence arms of finding 17 could not see. `ADDI;ADD` — one literal
decoded from the essence and combined with two values the head is handed as bits
— gains ~20 points over the better seed, far outside a 7.8-point spread.

**`ADDI;ADDI` is the sub-case nothing reaches.** It needs two literals folded
into one composite, and every intervention that helps the first case drives it
further down. That is the subset equivalence collapse covers, and the only
subset it covers.

Caveat on attribution: both runs retrain the encoder and the head together, so
they show what the joint system does when one knob moves, not what the readout
does with the encoder held fixed. The clean head-only test is still blocked by
the data limit above.

**23. T1 does not collapse cross-opcode equivalences either.** Distances between
single-instruction T1 essences, where each pair has an identical binding — one
live input, the same register, the same destination — so the whole difference is
in the essence. Measured by `probes/t1_scorecard.py` on the frozen
`runs/t1_writers_250k_rank`, 2000 pairs per row:

| pair | same function | essence L2 |
|---|---|---|
| `ADD x5,x3,x3` vs `SLLI x5,x3,1` | yes | **0.5696** |
| `ADDI x5,x3,0` vs `{ORI,XORI,SLLI} x5,x3,0` | yes | 0.2794 |
| `ADD x5,x3,x4` vs `ADD x5,x4,x3` | yes | **0.0083** |
| `SLLI x5,x3,1` vs `SLLI x5,x3,2` | **no** | 0.6948 |
| `ADD x5,x3,x4` vs `SUB x5,x3,x4` | **no** | 0.9675 |

Pairs computing the same function sit within a factor of 2 of pairs computing
different ones. The one pair that does collapse is the commutative swap (0.0083,
117× separation), which is free because the pool is a multiset over register
slots, so exchanging two operands is invisible by construction.

**Read the ratio, not the raw distance, and not from one checkpoint.** Across
ten T1 checkpoints the `ADD x,y,y` vs `SLLI x,y,1` distance ranges 0.32–1.01 and
its denominator 0.55–0.76, putting the ratio anywhere in 0.63–2.07× with no
relation to the settings — on two of the ten the equivalent pair lands *further*
apart than the inequivalent one. What holds on every checkpoint is that the
ratio never approaches the commutative row's.

**The pattern across both tiers is the same.** An equivalence collapses when the
architecture cannot represent the difference, and does not collapse when
recognising it would require knowing that two different operations compute the
same function. Nothing in the objective asks for the second kind at any tier —
vp gives both members of a pair the same target without ever comparing them, and
the equivalence manifest holds only rewrites. So this is not a T2 deficit being
inherited from a clean T1; it is absent at T1 and absent at T2, for the same
reason.

**24. A structural question answered by value comparison made the oracle's
supervision depend on register NAMES.** `_actually_modified` decided "did the
chunk write this register?" by comparing the register's post-execution value
against its pre-execution value. Canonical position *i* **is** register x*i*, so
a canonicalized input value is bit-identical to some untouched register's own
value, and a chunk that copies canonical input 1 into x1 writes exactly the
value x1 already held. The value test then reported no write.

That made a chunk and its rename disagree:

```
ADD x1, x7, x0    behavioral_inputs=[]     in_slots=[]    out_slots=[1]
ADD x2, x9, x0    behavioral_inputs=[9]    in_slots=[9]   out_slots=[2]
```

The model is exactly equivariant, so it represents those two identically and is
then graded against two different targets — the same defect shape as finding 14,
one layer down. The consequence compounded: dropping the output emptied
`output_regs`, which zeroed every `input_mags` entry, which emptied the
behavioral set, which made the re-executed frame the RAW anchor state, which
restored the output with a target no encoder can produce (the head is fed no
inputs and asked for a specific register's raw value).

**The fix distinguishes two cases the value test conflated.** A register the
chunk never READ is modified by any write to it — whether its new value happens
to equal its old one is a property of the anchor state, not of the chunk. A
register the chunk DID read keeps the value test, because there "does the output
equal the input" is a real question about the chunk, and answering it yes is
what makes `ADD x5,x5,x0` a genuine no-op with no inputs and no outputs.

Corpus impact is small and was verified rather than assumed: old and new rules
disagree on 0.10% of len=2 chunks and 0.30% at len=4, and every disagreement is
the predicted shape. It grows with chunk length.

**The generalizable rule: never answer a structural question by comparing
values, in a space where canonical values alias concrete ones.** The identity
needed was already available and untouchable by any instruction —
`SSAGraph.output_versions` and `input_node_id`, built by walking the instruction
list before any value exists. Threading an identity tag through
`_eval_ssa_numpy` would recompute the same answer; the two could only differ
under intra-chunk control flow, which that evaluator does not model either.

**Nothing checked that the oracle's output is rename-equivariant.** The project
asserts `state_max_err = 0.00e+00` for the model and had no analogue for the
targets, which is why this sat. `test_supervision_is_rename_equivariant` now
asserts that `in_slot_regs`, `out_slot_regs` and the live masks permute under a
register permutation while `out_regs` values come out EQUAL — the canonical
frame's whole purpose. It was checked against the old rule and fails there.

**25. T1→T2 transfer varies by 0.023 points from the frozen T1 alone, and no
core setting is distinguishable against that.** Retraining the T2 on a fresh
corpus over the same frozen T1 moves len=2 exact match by **0.002–0.009** points
(two configurations, each replicated). Retraining the frozen T1 moves it by
**0.023** (σ over five same-config T1s, 0.0553 range), and depth-2 exact match
by **0.24** (0.64 range). A comparison between T1s needs the larger figure — and
the 4.5× gap between the two T2-level measurements is itself why a floor from
one configuration cannot be reused on another: replicate the arm under test.

Each row is a T2 at len=2 / `d_out` 512 / 1M steps, every hyperparameter
identical, differing only in the frozen T1 beneath it; the T1's settings are
given as read-position / seeding / pool-channel. `n` counts same-configuration
T1s retrained on separately generated corpora:

| T1 configuration | n | mean exact | range |
|---|---|---|---|
| **rank / tags / write_time** | **5** | 99.8264% | **99.7983 – 99.8536** |
| rank / rank / none | 2 | 99.8285% | 99.8272 – 99.8298 |
| rank / rank / write_time | 1 | 99.8283% | — |
| stamp / tags / write_time | 1 | 99.8537% | — |
| stamp / tags / none | 1 | 99.8570% | — |
| stamp / tags / none, pre-correction oracle (findings 14, 24) | 1 | 99.8497% | — |

Every row sits within **1.4σ** of the five-replicate distribution (z = +0.03 to
+1.32), as does the oracle correction of finding 24, whose apparent +0.0073
exact and +0.42 depth-2 are both inside it.

**T1's own metrics do not predict what T2 extracts.** All six T1s reach
**100.0000%** exact 32-bit match on their own single-instruction value
prediction (`probes/t1_scorecard.py`), while the T2s built on them span 0.059
points — an order of magnitude beyond any T2-level spread measured. A tier can be perfect on its own objective and
still be a measurably different input to the next one. Operationally: **a T1
cannot be selected without training a T2 over it** (~2.4h at len=2);
`t1_scorecard.py` verifies a T1 is intact, it cannot rank T1s.

**Two structural facts about the core:**

- **`read_time` is absent from the essence computation.** Scaling it 100× leaves
  T1's essence bit-identical (`0.000e+00`); it reaches only the event channels,
  and hence the binding heads.
- **T1's essence carries its instruction's content in full.** A fitted readout
  recovers the opcode at 99.65–100% and the 12-bit immediate at 99.96–100%
  exact, against base rates of 5.90% and 0.024%.

The settings themselves are fixed on first principles rather than on this table
— see RISCV_DESIGN.md, "Leaf identity for input registers".
`runs/t1_writers_250k_rank` is the frozen T1; checkpoints trained at other
values do not load.

**26. The adder works; the CONSTANT it adds is wrong.** The head's prediction
for a composed ADD-family slot equals the known summands plus some plausible
12-bit constant on 95.4–99.6% of triples, so the 32-bit arithmetic is intact and
what fails is the scalar. Recovered by subtracting the known summands from the
head's own output — no fitting, no execution (`probes/vp_compose_control.py`
shapes; the subtraction is arithmetic on the head's prediction):

| arm | target | exact | prediction = summands + a plausible constant | median miss |
|---|---|---|---|---|
| `ADD;ADDI` | a + x + i2 | 96.18% | 99.61% | 0 |
| `ADDI;ADD` | a + i1 + y | 68.56% | 99.08% | 0 |
| `ADDI;ADDI` | a + i1 + i2 | 19.48% | 95.42% | 144 |

A median miss of 144 against ~1365 for a random 12-bit value says the head
APPROXIMATES the constant rather than failing to produce one. The error sits at
0.0% on bit 0 of that constant and climbs monotonically with bit position
(1.8% / 6.7% / 16.1% … 29.7% at bit 8 for `ADDI;ADDI`), which is the carry
signature of an addition: bit 0 needs no carry, every higher bit needs the carry
from all below it.

**The head behaves like a LINEAR readout of the constant it needs.** Per arm,
against a linear readout of that constant from the same frozen essence:

| arm | needs | head | linear | MLP |
|---|---|---|---|---|
| `ADD;ADDI` | `i2` | 96.28% | **96.31%** | 100.000% |
| `ADDI;ADD` | `i1` | 68.45% | 95.31% | 100.000% |
| `ADDI;ADDI` | `i1+i2` | 19.52% | **23.84%** | 77.84% |

Two of three agree to within 4 points. So `ADDI;ADDI` fails because the quantity
it needs is the SUM of two constants, which exists in no instruction and which
the encoder never forms: only 23.8% linearly available, and the head gets
19.5%. It is not an arithmetic deficit — the same head adds three given 32-bit
values at 100.0000% (finding 29) and decodes-and-adds one constant at a single
instruction at 100.0000%.

**`ADDI;ADD` is the open anomaly.** It needs only `i1`, which is 95.3% linearly
available, and the head gets 68.45% — while predicting the PRODUCER's slot in
the same forward pass at 99.91%. The same constant, from the same essence, read
correctly for one output slot and wrongly for another. That is not about how
`i1` is stored. What distinguishes it from `ADD;ADDI` is that there the constant
arrives in the composed slot's own instruction content, while here it must be
inherited through the operand slot.

**27. Feeding the vp head the composed slot's intermediate value changes
nothing, and moving the slot axis into the head's batch is worse.** Two cells at
the adopted config over the same frozen T1, 1M steps, identical but for where
each composed slot's intermediate value comes from — its own row, or another
row's register file read at the same register (same width, same distribution,
same zero pattern, no information). Both run the vp head once per OUTPUT SLOT
instead of once per row, which is what lets an intermediate be fed to the slot
it belongs to without handing any slot its own answer.

| | shared head (adopted) | per-slot, shuffled | per-slot, true |
|---|---|---|---|
| `ADDI;ADDI` composed slot | **19.52%** | 10.11% | 11.51% |
| `ADDI;ADD` composed slot | **68.18%** | 60.13% | 49.53% |
| `ADD;ADDI` composed slot | 96.24% | 96.47% | 96.14% |
| `ADD;ADD` composed slot | 100.0000% | 100.0000% | 100.0000% |
| corpus `distinct` exact | 99.8290% | 99.8278% | 99.8225% |
| `distinct \| imm \| depth 2` | 93.11% | 92.70% | 92.46% |
| GVN behavioral (equivalent pairs) | 0.0352 | 0.0392 | 0.0385 |
| GVN behav-diff (must stay large) | 0.9584 | 0.9338 | 0.9411 |

**The intermediate is null**, as a head-only readout on the frozen essence had
already indicated (79.52% handed the intermediate against 79.80% without it,
finding 21): the two cells differ by 0.005 points of corpus exact match, inside
the floor, and the constructed arms move +1.4 and −10.6 points in opposite
directions. Supervising the intermediate is a lower-prior version of the same
idea and inherits the result.

**The per-slot head is a regression on the population it targets.** Its corpus
aggregate is inside the floor, which on its own would read as no change — the
arms are what show the cost, and they halve the worst cell. All slots sharing
the head's parameters and gradient does not help a rare slot population, which
agrees with the other measurement of that idea: fine-tuning the trained shared
head on a balanced mixture of the five depth-2 shapes, encoder frozen, moves
`ADDI;ADDI` 19.68% → 22.02% while the natural corpus slips 99.797% → 99.619%.
The head's deficit on these shapes is not a matter of how its capacity or its
training signal is divided between them.


**28. Census of the corpus: dataflow composition is scarce, and foldable
composition is scarcer.** Counted over 15,744 rows of
`configs/instr_alu_writers.json` at `cap=2` by decoding the tokens and taking
chains where the second instruction reads the first's destination. A chain has a
COMPOSITE — a single constant the two instructions' literals collapse into —
only when both are in the same algebraic family: two additive ops fold to one
offset, two of the same bitwise op to one mask, two shifts in the same direction
to one amount.

| | rows | share |
|---|---|---|
| composed chains, any opcode pair | 745 | 4.73% |
| ...whose two ops are in the same family | 57 | 0.36% |
| ...both additive | 15 | 0.095% |

At a batch of 384 rows against vp's ~760 supervised slots, the same-family
population is ~0.2 rows per family per batch. This scarcity is what the
composed-slot deficit was (finding 30), and it bounds any objective supervised
on the foldable subset specifically: such an objective would need the generator
to produce those chains deliberately, since the equivalence manifest holds only
single-instruction rewrites.

**A fold also addresses only a minority of the error.** `ADDI;ADDI` has a
composite; `ADDI;XORI` does not, and both fail. Against the per-opcode-class
error shares of finding 21, the foldable pairs — add→add at 5.8% of all error,
logic→logic at 0.8%, and the same-direction part of shift→shift's 12.2%, which
is not measured separately — bound the reachable share at roughly 7–19%, against
the ~56% of error sitting on a depth-2 chain that carries a literal.

**29. The composed-chain deficit is RARITY, and what sorts the arms is a
carry-propagating WRITER.** Head-only readouts on the frozen essence, one per
(arm, depth), same data and step budget, predicting the composed slot's value
from (essence, canonical inputs) — the vp head's own task:

| arm | trained head | depth 1 | depth 2 | depth 4 |
|---|---|---|---|---|
| `ADDI;ADDI` — a composite exists | 19.25% | 47.25% | 69.31% | 80.28% |
| `XORI;XORI` — a composite exists | 99.17% | 100.000% | 100.000% | 100.000% |
| `ADDI;XORI` — no composite | 48.70% | 94.08% | 98.04% | 98.19% |
| `XORI;ADDI` — no composite | 15.07% | 26.39% | 44.16% | 59.38% |

**Depth helps every arm, so depth and folding do not address disjoint
subsets.** What separates the arms is whether the WRITER carries a carry: an
`ADDI` writer stays at 59–80% even at depth 4, a bitwise writer reaches
98–100%. Whether the pair is foldable does not order them.

**The trained head is below a specialised depth-1 readout on EVERY arm**, by 7
to 45 points, on the same frozen essence it reads itself. Read that against the
natural distribution, where a fitted readout LOSES to the trained head (76.6%
exact on depth-2 slots against 94.69%, finding 17's staging arms): the head is
near-optimal for the distribution it serves, and these arms are a measure-zero
slice of it. The gap is the cost of one head serving the whole corpus, which is
why rebalancing the head alone does not recover it — fine-tuning it on a
balanced mixture of the five shapes with the encoder frozen moves `ADDI;ADDI`
19.68% → 22.02% while the corpus slips 99.797% → 99.619% (finding 27).

So the residual is a property of how rare these chains are in training, not of
the essence's content (finding 26) or of available capacity (finding 17);
generating more of them resolves it (finding 30). Weighting the loss by dataflow
depth does NOT, and the distinction is the reason: `--vp-depth-weight` amplifies
the gradient from the same rare rows, where generation supplies new ones.


**30. The len=2 residual is a CORPUS DENSITY problem, and raising the density
of dataflow-composed chunks cuts the error by a third to a half — replicated.**
A uniform register draw composes only ~4.7% of two-instruction chunks, and the
self-referential immediate chains that carry most of the error are ~0.1% of rows
(finding 28). A corpus config's `chain.rate` R rewrites that fraction of
chunks so each instruction reads its predecessor's result, leaving every
opcode untouched, which lifts composed chunks to 53% of rows at R=0.5 and
both-additive chains
14×. Cells at the adopted config over the same frozen T1, 1M steps, all graded
on a NATURALLY drawn corpus so only the training mix differs:

| training mix | overall exact | failing triples | `cross-instr \| imm \| depth 2` | GVN separation |
|---|---|---|---|---|
| adopted (~4.7%) | 99.8270% | 1044 | 76.79% | 35.5× |
| R=0.25 | 99.8840% | 700 | 92.07% | 27.5× |
| R=0.25, second corpus draw | **99.9045%** | **576** | 93.75% | 33.5× |
| R=0.35 | 99.8774% | 740 | **96.25%** | 29.4× |
| R=0.50 | 99.5518% | 2704 | 95.94% | 29.4× |

The two R=0.25 draws spread 0.021 points and the effect is ~0.067, so it clears
its own replicated floor rather than one borrowed from another configuration.

And on the constructed arms, where the mechanism shows:

| arm | adopted | R=0.25 | R=0.25 (s7) | R=0.35 | R=0.50 |
|---|---|---|---|---|---|
| `ADDI;ADDI` | **19.52%** | 93.20% | 93.61% | 96.16% | **97.75%** |
| `ADDI;ADD` | 68.18% | 99.94% | 99.95% | 100.0000% | 100.0000% |
| `ADD;ADDI` | 96.24% | 100.0000% | 99.996% | 100.0000% | 100.0000% |

**At a FIXED step budget it is a trade, and the trade is decided by bucket
size.** Scored per bucket with every model on ONE fixed corpus file, so the
populations are identical and a difference is the model. `depth 1` is an output
slot whose instruction read only chunk inputs; `depth 2` is one whose operand
came from the other instruction's result:

| bucket | triples | ~4.7% | R=0.10 | R=0.25 | R=0.35 | R=0.50 |
|---|---|---|---|---|---|---|
| depth 1 | 786,192 | **99.9349%** | 99.9183% | 99.9143% | 99.8775% | 99.5426% |
| `distinct \| imm \| depth 1` | 521,440 | **99.9653%** | 99.9448% | 99.9381% | 99.8909% | 99.4112% |
| `cross-instr \| imm \| depth 1` | 29,448 | **99.4329%** | 99.4125% | 99.3786% | 99.2937% | 98.8522% |
| depth 2 | 19,216 | 96.3312% | 99.2558% | 99.7710% | 99.8439% | **99.8855%** |
| `distinct \| imm \| depth 2` | 10,984 | 94.4556% | 99.1806% | 99.7815% | 99.8634% | **99.9272%** |
| `cross-instr \| imm \| depth 2` | 640 | 86.5625% | 92.9688% | 97.1875% | 97.9688% | **98.1250%** |
| OVERALL | 805,408 | 99.8489% | 99.9025% | **99.9109%** | 99.8767% | 99.5508% |

Every depth-1 bucket declines monotonically with the rate and every depth-2
bucket improves, so the aggregate is decided by their sizes: depth-1 is 97.6% of
supervised triples and depth-2 is 2.4%. In failing triples against the
unmodified corpus, R=0.25 is +163 on depth-1 against −661 on depth-2 while
R=0.50 is +3085 against −683. The no-immediate buckets barely move at any rate
(99.928% → 99.922%), so the exchange is confined to immediate-carrying chunks.

**The trade is NOT intrinsic — it is an artifact of holding the step budget
fixed.** Depth-2 accuracy tracks the absolute number of chained rows seen rather
than their proportion; the proportion only sets how fast they accumulate and how
much depth-1 exposure is surrendered per step. A cell at R=0.10 for 2M steps sees
the same ~109M chained rows as R=0.25 for 1M:

| bucket | ~4.7% @1M | R=0.10 @1M | **R=0.10 @2M** | R=0.25 @1M |
|---|---|---|---|---|
| OVERALL | 99.8489% | 99.9025% | **99.9364%** | 99.9109% |
| depth 1 | 99.9349% | 99.9183% | **99.9411%** | 99.9143% |
| `distinct \| imm \| depth 1` | 99.9653% | 99.9448% | **99.9724%** | 99.9381% |
| `cross-instr \| imm \| depth 1` | 99.4329% | 99.4125% | **99.4499%** | 99.3786% |
| depth 2 | 96.3312% | 99.2558% | 99.7450% | **99.7710%** |

It reaches R=0.25's depth-2 accuracy and leaves EVERY depth-1 bucket better than
the unmodified corpus: depth-1 failing triples 512 → 463 and depth-2 705 → 49.
Both populations improve, so raising the rate does not displace depth-1 learning
and the shared op-cell shows no interference between the two — it only slows
depth-1 by giving it fewer examples per step, and buying the steps back recovers
it. What is being chosen is therefore a (rate, steps) pair.

**The control that is missing is R=0 at 2M**, without which the step budget and
the rate are not separated. A budget increase alone has moved composed-slot
accuracy substantially before, though only at `d_out` 256 and against the
syntactic-frame target, which makes that figure incomparable to the ones here
and indicative of a direction rather than a magnitude (finding 17). If the
budget alone reaches most of this, the rate is an efficiency knob rather than a
necessary one.

**There is no established collapse cost.** One draw at R=0.25 reads separation
27.5× and another 33.5×, against the adopted 35.5× — the row's draw-to-draw
variation swamps the difference, which is what METRICS says to expect of a mean
over a few hundred near-zero distances. `behav-diff` stays large throughout
(0.86–0.94). An earlier reading of this as a small but real cost was a
single-draw artifact.

**This settles what the composed-chain deficit was.** The architecture was never
short of anything: the same model, trained on the same objective at the same
width and depth over the same frozen T1, reaches 97.75% on an arm that scores
19.52% at the corpus's natural composition. The missing ingredient was examples. It also dissolves the
`ADDI;ADD` anomaly of finding 26 — a constant that was linearly present but
unread reaches 100.0000% once the shape is common — so that anomaly was a
symptom of the same cause rather than a separate defect.

**And FOLDING is not the mechanism.** A same-family chain has a composite the
encoder can collapse (`ADDI;ADDI` to one offset); a mixed-family chain has none
— `ADDI;XORI` is `(x+i1)^i2`, `ADDI;SLLI` is `(x+i1)<<s`, neither reducible to
"input plus a constant". Density lifts both kinds by comparable amounts:

| composed slot | foldable | adopted | R=0.25 | R=0.50 |
|---|---|---|---|---|
| `ADDI;ADDI` | yes | 19.18% | 93.03% | 97.64% |
| `XORI;XORI` | yes | 99.13% | 100.00% | 100.0000% |
| `ADDI;XORI` | **no** | 48.73% | 98.58% | 99.71% |
| `XORI;ADDI` | **no** | 14.49% | 85.27% | 95.39% |
| `SLLI;ADDI` | **no** | 71.80% | 97.44% | 99.34% |
| `ADDI;SLLI` | **no** | 16.23% | 70.19% | 89.97% |

So the encoder learns whatever multi-step structure it is shown enough of. That
`i1+i2` becomes linearly recoverable (23.8% → 86.8%) is what it happens to learn
where a fold exists, not the reason density works — which supersedes the reading
of finding 28 that a fold objective was the one encoder-side lever. Arm
difficulty tracks how much structure the function has rather than whether it
collapses: `ADDI;SLLI`, a shift applied to a composed sum, is the worst arm at
every density and the only one still below 95% at R=0.50.

**Not yet established:** a replicate. The aggregate gain is 0.053 points against
a T2-level run-to-run floor of ~0.01 (finding 25), so ~5×; the hard-cell gain
(+16 points) and the arm gain (+74) are far outside any floor measured here, but
the aggregate figure wants the null before R=0.25 is adopted. The collapse cost
is real and small — separation 35.5× → 27.5×, with `behav-diff` holding at 0.89
— and has to be watched rather than ignored (METRICS: a vp gain that craters the
GVN rows is a regression).


## Decisions taken

- **The pool reads each slot's write step.** Unconditional: the core carries no
  alternative.
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
  step — at every tier, with no option to do otherwise.** Settled on first
  principles rather than on measurement, which cannot resolve differences this
  small (finding 25): the compressor must be a deterministic function of its
  input, the rank *is* the canonical position the target is expressed in, and a
  symmetric pool is structurally blind to write order. RISCV_DESIGN.md, "Leaf
  identity for input registers", carries the argument. Feeding the positions to
  the per-instruction cell as extra inputs (`--op-cell-rank`) is a redundant
  second route to the same binding result. The commutative-collapse cost
  recorded in finding 11 is accepted, though its size is not established.
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
  twice on separately generated corpora. Without it no row is interpretable; a
  len=2 cell at 1M steps costs ~2 hours of GPU time (7 ms/step solo, 11 ms/step
  each when two cells share the device).
- **Run the null on the arm under test — floors do not generalise between
  cells.** Two configurations here differ 20× in their own run-to-run spread
  (0.0026 against 0.0553, finding 25), so a floor borrowed from a quiet cell
  inflates every significance claim made against it. Where the quantity is a
  difference between frozen LOWER tiers, the lower tier is what needs
  replicating.
- **Rows sharing one setting but differing in another are not replicates.**
  Neither are rows trained against different oracles. Pooling them to reach a
  usable `n` manufactures confidence and can count the same quantity as signal
  in one comparison and as noise in another.
- **Report an error BUDGET, not an accuracy.** A bucket's rate and its share of
  total error are different facts and neither implies the other:
  immediate-carrying instructions at depth 1 are at 99.909% and are 22.5% of the
  error, while composed slots with an immediate are at 93.01% and are 39.5% of
  it. `probes/vp_budget.py` reports both, crossed. Localise with probes on frozen checkpoints
  first; train only to test a specific mechanism.
- **`--vp-head-depth 4` and `--value-norm` are both rejected at the current
  config**, each leaving the aggregate inside the noise floor; the deep head
  additionally costs GVN separation. Neither is a null result, though — both
  move individual arms by 15–25 points in the same direction as each other
  (finding 22), so the cell they target is a trade-off rather than a deficit.
- **Score a composed-operand change on the constructed arms, not on the
  aggregate or on the depth-2 bucket.** Depth-2 as a whole is 94.9% and both
  interventions above leave it flat while moving `ADDI;ADD` by 20 points and
  `ADDI;ADDI` by 14 in the opposite direction. An aggregate that does not move
  is not evidence that nothing happened.
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

No option exists for any mechanism below. A flag name identifies the arm that
produced the number; recover it from git history to re-measure one.

- **Feeding the vp head a composed slot's intermediate value**
  (`--vp-per-slot`, which runs the head once per output slot, with
  `--vp-inter-shuffle` as the paired control). Null, and the per-slot head it
  requires is itself a regression on the arms it targets (finding 27).
- **Supervising the intermediate value** in the RVT batch. Not run, and ruled
  out by the above: handing the head the intermediate outright is already null,
  so supervising it into a slot whose own vp is 99.99%+ has nothing to add.
- **Capacity.** Scaling the model's dimensions and steps together left
  aggregate vp essentially unchanged at a given chunk length, and `d_out`
  512 → 1024 leaves even the specific error cells unchanged (finding 17). The
  one exception was 256 → 512, which bought a fixed amount and did not repeat.
- **Buying gradient share for the hard slots.** Weighting vp by dataflow depth
  (`--vp-depth-weight`) at 8× and 32× leaves depth-2 flat and costs depth-1
  (finding 17).
- **Normalizing values into the register slot** (`--value-norm`), to put a
  seeded leaf and a computed op-cell output on one scale. The scale gap is real
  and measured — a composed operand excites the op-cell's operand weights at
  0.08-0.14x a leaf's energy — but closing it leaves depth-2 slightly worse
  (I-type 95.49% -> 94.79%; finding 17).
- **Slot-competition pooling** (`pool_mode=compete`), which made each register
  slot softmax-assign across the pooling heads so the heads had to divide the
  register file between them instead of piling onto the salient few. No vp
  benefit in either the original or the current configuration, and with
  read-order recorded in the slots it made the essence sensitive to the
  per-chunk random leaf tags of the time — rename distance 0.0251 against
  0.0014 — which it must not be.
- **Feeding the op-cell each operand's canonical read position as extra scalars**
  (`op_cell_rank`). A redundant second route to a position the rank-seeded slots
  already supply through the operand values the cell reads, and enabling both
  scores worse than seeding alone.
- **A content bypass into the slot** (`--content-skip`, a direct linear path
  from an instruction's content to the slot it writes, so the content does not
  share the op-cell's first layer with decoding the operands). Inside the noise
  floor. The composed instruction's content is not blocked on its way into the
  slot — it is fused with the operand once there.
- **Supplying the pool more ordering information than the per-slot write step.**
  Feeding it read-order, or the whole event block, leaves the shared-source
  deficit at ~83% and costs equivalence collapse; read-order plus canonical
  ranks is the worst on vp (0.0434 against the write-step baseline's 0.0405)
  while paying that cost (behavioral 0.0084 → 0.0115, separation 310× → 224×).
  The pool cannot see `content`, so it must use any channel unconditionally;
  that is the reason, and it does **not** extend to content-conditioned
  consumers such as the op-cell.
- **Injecting commutative equivalents (a corpus config's `equivalences.rate`)
  to train the invariance.** Worse, not better: on genuinely-equivalent pairs
  0.0110 → 0.0235 and separation 119× → 54×, with vp 0.0082 → 0.0252. Nothing
  in the loss ties an injected variant's essence to its base, so it adds
  correlated rows while reducing effective diversity. Partially confounded —
  the injected rows
  displace others, so that run saw 545 instructions per batch against 768 — but
  the direction is unambiguous.
- **More training as a fix for commutative collapse.** 250k → 1M moves it 0.0426
  → 0.0373, while an encoder seeded from random tags moves 0.0027 → 0.0023 over
  the same 4×; the gap is untouched.
- **Treating tag-sensitivity as the cause of the commutative-collapse
  regression.** Removing the random tags entirely takes tag-sensitivity to the
  architectural floor and leaves collapse unchanged (0.0426, against
  `op_cell_rank`'s 0.0366 and a 0.0027 baseline). Both are downstream of
  read-order reaching the essence.
- **Treating the behavioral row as contaminated.** Commute pairs share the vp
  target on 100% of len=1 and 97% of len=2 chunks, so the row measures what it
  claims and the regression is not an artifact of it.

## Where the methodology went

The probe-design rules this phase produced — calibrate against a known
quantity, check the base rate, a probe below the trained head is measuring
itself, measure one quantity at every stage to locate a loss, and the
confound-breakers that work — are durable rather than phase-specific, and are
in METRICS.md ("Probe methodology"). So is what `vp = 0` does and does not
certify, and the noise floors every comparison here is measured against.

## Open threads

1. **The acceptance criterion for milestone 1.** On the cell carrying the whole
   len=2 residual the essence is a faithful description and the vp head cannot
   evaluate it (finding 26), so `vp = 0` is a stronger demand than "the essence
   preserves the chunk's behavior". Either the criterion has to say that it
   scores the essence composed with an evaluator, or it has to be replaced by
   something that reads the description rather than the evaluation. That is a
   definition, and it gates whether the milestone is reachable at all.

2. **Folding a chain's literals in the op-cell** is the one intervention the
   measurements point at, and it is the mechanism of the equivalence-collapse
   task (TODO task 3) rather than a separate lever: handed the composite, a
   readout on the same essence goes from 79.80% to 99.95% (finding 21). What is
   unsettled is the objective that would produce the fold, and what it costs
   the discriminative rows.

3. **Decoder + reconstruction loss** — the standing untried lever for goal 2.
4. **Everything after the Round-1 sweep is len=2 only.** In particular,
   input-ordering recovery sits at chance (0.518) at len=8 and is unexplained.
