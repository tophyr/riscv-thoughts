# Phase 16 (IN PROGRESS): the register frame

**Status: open.** The essence is graded in the register frame: leaves are
identified by their structural colour in the live dataflow graph with every
operand role ordered, the essence is pooled over the written slots only, and
value prediction reads the essence — for each written output, a readout
network (the evaluator) is handed the essence, a pointer (a one-hot naming
which of the chunk's distinct outputs to evaluate) and the
values and colours of the output's inputs
gathered by register, never the output's own description (RISCV_DESIGN, "GVN",
"Leaf identity for input registers" and "Value prediction reads the
essence"). Every equivalence that leaves the live dataflow graph unchanged —
rename, reordering, dead code — collapses by construction and reads zero on
every trained model. Without a collapse objective, value prediction from the
essence is exact at T1 and 99.83% at T2 len=2, and nothing the op-cell has to
compute collapses (findings 37–39). Collapse is trained with a behavioral
distance term: every two rows' essence distance is matched to how differently
they behave at shared witness states, their inputs placed by canonical role,
an order computed from the function (findings 41 and 43); training anchors
aimed at each row's boundaries keep value prediction exact under it (finding
42). The frozen T1, `runs/t1_distance_cosbal_500k`, is exact at held-out
anchors. A T2 at len=2 over it, trained on partner chunk pairs drawing every
rewrite with the distance target's chord √(2d) and equivalent pairs averaged
apart from the rest, collapses every equivalence class tried at vp 99.79%,
one seed (finding 50); a collapse holds only while its pairs stay in the
training mix (finding 47), and reaches no equivalence class the mix does not
draw (finding 53). That target departed from the behaviour it stands for —
the order outputs were paired in, same-value writes, a spelling-dependent
bound — and the encoder could not tell an in-place write from a copy
(finding 51); the code trains toward a target that tracks the ideal
distance d\* (finding 52), and partner pairs come from the rewrites and from a
library found by search and proved: the T2 record on it collapses every class
the probes build, named or not, at vp 99.80% (finding 57), and a family whose
pairs are held out of the search does not collapse (finding 56). Open:
equivalences with no single-instruction form, the looser collapse of the
rewrite classes, a second seed, the binding's wire format, and the learned
replacement for the colouring.

Of the runs named below only `runs/t1_distance_cosbal_500k`, finding 50's T2,
and the runs of findings 54, 56 and 57 are kept, and of the corpus configs only those still in `configs/`; a
deleted name is the record of what was measured, not a loadable artifact.
`runs/eval/` holds the latest regeneration of the held-out corpora; a finding
that cites them scored the regeneration current at the time.

Findings 1–36 (phases 14 and 15) were measured in the first-read canonical
frame; their numbers are not comparable to these (METRICS.md).

**37. In the register frame — with the encoder told which operations
commute — T1 is exact and every structural equivalence collapses to zero; T2
at len=2 reaches 99.8385% after 1M steps, with the composed slots carrying the
residual, and its essence varies 30× less between unrelated chunks than in
the first-read frame.** The models here carried a list of commutative
opcodes into the encoder: T1 learned to emit a symmetry bit against it and
T2's colouring treated such an operation's two roles as one, which is what
made the commutative-swap rows below exactly zero. The list reaches only the
oracle in the code (Decisions); the swap rows are the only figures it
touched. Every figure is at the fixed held-out anchor set, 64 states, on
corpora generated with `datagen --heldout-anchors` from
`configs/instr_alu_writers.json`: `runs/eval/t1_heldout64_single.rvt` and
`runs/eval/t2_heldout64_len2.rvt`, 100 batches of 384 rows each (twins
included). Scores are `probes/frame_scorecard.py`'s; a value is one output
register of one row at one anchor state; essence distances are L2 between
unit-normalized essences, 2,000 pairs per class.

*The T1*, `runs/t1_frame_500k`: `d_model` 128, `d_out` 64,
`imm_weight` 0.3, `sym_weight` 0.1, three-layer residual vp head, 500k steps
at 4 ms/step (33 min).

| T1 | value |
|---|---|
| vp exact at held-out anchors | **100.0000%** (0 failing of 2,457,600) |
| emitted wiring exact | 100.0000% of 38,400 instructions |
| emitted symmetry bit exact | 100.0000% of 38,400 |
| rename / commutative swap, fresh or aliased destination | 0 (float noise ≤ 3.5e-7; swaps bit-exact on 2000/2000) |
| aliasing control (`x+x` vs `x+y`) | 0.3220 |
| unrelated instructions | 0.8332 |

*The T2 cell*, `runs/t2_len2_d512_s1000000_dep4_vp12_pool16_alu_writers_chain010`:
`scripts/run_t2.sh` at len=2, `d_out` 512, op-cell depth 4, 16 pool heads,
twelve-layer residual vp head, chain rate 0.10, 1M steps at 15 ms/step (4.0 h;
8 ms in the first-read frame — the evaluator's input is 3,616 wide per slot
and the colouring runs eight rounds), one instruction-stream seed.

| T2 | value |
|---|---|
| vp exact at held-out anchors | **99.8385%** (7,809 failing of 4,834,176) |
| — outputs depending on 3 inputs / 2 / 1 / 0 | 90.632% / 99.866% / 99.969% / 100.000% |
| — by writing opcode, lowest | `SUB` 99.201%, `SRL` 99.588%, `ADD` 99.624%, `SLL` 99.627%, `SRA` 99.772% |
| binding, inputs and outputs exact per chunk | 100.0000% of 38,400 |
| rename / reorder / dead-code insertion / commutative swap of fresh operands | **0 on 2000/2000 each** (≤ 2.7e-7) |
| commutative swap whose operand is another instruction's input or result | 0.0051 (max 0.0223; 69/2000 at zero) |
| aliasing control | 0.0192 |
| unrelated chunks | 0.0233 |

A three-input output is a composed slot whose two instructions read disjoint
inputs; it is the residual's home, as depth 2 was in the first-read frame
(96.1% at 1M there, finding 35; a different evaluator input, so not a
like-for-like figure). `scripts/eval.py`'s rows agree: rename 0.0000, swap on
equivalent pairs 0.0004 (n=194), distinct 0.0231.

*The op-cell classes, on the same pair.* `probes/additive_fusion.py`:
`ADDI` chains with one composite 0.0236 against 0.0072 for a different
composite (0.3×); `XORI` 0.0200 against 0.0049 (0.2×). `probes/write_position.py`:
a write at step 1 against step 2, and a single instruction alone against padded
to two slots, both 0.0000; `ADDI r,r,0` against `ADD r,r,x0` 0.0227, against
`ADDI r,r,k; ADDI r,r,-k` 0.0305, against the empty chunk 0.0198, with
denominators 0.0181 (`ADDI r,r,1`) and 0.0071 (`+k;-k+1`). `probes/swap_classes.py`:
the non-fresh swap rows at 1.6e-3 and 6.6e-4 trained against 2.4e-2 and
5.5e-2 at random initialisation. So nothing the op-cell has to compute
collapses, as in the first-read frame; in the register frame nothing the
graph determines is left to training.

*The essence's dynamic range.* Unrelated chunks sit 0.0233 apart, against
0.67–1.19 in the first-read frame, and the aliasing control at 0.0192 is
within that. The essence receives no vp gradient: value prediction reads the
written slot's description in `T`, and the only losses on the essence are the
validity magnitude and `pc_writes`, so its direction is shaped by nothing
that asks it to tell chunks apart. The ratios between classes are of the
same order as in the first-read frame (the aliasing control and the fusion
rows separate by similar factors), so the structure may survive at a
thirtieth of the scale rather than be gone; what shaped it in the first-read
frame was vp reading the essence (the opcode clustering) and T1's immediate
head (the rotary immediate axis), which this T1 trains and T2 does not have.
Finding 39 has value prediction read the essence.

*A pool defect.* The pool projects only the written slots and fills a short
row's spare picks from the lowest-index unwritten slots, which is sound only
if those rows are zero; a read-only leaf's row is not (it holds the leaf's
seed), and x1 is the first candidate, so a chunk with one written slot that
read x1 projected that leaf into the pool, and dead-code pairs whose inserted
instruction read x1 sat 0.02 apart. The rows are masked before selection
(`_project_touched`, `compressor/model.py`). The T2 cell above was trained
with the defect — it touches rows with one written slot that read x1 — and
scored with the mask; T1 is unaffected (one instruction, one write, no spare
pick).

**38. Without the commutative list in the encoder: T1 exact, T2 at len=2
99.8322% after 1M steps, every graph-identity class exactly zero, and a swap
whose operand is shared stays apart.** Same corpora, scorer, cell and budget
as finding 37; the only change is that every operand role is ordered in the
colouring, the list reaching only the oracle's labels, and the pool's rows are
masked in training as well as scoring. T1 `runs/t1_register_500k`: vp
100.0000% (0 of 2,457,600), wiring 100.0000%. T2: finding 37's cell
configuration:

| T2 | value |
|---|---|
| vp exact at held-out anchors | **99.8322%** (8,113 failing of 4,834,176; finding 37: 7,809) |
| — outputs depending on 3 / 2 / 1 / 0 inputs | 90.324% / 99.864% / 99.964% / 100.000% |
| — lowest writers | `SUB` 99.249%, `SRL` 99.562%, `SLL` 99.584%, `ADD` 99.625% |
| binding exact per chunk | 100.0000% of 38,400 |
| rename / reorder / dead-code insertion | **0 on 2000/2000 each** (≤ 1.9e-7) |
| commutative swap of two fresh leaves | 0 on 2000/2000 — a rename of the two leaves, not a commutativity test |
| commutative swap whose operand is shared or a result | **0.0125** (max 0.041; 66/2000 at zero) — the commutativity test, not collapsed |
| aliasing control | 0.0198 |
| unrelated chunks | 0.0281 |

So the list bought nothing but the shared-operand swap row, which is the only
place commutativity is tested at len=2, and value prediction alone leaves it at
2.2× separation from unrelated chunks — the op-cell is not commutative. The
essence's range is as in finding 37 (0.028 between unrelated chunks).
`scripts/eval.py` agrees: swap on equivalent pairs 0.0012, distinct 0.0229.

**39. With value prediction read from the essence through each output's
pointer, T1 is exact, T2 at len=2 is as faithful as finding 38's per-slot
cell, and the essence spans the first-read frame's range — 25× finding 38's;
nothing the op-cell has to compute collapses, and the essence separates
those equivalents by their spelling.** Each written output carries a pointer,
a one-hot naming its position among the chunk's distinct outputs. In this
finding and findings 40–45 the order is a content hash: each output's data
path is coloured by the leaf colouring's pass with every operation also
seeded from its content, and the chunk's distinct paths are sorted by a fixed
projection of that colour; finding 46 replaces it with an order by the
outputs' descriptions. The pool adds each written slot's pointer
to its attention key, and the evaluator is handed the unit-normalized essence,
the pointer and the inputs' values and colours in place of the slot's
description (`vp_loss`, `compressor/train.py`). Training used only this term;
finding 38's cells used only the per-slot one. Both models here were trained,
and scored, with the evaluator's inputs in the register-dependent order
finding 40 describes, on corpora carrying two register-relabelled copies of
every chunk. Scores are `probes/frame_scorecard.py`'s at the 64 held-out
anchor states, on that finding's corpora; essence distances are L2 between
unit-normalized essences, 2,000 pairs per class. A second exact-match rate,
reported as the leak check, hands each row's evaluator another row's essence
with the row's own pointer and inputs; it tests that the function reaches the
evaluator only through the essence, and its chance floor is not measured
(METRICS.md, "vp").

*The T1*, `runs/t1_essonly_500k`: as finding 38's T1, 500k steps at
3–4 ms/step.

| T1 | finding 38 (per-slot evaluator) | essence evaluator |
|---|---|---|
| vp exact at held-out anchors | 100.0000% | **100.0000%** (0 of 2,457,600) |
| leak check | — | 12.14% |
| emitted wiring exact | 100.0000% | 100.0000% of 38,400 |
| rename | 0 | 0 (≤ 4.3e-7) |
| aliasing control (`x+x` vs `x+y`) | 0.2571 | 0.6308 |
| unrelated instructions | 0.9197 | 1.3688 |

*The T2 cell*, `runs/t2_len2_d512_s1000000_dep4_vp12_pool16_essonly_alu_writers_chain010`:
as finding 38's cell, 1M steps at 14 ms/step (3.9 h).

| T2 | finding 38 | essence evaluator |
|---|---|---|
| vp exact at held-out anchors | 99.8322% (8,113 failing of 4,834,176) | **99.8276%** (8,333 failing) |
| — outputs depending on 3 / 2 / 1 / 0 inputs | 90.32 / 99.86 / 99.96 / 100% | 90.54 / 99.86 / 99.95 / 100% |
| leak check | — | 16.50% |
| binding exact per chunk | 100% | 100% of 38,400 |
| rename / reorder / dead-code insertion / fresh swap | 0 | **0** (≤ 6.5e-7) |
| commutative swap, shared operand | 0.0125 | 0.3760 |
| aliasing control | 0.0198 | 0.5051 |
| unrelated chunks | 0.0281 | **0.6901** |

The two vp figures differ by 0.005 points, inside the 0.023-point floor
between T2s over different T1s (METRICS.md, "Current calibration"). The
unrelated-chunk distance matches the first-read frame's residual-headed 10M
model (0.67–0.69, finding 35).

*The op-cell classes*, `probes/additive_fusion.py` and
`probes/write_position.py` on the same pair:

| pair | finding 38 | essence evaluator |
|---|---|---|
| `ADDI` chains, same composite / different composite | 0.0303 / 0.0091 | 0.7937 / 0.3221 (0.4×) |
| `XORI` chains, same composite / different composite | 0.0355 / 0.0076 | 0.7651 / 0.1927 (0.3×) |
| `ADDI r,r,0` vs `ADD r,r,x0` | 0.0187 | 0.1029 |
| `ADDI r,r,0` vs `ADDI r,r,k; ADDI r,r,-k` | 0.0473 | 0.9801 |
| `ADDI r,r,0` vs `ADDI r,r,1` (inequivalent) | 0.0155 | 0.2756 |
| unrelated two-instruction chunks | 0.0283 | 0.6934 |

A fusion pair changes both literals and its denominator changes one, and the
fusion pair sits further apart; an identity sits further from a cancelling
pair than unrelated chunks sit from each other. The essence describes the
instructions as written: the pressure that makes it carry every output's
function does not make it fold one, as in the first-read frame (ratios
0.8–1.0×), and here at that frame's scale.

**40. Register-relabelled copies of a chunk carried training signal only
through a register-number leak in the evaluator's input order; with the order
taken from the encoder's colouring they duplicate their source exactly.**
The corpora of findings 37–39 gave each chunk two register-relabelled copies
sharing its anchor draw. On finding 39's T2, over 40 chunks of one training
batch, 68 of the 80 copies gave their source's loss and full parameter
gradient to float noise (median relative gradient difference 3.8e-5) and 12
differed by up to 2.8× (cosine down to −0.64); two unrelated chunks differ by
0.90. All 12 contained `ADD`, `AND`, `OR` or `XOR`, and in each the two
operands of that operation were listed in the other order in `out_inputs`,
the per-output list of input registers in the order the evaluator reads them:
that order came from the oracle's colouring, which merges a commutative
operation's two roles, with the resulting tie broken by register number,
while the encoder colours the two roles apart — so a rename moved the two
values between evaluator positions. With the order computed from the
colouring with every role ordered (`datagen.signature.output_inputs`), on a
batch drawn from the same seed, every copy matches its source (relative
gradient difference ≤ 2.2e-4, cosine ≥ 0.99999976, against 0.81 between
unrelated chunks), and a test pins the property
(`test_a_rename_never_moves_an_input_between_gather_positions`,
`datagen/tests/test_signature.py`); the canonical role order of finding 43
keeps it (`datagen/tests/test_roles.py`). Batches carry 384 distinct chunks
and no copies. The models of findings 37–39 were trained and scored with the
register-dependent order. `runs/t1_essence_500k` is finding 39's T1 trained
with the encoder's input order on batches of distinct chunks (500k steps, 4
ms/step), scored on held-out corpora regenerated the same way,
38,400 distinct instructions:

| T1 | finding 39's T1 | `runs/t1_essence_500k` |
|---|---|---|
| vp exact at held-out anchors | 100.0000% | **100.0000%** (0 of 2,457,600) |
| leak check | 12.14% | 11.44% |
| emitted wiring exact | 100.0000% | 100.0000% |
| rename | 0 | 0 (≤ 3.9e-7) |
| aliasing control | 0.6308 | 0.7363 |
| unrelated instructions | 1.3688 | 1.3658 |

No T2 was trained over it.

**41. At T1, a behavioral distance term collapses equivalent spellings while
keeping inequivalent ones graded by how differently they behave; InfoNCE
collapses them but flattens everything else; and the immediate head
fights the distance term while also masking the one place its target is
wrong.** Four T1 arms over `runs/t1_essence_500k`'s configuration (finding 40),
each 500k steps on the writers corpus with a quarter of every batch made of
partner pairs (`configs/instr_alu_writers_partners025.json`): two different
spellings of one register function on the same registers, drawn from groups
of the equivalence manifest merged by the function computed — copying a
register, writing zero, doubling, the sign test, loading a constant
(`datagen.generate.PARTNER_GROUPS`, every spelling checked on the emulator).
For the InfoNCE arms each row also carried an equivalence key, shared by rows
computing one function: the constant written when no source matters, else the
partner group whose spelling it is, else its structural signature, so a
register rename shares its key. The arms differ in the collapse term:

- *InfoNCE*: supervised InfoNCE on the unit-normalized essence — each row
  pulled toward the rows sharing its key and pushed from every other row
  (weight 0.1, temperature 0.1);
- *InfoNCE + uniformity*: the same plus a term repelling every two rows
  of different keys, log-mean-exp(−2·distance²) (Wang & Isola);
- *distance*: an MSE between every two rows' essence distance (chord, 0–2) and
  twice their **behavioral distance** (weight 1.0; the target form in
  findings 41–48, which findings 48–50 change). The
  behavioral distance executes both rows at 64 shared witness states, their
  inputs placed in colour order (finding 43 replaces it with canonical role
  order) so that inputs correspond positionally, and
  averages log1p(log1p(d)) / log1p(log1p(max_dist)) over the states, d being
  the wrap-around difference of the two outputs and max_dist the largest
  difference those two outputs could ever have, from a known-bits analysis of
  each output (`datagen/knownbits.py`: two comparisons are at most 1 apart);
- *distance, no immediate head*: the same with the immediate head — a linear
  readout of the essence trained to a rotary code of the written immediate —
  switched off.

Scores at the 64 held-out anchors on held-out corpora of 38,400 distinct
instructions, `probes/frame_scorecard.py`; the T1 baseline and the two
InfoNCE arms were scored on one regeneration of that corpus and the
distance arms on the next, so their vp and leak rows come from different draws
while their distance rows come from the probe's own seeded pairs. Distances
are essence chord distances, 2,000 pairs per row; a partner-group row is two
spellings bound to the same registers against the group's contrast spelling,
a near-miss computing something else.

| | baseline | InfoNCE | InfoNCE + uniformity | distance | distance, no imm head |
|---|---|---|---|---|---|
| vp failing of 2,457,600 | 0 | 2 | 6 | 23 | 108 |
| leak check | 11.19% | 12.45% | 10.91% | 11.10% | 10.45% |
| copy: equivalent / contrast | 0.709 / 0.963 | 0.049 / 1.475 | 0.027 / 1.462 | 0.016 / 0.952 | **0.0019** / 0.947 |
| zero | 1.150 / 1.282 | 0.131 / 1.481 | 0.145 / 1.448 | 0.148 / 1.472 | 0.107 / 1.471 |
| double | 1.418 / 1.042 | 0.0055 / 1.460 | 0.0001 / 1.428 | 0.0012 / 1.300 | 0.0008 / 1.302 |
| sign | 1.172 / 0.835 | 0.0076 / 1.474 | 0.082 / 1.432 | 0.062 / 0.307 | **0.0004** / 0.295 |
| constant | 0.196 / 1.263 | 0.0053 / 1.368 | 0.023 / 1.358 | 0.160 / 1.106 | 0.084 / 1.123 |
| aliasing control | 0.736 | 1.130 | 1.454 | 1.156 | 1.155 |
| unrelated instructions | 1.366 | **0.614** | 1.400 | 1.360 | 1.362 |

Near-misses, 2,000 pairs each, `probes/near_miss.py` (the InfoNCE arm
without uniformity was not probed); the target is twice the pairs' behavioral
distance, the value the distance term trains toward:

| pair | baseline | InfoNCE + uniformity | distance | distance, no imm head | target |
|---|---|---|---|---|---|
| `ADDI` k vs k+1 | 0.645 | 0.816 | 0.232 | **0.335** | 0.34 |
| `ADDI` k vs k+512 | 0.654 | 0.685 | 0.742 | 0.785 | 1.27 |
| `ADD` vs `SUB`, same registers | 1.816 | 1.451 | 1.277 | 1.277 | 1.48 |
| `SLLI 3` vs `SLLI 4` | 0.771 | 1.287 | 1.048 | 1.059 | 1.49 |

*InfoNCE* collapses every group, but rows sharing their key
with no other row of the batch are never pulled or pushed, so unrelated
instructions drift together (0.61). With uniformity they keep their spread,
and every inequivalent pair is driven toward orthogonal, closest pairs
hardest: `SLLI 3` against `SLLI 4` goes from 0.77 to 1.29 and `imm+1` sits
further than `imm+512`. Neither term says how far apart two different
functions should be.

*The distance term* collapses the groups, keeps the unrelated spread, and
grades the rest: `imm+1` lies well inside `imm+512`, and a contrast that
differs from its group only at one input (`SLTI rd,rs,1` against the sign
test) sits at 0.30 rather than at the far end. Achieved distances sit below
their targets at the far end — the MSE plateaus near 0.03 from 50k steps —
with the ordering kept.

*The immediate head* fights the distance term wherever two equivalent
spellings carry different immediates (`ANDI rd,rs,0` against
`SLLI rd,x0,k`), since a linear readout cannot decode two codes from one
point; without it copy and sign collapse a further 8× and 150×, and `imm+1`
lands on its target. It was also holding apart pairs the distance target
wrongly calls identical:

| `k` vs `k+1` | target | pairs with target 0 | baseline | distance | no imm head |
|---|---|---|---|---|---|
| `SLTI` | 0.0000 | 100% | 0.647 | 0.100 | **0.0012** |
| `SLTIU` | 0.0033 | 99.7% | 0.652 | 0.104 | 0.0063 |
| `ADDI` | 0.169 | 0% | 0.649 | 0.240 | 0.343 |

The two comparisons differ only at x = k, which no witness state holds, so the
term pulls two different functions together, and value prediction, which
rarely sees x = k either, fails on them: the no-head arm's 108 failures are all
`SLTI` (99.948%), `SLTIU` (99.967%) and `ADDI` (99.998%). Averaging over states
cannot separate functions that disagree on one input in 2³², and putting them
close is the geometry intended; what keeps them distinguishable has to come
from value prediction seeing the boundary.

The distance arms ran at 10–12 ms/step against 4 ms, with the GPU at 71–76%,
their targets computed eagerly before each compiled step (finding 42 replays
them as one CUDA graph).

**42. A T1 trained with the behavioral distance term evaluates exactly at
held-out anchors once half of each row's anchor states straddle its own
boundaries at every scale; straddling only the step where an output flips
leaves the comparisons misplaced by hundreds.** The distance term grades
`SLTI k` against `SLTI k+δ` by how many witness states fall between the two
thresholds, which is few for any δ inside the immediate's range, so it puts
their essences close together, as intended:

| `SLTI` pair | target | no distance term (`runs/t1_essence_500k`) | distance, unaimed | distance, aimed at every scale |
|---|---|---|---|---|
| k vs k+1 | 0.0001 | 0.644 | 0.0010 | 0.0103 |
| k vs k+64 | 0.0125 | 0.645 | 0.0154 | 0.0227 |
| k vs k+256 | 0.0436 | 0.627 | 0.0472 | 0.0458 |
| k vs k+1024 | 0.2086 | 0.633 | 0.2002 | 0.1997 |

(chord distances between unit-normalized essences, 500 pairs per row on shared
registers; the target is twice the pairs' behavioral distance.) The evaluator
then has to place k to the unit from that geometry, and a structured draw puts
an input within 256 of k on about 0.4% of states.

The aiming (`datagen/aim.py`; RISCV_DESIGN, "Training") rewrites a fraction
of each row's states in pairs, each pair straddling one of the row's
boundaries: a comparison's threshold exactly (k−1 and k), the same threshold
at a distance δ drawn log-uniformly from 2 to 2¹¹ (k−δ and k−1+δ), and an add
whose result crosses 2ʰ, for h drawn from 4 to 32. Three arms over finding
41's distance arm without the immediate head, each 500k steps, everything else
equal:

- *unaimed*: `runs/t1_distance_noimm_500k` (finding 41);
- *exact threshold*: a quarter of the states aimed, enough for one pair, so
  comparisons get only k−1 and k and adds their carry;
- *every scale*: half the states aimed, so a comparison gets both pairs
  (`configs/instr_alu_writers_partners025_aim050.json`), trained on two
  instruction-stream seeds (`runs/t1_distance_noimm_aim050_500k`, and
  `_s7` on a separately drawn corpus).

All four are scored on one regeneration of the held-out corpus
(`runs/eval/t1_heldout64_single.rvt`, 38,400 distinct instructions at the 64
held-out states, which are never aimed), with `probes/frame_scorecard.py` and
`probes/near_miss.py`; the unaimed arm reads 106 failing triples here and 108
on the regeneration finding 41 used.

| | unaimed | exact threshold | every scale, seed 0 | every scale, seed 7 |
|---|---|---|---|---|
| vp failing of 2,457,600 | 106 | 181 | **0** | **0** |
| — `SLTI` / `SLTIU` / `ADDI` | 99.949 / 99.968 / 99.998% | 99.929 / 99.927 / 100% | 100% | 100% |
| leak check | 10.43% | 11.18% | 10.75% | 10.83% |
| copy / zero / double / sign / constant, equivalent | 0.0019 / 0.109 / 0.0008 / 0.0004 / 0.084 | 0.0024 / 0.109 / 0.0016 / 0.0003 / 0.078 | 0.0019 / 0.108 / 0.0008 / 0.0009 / 0.094 | 0.0022 / 0.108 / 0.0008 / 0.0004 / 0.079 |
| aliasing control (`x+x` vs `x+y`) | 1.155 | 1.155 | 1.155 | 1.154 |
| unrelated instructions | 1.362 | 1.360 | 1.360 | 1.362 |
| `ADDI` k vs k+1 (target 0.338) / k vs k+512 (target 1.272) | 0.335 / 0.785 | 0.315 / 0.808 | 0.358 / 0.813 | 0.339 / 0.790 |

The carry pair alone fixes `ADDI`. The exact threshold alone removes every
failure at x = k or k−1 (12 in the unaimed arm, 0 with it) and leaves the
comparisons worse overall: 142 of its 181 failures, like 50 of the unaimed
arm's 104 comparison failures, sit 8 to 4,096 from k, on a handful of
mid-range held-out inputs (8 and 13 distinct values). Aiming at every scale
removes them on both seeds, and the collapse, the partner contrasts and the
near-miss grading stay where finding 41 left them.

The aimed corpus costs 14% more generator time per batch at len=1 (one
process, 384-row batches, 8 states). The distance arms here ran at 8 ms/step
with the GPU at 98–99%; finding 41's arms ran the same targets eagerly at
10–12 ms and 71–76%, and replaying them as one CUDA graph gives bit-identical
targets (`GraphedDistanceTargets`, `compressor/train.py`).

**43. Placing the witness states by the dataflow graph's colour order gives
equivalent chunks different distance targets whenever a commutative swap
changes the graph; placing them by canonical role, an order computed from
the function, gives every equivalent pair tried identical targets.** The behavioral distance compares two
rows at the witness states, each row's inputs placed in some order. The
colour order is computed from the wiring with every operand role ordered, so
swapping the operands of an `ADD` whose operand another instruction also
reads changes it. Canonical roles (`datagen/roles.py`; RISCV_DESIGN,
"Collapse"): each input is fingerprinted by the chunk's outputs on 64 fixed
states where it holds one value and every other input a second, each ordered
pair likewise with a third value; inputs are ordered by Weisfeiler–Lehman
refinement over these fingerprints, singling out one member of any class
left with several and refining again; inputs the function ignores are
dropped first; outputs are sorted by the raw bytes of their values at the
witness states.

Oracle only, no model. Pairs equivalent on the emulator, 300 per class: the
fraction whose outputs at the witness states agree exactly.

| class | colour order | canonical roles |
|---|---|---|
| commutative swap of every `ADD`/`AND`/`OR`/`XOR` in a natural chunk, len 2 / 4 / 8 | 88.3 / 62.3 / 36.3% | 100 / 100 / 100% |
| associativity of `ADD`, `XOR`, `AND`, `OR`; `(a−b)−c` vs `a−(b+c)`; `(a−b)+c` vs `(c−b)+a`; `ADDI` folds; `+k;−k` vs `+0`; two parallel copies of one operation; renames at len 4 and 8 | 100% | 100% |
| a chunk reading an input its function ignores against one that does not read it (four forms, 200 each) | — | 100% |

The colour order's 100% rows are functions symmetric in the inputs that
move, which any placement satisfies.

Natural chunks of the writers corpus at chain rates 0.10 and 1.00, 400 per
cell at len 2, 4 and 8: 20–54% of chunks leave a class with several inputs
that has to be singled out, and in none of the 2,400 did the choice of
member change the outputs at the witness states (checked at the first
choice in each chunk). Fingerprinting on 16 states instead of 64 leaves more
inputs tied before the singling out; 256 changes nothing. Changing one
I-type immediate by one reorders the canonical inputs in 1.0, 1.8 and 5.8%
of pairs with two or more inputs at len 2, 4 and 8.

At len=1, canonical roles place the two operands of `SUB`, `SLT` and `SLTU`
differently from the colour order (203 of 282 such rows over five batches of
the T1 corpus), so the T1 distance targets move: mean change 0.0115, 2.8% of
pairs by more than 0.1, correlation 0.953 with the targets finding 42's T1s
trained on. A T1 retrained on them (`runs/t1_distance_noimm_aim050_roles_500k`, finding 42's
configuration otherwise, 500k steps, 7 ms/step) follows them: on the 16,819 pairs of ten writers-corpus batches whose
target moved by more than 0.2 — almost all `SUB`, `SLT` or `SLTU` against
another operation — its essence distance sits 0.140 from the new target and
0.459 from the old, while finding 42's two T1s sit 0.530 and 0.529 from the
new and 0.093 and 0.094 from the old. Everything else holds: vp exact on all
2,457,600 held-out triples, partner groups equivalent at 0.0009–0.105
against contrasts of 0.29–1.47, aliasing control 1.152, unrelated
instructions 1.360, `ADDI` k vs k+1 at 0.333 (target 0.338).

Computing the roles for every generated row makes one generator core take
66, 102, 193 and 401 ms per 384-row batch at len 1, 2, 4 and 8, against 49,
70, 125 and 239 without them, with every fingerprint of a chunk executed in
one stacked call; 96 generator workers deliver at least 450 and 290 batches/s
at len 1 and 2.

The evaluator takes each output's inputs in the same canonical role order,
so two equivalent chunks of different dataflow shapes present corresponding
inputs at the same positions; a class of interchangeable inputs is ordered by
the encoder's colour, the singled-out member first, which leaves a
commutative operation's two operands in the colour order the evaluator had.
On a held-out corpus in that order, `runs/t1_distance_noimm_aim050_roles_500k` scores
87.87%, every failure on `SUB` (8.7%), `SLT` (31.5%) or `SLTU` (41.2%), whose
two operands the roles re-pair; retrained with it
(`runs/t1_distance_noimm_aim050_canon_500k`, otherwise as above), T1 is exact
on all 2,457,600 held-out triples, with partner groups equivalent at
0.0003–0.102 against contrasts of 0.30–1.46, aliasing control 1.152,
unrelated instructions 1.361 and `ADDI` k vs k+1 at 0.377.

**44. A T2 trained with the behavioral distance term on one-output rows and
partner chunk pairs is as faithful as finding 39's T2 and collapses every kind
of partner pair it trained on, by 5× to 3,500× against a near-miss; it does
not collapse a commutative swap whose operand is the previous instruction's
result, a shape no partner kind produces, and it pushes the identity away
from the empty chunk.** The cell is finding 39's (`scripts/run_t2.sh`: len=2,
`d_out` 512, op-cell depth 4, twelve-layer residual vp head, 16 pool heads, 1M
steps, one seed) over `runs/t1_distance_noimm_aim050_canon_500k`, on
`configs/instr_alu_writers_chain010_partners025_aim050.json` — the chain-0.10
writers corpus with a quarter of each batch partner chunk pairs and half the
anchor states aimed — with the distance term at weight 1.0 on every two rows
that write one register (`runs/t2_len2_d512_s1000000_dep4_vp12_pool16_dist1.0_alu_writers_chain010_partners025_aim050`;
20 ms/step, 5.5 h). Scored with `probes/frame_scorecard.py` on
the regenerated `runs/eval/t2_heldout64_len2.rvt`, and with
`probes/partner_chunks.py`, `probes/additive_fusion.py` and
`probes/write_position.py`, 1,000–2,000 pairs per row, beside finding 39's T2
(which reads 99.55% on this corpus, trained with an evaluator input order
finding 40 removed).

| | distance term | finding 39's T2 |
|---|---|---|
| vp exact at held-out anchors | **99.8416%** (7,652 of 4,831,744 failing) | 99.8276% on its own corpus |
| — outputs depending on 3 inputs | 91.3% | 90.5% |
| binding exact per chunk | 100% | 100% |
| rename / reorder / dead code / fresh swap | 0 (≤ 1.5e-6) | 0 |
| aliasing control / unrelated chunks | 1.024 / 1.332 | 0.505 / 0.690 |

The corpus's five partner kinds (a generator since replaced by finding 48's
rewrites), equivalent pair against the second spelling with one immediate +1
or its first operation changed:

| kind | distance term | finding 39's T2 |
|---|---|---|
| respelled first step | 0.022 / 0.810 | 0.203 / 0.441 |
| fold of two literals | 0.075 / 0.405 | 0.779 / 0.786 |
| a copy through `+k;−k` against the plain copy | 0.002 / 0.333 | 1.009 / 1.016 |
| reassociation | 0.0003 / 1.058 | 0.185 / 1.105 |
| commutative first step whose operand is read again | 0.0006 / 0.984 | 0.169 / 0.813 |

Beyond the partner kinds: `ADDI` chains of one register reading itself, same
total against a total differing by one, 0.163 against 0.418 (2.6×; finding 39:
0.4×); `XORI` chains 0.060 against 0.354 (5.9×; 0.3×); the identity against
`+k;−k` in place, 0.002 (0.980). Commutative swaps in naturally drawn len-2
chunks, against unrelated chunks at 1.344 (1.063):

| how the swapped instruction meets the other | one output | two outputs |
|---|---|---|
| shares an input | 0.000 (the swapped instruction is overwritten) | 0.738 (0.337) |
| its result is read by the other | 0.003 (0.009) | 0.012 (0.019) |
| its operand is the other's result | **0.289 (0.271)** | 0.598 (0.445) |

The swap in the partner kinds is always the first instruction, on two inputs
from outside; a swap on a composed operand is the same algebra in an operand
context no partner pair reaches, and it moves not at all. Two-output chunks,
which the term never pairs, keep or lose ground as the essence spreads. The
identity `ADDI r,r,0` sits 1.551 from the empty chunk (0.261 in finding 39),
against 0.333 from `ADDI r,r,1`: the empty chunk has no output, so the term
never places it.

**45. Two equivalent two-output chunks file their functions under different
pointers in about 40% of commutative swaps whose operand is shared or
composed, and almost never otherwise, when pointers are ordered by content
hash; an order by a projection of each output's description agrees far more
often.** The content-hash pointer (finding 39) orders a chunk's outputs by a
fixed hash of each output's content-aware path colour, so whether it flips
depends on the T1 and the fixed hash, not on T2's training. Equivalent pairs
of two-output len-2 chunks, checked on the emulator, over
`runs/t1_distance_noimm_aim050_canon_500k` (read through finding 44's T2);
1,000 pairs per row; a flip is any output register whose pointer differs:

| pair | pointer flips |
|---|---|
| one instruction respelled within a T1 partner group, step 1 / step 2 | 0.7% / 0.7% |
| commutative swap, no register shared with the other instruction | 0.1% |
| commutative swap, its result read by the other | 1.4% |
| commutative swap, sharing an input with the other | 43.6% |
| commutative swap, its operand the other's result | 40.8% |

A respelling barely moves the hash because the T1 already places equivalent
spellings at one essence (finding 43). A swap reorders operand roles, which
the colouring keeps ordered because the encoder is told nothing about
commutativity, and the swapped output's hash lands on either side of the
other's at nearly even odds. These are the two-output classes finding 44
left apart (0.738 and 0.598).

Ordering the outputs instead by a fixed projection of each written slot's
description (its value channels, which the op-cell computes before any
pointer enters) agrees between the two spellings far more often. On the same
pairs through finding 44's T2, over 8 fixed random unit directions: 99.1% and
98.9% for the respellings, 100% and 99.8% for the swaps without a shared or
composed operand, 86.2% for a swap whose operand is the other's result and
77.3% for one sharing an input (hash: 58.1% and 56.3%). The median ratio of
the two spellings' difference along the direction to the gap between a
chunk's two outputs is 0.003 where the essences already coincide, and 0.443
and 0.709 on the two swap classes; 0.4–2.6% of chunks have their two outputs
within 1% of the median gap. This T2 was never trained on two-output pairs.

**46. Ordering pointers by a projection of each output's description keeps
two-output equivalents' pointers in agreement more often than the hash and
settles as training runs, but does not by itself collapse the swaps whose
operand is shared or composed, because the op-cell does not describe a
commutative operation alike when one operand is another instruction's result — with one output and
no pointer to disagree on, that swap stays where it was.** Finding 44's T2,
its pointer replaced by the description order (`RegisterStateMachine._pointer`:
a fixed unit direction, ties within 1e-4 sharing a position) and its distance
target by the root-mean-square over canonically ordered outputs
(`chunk_distance_targets`, `compressor/train.py`), same-value writes dropped
from the outputs (`precompute_chunk`, `datagen/compare.py`),
trained 100k steps more (learning rate 5e-5, 2k warmup, cosine) on its corpus
with a partner kind added that swaps one commutative instruction of a chunk
of any shape (`runs/t2_ptrdesc_rms_resume100k`, 21 ms/step).
Tracked every 10k steps on fixed sets of finding 45's pairs (1,000 per class)
and 2,000 natural chunks; vp on 20 held-out batches:

| step | vp | swap, operand a result: flips / distance | swap, sharing an input: flips / distance | churn |
|---|---|---|---|---|
| 0 | 99.70% | 15.5% / 0.446 | 21.3% / 0.503 | |
| 20k | 99.71% | 18.2% / 0.401 | 16.1% / 0.418 | 7.1% |
| 50k | 99.76% | 21.2% / 0.377 | 17.3% / 0.359 | 5.8% |
| 100k | 99.81% | 17.8% / 0.341 | 15.4% / 0.337 | 0.7% |

(churn: the natural chunks whose pointer assignment changed since the last
snapshot; respellings flip 0.4–2.5% throughout; unrelated chunks 1.34–1.36.)
At step 0 the description order flips 15.5% and 21.3% where the hash flips
41–44%; churn decays as the learning rate does, and the flips stay between
15.4% and 21.3%. `probes/partner_chunks.py` on the same two models:

| | step 0 | step 100k |
|---|---|---|
| swap, operand the other's result, **one output** | 0.289 | 0.310 |
| swap, operand the other's result, two outputs | 0.457 | 0.348 |
| swap, sharing an input, two outputs | 0.515 | 0.342 |
| swap partner kind, equivalent / near-miss | 0.322 / 0.838 | 0.207 / 0.808 |

A one-output chunk has one pointer, so its swap cannot be held apart by the
order; it stays apart all the same, and the two-output swaps end at its
level. The swap partner kind draws its operands from registers already in the
chunk half the time, so composed operands are in its training pairs; how many
per batch is not counted.

**47. The op-cell learns a commutative operation on a composed operand once
pairs of that kind are in the training mix, and loses a collapse it had when
its pairs leave the mix.** A swap of a commutative instruction whose operand
is another instruction's result changes the leaf colours of the chunk in a
way that is not a permutation — the colouring passes the swapped role down to
the producer's inputs — on 500 of 500 generated pairs (37% of swaps of any
shape), so the two chunks hand the op-cell different seeds and it has to
describe them alike regardless. Finding 46's final T2 trained a further 100k
steps (learning rate 5e-5, cosine) on a corpus whose partner pairs, half of
every batch, are all such swaps (a partner kind not kept in the generator;
`configs/instr_alu_writers_chain010_composedswap050_aim050.json`;
`runs/t2_composedswap_resume100k`). `probes/partner_chunks.py`, before and
after:

| pair | before | after |
|---|---|---|
| swap, operand the other's result, one output | 0.310 | **0.089** |
| swap, operand the other's result, two outputs | 0.348 | **0.081** |
| composed-swap partner kind, equivalent / near-miss | — | 0.085 / 0.875 |
| a copy through `+k;−k` against the plain copy / near-miss | 0.002 / 0.333 | **0.809** / 0.822 |
| fold of two literals / near-miss | 0.075 / 0.405 | 0.307 / 0.418 |
| reassociation / near-miss | 0.0003 / 1.058 | 0.090 / 0.968 |
| commutative first step read again / near-miss | 0.0006 / 0.984 | 0.047 / 0.838 |

vp on 20 held-out batches 99.80%; unrelated chunks 1.37. The swap the op-cell
did not describe alike collapses tenfold once trained, so its failure was
the absence of training pairs, not the op-cell's capacity. Every partner kind
absent from this corpus moved apart, the cancelling pair all the way to its
near-miss: a collapse is held by the distance term acting on it, not retained
once learned, so the partner mix has to keep every equivalence in it.

**48. A T2 trained from scratch with rewrite partner pairs collapses
commutative swaps in every operand context but folds worse than finding 44's
cell, does not generalize to a rewrite held out of training, and is trained
against distance targets no essence geometry can meet.** The cell is finding
44's with the description-ordered pointer, the root-mean-square target over
all rows (finding 46), and half of every batch partner pairs from the
`rewrite` kind (`datagen.generate._rewrite_pair`: one rewrite — commutative
swap, respelling within a T1 partner group, fold of two literals, or
reassociation — embedded in random context whose registers it reads half the
time; cancellation held out), 1M steps
(`runs/t2_len2_d512_s1000000_dep4_vp12_pool16_dist1.0_alu_writers_chain010_rewrites050_nocancel_aim050`,
`configs/instr_alu_writers_chain010_rewrites050_nocancel_aim050.json`, 22
ms/step). vp 99.8229% at held-out anchors (outputs with three
inputs 91.0%), binding exact; the scorecard's shared-operand swap 0.170
against unrelated chunks 1.318 (finding 44: 0.659 / 1.332); natural swaps on
a composed operand 0.207 with one output, 0.182 with two (0.289 / 0.598).
Partner kinds, equivalent / near-miss: respelling 0.028 / 0.802, reassociation
0.0002 / 1.086, swap 0.047 / 0.926, fold 0.260 / 0.368 (finding 44: 0.075 /
0.405), held-out cancellation 0.616 / 0.817; the `ADDI` and `XORI` fusion
probes 0.9× each (2.6× and 5.9×); the identity against the empty chunk 0.975.

The targets explain part of it. The term matches the chord to 2 × the
behavioral distance, so two maximally different rows are asked to be
antipodal, and rows typically sit near 0.8 apart behaviourally. On real
batches the mean target chord is 1.46 for the T1's corpus, 1.61 for finding
44's and 1.69 for this one; no set of points on a sphere has a mean pairwise
chord above √2, since the mean squared chord is 2 − 2‖mean point‖². Pairs
whose output counts differ, which score an unmatched output as 1, are 47%
of this corpus's pairs, with a mean target chord of 1.83 and 54% of the loss.

Resuming this T2 for 100k steps (learning rate 5e-5) at scale 2 and at
scale √2 — distance 1 mapped to orthogonal — makes the targets satisfiable
(final distance loss 0.104 against 0.002) and shrinks the whole space by the
same factor (unrelated chunks 1.32 against 1.10) without changing collapse
relative to the near-misses: fold 0.269 / 0.371 against 0.232 / 0.333,
held-out cancellation 0.607 / 0.797 against 0.502 / 0.663, vp 99.79% in both
(`runs/t2_rewrites_resume100k_scale2`, `_scale141`). Equivalent pairs (target
0) are 0.83% of a batch's pairs (613 of 73,536), so they carry a matching
share of the loss.

**49. Averaging the distance loss separately over equivalent pairs and the
rest collapses every rewrite the mix trains — folds included — and half of
the rewrite held out, at unchanged vp and aliasing control.** Equivalent pairs
(target 0) are under 1% of a batch's pairs (finding 48), so in a loss
averaged over all pairs they carry under 1% of the gradient. Finding 48's √2
arm, resumed identically except that the squared errors of target-0 pairs
and of all other pairs are each averaged and the two means summed
(the balancing in `distance_loss`, `compressor/train.py`, here on the
√2 × distance target; `runs/t2_rewrites_resume100k_scale141_bal`), against
that arm:

| | balanced | averaged over all pairs |
|---|---|---|
| fold of two literals / near-miss | **0.033** / 0.296 | 0.232 / 0.333 |
| `ADDI` / `XORI` fusion separation | **4.1× / 7.4×** | 0.9× / 0.9× |
| rewrite: commutative swap / respelling | **0.005 / 0.009** | 0.044 / 0.027 |
| composed-swap partner / near-miss | **0.036** / 0.684 | 0.204 / 0.734 |
| natural swap on a composed operand, 1 / 2 outputs | **0.053 / 0.035** | 0.203 / 0.171 |
| scorecard's shared-operand swap / unrelated chunks | **0.017** / 1.110 | 0.148 / 1.110 |
| cancellation, held out of training / near-miss | **0.194** / 0.726 | 0.502 / 0.663 |
| aliasing control | 0.808 | 0.819 |
| vp, 20 held-out batches | 99.79% | 99.79% |

The identity `ADDI r,r,0` against the empty chunk stays at 0.856: no partner
pair relates a same-value write to its absence.

**50. With the distance term matching the chord to √(2d), equivalent pairs
averaged apart, and every rewrite in the partner mix, a T2 at len=2 collapses
every equivalence class tried — folds, cancellations, reassociations,
respellings, identities and commutative swaps in every operand context — at
vp 99.79%.** The target chord √(2d) makes d the target cosine distance, so
rows that behave entirely differently are held orthogonal, never antipodal
(`distance_loss`, `compressor/train.py`). The T1 is finding 43's canonical-role
configuration retrained under this term
(`configs/instr_alu_writers_partners025_aim050.json`;
`runs/t1_distance_cosbal_500k`): vp exact on all 2,457,600
held-out triples, wiring exact, partner groups equivalent at 0.0001–0.0068
against contrasts of 0.48–1.30, aliasing control 1.088, unrelated
instructions 1.213. The T2 over it is finding 44's cell with the
description-ordered pointer, the root-mean-square target over every row,
this term, and half of every batch rewrite partner pairs drawing every
rewrite — commutative swap, respelling, fold, cancellation, reassociation, a
same-value write against its absence
(`configs/instr_alu_writers_chain010_rewrites050_aim050.json`;
`runs/t2_len2_d512_s1000000_dep4_vp12_pool16_dist1.0_cosbal_alu_writers_chain010_rewrites050_aim050`,
1M steps, 24 ms/step). Scored by `scripts/run_t2.sh` on the
regenerated held-out corpus and by the probes, against finding 44's cell:

| | this T2 | finding 44 |
|---|---|---|
| vp exact at held-out anchors | 99.7857% (10,355 of 4,831,488 failing) | 99.8416% |
| — outputs depending on 3 / 2 / 1 inputs | 89.6 / 99.78 / 99.95% | 91.3 / 99.85 / 99.97% |
| binding exact per chunk; rename, reorder, dead code, fresh swap | 100%; 0 | 100%; 0 |
| respelling / fold / cancellation / reassociation, equivalent | 0.003 / 0.016 / 0.003 / 0.0001 | 0.022 / 0.075 / 0.002 / 0.0003 |
| — their near-misses | 0.89 / 0.57 / 1.07 / 1.26 | 0.81 / 0.40 / 0.33 / 1.06 |
| identity write against its absence (rewrite) / near-miss | 0.027 / 1.03 | — |
| `ADDI` / `XORI` fusion separation | 5.8× / 44.8× | 2.6× / 5.9× |
| `ADDI r,r,0` against `ADD r,r,x0`; against `+k;−k` | 0.0002; 0.003 | —; 0.002 |
| scorecard's shared-operand swap | 0.014 | 0.659 |
| natural swap: shared input / result read / operand a result, 2 outputs | 0.005 / 0.0001 / 0.034 | 0.738 / 0.012 / 0.598 |
| natural swap on a composed operand, 1 output | 0.080 | 0.289 |
| aliasing control / unrelated chunks | 0.907 / 1.253 | 1.024 / 1.332 |

The identity against the empty chunk sits at 1.013: no chunk in the corpus
is empty, so the empty chunk — the pool with nothing written — is never
trained against, while chunks consisting of identities collapse with one
another and an identity inside a chunk collapses with its absence. The vp
difference is 0.056 points, over a different T1 and on one seed each.

**51. The distance target and the encoder disagree with register-file
behaviour in three places, each shown by a single constructed case.** A
chunk's behaviour is its map on the register file up to a rename, so a write
that leaves its register unchanged is no output and a copy into another
register is one (RISCV_DESIGN, "Collapse", "The behaviour the distance
measures").

- *The bound depends on the spelling.* `ADDI x5,x0,0` and `SUB x5,x6,x6`
  always write 0. Known bits prove the first constant and leave every bit of
  the second unknown, so against the constant 1 (`ADDI x5,x0,1`) the first
  pair's `max_dist` is 1 and the second's 2³¹, and the per-state distances are
  1.0 and 0.169 (`datagen.knownbits.output_distance`).
- *The same-value test misses rare changes.* In `SLTIU x6,x5,1 ; ADD x7,x7,x6`
  x7 gains 1 only when x5 is 0, which none of the 16 uniform states holds, so
  `precompute_chunk` drops x7 as a same-value write; the chunk's witness
  outputs equal those of `SLTIU x6,x5,1` alone, a distance target of 0 between
  two inequivalent chunks.
- *The essence cannot see a self-loop.* On finding 50's T2, chord distances
  between unit-normalized essences:

  | `ADDI x4,x3,42` followed by | nothing | `ADDI x7,x7,0` | `ADDI x8,x7,0` | `ADDI x7,x7,1` | `ADDI x8,x7,1` |
  |---|---|---|---|---|---|
  | distance from `ADDI x4,x3,42` alone | 0 | 0.0345 | 0.0345 | 0.7115 | 0.7115 |

  Each in-place write and its copy into x8 have bit-identical essences. The
  pool reads value channels and pointers only, and a written slot's value is
  the same description whichever register holds it; the colour channel, which
  is nonzero on a written slot exactly when that register was also a live
  input, never reaches the pool. The trained identity rewrite pulls the copy to
  the chunk alone with the identity write, and `x7 += 1` sits as far from it as
  `x8 = x7 + 1`.

On naturally drawn chunks the departures are rare at len=2.
`probes/target_vs_ideal.py` compares the target the trainer builds against the
reference d\* (`datagen/ideal.py`: exact over renames, 256 states with each
state's registers shuffled) on len-2 chunks of the T2 training corpus's
natural stream, 1,000 pairs per population:

| population | target 0 / d\* 0 | target 0, d\* > 0 | target > 0, d\* = 0 | Spearman over pairs both > 0 |
|---|---|---|---|---|
| unrelated chunks | 0.1% / 0.1% | 0 | 0 | 0.708 |
| near-misses: one immediate moved by 1 or by 2–511, or one R-type opcode changed | 12.2% / 11.5% | 10 (d\* median 0.004) | 3 (target 0.011) | 0.711 |
| partner pairs | 100% / 100% | 0 | 0 | — |

The two zero-mismatch columns are the two state sets' own blind spots: each
holds a few one-state differences the other misses. The correlation is not
1 partly by construction — the target is a root-mean-square over outputs
divided by the larger output count, d\* a root-sum-square over registers. The
departures that matter sit in the tail: near-misses with d\* in (0.05, 0.2]
have targets at 0.058 / 0.120 / 0.643 (5th / 50th / 95th percentile, n = 75).
The one pair of 600 with a target above 3× its d\*,
`XORI x10,x20,-312 ; ANDI x19,x13,1349` against the same with 1348 (d\* 0.116,
target 0.786), is the output byte-sort: the changed bit reorders the two
outputs, so each is compared with the other chunk's unrelated output.

**52. The register-file target tracks d\* on near-misses, where the
output-sort target jumped.** The target the trainer builds is the
register-file distance of RISCV_DESIGN, "Collapse": each row's register file
after the chunk at the 64 witness states under every placement of its
canonical inputs, an unchanged register no output, bounds read from the bits a
column never changes across 1,024 states, the second row's placement and the
written-only registers' matching chosen to minimise, symmetrised, tanh
(`register_distance_targets`). `probes/target_vs_ideal.py` against d\*, same
populations as finding 51, 1,000 pairs each:

| population | target 0, d\* > 0 | target > 0, d\* = 0 | Spearman | targets at d\* in (0.05, 0.2], p5 / p50 / p95 |
|---|---|---|---|---|
| unrelated chunks | 0 | 0 | 0.720 (0.708) | — |
| near-misses | 8 (d\* median 0.004) | 2 (target 0.023) | **0.980** (0.711) | 0.077 / 0.162 / 0.179 (0.058 / 0.120 / 0.643) |
| partner pairs | 0 | 0 | — | — |

(the output-sort target of finding 51 in parentheses.) The zero mismatches are
the two state sets' blind spots, as there. Unrelated pairs fall in a narrow
band of d\* (0.5–1.3), where both state sets' sampling noise decides most of
the ranking. On the constructed cases of finding 51 the target is tanh(d\*) to
within 0.02: the identity write against its absence 0, the copy 0.707, `x7 +=
1` and an immediate moved by one both 0.168, the two spellings of zero both
0.762 from the constant 1, the byte-sort pair 0.117 (d\* 0.116)
(`compressor/tests/test_train.py::test_register_distance_targets_track_the_reference_distance`).
A len-2 batch carries 24 placements and is 15 MB; the trainer builds the
target as two fused kernels.

**53. Finding 50's T2 collapses the equivalences its partner rewrites draw
and none outside them: its collapse is of the trained edits, not of the
function.** `probes/novel_equivalences.py` builds seven classes of equivalent
two-instruction chunks that no rewrite in the training mix produces, each
against a near-miss (the second spelling with one immediate moved or its
operation changed), every pair checked on the emulator; chord distances
between unit-normalized essences of finding 50's T2, 1,000 pairs per class,
against 1.211 for unrelated chunks:

| class | equivalent | near-miss |
|---|---|---|
| `ANDI t,a,k1 ; ANDI t,t,k2` vs `ANDI t,a,k1&k2` | 1.035 | 0.404 |
| `ORI` likewise | 1.118 | 0.487 |
| `SLLI t,a,s1 ; SLLI t,t,s2` vs `SLLI t,a,s1+s2` | 1.271 | 1.119 |
| `SRLI` likewise | 1.186 | 1.017 |
| `SUB t,a,b ; SUB t,x0,t` vs `SUB t,b,a` | 1.041 | 1.286 |
| `ADD t,a,a ; ADD t,t,t` vs `SLLI t,a,2` | 1.224 | 1.076 |
| `ADD t,a,b ; SUB t,t,b` vs `ADDI t,a,0` | 0.060 | 1.071 |

Six classes sit at the distance of unrelated chunks — the AND and OR folds
further apart than their near-misses — and the one that collapses is the
`ADD`/`SUB` form of the cancellation the mix trains in its `ADDI`/`XORI`
forms. The same T2 collapses the trained `ADDI` and `XORI` folds to 0.016
(finding 50). So the distance term pulls together the pairs it is shown, and
the op-cell has not learned a description of the function that makes an
untrained equivalent meet: finding 47's lapse, where a collapse is held only
while its pairs stay in the mix, is the same fact.

**54. A T1 trained on the register-file target, with the pool reading each
written slot's colour, keeps value prediction and its wiring and collapses
its partner groups at least as tightly; 2 of 2.46M held-out values fail.**
`runs/t1_regfile_500k`: finding 50's T1 configuration
(`configs/instr_alu_writers_partners025_aim050.json`, 500k steps) under the
target of finding 52 and the Sammon-weighted loss, generation on odin, 4–7
ms/step. Scored on the held-out corpus regenerated in the register-file
format (`runs/eval/t1_heldout64_single_regfile.rvt`, the same 38,400
instructions), `probes/frame_scorecard.py` and `probes/near_miss.py`:

| | `runs/t1_regfile_500k` | finding 50's T1 |
|---|---|---|
| vp exact at held-out anchors | 99.9999% (2 of 2,457,600, both `ADDI`) | 100% |
| leak check | 11.85% | — |
| wiring exact | 100% of 38,400 | 100% |
| partner groups equivalent (copy / zero / double / sign / constant) | 0.0021 / 0.0006 / 0.0030 / 0.0002 / 0.0091 | 0.0001–0.0068 |
| their contrasts | 0.85 / 1.22 / 1.12 / 0.42 / 1.16 | 0.48–1.30 |
| aliasing control / unrelated instructions | 1.085 / 1.120 | 1.088 / 1.213 |
| `ADDI` k vs k+1 / k vs k+512 / `ADD` vs `SUB` / `SLLI 3` vs `4` | 0.442 / 0.688 / 1.095 / 0.973 | — |
| — their targets (chord) | 0.579 / 1.060 / 1.126 / 1.145 | — |

The grading holds, below its targets at the far end as in finding 41. One
seed; whether the two `ADDI` failures are the target's or the draw's is
unmeasured.

**55. Of the pairs a training batch gives target 0, every partner pair and
99.4% of the rest are equivalent; the rest are comparisons whose thresholds
no witness state separates.** Ten batches of
`configs/instr_alu_writers_chain010_rewrites025_mined025_aim050.json` at
len=2 (384 rows each), every pair whose register-file target
(`register_distance_targets`) is 0, proved with z3 under the rename that
minimises d\* (`datagen/prove.py`, `datagen/ideal.py`):

| | proved equivalent | not equivalent |
|---|---|---|
| partner pairs (rewrites and mined) | 996 | 0 |
| every other pair in the batch | 5,946 | 41 |

The other equivalent pairs, about 600 per batch, are naturally drawn chunks
computing one function on different registers — two copies, two constant
writes, two comparisons that are constant over every input. Each of the 41 is
a pair of comparisons against thresholds no witness state falls between
(`AND x6,x21,x18 ; SLTI x31,x26,-1208` against the same shape with -1360), or
a comparison almost never true (`XORI x25,x18,-1267 ; SLTIU x25,x25,851`)
against a constant 0: near-misses whose true distance is small and nonzero,
given 0 and the largest weight the loss gives.

**56. A T2 trained on the register-file target with half its partner pairs
mined by search collapses every equivalence class the search covers, at
finding 50's vp, and none in the operation family held out of the search.**
The cell is finding 50's (len=2, `d_out` 512, op-cell depth 4, twelve-layer
residual vp head, 16 pool heads, 1M steps, one seed) over finding 54's T1,
on the register-file target and the Sammon-weighted loss, with half of every
batch partner pairs, half of those from the rewrites and half drawn from
400,008 pairs proved equivalent (`scripts/mine_partners.py`: a
two-instruction chain writing one register, every single ALU instruction
computing the same register from the chain's inputs, proved with z3), no
shift operation in any mined pair
(`configs/instr_alu_writers_chain010_rewrites025_mined025_aim050.json`,
`runs/mined/noshift_400k.txt`;
`runs/t2_len2_d512_s1000000_dep4_vp12_pool16_dist1.0_regfile_alu_writers_chain010_rewrites025_mined025_aim050`,
34 ms/step). Scored on the held-out corpus regenerated in the register-file
format, same 38,400 chunks:

| | this T2 | finding 50 |
|---|---|---|
| vp exact at held-out anchors | 99.7890% (10,198 of 4,833,600 failing) | 99.7857% |
| — outputs depending on 3 / 2 / 1 inputs | 89.2 / 99.77 / 99.97% | 89.6 / 99.78 / 99.95% |
| binding exact per chunk | 100% | 100% |
| rename / reorder / dead code | ≤ 3.5e-6 | 0 |
| commutative swap of two leaves (fresh operands or an aliased destination) | 0.016 (max 0.54; 1,791/2,000 at 0) | 0 |
| scorecard's shared-operand swap / aliasing control / unrelated | 0.045 / 0.909 / 1.258 | 0.014 / 0.907 / 1.253 |

`probes/novel_equivalences.py` (finding 53's classes; equivalent / near-miss,
unrelated chunks 1.206):

| class | in the mined pairs | this T2 | finding 50 |
|---|---|---|---|
| `ANDI` fold | yes | **0.022** / 0.430 | 1.035 / 0.404 |
| `ORI` fold | yes | **0.025** / 0.392 | 1.118 / 0.487 |
| negation flip through `SUB` from x0 | yes | **0.016** / 1.137 | 1.041 / 1.286 |
| `ADD`/`SUB` cancellation | yes | **0.001** / 0.580 | 0.060 / 1.071 |
| `SLLI` fold | held out | 1.166 / 1.017 | 1.271 / 1.119 |
| `SRLI` fold | held out | 1.096 / 0.891 | 1.186 / 1.017 |
| `ADD t,a,a ; ADD t,t,t` vs `SLLI t,a,2` | held out | 1.098 / 0.856 | 1.224 / 1.076 |

At 100k steps the same classes read 0.09–0.18 and 0.90–1.11. `probes/partner_chunks.py`
(equivalent / near-miss; finding 50 in parentheses): swap 0.010 / 0.949,
respelling 0.002 / 0.808 (0.003 / 0.89), fold 0.050 / 0.530 (0.016 / 0.57),
cancellation 0.002 / 0.580 (0.003 / 1.07), reassociation 0.0003 / 1.146
(0.0001 / 1.26), identity write 0.070 / 0.893 (0.027 / 1.03); natural swaps
on a shared input 0.000 / 0.021 with one and two outputs, on an operand that
is a result 0.148 / 0.091 (0.080 / 0.034), on a result read 0.007 / 0.013.

The search makes collapse follow from the function over the space it
searches — whatever single instruction a chain reduces to, named or not — and
a family no training pair ever draws stays apart: collapse transfers across
spellings within the families the pairs reach, not across operation families.
The rewrite classes collapse 2–3× less tightly than in finding 50, with a
quarter of each batch rewrite pairs against half there and the loss weighting
equivalent pairs by 1/ε (about 15% of the gradient on these batches) against
half there; this run does not separate the two. A commutative swap whose
destination is one of the swapped operands is not equivalent by
construction: the pool reads the overwritten leaf's colour, which names the
operand role, so the swap changes it and the op-cell has to learn that the
two describe one function, as it does for a swap on a shared operand.

**57. With every operation in the mined pairs, the T2 collapses all seven of
finding 53's untrained-by-rewrite classes, shift folds included, at vp
99.7975%.** Finding 56's cell with the library mined over every ALU operation
(`configs/instr_alu_writers_chain010_rewrites025_minedall025_aim050.json`,
`runs/mined/all_400k.txt`: 400,008 proved pairs, 227,154 involving a shift;
`runs/t2_len2_d512_s1000000_dep4_vp12_pool16_dist1.0_regfile_alu_writers_chain010_rewrites025_minedall025_aim050`,
34 ms/step), scored as finding 56:

| | this T2 | finding 56 (shifts held out) | finding 50 |
|---|---|---|---|
| vp exact at held-out anchors | **99.7975%** (9,787 of 4,833,600 failing) | 99.7890% | 99.7857% |
| — outputs depending on 3 / 2 / 1 inputs | 90.1 / 99.77 / 99.97% | 89.2 / 99.77 / 99.97% | 89.6 / 99.78 / 99.95% |
| binding exact per chunk | 100% | 100% | 100% |
| commutative swap of two leaves / on a shared operand | 0.003 / 0.050 | 0.016 / 0.045 | 0 / 0.014 |
| aliasing control / unrelated | 0.872 / 1.257 | 0.909 / 1.258 | 0.907 / 1.253 |

`probes/novel_equivalences.py`, equivalent / near-miss (unrelated 1.208):

| class | this T2 | finding 56 | finding 50 |
|---|---|---|---|
| `ANDI` fold | 0.035 / 0.427 | 0.022 / 0.430 | 1.035 / 0.404 |
| `ORI` fold | 0.037 / 0.415 | 0.025 / 0.392 | 1.118 / 0.487 |
| `SLLI` fold | **0.043** / 1.070 | 1.166 / 1.017 | 1.271 / 1.119 |
| `SRLI` fold | **0.029** / 0.978 | 1.096 / 0.891 | 1.186 / 1.017 |
| negation flip | 0.033 / 1.137 | 0.016 / 1.137 | 1.041 / 1.286 |
| doubling twice vs `SLLI 2` | **0.041** / 0.876 | 1.098 / 0.856 | 1.224 / 1.076 |
| `ADD`/`SUB` cancellation | 0.002 / 0.580 | 0.001 / 0.580 | 0.060 / 1.071 |

At 100k steps the shift classes already read 0.14–0.17, where finding 56's
read 0.90–1.11. `probes/partner_chunks.py`: swap 0.013 / 0.930, respelling
0.001 / 0.786, fold 0.056 / 0.513, cancellation 0.002 / 0.580,
reassociation 0.0004 / 1.164, identity write 0.072 / 0.893; natural swaps on
an operand that is a result 0.162 / 0.096 with one and two outputs — finding
56's levels, 2–3× looser than finding 50's on folds, identities and composed
swaps. A family enters the collapse when its pairs enter the search; with
every ALU operation searched, every two-to-one equivalence the probes build
collapses, and the shared vp is unchanged.

**58. On the record T2 the description-ordered pointer agrees on equivalent
pairs, and is a discontinuity on near-misses: a small change that swaps two
outputs' ranks moves the essence about 0.3 further from its target.** Each
written register's pointer is its rank when the chunk's outputs are ordered
by a fixed projection of their descriptions (`RegisterStateMachine._pointer`)
— a canonical order of a set, like the output byte-sort finding 51 measured in
the target. `probes/pointer_order.py` on finding 57's T2, two-output len-2
chunks, 2,000 pairs per row:

| pairs | pointers differ | essence chord |
|---|---|---|
| a commutative swap in a natural chunk (equivalent) | 0.8% | 0.352 where they differ, 0.008 where they agree |
| partner rewrites (equivalent) | 1.2% | 0.265 / 0.011 |

| near-misses: one immediate moved | ranks swap | chord − target chord, ranks swap / ranks kept |
|---|---|---|
| by 1 | 6.2% | +0.134 / −0.175 (targets 0.676 / 0.616) |
| by 2–64 | 8.3% | −0.075 / −0.364 (targets 0.961 / 0.935) |

Finding 46 measured 15–18% of swaps flipping before the op-cell described
the two spellings alike; with collapse trained, the order agrees, and a flip
is where a collapse failed. Among near-misses whose targets are alike, those
whose ranks swap sit about 0.3 further from their targets. A pair whose
descriptions move most is also the likeliest to swap, so the split is
correlational; the mechanism — the pointer enters the pool's attention key, so
a swap refiles both outputs — predicts a jump of this kind.

**59. A T1 whose evaluator predicts its output as a set, with no pointer in
the pool, scores as finding 54's.** `runs/t1_setvp_500k`: finding 54's
configuration with the pool keyed on descriptions alone and the evaluator
reading the essence and the instruction's inputs in canonical role order
(`vp_loss`, `compressor/train.py`), 4–12 ms/step. Scored on the held-out corpus
regenerated with role-ordered inputs (`runs/eval/t1_heldout64_single.rvt`,
the same 38,400 instructions):

| | `runs/t1_setvp_500k` | finding 54 |
|---|---|---|
| vp exact at held-out anchors | 99.9998% (5 of 2,457,600: `ADDI` 3, `SLTIU` 1, `ANDI` 1) | 99.9999% (2) |
| leak check | 10.74% | 11.85% |
| wiring exact | 100% | 100% |
| partner groups equivalent (copy / zero / double / sign / constant) | 0.0024 / 0.0002 / 0.0025 / 0.0001 / 0.0103 | 0.0021 / 0.0006 / 0.0030 / 0.0002 / 0.0091 |
| aliasing control / unrelated instructions | 1.086 / 1.120 | 1.085 / 1.120 |
| `ADDI` k vs k+1 / k+512 / `ADD` vs `SUB` / `SLLI 3` vs `4` | 0.468 / 0.658 / 1.095 / 0.964 | 0.442 / 0.688 / 1.095 / 0.973 |

One seed each; 5 failures against 2 is unresolved.

**60. With no pointer and the evaluator predicting a set, the essence's
near-miss discontinuity shrinks 4–5×, and value prediction fails 2.2× as
often.** Finding 57's cell over finding 59's T1, the pool keyed on
descriptions alone and the evaluator reading the essence and the chunk's
role-ordered inputs, predicting every output at once, matched by least loss
(`runs/t2_len2_d512_s1000000_dep4_vp12_pool16_dist1.0_setvp_alu_writers_chain010_rewrites025_minedall025_aim050`,
28 ms/step), scored as finding 57 on the held-out corpus regenerated with
role-ordered inputs (the same 38,400 chunks):

| | set evaluator | finding 57 (pointer) |
|---|---|---|
| vp exact at held-out anchors | 99.5605% (21,248 of 4,835,008 failing) | 99.7975% (9,787) |
| — outputs depending on 3 / 2 / 1 / 0 inputs | 89.18 / 99.49 / 99.80 / 99.997% | 90.09 / 99.77 / 99.97 / 100% |
| — lowest writers | `SUB` 97.40%, `ADD` 99.04%, `SLTU` 99.50% | `SUB` 99.04%, `ADD` 99.38%, `SLL` 99.59% |
| binding exact per chunk | 100% | 100% |
| commutative swap of two leaves / on a shared operand | 0.013 / 0.109 | 0.003 / 0.050 |
| aliasing control / unrelated | 0.811 / 1.254 | 0.872 / 1.257 |
| finding 53's classes, equivalent | 0.002–0.089 | 0.002–0.043 |
| partner rewrites, equivalent | 0.0006–0.065 | 0.0004–0.072 |

`probes/pointer_order.py`, the description rank computed from each model's
value channels (finding 58's split):

| near-misses | ranks swap | chord − target chord, ranks swap / kept | finding 58 |
|---|---|---|---|
| immediate by 1 | 7.7% | −0.162 / −0.239 (gap 0.077) | +0.134 / −0.175 (gap 0.309) |
| immediate by 2–64 | 13.0% | −0.293 / −0.347 (gap 0.055) | −0.075 / −0.364 (gap 0.289) |

Equivalent two-output pairs whose ranks differ sit 0.13–0.16 apart, against
0.27–0.35 under the pointer. The remaining gap is of the size the selection
the split carries — a pair whose descriptions move most is likeliest to swap
— would leave. The vp loss is spread over every input count, worst on
`SUB`: handed the essence and all of the chunk's inputs, the evaluator has to
work out which inputs each output reads, which the pointer and each output's
own input list told it. One seed each, over T1s that differ only in their
pools and evaluators.

**61. The set evaluator trained 3M steps on freshly mined pairs fails half as
often as finding 57's pointer model and collapses every class tighter, the
near-miss discontinuity staying small.** Finding 60's cell, 3M steps (cosine
over 3M), with every mined pair searched and proved fresh and drawn twice from
a rolling pool, renamed each draw
(`configs/instr_alu_writers_chain010_rewrites025_minedlive025_aim050.json`,
`partners.mined.reuse` 2; `runs/t2_len2_d512_s3000000_dep4_vp12_pool16_dist1.0_setvp_alu_writers_chain010_rewrites025_minedlive025_aim050`,
generation on odin's pool of 56 workers and 5 local, 28 ms/step; the trainer
hung at step 2,901,400 and the run resumed from its step-2,900,000 checkpoint
on a stream drawn with another seed, `resume.txt` in the run directory).
Scored on the same held-out corpus as finding 60:

| | this T2 | finding 57 (pointer, 1M) | finding 60 (set, 1M) |
|---|---|---|---|
| vp exact at held-out anchors | **99.8965%** (5,006 of 4,835,008 failing) | 99.7975% (9,787) | 99.5605% (21,248) |
| — outputs depending on 3 / 2 / 1 / 0 inputs | 96.07 / 99.92 / 99.94 / 100% | 90.09 / 99.77 / 99.97 / 100% | 89.18 / 99.49 / 99.80 / 99.997% |
| — lowest writers | `SUB` 99.52%, `ADD` 99.76%, `SLL` 99.84% | `SUB` 99.04%, `ADD` 99.38% | `SUB` 97.40%, `ADD` 99.04% |
| binding exact per chunk | 100% | 100% | 100% |
| rename / reorder / dead code | ≤ 6.2e-6 (dead code under 1e-6 on 900 of 2,000) | ≤ 1.9e-6 | ≤ 4.3e-6 |
| commutative swap of two leaves / on a shared operand | 0.007 / 0.031 | 0.003 / 0.050 | 0.013 / 0.109 |
| aliasing control / unrelated | 0.825 / 1.255 | 0.872 / 1.257 | 0.811 / 1.254 |
| finding 53's classes, equivalent | 0.002–0.035 | 0.002–0.043 | 0.002–0.089 |
| partner rewrites, equivalent (fold / identity write) | 0.0002–0.046 (0.031 / 0.046) | 0.0004–0.072 (0.056 / 0.072) | 0.0006–0.065 |
| natural swap on an operand that is a result, 1 / 2 outputs | 0.187 / 0.048 | 0.162 / 0.096 | 0.194 / 0.075 |
| near-misses: chord − target, ranks swap / kept, by 1 / by 2–64 | −0.153 / −0.188; −0.167 / −0.270 | gap 0.31 / 0.29 (finding 58) | −0.162 / −0.239; −0.293 / −0.347 |

The novel classes over training (finding 53's probe, equivalent): 0.012–0.173
at 500k, 0.007–0.100 at 1M, 0.006–0.074 at 2M, 0.002–0.035 at 3M; the
training vp loss passed finding 57's final 0.0012 near 1.9M steps. One seed;
against finding 57 it changes the evaluator, the step budget and fresh
against frozen pairs together, so it ranks the configurations and attributes
nothing to one change. The dead-code and rename rows carry float noise up to
6e-6 where finding 57's stayed under 2e-6; both are zero by construction.

## Decisions taken

- **The essence is graded in the register frame** — colour-identified leaves,
  raw targets, the pool over written slots, and value prediction read from
  the essence (findings 38 and 39). The liveness pass and the colouring are
  exact structural rules, built in as scaffolding for a learned pass over the
  same graph (RISCV_DESIGN, "What is built in for good, and what is built in
  for now").
- **The encoder is told nothing about which operations commute.** Every
  operand role is ordered in its colouring and in the evaluator's input
  order; the list of commutative opcodes reaches only the oracle's
  signature. Commutativity is to be learned under a collapse objective (the
  user's ruling); finding 37's swap rows read zero only because its encoder
  carried the list.
- **The structural signature is the collapse label** (`datagen/signature.py`)
  and never a target.
- **Value prediction reads the essence, never a written slot's description**
  (finding 39). The per-slot evaluator is deleted: it certifies each written
  slot and leaves the pool unshaped, while an exact essence evaluator implies
  the written slots carry their functions, since each slot's row is the only
  route its function has into the essence.
- **A batch holds distinct chunks only** (finding 40).
- **T1 has no immediate head** (finding 41). Its purposes do not hold: the
  immediate-field vp deficit it targeted came from the essence pool's shared
  value projection (fixed by per-head projections), the rotary composition of
  immediates it was to give the essence is not observed (finding 21: a chain's
  summed immediate is no more linearly recoverable than either addend), and it
  pins the essence to how an instruction is written, which collapse requires
  it not to encode.
- **Collapse at T1 is trained with the behavioral distance term** (finding
  41): the user's requirement is a smooth space, graded by how differently two
  functions behave, and the distance term states it; InfoNCE alone, with or
  without uniformity, measured as flattening that grading.
- **Training anchors are aimed at each row's boundaries, at every scale**
  (finding 42), so value prediction sees the inputs where functions the
  distance term places close together differ.
- **Distance targets place inputs by canonical role** (finding 43), never by
  the graph's colour order, and take the minimum over every placement of the
  second row's inputs where a batch can carry them (len 2; finding 52).
- **The distance target is the register-file distance** (findings 51 and 52):
  a register the chunk leaves unchanged is no output, a written register is
  compared with the other row's register or with its own old value, bounds are
  read from the bits a column never changes across a fixed state family, the
  combination is a root-sum-square mapped by tanh.
- **The pool reads each written slot's colour** (finding 51), so an output that
  overwrites an input is told from a copy.
- **Equivalent pairs used as training labels are proved** on every register
  file (`datagen/prove.py`), never accepted from sampled states.
- **The evaluator takes inputs in canonical role order** (finding 43), ties
  between interchangeable inputs by colour.
- **The frozen T1 is `runs/t1_regfile_500k`** (finding 54): the
  register-file distance target with Sammon weighting, the evaluator in
  canonical role order, no immediate head, aimed anchors, the pool reading
  each written slot's colour.
- **The distance term matches the chord to √(2d)** (findings 48–50), each
  pair's error weighted by 1 / (target chord + ε), ε the chord of the smallest
  nonzero target — continuous in the target, where findings 49–50 averaged the
  equivalent pairs apart from the rest.
- **The T2 record is finding 61's cell**: the set evaluator with no pointer,
  half of every batch partner pairs — half of those rewrites, half searched
  and proved fresh, each drawn at most twice — 3M steps (findings 56, 57, 60
  and 61).
- **The evaluator predicts a set and nothing ranks the outputs** (findings 58,
  60 and 61): the discontinuity a rank put in the essence is gone, and with
  3M steps value prediction exceeds the pointer model's.
- **Mined pairs are searched and proved per draw**, never taken from a fixed
  library (finding 61).
- **T2 trains with the distance term on every row and partner chunk pairs**
  (findings 44 and 46).
- **Pointers are ordered by a fixed projection of each output's description**
  (finding 46), not by a hash of its spelling.
- **The partner mix covers every equivalence at once** (finding 47): a
  collapse lapses when its pairs leave the mix.

## Dead ends (do not re-run)

- **Scoring the pool's spare picks on unmasked rows.** A selection over a 0/1
  mask breaks ties by index; any slot the selection may land on must be
  exactly zero.
- **Grading the essence through each written slot's description alone**
  (finding 38). Nothing reads the pool, so the essence's range falls to a
  thirtieth of the first-read frame's while value prediction stays exact.
- **Aiming a comparison's anchors only at the step where it flips** (finding
  42). It fixes the step and leaves the threshold misplaced by hundreds
  under the distance term.
