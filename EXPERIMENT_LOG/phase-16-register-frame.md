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
training mix (finding 47). Open: whether the decisions behind that collapse
hold up — the balanced loss, the order outputs are paired in, the pointer
order — a second seed, rewrites never trained, the composed slots' residual,
the binding's wire format, and the learned replacement for the colouring.

Of the runs named below only `runs/t1_distance_cosbal_500k` and finding 50's
T2 are kept, and of the corpus configs only those still in `configs/`; a
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
  the graph's colour order.
- **The evaluator takes inputs in canonical role order** (finding 43), ties
  between interchangeable inputs by colour.
- **The frozen T1 is `runs/t1_distance_cosbal_500k`** (finding 50): the
  √(2d) balanced distance term on canonical-role targets, the evaluator in
  canonical role order, no immediate head, aimed anchors.
- **The distance term matches the chord to √(2d) and averages equivalent
  pairs apart from the rest** (findings 48–50).
- **The T2 record is finding 50's cell**, trained on rewrite partner pairs
  drawing every rewrite.
- **T2 trains with the distance term on every row and partner chunk pairs**
  (findings 44 and 46): outputs in canonical order, combined by
  root-mean-square, same-value writes not counted.
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
