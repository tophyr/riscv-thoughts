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
essence"). Every equivalence that leaves the live dataflow graph unchanged
collapses by construction and reads zero on the trained models; T1 is exact at
held-out anchors; T2 at len=2 is 99.83% after 1M steps on one seed, with the
essence spanning the range the first-read frame's models had (finding 39).
At T1 a behavioral distance term collapses equivalent spellings and grades
inequivalent ones by how differently they behave, at a cost of 108 failing
vp triples on comparisons the target cannot tell apart (finding 41). Open: the composed slots' residual, every equivalence the op-cell has to
compute — commutativity first — the binding's wire format, the pointers'
order under an equivalence that respells one of several outputs, and the
learned replacement for the colouring.

Findings 1–36 (phases 14 and 15) were measured in the first-read canonical
frame; their numbers are not comparable to these (METRICS.md).

**37. In the register frame — with the encoder told which operations
commute — T1 is exact and every structural equivalence collapses to zero; T2
at len=2 reaches 99.8385% after 1M steps, with the composed slots carrying the
residual, and its essence varies 30× less between unrelated chunks than
before.** The models here carried a list of commutative opcodes into the
encoder: T1 learned to emit a symmetry bit against it and T2's colouring
treated such an operation's two roles as one, which is what made the
commutative-swap rows below exactly zero. The list reaches only the oracle
in the code (Decisions); the swap rows are the only figures it touched. Every
figure is at the fixed held-out anchor
set, 64 states, on corpora generated with `datagen --heldout-anchors` from
`configs/instr_alu_writers.json`: `runs/eval/t1_heldout64_single.rvt` and
`runs/eval/t2_heldout64_len2.rvt`, 100 batches of 384 rows each (twins
included). Scores are `probes/frame_scorecard.py`'s; a value is one output
register of one row at one anchor state; essence distances are L2 between
unit-normalized essences, 2,000 pairs per class.

*The frozen T1*, `runs/t1_frame_500k`: `d_model` 128, `d_out` 64,
`imm_weight` 0.3, `sym_weight` 0.1, three-layer residual vp head, 500k steps
at 4 ms/step (33 min), 24 local generator workers, GPU at 96–97%.

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
and the colouring runs eight rounds), 26 local workers, GPU at 99%, one
instruction-stream seed.

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
collapses, as before; what the frame changed is that nothing the graph
determines is left to training.

*The essence's dynamic range.* Unrelated chunks sit 0.0233 apart, against
0.67–1.19 in the first-read frame, and the aliasing control at 0.0192 is
within that. The essence receives no vp gradient: value prediction reads the
written slot's description in `T`, and the only losses on the essence are the
validity magnitude and `pc_writes`, so its direction is shaped by nothing
that asks it to tell chunks apart. The ratios between classes are of the
same order as before (the aliasing control and the fusion rows separate by
similar factors), so the structure may survive at a thirtieth of the scale
rather than be gone; what shaped it in the first-read frame was vp reading
the essence (the opcode clustering) and T1's immediate head (the rotary
immediate axis), which T1 still trains and T2 never had. Finding 39 gives
the essence that pressure back.

*A pool defect found by the scorecard and fixed before scoring.* The pool
projects only the written slots and fills a short row's spare picks from the
lowest-index unwritten slots, which assumed those rows are zero; a read-only
leaf's row is not (it holds the leaf's seed), and x1 is the first candidate,
so a chunk with one written slot that read x1 projected that leaf into the
pool. Dead-code pairs whose inserted instruction read x1 sat 0.02 apart. The
rows are masked before selection now (`_project_touched`,
`compressor/model.py`); the T2 cell above was trained with the defect — it
touches rows with one written slot that read x1 — and scored after the fix;
T1 is unaffected (one instruction, one write, no spare pick).

**38. Without the commutative list in the encoder: T1 exact, T2 at len=2
99.8322% after 1M steps, every graph-identity class exactly zero, and a swap
whose operand is shared stays apart.** Same corpora, scorer, cell and budget
as finding 37; the only change is that every operand role is ordered in the
colouring, the list reaching only the oracle's labels, and the pool fix is in
for training as well as scoring. T1 `runs/t1_register_500k`: vp 100.0000%
(0 of 2,457,600), wiring 100.0000%. T2
`runs/t2_len2_d512_s1000000_dep4_vp12_pool16_alu_writers_chain010` (this
directory held finding 37's cell and was overwritten by this one):

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
nothing the op-cell has to compute collapses, and the essence now separates
those equivalents by their spelling.** Each written output carries a pointer,
a one-hot built in three steps: each output's data path is coloured by the
leaf colouring's pass with every operation also seeded from its content; the
chunk's distinct paths are sorted by a fixed projection of that colour (their
content-hash order); and the pointer marks the output's position in that
order. The pool adds each written slot's pointer
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

*The frozen T1*, `runs/t1_essonly_500k`: as finding 38's T1, 500k steps at
3–4 ms/step, 96–97% GPU.

| T1 | finding 38 (per-slot evaluator) | essence evaluator |
|---|---|---|
| vp exact at held-out anchors | 100.0000% | **100.0000%** (0 of 2,457,600) |
| leak check | — | 12.14% |
| emitted wiring exact | 100.0000% | 100.0000% of 38,400 |
| rename | 0 | 0 (≤ 4.3e-7) |
| aliasing control (`x+x` vs `x+y`) | 0.2571 | 0.6308 |
| unrelated instructions | 0.9197 | 1.3688 |

*The T2 cell*, `runs/t2_len2_d512_s1000000_dep4_vp12_pool16_essonly_alu_writers_chain010`:
as finding 38's cell, 1M steps at 14 ms/step (3.9 h), GPU at 99%.

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
0.8–1.0×), now at full scale.

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
`datagen.signature.output_inputs` ordered an output's inputs by the oracle's
colouring, which merges a commutative operation's two roles, and broke the
resulting tie by register number, while the encoder colours the two roles
apart — so a rename moved the two values between evaluator positions. With the
order computed from the colouring with every role ordered, on a batch drawn
from the same seed, every copy matches its source (relative gradient
difference ≤ 2.2e-4, cosine ≥ 0.99999976, against 0.81 between unrelated
chunks), and a test pins the property
(`test_a_rename_never_moves_an_input_between_gather_positions`). Batches now
carry 384 distinct chunks and no copies; the generator with copies set to zero
and the generator without them emit byte-identical batches on one seed. The
models of findings 37–39 were trained and scored with the register-dependent
order. The frozen T1, `runs/t1_essence_500k`, is finding 39's T1 trained with
the encoder's input order on batches of distinct chunks (500k steps, 4 ms/step,
86–90% GPU), scored on held-out corpora regenerated the same way, 38,400
distinct instructions:

| T1 | finding 39's T1 | `runs/t1_essence_500k` |
|---|---|---|
| vp exact at held-out anchors | 100.0000% | **100.0000%** (0 of 2,457,600) |
| leak check | 12.14% | 11.44% |
| emitted wiring exact | 100.0000% | 100.0000% |
| rename | 0 | 0 (≤ 3.9e-7) |
| aliasing control | 0.6308 | 0.7363 |
| unrelated instructions | 1.3688 | 1.3658 |

No T2 has been trained over it; finding 39's T2 is the T2 record.

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
  twice their **behavioral distance** (`distance_loss`, weight 1.0). The
  behavioral distance executes both rows at 64 shared witness states, their
  inputs placed in colour order so that inputs correspond positionally, and
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

The distance arms ran at 10–12 ms/step against 4 ms, with the GPU at 71–76%:
the targets are computed eagerly before each compiled step.

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
  signature. Finding 37's models carried that list into the encoder (a
  symmetry bit T1 emitted and T2's colouring consumed), which is why their
  swap rows read exactly zero; the user's ruling is that commutativity has to
  be learned, under a collapse objective, and finding 38 is the same cells
  without the list.
- **The structural signature is the collapse label** (`datagen/signature.py`)
  and never a target.
- **Value prediction reads the essence, never a written slot's description**
  (finding 39). The per-slot evaluator is deleted: it certifies each written
  slot and leaves the pool unshaped, while an exact essence evaluator implies
  the written slots carry their functions, since each slot's row is the only
  route its function has into the essence.
- **A batch holds distinct chunks only** (finding 40).
- **T1 has no immediate head** (finding 41). It was added to give the essence
  an immediate axis on which imm+δ is a consistent rotation, and to fix an
  immediate-field vp deficit; the deficit was the essence pool's shared value
  projection (fixed by per-head projections), the rotary composition was never
  observed (finding 21: a chain's summed immediate is no more linearly
  recoverable than either addend), and it pins the essence to
  how an instruction is written, which collapse requires it not to encode. The
  code is deleted; every kept checkpoint loads without its tensors and
  reproduces its outputs bit for bit.
- **Collapse at T1 is trained with the behavioral distance term** (finding
  41): the user's requirement is a smooth space, graded by how differently two
  functions behave, and the distance term states it; InfoNCE alone, with or
  without uniformity, measured as flattening that grading.

## Dead ends (do not re-run)

- **Scoring the pool's spare picks on unmasked rows.** A selection over a 0/1
  mask breaks ties by index; any slot the selection may land on must be
  exactly zero.
- **Grading the essence through each written slot's description alone**
  (finding 38). Nothing reads the pool, so the essence's range falls to a
  thirtieth of the first-read frame's while value prediction stays exact.
