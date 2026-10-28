# Phase 16 (IN PROGRESS): the register frame

**Status: open.** The essence is graded in the register frame: value
prediction per written slot from that slot's description and its inputs'
values and colours gathered by register, leaves identified by their structural
colour in the live dataflow graph with every operand role ordered, the essence
pooled over the written slots only (RISCV_DESIGN, "GVN" and "Leaf identity
for input registers"). Every equivalence that leaves the live dataflow graph
unchanged collapses by construction and reads zero on the trained models; T1
is exact at held-out anchors; T2 at len=2 is 99.8322% after 1M steps on one
seed (finding 38). Open: the composed slots' residual, every equivalence the
op-cell has to compute — commutativity first — the binding's wire format, the
essence's dynamic range (finding 37), and the learned replacement for the
colouring (TODO.md).

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
immediate axis), which T1 still trains and T2 never had. Whether the essence
still carries the written descriptions — the content a next tier would
consume — is untested here; a
readout fitted on frozen essences against the written slots' values is the
measurement, and an objective that makes the essence answer for its
descriptions is the decision (TODO.md).

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

## Decisions taken

- **The essence is graded in the register frame** — per-slot evaluation, raw
  targets, colour-identified leaves, the pool over written slots (TODO.md
  task 1 records the frame and the scaffolding it is built with).
- **The encoder is told nothing about which operations commute.** Every
  operand role is ordered in its colouring; the list of commutative opcodes
  reaches only the oracle's labels. Finding 37's models carried that list
  into the encoder (a symmetry bit T1 emitted and T2's colouring consumed),
  which is why their swap rows read exactly zero; the user's ruling is that
  commutativity has to be learned, under a collapse objective, and finding
  38 is the same cells without the list.
- **The structural signature is the collapse label** (`datagen/signature.py`)
  and never a target.

## Dead ends (do not re-run)

- **Scoring the pool's spare picks on unmasked rows.** A selection over a 0/1
  mask breaks ties by index; any slot the selection may land on must be
  exactly zero.
