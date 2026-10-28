# Probes

Diagnostics that read a frozen checkpoint and answer one question each. None
train a model; the usage line in each docstring gives its arguments. Each file's docstring is the authority on how to
read its output — this index only says which one to reach for.

How to trust what a probe reports, and the failure modes that have produced
withdrawn conclusions here, are in [METRICS.md](../METRICS.md) ("Probe
methodology").

Run them with the venv active:

```bash
source .venv/bin/activate
python probes/frame_scorecard.py runs/<t2_dir> runs/<t1_dir>/encoder.pt runs/eval/t2_heldout64_len2.rvt
```

The scorecard reads its anchor states from the batches it scores, which must
be generated with `datagen --heldout-anchors`, so a score is at input states no
model trained on (METRICS.md). Probes that rebuilt one fixed anchor set from a
seed, and the probes of the first-read frame (`vp_budget.py`, `vp_stage.py`,
`vp_miss_kind.py`, `vp_head_refit.py`, `vp_gt_depth.py`, `vp_gt_misses.py`,
`t1_scorecard.py`), are deleted; their findings are in EXPERIMENT_LOG, and one
can be recovered from git history and ported to the register frame.

| probe | the question it answers |
|---|---|
| `frame_scorecard.py` | **reach for this first** — one tier on the register frame: value prediction per written slot at held-out anchors (exact match, by writing opcode and input count), the emitted binding, and essence distance over the signature's equivalence classes (rename, reordering, dead-code insertion, commutative swaps with fresh and with shared operands) against an aliasing control and unrelated chunks. `-` as the T2 directory scores a T1 alone on a `--rule single` corpus |
| `additive_fusion.py` | whether two chunks computing the same function land on the same point. Builds the chunks directly and takes a distance — no fitted readout — and asserts equivalence through the emulator. Add a family with a `combine`/`resplit`/`perturb` triple in `FAMILIES` |
| `write_position.py` | whether the essence depends on WHERE a chunk's write happens and on how many instructions it holds rather than on what it computes: moves one `ADDI` between step 1 and step 2 of the two-slot grid beside an inert x0 no-op, and compares the five spellings of the identity on one register (`ADDI r,r,0`, `ADD r,r,x0`, `ADDI r,r,k; ADDI r,r,-k`, the empty chunk). Direct distances, no fitted readout; the emulator checks every claimed equivalence |
| `swap_classes.py` | whether invariance to a commutative operand swap is structural or learned: four swap classes (fresh operands; destination overwriting a swapped source; a second instruction whose swapped operand already carries a rank, or is the first instruction's result), each encoded by the trained T1/T2 pair and by a randomly initialised pair of the same architecture. Float noise on the random pair is structural; a reduction from the random pair's distance is what training supplied |
