# Probes

Diagnostics that read a frozen checkpoint and answer one question each. None
train a model except `vp_gt_depth.py`, which trains readouts on ground-truth
chunk features in place of an essence; the usage line in each docstring gives its arguments. Each file's docstring is the authority on how to
read its output — this index only says which one to reach for.

How to trust what a probe reports, and the failure modes that have produced
withdrawn conclusions here, are in [METRICS.md](../METRICS.md) ("Probe
methodology").

Run them with the venv active, since several shell out to `python -m datagen`
to generate their own corpus:

```bash
source .venv/bin/activate
python probes/vp_budget.py runs/<t2_dir> runs/<t1_dir>/encoder.pt 2 300
```

The two that score vp read their anchor states from the batches they score and
generate their corpus with `datagen --heldout-anchors`, so a score is at input
states no model trained on (METRICS.md). Probes that rebuilt one fixed anchor set from a
seed are deleted; recover one from git history and port it to `batch.anchors`
before relying on it.

| probe | the question it answers |
|---|---|
| `vp_budget.py` | **reach for this first** — the whole vp error budget from the model's own head at held-out anchors, crossed over sources × immediate × depth plus opcode and bit position, each bucket as a rate AND as its share of total error. Takes a corpus config, so a past claim can be re-scored on its own population, and an optional corpus path, so several models can be scored on identical data |
| `t1_scorecard.py` | scores a T1 ALONE: its own vp head at held-out anchors, by opcode, and the essence distance between instructions computing the same function and different ones. The geometry half's raw distances move across checkpoints with no relation to the settings, so read its ratios, and only against a checkpoint's own numbers. `n_batches=0` runs the geometry half alone, in-process, without spawning a `datagen` that would perturb a training run sharing the box. T1's binding is scored by `scripts/eval.py` |
| `vp_gt_depth.py` | whether a readout of a given shape can evaluate a two-instruction chunk at all: trains readouts of several depths (plain Linear+ReLU, or residual with LayerNorm) on ground-truth chunk features instead of an essence, scores them at held-out anchors beside the trained T2's own head, and saves per-cell results and weights at every evaluation. Trains for as many steps as it is given, so it occupies the GPU like a training run |
| `vp_gt_misses.py` | where a saved `vp_gt_depth.py` checkpoint still fails: every missed triple with its operands recovered by emulation, bucketed by writer, producer, operand roles and held-out state, with the carry-chain length of each `ADD`/`SUB` miss |
| `additive_fusion.py` | whether two chunks computing the same function land on the same point. Builds the chunks directly and takes a distance — no fitted readout — and asserts equivalence through the emulator. Add a family with a `combine`/`resplit`/`perturb` triple in `FAMILIES` |
| `write_position.py` | whether the essence depends on WHERE a chunk's write happens and on how many instructions it holds rather than on what it computes: moves one `ADDI` between step 1 and step 2 of the two-slot grid beside an inert x0 no-op, and compares the five spellings of the identity on one register (`ADDI r,r,0`, `ADD r,r,x0`, `ADDI r,r,k; ADDI r,r,-k`, the empty chunk). Direct distances, no fitted readout; the emulator checks every claimed equivalence |
| `swap_classes.py` | whether invariance to a commutative operand swap is structural or learned: four swap classes (fresh operands; destination overwriting a swapped source; a second instruction whose swapped operand already carries a rank, or is the first instruction's result), each encoded by the trained T1/T2 pair and by a randomly initialised pair of the same architecture. Float noise on the random pair is structural; a reduction from the random pair's distance is what training supplied |
