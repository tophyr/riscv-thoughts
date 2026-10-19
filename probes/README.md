# Probes

Diagnostics that read a frozen checkpoint and answer one question each. None
require training; the usage line in each docstring gives its arguments. Each file's docstring is the authority on how to
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
| `additive_fusion.py` | whether two chunks computing the same function land on the same point. Builds the chunks directly and takes a distance — no fitted readout — and asserts equivalence through the emulator. Add a family with a `combine`/`resplit`/`perturb` triple in `FAMILIES` |
