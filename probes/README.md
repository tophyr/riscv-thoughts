# Probes

Diagnostics that read a frozen checkpoint and answer one question each. None
require training; all take `<t2_dir> <t1_path>` and most take a chunk length
and a batch count after that. Each file's docstring is the authority on how to
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

## Where the vp error is

| probe | the question it answers |
|---|---|
| `vp_budget.py` | **reach for this first** — the whole vp error budget from the model's own head, crossed over sources × immediate × depth plus opcode and bit position, each bucket as a rate AND as its share of total error. Takes a corpus config, so a past claim can be re-scored on its own population |
| `vp_depth2.py` | splits depth-2 error by WHICH instruction carries the immediate (producer, writer, both) and by opcode class — an axis a chunk-level "has an immediate" split cannot show |
| `vp_imm_share.py` | R-type vs I-type share, immediate bit positions, error crossed with depth. The mixed-corpus counterpart to `vp_slot_b_split` |
| `vp_slot_b_split.py` | operation vs operands, disjoint vs shared sources. Generates from an R-only corpus by construction, so it cannot see an immediate failure at all |
| `vp_superposition.py` | one output vs two; asserts a slot's values are unchanged by the other instruction |

## Isolating a cause

| probe | the question it answers |
|---|---|
| `vp_compose_control.py` | one depth-2 shape, six arms, only the literals moving, with the producer's own slot scored beside the consumer's — distinguishes a literal lost on consumption from one never encoded |
| `vp_drop_control.py` | the same question causally: the second instruction held identical while only the first varies, so a difference cannot be explained by the chunks being harder. Use this shape whenever a defect can make two different chunks produce the same encoder input |
| `vp_frame.py` | whether the essence and the vp target describe the SAME chunk. Splits vp by how T1's emitted wiring differs from the oracle's and by whether the canonical frames coincide. Reach for it when a cell resists every capacity lever |
| `imm_ceiling.py` | recovers one quantity at every pipeline stage (T1 essence → T2's written slot → the pool's view of it → T2 essence) to localize WHERE something is lost |
| `pool_bottleneck.py` | refits a bottleneck for one task and sweeps its width, to size a suspected capacity limit before changing the architecture |
| `opcell_rank_sufficiency.py` | whether a slot's state holds everything but the operand→position association, with a no-rank calibration arm |
| `depth_composition.py` | whether a vp deficit is in the essence or the head, by refitting and fine-tuning heads on frozen essences |
| `depth_stage.py` | the same quantity recovered at each encoder stage (slot → pool view → essence), split by dataflow depth |
| `depth_intermediate.py` | removes the "composition is a harder function" confound by handing the readout the true intermediate value |

## Geometry and whole-model checks

| probe | the question it answers |
|---|---|
| `additive_fusion.py` | whether two chunks computing the same function land on the same point. Builds the chunks directly and takes a distance — no fitted readout — and asserts equivalence through the emulator. Add a family with a `combine`/`resplit`/`perturb` triple in `FAMILIES` |
| `t1_scorecard.py` | scores a T1 ALONE, which nothing else does (`eval.py`'s collapse rows need a T2). **Verifies a T1 is intact; cannot rank T1s** — every candidate scores 100.0000% on the vp half, and the geometry half's raw distances move 3× across checkpoints with no relation to the settings, so read its ratios and only against a checkpoint's own numbers. `n_batches=0` runs the geometry half alone, in-process, without spawning a `datagen` that would perturb a training run sharing the box |
| `t2_reconstruct.py` | essence → recovered T1 vectors → instruction opcode/immediate accuracy, plus effective dimensionality. Regression check |
| `t2_faithfulness.py` | regression check on faithfulness and on how far essences sit off the unit sphere |
