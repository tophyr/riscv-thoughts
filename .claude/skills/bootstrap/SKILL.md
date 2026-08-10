---
name: bootstrap
description: Orient on this project before doing substantive work — read the docs in order, verify them against the code, then ask the user clarifying questions before starting. Use at the start of a fresh session, when picking up the current mission, or whenever you cannot already state what the mission is and why it is the mission.
---

# Bootstrap

Goal: reach the point where you can state, without looking, **what the current
mission is, why it is the mission, and what has already been ruled out** — and
where every premise you are about to build on has been checked against code
rather than prose. Then surface what is still unclear, *before* starting work.

Do not begin the task until step 4 is answered.

## 1. Read the docs, in order

Follow README's **Reading order** section. It ends at `TODO.md`, which holds the
live mission, the open tasks, the constraints, and the dead ends.

`EXPERIMENT_LOG.md` is an index over `EXPERIMENT_LOG/`, one file per phase. Read
the index, then the phases it marks live. Older phases are superseded design
kept because their dead ends are load-bearing — open one when you are about to
revisit the question it settled, not by default.

`CLAUDE.md` (already in context) carries the verification protocol and the data
pipeline's silent-corruption invariants. The rest of the pipeline guide —
commands, flags, the RVT format, multinode operation — is the `datagen` skill;
invoke it if the task touches `datagen/`, the RVT format, or the stream tools,
rather than working from memory.

## 2. Verify the docs against the code

Docs summarize, and summaries drift. Before relying on any claim, read the thing
it describes. At minimum:

- the encoder core (`compressor/model.py`) — the state machine, what each
  consumer actually reads, and what the pool is fed;
- whichever loss or metric you are about to move (`compressor/train.py`);
- the `hparams.json` of **every** run whose numbers you intend to cite.

**A run's directory name is not its configuration.** Names encode intent at
launch; `hparams.json` records what ran. Today's example: a run named `…rtrank…`
had `pool_extra=write_time`, so the thing its name advertises never reached the
essence — and it had been recorded in three documents as a failed experiment.

Cite `path:line` for any claim you make about what the code does. If you cannot
cite it, you have not verified it — say so instead of asserting it.

## 3. Check the metrics before you trust them

Numbers in this project have scope conditions that are easy to miss:

- **Aggregate metrics hide subset failures.** The current open deficit lives in
  ~12% of a subset and is invisible in the training-log number. Use the probe
  the task names, not the aggregate.
- **Know what a metric certifies.** `vp` is scored against 8 fixed anchor
  states, so `vp = 0` means "correct at 8 input points", not "encodes the
  function".
- **Calibrate every probe** against a quantity known independently, and against
  the base rate. A probe scoring at the majority-class frequency has learned
  nothing; one scoring below the model's own trained head is measuring itself.

## 4. Ask, then start

Ask the user the questions where **two plausible answers would send you down
different roads** — not everything you are curious about. Include anything the
docs answer inconsistently, with each other or with the code; those are bugs in
the docs and the user wants them fixed, not worked around.

"I don't know yet, I need to check X" is a complete answer to give.

The user would much rather spend time clarifying up front than have you spend
hours building on a wrong premise and both of you unwinding it. Cheap questions
now; expensive backtracking avoided.

## When the mission lands

Update the docs and the memory notes to match what is now true — correct the
claim in place, do not append a note about the correction — then hand off to a
fresh session rather than rolling on.
