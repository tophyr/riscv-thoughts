# Roadmap

What is left to do, in order. Nothing here is a history.

- How a number is measured, what it certifies, and how far it moves when
  nothing changed: [METRICS.md](METRICS.md).
- The evidence behind any result named here:
  [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
- The architecture and why it is shaped this way:
  [RISCV_DESIGN.md](RISCV_DESIGN.md).
- Running a training cell or a corpus: the `datagen` skill. Scoring a frozen
  checkpoint: [probes/README.md](probes/README.md).

## Mission

**T2 is done when its essence is a lossless behavioral description of its
instruction chunk AND two chunks that compute the same function land on the same
point.** Both are required; neither implies the other, and a model can score
perfectly on one while failing the other completely.

**Faithfulness — the essence determines the chunk's output values.** Can the
chunk's output register values be recovered from the essence, given the input
values? This is what a future T3 consumes: a next tier can only recover what T2
preserved. `vp` is how it is scored, and it scores the essence composed with the
vp head's ability to EVALUATE it — the encoder never sees a register value, so a
slot holds a symbolic description of the function that produced it and the head
is asked to evaluate that description at anchor input states. Those states have
to be ones the head never trained on: with a fixed set it stores answers instead
of evaluating (METRICS.md). `vp = 0` at held-out anchors is sufficient for
faithfulness and not necessary.

**Equivalence — computed equivalents collapse.** Two chunks are equivalent when
their externally visible register inputs and outputs are identical — whatever
their instruction count, final PC, instruction order, or dead and same-value
writes; PC is not an independent output. The binding is the interface: a graph
from external registers through operation roles to results, not an ordered
list. "Two token sequences mean the
same thing if and only if they compress to the same point" is the framework's
definition of meaning (WHAT_IS_A_THOUGHT.md), so an essence that separates
`ADDI r1,r1,256; ADDI r1,r1,-16` from `ADDI r1,r1,32; ADDI r1,r1,208` is not a
thought vector regardless of its vp.

Where each equivalence class stands. The essence is graded in the register
frame (RISCV_DESIGN.md, "GVN"): every class that leaves the chunk's live
dataflow graph unchanged collapses **by construction**, and the trained models
read exactly zero on them (`probes/frame_scorecard.py`, 2000 pairs per class);
every class that needs the op-cell to compute is taught by the behavioral
distance term over a partner mix drawing every rewrite (RISCV_DESIGN.md,
"Collapse"). The figures are the T2 record (EXPERIMENT_LOG, "The register
frame", finding 50; essence chord distances, unrelated chunks at 1.25,
equivalent pair against a near-miss):

| equivalence class | status |
|---|---|
| register rename, reordering of independent instructions, dead code, an x0-write beside live instructions, a commutative swap of two fresh operands | 0 by construction |
| a respelling, a cancelling pair, a reassociation | 0.0001–0.003 against 0.89–1.26 |
| a fold of two literals | 0.016 against 0.57; `ADDI` / `XORI` chains 5.8x / 44.8x |
| a same-value write against its absence; one no-op for another | 0.027 against 1.03; 0.0002 |
| a commutative swap whose operand is shared, read again or another instruction's result | 0.0001–0.034; 0.080 for a composed operand in a one-output chunk |
| an identity write against the empty chunk | 1.01 — the empty chunk occurs in no corpus |

The guard on all of this is the opposite row: pairs a transformation did NOT
leave equivalent must stay far apart — the aliasing control (`x+x` against
`x+y`) reads 0.907 against 1.253 for unrelated chunks at T2 and 1.088 against
1.213 at T1 (finding 50). A collapse objective that pulls those together is a
worse failure than loose collapse (METRICS.md).

## Milestones

1. **[COMPLETE] Faithful at len=2, at held-out anchors, from the essence.** T1
   holds exactly: value prediction read from its essence is 100.0000% exact on
   2,457,600 held-out triples with a three-layer residual vp head, its emitted
   wiring exact on every instruction (`probes/frame_scorecard.py`; frozen as
   `runs/t1_distance_cosbal_500k`, finding 50). The T2 over it, trained with
   the behavioral distance term and rewrite partner pairs, is at **99.7857%**,
   10,355 failing triples of 4,831,488, after 1M steps, one seed, binding
   exact on every chunk, with outputs that depend on three inputs (a composed
   slot over disjoint inputs) at 89.6% and everything else above 99.78%
   (EXPERIMENT_LOG, "The register frame", finding 50). The residual falls
   with training budget — 5,418–5,511 failing at 1M and 36 at 10M in the
   first-read frame (finding 35) — so 1M is the budget a configuration is
   judged at.

2. **[VALIDATING] Computed equivalents collapse at len=2.** Every class tried
   collapses at vp 99.79%, one seed (Mission's table; finding 50); the
   decisions that produced it are under review (task 1). Settled
   before chunks get longer because every added instruction multiplies the
   equivalences a chunk participates in.

3. **vp = 0 and collapse at len=4, then len=8.** Unmeasured in the register frame. A dead
   instruction is 15% of len=4 and 46% of len=8 chunks (finding 33); its reads
   are in the binding and pruned from the essence at every length.

4. **Beyond the happy path.** Everything above is writers-only (R/I-type ALU,
   valid, non-x0). Then: no-write instructions (task 3), PC/branches, memory,
   longer and looser chunks.

5. **T3.** Sequences of T2 blocks. Prerequisite: decide what a *composed*
   binding is — T3 is the first tier that must combine bindings rather than just
   emit them (WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface"). Settle
   that before building, not during.

## Open tasks

In order. Tasks 1 and 2 are required for the MVP. The rest are queued, not parallel
work.

### 1. Validate milestone 2

The collapse at len=2 (finding 50) rests on decisions that each change what
the encoder is asked to learn, measured on one seed. Each is examined for
whether it is right, not only whether it worked, before the len=4 work builds
on it.

- **The balanced distance loss** (finding 49) is the critical one, and
  nonstandard: squared errors of equivalent pairs (target 0) and of all other
  pairs are averaged separately and summed. Its nearest standard relative is
  class-balanced loss, applied here to a regression target split at exactly
  0. Open questions:
  - The 1:1 split is untuned.
  - An equivalent pair's weight is set by how many equivalent pairs the batch
    holds, about 119× an ordinary pair's at 613 of 73,536, so the corpus mix
    sets it.
  - The split is a step at d = 0: a near-miss at d = 0.01 gets the ordinary
    weight, so separating near-misses is not boosted.
  - A pair wrongly given d = 0 is pulled together 119× harder, so the d = 0
    set needs auditing for false equivalents.
  - Its only isolated evidence is a 100k-step resume at the linear target. The
    1M-step cell combines it with the √(2d) target and the identity rewrite,
    so no measurement isolates it there.
- **The canonical input and output orders** (RISCV_DESIGN, "Collapse").
  Both are exact for equivalent chunks; neither is designed for chunks that
  differ, whose distance targets grade separation.
  - Changing one immediate by one reorders the canonical inputs in 1–6% of
    pairs (finding 43), comparing those pairs at non-corresponding inputs;
    how much that inflates their targets is unmeasured.
  - Outputs are sorted by the raw bytes of their witness values, effectively a
    hash, so two different chunks' outputs can be paired arbitrarily; how
    often near-misses are mispaired, at len 2, 4 and 8, is unmeasured. The
    sort avoids a matching across output permutations, which matters at 8
    outputs. Alternatives: a minimum-cost matching (an assignment per row
    pair) or a sort keyed first on the canonical inputs an output depends on.
- **Same-value writes dropped from the outputs,** decided at 16 uniform states
  (finding 46).
- **The pointer ordered by a projection of each output's description** (finding
  46): 15–18% of commutative swaps still flip it.
- **The rewrite mix:** half of every batch is partner pairs. Rewrites held out
  of training generalize only partly, and a collapse lapses when its pairs leave
  the mix (finding 47).
- **A second seed,** to tell the 0.056-point vp gap to finding 44's cell from
  noise.

### 2. Carry len=2's collapse to len=4 (milestone 3)

The distance term, the partner mix and the pointer order are settled at
len=2 (findings 47–50); what changes at len=4 is the anchor and witness
budget per row, the canonical roles' cost and near-miss flips (findings 42
and 43: comparisons through operations the aiming cannot solve, carries at
16 states), and the output counts a chunk can have. Before training at
len=4: aim the anchors with 16 states if 8 leave carries uncovered, and
measure the rewrite generator's operand contexts at that length.

Decisions that remain:

- **Mining equivalents from the natural stream.** Two generated chunks whose
  canonical witness outputs are identical compute one function, so datagen
  could pair them without enumerating any rewrite. Most such matches are
  renames, which collapse by construction; how many are algebraic
  equivalents is unmeasured and can be counted from the oracle alone.
- **The partner fraction.** How much of a batch is partner pairs against
  naturally drawn chunks: half at T2, a quarter at T1.
- **The baseline to beat** is finding 50's cell, one seed. Its aliasing control
  (0.907 against 1.253 unrelated) and its near-miss distances are the guard —
  they must not fall with the rest.
- **The binding on the wire.** Per output register, its external input
  registers in canonical role order (`out_inputs` already ships it); the
  ordered input and output lists (`in_slot`/`out_slot` ListMLE) go away. What
  the heads predict instead, and how T3 consumes it, is open; until then the
  binding heads and their ordered targets are unchanged.
- **Removing the scaffolding.** The liveness pass and the colouring are exact
  structural rules in the encoder, to be replaced by a learned message pass
  over the same graph once the collapse scorecard holds with them, checked
  against them, and deleted (RISCV_DESIGN.md, "What is built in for good, and
  what is built in for now").

### 3. No-write instructions are invisible to the essence

Blocks milestone 4 (~⅓ of the ISA). Content reaches the essence only through a
register write (`value[out] = res` is gated on `out != 0`), so for branches,
stores and x0-writes the op-cell result is computed and discarded. For an
x0-write that is correct — PC is not an output, so the instruction is equivalent
to its absence (below) — but for branches and stores, whose effects are on the
PC and memory, it loses the behavior. Verified:
valid-no-output and invalid windows produce the *identical* essence (cos
1.0000), so validity cannot separate them and all branches collapse to one
point.

Designed, not built: route the op-cell result into a non-renameable **operation
slot** (seeded zero, fixed under rename, like x0) for every active instruction,
so content reaches the essence regardless of register write. Open: whether a
single slot last-write or an accumulation across instructions; and what a taken
branch contributes to the essence. Routing the result must not make an
x0-write no-op visible at all: it is equivalent to its absence (below), so a
slot that carried its content would separate equivalents.

**PC is not an independent output.** Two chunks with identical externally
visible register inputs and outputs are equivalent whatever their instruction
count or final PC. So an x0-write, a same-value write or a dead instruction
beside a live chunk leaves the chunk equivalent to the live chunk alone, and
the distance between them is a collapse failure (milestones 2 and 3), not a missing
target here; no PC-effect vp target is shipped (`out_pcs` stays unshipped). The
one case the decision does not settle by itself is a taken branch, whose only
effect is on the PC; confirm what it contributes when branches enter scope,
before building any target for them.

### 4. Give T1 a validity task

T1's output norm is meant to say whether its token window is a complete, valid
instruction — near 1 if so, near 0 for a partial instruction, a window spanning
an instruction boundary, several instructions run together or random tokens —
with the meaning in the direction (Phase 10, magnitude-as-validity). That is
what a segmenter reading a raw token stream needs, and a gate trained on an
encoder without it plateaued at ~70% (Phase 9). The loss exists
(`(‖essence‖ − valid)²`, `valid_weight` 0.1), but the writers corpus declares
no `invalidity` block, so every row is valid, the target is always 1 and the
loss teaches nothing: the current T1 cannot reject invalid input. Every chunk T2
sees is valid by construction, so this does not block the milestones above; it
is needed before anything leaves the happy path. Invalid windows are generated
by `datagen/invalidity.py` from a config's `invalidity` block.

## Open ideas

Not scheduled and not yet a task; each needs a decision before it becomes one.

### An evaluator that does not depend on its inputs' order

The evaluator concatenates an output's inputs in canonical role order,
interchangeable inputs ordered by the encoder's colour. From three
instructions two interchangeable inputs can also share a colour, and the
register number then decides their positions, so a rename would move values
between positions again (EXPERIMENT_LOG, "The register frame", finding 40).
Embedding each (colour, value) pair with one shared layer and summing the
results removes the order entirely, and removes most of the evaluator's input
layer as well (next idea). It changes the evaluator's architecture, so it
needs a remint and a comparison against finding 44. Settle before len=3.

### Evaluator throughput

The twelve-layer evaluator dominates a T2 step: it runs once per (row, anchor
state, output slot), 6,144 times per step, and about 3.2M of its parameters
are input-layer weights for 32 input positions when a len=2 output reads at
most three. Slicing the input positions to the chunk's bound, as the output slots
already are, is the same math at a fraction of the input layer's cost; BF16
autocast (branch `experiment/bf16-autocast`) needs an equivalence check. A
profile of one compiled step would size both before either is built.

### Steer the corpus mix from the training signal, while training runs

A corpus config fixes its composition (opcode mix, `chain.rate`) for the whole
run, which presumes the right proportions are known in advance. They are not:
at a fixed step budget, raising the composed-chunk rate buys depth-2 accuracy by
spending depth-1 exposure, and depth-2 accuracy tracks the absolute number of
composed rows seen rather than their share (EXPERIMENT_LOG, "T2 faithfulness",
finding 30). A static rate is a guess at a trade a controller could manage
continuously: generate less of what the model has mastered, more of what it is
still learning, and raise a cell's rate again if its accuracy slips.

The lever has to be on the DATA side. Reweighting the loss toward hard slots
(`--vp-depth-weight`) failed because it amplifies gradient from the same rare
rows, while generation supplies new ones (finding 17).

Prior work to start from: Teacher–Student Curriculum Learning (Matiisen et al.,
2017) samples tasks by the absolute slope of their learning curves, so a
plateaued task fades and a forgotten one returns; Automated Curriculum Learning
(Graves et al., 2017) treats the mix as a bandit rewarded by learning progress;
DoReMi (Xie et al., 2023) and Online Data Mixing (Albalak et al., 2023) reweight
pretraining domains online; Prioritized Level Replay (Jiang et al., 2021) does
it over a procedural generator's settings, the closest structural match here.

What would have to be settled:

- **Drive on learning progress, not error level.** Some rows cannot be
  satisfied — where T1's emitted wiring disagrees with the tokens (~3% of len=2
  chunks), T2 is graded against a chunk it was not shown — and an error-chasing
  controller would pour data into them indefinitely.
- **Rare cells give noisy readings.** The rarest error bucket at len=2 is about
  one supervised triple per batch, so the controller must be slow and smoothed.
- **The knobs are not the buckets.** Error is measured on sources × immediate ×
  depth (METRICS.md); the generator controls opcode mix and `chain.rate`, and
  nothing controls shared sources directly. It needs a knob-to-bucket mapping or
  new generator knobs.
- **Plumbing.** The pipeline is a one-way filter with generation on odin behind
  a deep prefetch queue, so it needs a back-channel, or per-bucket streams that a
  trainer-side mux selects among (`datagen --mode weighted` biases order, not
  count, so it does not do this as-is).
- **Reproducibility.** A run would no longer be described by one config file; it
  would have to record its mix trajectory. The cosine LR schedule also assumes a
  fixed distribution.

The comparison it has to win is against the best fixed (rate, steps) pair, scored
on the naturally drawn corpus.

### Aimed anchors for chunks

Training anchors are aimed at each row's own boundaries (`datagen/aim.py`;
RISCV_DESIGN.md, "Training"), which makes the distance-term T1 exact
(EXPERIMENT_LOG, "The register frame", finding 42). The generator walks a
boundary back through chunks too, and on the chain-0.10 writers corpus at
half the states aimed it straddles a comparison's exact threshold on nearly
every row up to len=4 at 8 states and up to len=8 at 16; an add's carry on
75% of rows at len=2 and 20–23% at len=8 with 8 states, 52–60% with 16.
What it does not reach:

- **Comparisons of a value computed through an operation it cannot solve**
  for one operand — `AND`/`OR`/`ANDI`/`ORI`, shifts, another comparison — are
  straddled on about 20% of rows. Such an operation has no single input that
  puts its result at the threshold; aiming them means choosing the nearest
  values the operation can reach on either side.
- **The anchor budget at length.** Each boundary takes two states, so longer
  chunks need more states per row, and the T2 evaluator runs once per state.
  The count for T2 is open.
- **The held-out set is never aimed**, so the score reaches a boundary only by
  chance; a scoring corpus aimed on purpose would test them deliberately.

### A smaller vp head at T2

T2's vp head is twelve residual layers because no plain ReLU head evaluates a
perfectly described two-instruction chunk exactly (METRICS.md; finding 34). The
criterion T1's head was chosen by — the smallest head that still reaches
correct results, so the essence does as much of the work as possible — has not
been applied at T2. Applying it means putting pressure on the encoder to
describe the chunk well enough for a shallower head. The essence's geometry has
tightened with the deep head — mean distance between unrelated chunks 1.19 with
the plain four-layer head, 0.67–0.69 with the residual head at 10M, swap
collapse unchanged (finding 35) — and whether that costs a shallower consumer
is what this would measure.

Depth need not cost parameters: a recurrent head, one block applied repeatedly,
would be another way to shrink it, if an iterated fixed block can do what twelve
distinct ones do. Not tried.

It may also not need shrinking at all. The vp head is the measuring instrument,
not T3. If len=8 chunks turn out to need an absurd head — 64 layers, say — that
says the chunk's function is expensive to evaluate, not that the essence is
poor. What has to be ensured is that the encoder produces an essence the next
tier can USE; it does not have to be ensured that T3 can EXECUTE the chunk. On
that reading the head's size is a cost of scoring, not a property the essence
is graded on.

### Settings argued at 8 fixed anchors

The chain rate, the vp-as-evaluator analysis (findings 21, 26, 29, 30) and the
architecture settings argued from vp were measured at 8 fixed anchor states,
where the head stored answers (finding 32). None has been re-derived at held-out
anchors; the step budget has (1M against 10M, finding 35). Re-check one when a
decision turns on it.

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
