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
point.** Neither implies the other; a model can score perfectly on one and fail
the other completely.

**Faithfulness — the essence determines the chunk's output values.** A next
tier can only recover what T2 preserved. `vp` scores it: an evaluator handed
the essence and the input values must produce every output value. The encoder
never sees a register value, so the essence is a symbolic description and the
evaluator must evaluate it, at anchor input states it never trained on — with
a fixed set it stores answers instead (METRICS.md). `vp = 0` at held-out
anchors is sufficient for faithfulness, not necessary.

**Equivalence — computed equivalents collapse.** Two chunks are equivalent when
their externally visible register inputs and outputs are identical, whatever
their instruction count, final PC, instruction order, or dead and same-value
writes; PC is not an independent output. "Two token sequences mean the same
thing if and only if they compress to the same point" is the framework's
definition of meaning (WHAT_IS_A_THOUGHT.md), so an essence that separates
`ADDI r1,r1,256; ADDI r1,r1,-16` from `ADDI r1,r1,32; ADDI r1,r1,208` is not a
thought vector regardless of its vp. Collapse has to follow from the function,
including equivalences no generator enumerates; collapsing the classes the
training mix lists is necessary, not sufficient.

Where each equivalence class stands. Equivalences come in two kinds. Those
that leave the chunk's live dataflow graph unchanged — a rename, a reordering,
dead code — give the encoder identical input, so they collapse **by
construction** (RISCV_DESIGN.md, "GVN"). Those where two different graphs
compute one function — `ADDI x,x,1; ADDI x,x,1` against `ADDI x,x,2` — collapse
only if training pulls them together: the behavioral distance term (RISCV_DESIGN.md,
"Collapse") trains on *partner pairs*, two spellings of one function placed in
the same batch, and a family of equivalences collapses only when its pairs are
among them (findings 56 and 57). Partner pairs come from a hand-written list
of rewrites and from a search that finds, for a random two-instruction chain, a
single instruction computing the same thing, proved equal by a solver.

The table gives essence distances on the best T2 so far (EXPERIMENT_LOG, "The
register frame", finding 61), 2,000 pairs per class (`probes/frame_scorecard.py`,
`probes/novel_equivalences.py`). A distance is the chord between unit essences:
0 is the same point, and two unrelated chunks sit at about 1.25. Each class is
read beside a *near-miss*, the same pair with one detail changed so they are no
longer equivalent, which must stay far:

| equivalence class | equivalent pair, against its near-miss |
|---|---|
| register rename, reordering of independent instructions, dead code, an x0-write beside live instructions, a commutative swap of two fresh operands into a third register | 0 by construction |
| a respelling, a cancelling pair, a reassociation | 0.0002–0.002 against 0.58–1.10 |
| a fold of two literals (`ADDI`, `XORI`, `ANDI`, `ORI`, shifts) | 0.022–0.035 against 0.42–1.17 |
| a negation flip through `SUB` from x0; doubling twice against `SLLI 2` | 0.029; 0.026, against 1.13; 1.14 |
| a same-value write against its absence | 0.046 against 0.91 |
| a commutative swap whose destination or operand is a swapped register, shared, read again or another instruction's result | 0.007–0.048; 0.187 for a composed operand in a one-output chunk |
| an operation family no training pair draws | not collapsed: shift folds at 1.10–1.17 with shifts held out of the search (finding 56) |

The guard on all of this is that inequivalent pairs stay apart. The aliasing
control, `x+x` against `x+y` — one spelling with an operand repeated, which a
rename cannot reach — reads 0.825 against 1.255 for unrelated chunks at T2
(finding 61) and 1.086 against 1.120 at T1 (finding 59). A collapse objective
that pulls those together is a worse failure than loose collapse (METRICS.md).

## Milestones

1. **[COMPLETE] Faithful at len=2, at held-out anchors, from the essence.** The
   frozen T1 (`runs/t1_setvp_500k`, finding 59) evaluates 2,457,595 of
   2,457,600 held-out (chunk, state, output) triples exactly with a
   three-layer residual vp head, its emitted wiring exact on every
   instruction. The best T2 over it is at **99.8965%**, 5,006 failing triples
   of 4,835,008, after 3M steps, one seed, binding exact on every chunk;
   outputs that depend on three inputs (one instruction's result fed to
   another with a third input) are at 96.1% and everything else above 99.9%
   (EXPERIMENT_LOG, "The register frame", finding 61). The same configuration
   fails 21,248 at 1M steps (finding 60), so configurations are compared at a
   stated step budget.

2. **[COMPLETE] Computed equivalents collapse at len=2, within the families
   the training pairs reach.** Finding 61's T2 collapses every two-to-one
   equivalence the probes build and every named rewrite, at vp 99.90%, one
   seed (Mission's table). A family no pair draws does not collapse (finding
   56); that is task 1.

3. **vp = 0 and collapse at len=4, then len=8** (task 2). A dead instruction
   is 15% of len=4 and 46% of len=8 chunks (finding 33); its reads are in the
   binding and pruned from the essence at every length.

4. **Beyond the happy path.** Everything above is writers-only (R/I-type ALU,
   valid, non-x0). Then: no-write instructions (task 3), PC/branches, memory,
   longer and looser chunks.

5. **T3.** Sequences of T2 blocks. T3 is the first tier that must combine
   bindings rather than emit them, so what a *composed* binding is gets
   decided before building (WHAT_IS_A_THOUGHT.md, "The Thought and Its
   Interface"). With it, what the binding heads emit: per output register,
   its external input registers in canonical role order (`out_inputs` already
   ships it), replacing the ordered input and output lists (`in_slot` /
   `out_slot` ListMLE), which are unchanged until then.

## Open tasks

In order. Tasks 1 and 2 are required for the MVP.

### 1. Composition built into the map: operators, not vectors

Collapse reaches an operation family only when its pairs are
in training: with shift folds held out of the search, `ADD t,a,a ; ADD t,t,t`
and `SLLI t,a,2` stay 1.10 apart although `SLLI x,1` ≡ `ADD x,x` is trained
(finding 56). They would meet unpaired only if the op-cell combines
descriptions the way functions compose, and nothing asks it to. The candidate
prior is on how descriptions combine, not where they land — the same move as
the rename symmetry, built in because no flat vector can learn it.

- **The idea.** Each instruction is an operator K_f on a learned embedding of
  register values, fitted so K_f ψ(x) ≈ ψ(f(x)) (a Koopman representation);
  a chunk's description is the product of its instructions' operators along
  the dataflow. If K for `ADD x,x` equals K for `SLLI 1`, then K² equals K
  for `SLLI 2` with no pair shown.
- **What is hard.** Two-input operations are not maps of one value, so a
  chunk is a tensor contraction over its dataflow graph, not a product.
  Bitwise operations and comparisons are not linear on integers, so a finite
  lift is approximate and its error compounds with depth. T1 would emit an
  operator and a binding, and the evaluator would apply the operator: a
  change to the tier contract.
- **The first test.** Unary functions of one register (`ADDI`, `XORI`,
  `SLLI`, `SRLI`, `ANDI`, `ORI`) at len 2: learn ψ and the operators from
  single instructions alone, and score finding 56's held-out classes with no
  fold pair in training. It says whether the prior buys transfer before any
  redesign.

### 2. Carry len=2's collapse to len=4 (milestone 3)

Finding 61's cell at `LEN=4`, scored against the same guard: its aliasing
control (0.825 against 1.255 unrelated) and near-miss distances must not fall
with the rest. Before training:

1. **The rename in the distance target.** The target minimises over every
   mapping of one chunk's inputs onto the other's only up to 4 input
   positions (len 2); longer chunks ship only the canonical-order mapping,
   and changing one immediate by one reorders the canonical inputs in
   1.0–5.8% of pairs at len 2–8 (finding 43). Needed: an approximation of the
   minimum — a search started from the canonical mapping, or a matching of
   positions by per-position distance — checked against d\*
   (`probes/target_vs_ideal.py`).
2. **An evaluator that ignores its inputs' order.** The evaluator
   concatenates the chunk's inputs in canonical role order, ties broken by
   colour. From three instructions on, two interchangeable inputs can share a
   colour, the register number then decides their positions, and a rename
   moves values between positions (finding 40). Embedding each (colour, value)
   pair with one shared layer and summing removes the order, and most of the
   input layer with it. Needs a remint of T1 and a comparison against
   finding 61.
3. **Anchor states.** On the chain-0.10 writers corpus with half the states
   aimed (`datagen/aim.py`), aiming straddles a comparison's threshold on
   nearly every row up to len=4 at 8 states, and an add's carry on 75% of
   rows at len=2 but 20–23% at len=8 with 8 states (52–60% with 16),
   and a comparison of a value computed through `AND`/`OR`/`ANDI`/`ORI`, a
   shift or another comparison on about 20% of rows, since no single input
   puts such a result at the threshold. Measure at len=4; use 16 states if 8
   leave carries uncovered. The evaluator runs once per state, so this is the
   step-time cost.
4. **Equivalents with no single-instruction form.** The search finds only
   equivalences one instruction computes; one whose every spelling is two
   instructions or longer — reassociations, commutative swaps in context —
   comes only from the hand-written rewrites, whose operand contexts at len=4
   are unmeasured. A search over two-instruction candidates, or pairing
   generated chunks whose register-file fields agree (most such matches in
   the natural stream are renames, already collapsed), would draw them from
   the function.

### 3. No-write instructions are invisible to the essence

Blocks milestone 4 (~⅓ of the ISA). Content reaches the essence only through a
register write (`value[out] = res` is gated on `out != 0`), so for branches and
stores the op-cell result is computed and discarded: valid-no-output and
invalid windows produce the identical essence (cos 1.0000), and all branches
collapse to one point. For an x0-write this is correct — it is equivalent to
its absence.

Designed, not built: route the op-cell result into a non-renameable
**operation slot** (seeded zero, fixed under rename, like x0) for every active
instruction. Open: last-write or accumulation across instructions; what a
taken branch, whose only effect is on the PC, contributes; and keeping an
x0-write invisible, since a slot carrying its content would separate
equivalents. No PC-effect vp target ships (`out_pcs`); decide one only when
branches enter scope.

### 4. Give T1 a validity task

T1's output norm is meant to say whether its token window is one complete,
valid instruction — near 1 if so, near 0 for a partial one, a window spanning
a boundary, several run together, or random tokens (Phase 10) — which is what
a segmenter reading a raw token stream needs; a gate trained on an encoder
without it plateaued at ~70% (Phase 9). The loss exists (`(‖essence‖ −
valid)²`, `valid_weight` 0.1), but the writers corpus has no `invalidity`
block, so every target is 1 and the current T1 cannot reject invalid input.
Every chunk T2 sees is valid, so this blocks only leaving the happy path.
Invalid windows come from `datagen/invalidity.py` via a config's `invalidity`
block.

### 5. Replace the structural scaffolding with a learned pass

The liveness pass and the colouring are exact rules, to be replaced by a
learned message pass over the same graph once the collapse scorecard holds
with them, checked against them, and deleted (RISCV_DESIGN.md, "What is built
in for good, and what is built in for now").

## Open ideas

Not scheduled; each needs a decision before it becomes a task.

### Evaluator throughput

The twelve-layer evaluator dominates a T2 step: it runs once per (row, anchor
state), 3,072 times per step at 8 states, and its 1024×3616 input layer
(3.7M weights) is sized for 32 input positions when a len-2 chunk has at most
4, and 16 output slots when it has at most 2. Slicing both to the chunk
length's bound is the same math at a fraction of the cost; BF16 autocast
(branch `experiment/bf16-autocast`) needs an equivalence check. Profile one
compiled step before building either. Task 2's order-free evaluator removes
most of the input layer anyway.

### Steer the corpus mix from the training signal

A config fixes its composition (opcode mix, `chain.rate`) for the whole run,
but at a fixed budget a higher composed-chunk rate buys depth-2 accuracy with
depth-1 exposure, and depth-2 accuracy tracks the number of composed rows
seen, not their share (EXPERIMENT_LOG, "T2 faithfulness", finding 30). A
controller could generate less of what is learned and more of what is not.
The lever is the data: reweighting the loss toward hard slots failed because
it amplifies the same rare rows (finding 17). Prior work: Teacher–Student
Curriculum Learning (Matiisen et al., 2017), Automated Curriculum Learning
(Graves et al., 2017), DoReMi (Xie et al., 2023), Online Data Mixing (Albalak
et al., 2023), Prioritized Level Replay (Jiang et al., 2021, the closest
match: a procedural generator's settings). To settle:

- drive on learning progress, not error level — ~3% of len=2 chunks have T1
  wiring that disagrees with the tokens and can never be satisfied;
- the rarest error bucket is about one triple per batch, so smooth heavily;
- error is measured on sources × immediate × depth (METRICS.md) but the
  generator's knobs are opcode mix and `chain.rate`, so it needs a mapping or
  new knobs;
- the pipeline is one-way behind a deep queue (`--mode weighted` biases
  order, not count), so it needs a back-channel or per-bucket streams;
- a run must record its mix trajectory, and the cosine LR schedule assumes a
  fixed distribution.

It has to beat the best fixed (rate, steps) pair on the natural corpus.

### A smaller vp head at T2

T2's head is twelve residual layers because no plain ReLU head evaluates a
perfectly described two-instruction chunk exactly (finding 34); T1's was
chosen as the smallest that reaches exact vp, so the essence does the most
work. The same criterion at T2 would press the encoder toward descriptions a
shallower head can read; a recurrent head (one block applied repeatedly) would
shrink it without fewer layers. Whether it matters depends on what the head
is: it is the measuring instrument, not T3, and the next tier must USE the
essence, not EXECUTE the chunk — so its size may be a cost of scoring rather
than a property the essence is graded on.

### Settings argued at 8 fixed anchors

The chain rate and the architecture settings argued from vp (findings 21, 26,
29, 30) were measured at 8 fixed anchor states, where the head stored answers
(finding 32), and have not been re-derived at held-out anchors. Re-check one
when a decision turns on it.

## Long horizon

**Equivalence that composes.** For the essence to represent the function
rather than the pairs it was shown, the op-cell has to describe operations so
that descriptions compose. The test is finding 56's held-out family: a change
succeeds when the shift folds collapse without a shift fold in training. The
candidate is task 1.

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
