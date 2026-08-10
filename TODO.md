# Roadmap

Forward-looking only. Findings and their evidence live in EXPERIMENT_LOG.md;
architecture and rationale in RISCV_DESIGN.md. Nothing here is a history.

## Mission

**Build a T2 encoder whose essence is a lossless behavioral description of its
instruction chunk — vp = 0.0.**

vp measures whether the chunk's output register values can be predicted from
the essence given the input values. It is the metric that matters because T2's
essence is what a future T3 consumes: a next tier can only recover what T2
preserved.

**The vp number in a training log is already a held-out number.** `datagen`
generates every batch fresh and the trainer never revisits one, so there is no
train/test split to construct and no memorization to discount — quote the
training log directly. (`scripts/eval.py` exists to score *binding* and the GVN
rows on a fixed corpus, not to de-bias vp.)

## Milestones

1. **[CURRENT] vp = 0.0 at len=2.** Now at **0.0198**
   (`runs/t2_len2_d256_s1000000_dep4_vp1_pool16_pma_ord1_writers`), from 0.3525
   before the write-order fix. Single instructions are already exactly 0.000, so
   len=2 is the smallest case that is not yet solved.

   **`vp = 0.0` is meant literally, and it is scored discretely.** vp trains as a
   per-bit BCE, so the loss value is asymptotic and cannot itself reach 0. The
   *output* is discrete, so use that: the acceptance criterion is a **100% exact
   32-bit match rate on the `probes/vp_slot_b_split.py` subset** — including its
   overlap (shared-source) column, which is where the error lives. T1 and T2 at
   len=1 have already reached exact-0, so this is a demonstrated bar, not an
   aspiration.
2. **vp = 0.0 at len=4, then len=8.** Now 0.244 and 0.461. Do not start these
   until len=2 is solved — the len=2 case is the same mechanisms with less
   noise, and every diagnosis so far has transferred.
3. **Beyond the happy path.** Everything above is writers-only (R/I-type ALU,
   valid, non-x0). Then: no-write instructions, PC/branches, memory, longer and
   looser chunks.

   **Also strengthen the vp metric itself here.** vp is supervised *and* scored
   against **8 fixed anchor states** (`make_anchor_states(8, seed=0)`, identical
   gen-side, trainer-side, in `compressor/eval.py:410` and in every probe), and
   the head's input features are those same 8 vectors for every row
   (`in_vals_i`, `compressor/train.py`). So `vp = 0` certifies "the output values
   at 8 fixed input points are recoverable from the essence", which is weaker
   than "the essence encodes the chunk's I/O function" — in principle a model
   could satisfy it by storing 8×32 bits per output slot without representing
   operand binding at all. That is accepted as the success metric for the happy
   path: it is cheap and it has so far tracked the mechanisms we care about. It
   is *not* good enough to certify a lossless behavioral description once the
   corpus gets harder. Options when we get there: more anchors, a held-out
   anchor set the head never trained on, or anchors resampled per batch (which
   means shipping them in the RVT batch, i.e. a change to INVARIANT 1).
4. **T3.** Sequences of T2 blocks. Prerequisite: decide what a *composed*
   binding is — T3 is the first tier that must combine bindings rather than
   just emit them (WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface").
   Settle that before building, not during.

## Open tasks

**Task 1 is the exclusive focus until it is solved.** Tasks 2–5 are queued, not
parallel work. When task 1 lands: update these documents and the memory notes,
then hand off to a fresh session rather than rolling on.

### 1. [CURRENT] R-type operand binding when instructions share a source register

~65% of the remaining len=2 error. On R-type data-independent pairs the second
output slot loses **2.25 points** (100% → 97.75%), and *all* of it is the ~12%
of chunks where the two instructions read a common register: disjoint-source
pairs are at 99.999%, shared-source at ~83%. Operation identity is fine (opcode
99%+); what fails is knowing which input feeds which operand (~99.8% disjoint
vs ~51% shared).

**Measure this with `probes/vp_slot_b_split.py`, NOT with aggregate vp.** The
deficit lives in ~12% of a subset, so it is invisible in the training-log vp
number — a fix could work and aggregate vp would not move, or vice versa. Three
separate wrong conclusions in this phase came from reading the aggregate. Run:
`python probes/vp_slot_b_split.py <t2_run_dir> runs/t1_writers_250k/encoder.pt 400`
and read the `s1 overlap` column (currently ~83%, against ~100% disjoint).

**Two attempts to feed the pool more ordering both failed; a third run tested
nothing.** Check the hparams before generalizing from them:

- `pool_extra=order` — fed `read_time` (in the `stamp` encoding) + `write_time`
  to the pool. Unmoved at ~83%.
- `pool_extra=event` — fed the whole slot state to the pool. vp ~17% better,
  but doubled commutative distance and halved separation. Rejected.
- `read_time_mode=rank` — ran with **`pool_extra=write_time`**
  (`runs/t2_len2_d256_s250000_dep4_vp1_pool16_pma_pewrite_time_rtrank_writers/hparams.json`),
  so the corrected `read_time` never reached the pool; it changed only the event
  channels, which the essence does not see under `write_time`. Structurally
  incapable of moving essence-side operand recovery, so it says nothing about
  whether a corrected read-order would help.

What the two real pool experiments share is that the pool is
`softmax(pool_k(pool_src) · q)` over the register slots
(`compressor/model.py:277-295`) — it never sees `content`, so any channel it
receives must be used **unconditionally**. That one fact explains both: it cannot
help (read-order is recorded on the slot being *read*, while the fact needing it
lives on the slot doing the *reading*; associating them is a cross-slot join a
sum cannot perform) and `event` costs equivalence collapse (a commutative swap
changes read-order, and the pool cannot know to ignore that for `ADD` but not for
`SUB`).

**`pool_extra=order` + `read_time_mode=rank` is untried.** The `order` run used
`stamp`, which is off-by-one *exactly* on shared-source pairs — the pool was fed
a corrupted read-order on precisely the failing subset. Cheap, and not a repeat.
The cross-slot-join argument above predicts it still will not work, and it
carries the finding-9 equivalence risk (read_time moves under a commutative
swap), so it ranks below the op-cell hypothesis — but it is not a dead end.

**Leading hypothesis — the consumer has to be content-conditioned.** The op-cell
is the one consumer that sees the opcode and both operands in the same
activation, and it is the one consumer never modified. Give it each operand's
canonical read rank at read time (`compressor/model.py:196-253` — the rank is
already computed inside the loop; the op-cell currently reads only the `value`
channels, `model.py:204`). Then the association is *local*: no later cross-slot
comparison is needed. And the equivalence-collapse cost that killed
`pool_extra=event` is not implied here, because a content-conditioned cell *can*
use operand position for `SUB`/`SLT`/`SRL` and symmetrize it for
`ADD`/`AND`/`OR`/`XOR` — which is exactly what the pool could not do. That
distinction — conditional vs unconditional use of read-order — is what makes
this a different experiment from the pool runs, not another instance of them.

Falsifiable predictions, cheapest first:

- **(probe, no training — this probe does not exist yet; write it first)** A
  readout given the frozen model's per-slot state *plus* each operand's true
  canonical rank should recover operand→position on the overlap subset at ~100%.
  If it does not, the information is already gone upstream of the pool and the
  op-cell change cannot help either — that kills the hypothesis before any GPU
  time is spent. `probes/vp_slot_b_split.py` measures the deficit but does not
  test this; model the new probe on it, and calibrate it (see Dead ends) before
  believing either outcome.
- **(training)** With the rank fed to the op-cell, `s1 overlap` moves off ~83%
  while `state_max_err` stays `0.00e+00` and the GVN rows (rename /
  commutative / distinct) hold.
- **The failure mode to watch for** is finding 9 reappearing inside the value
  channels: commutative distance doubling and separation halving. Score every
  run on the GVN rows, not just vp.

**`read_time_mode=rank` is required for this experiment, and `stamp` should be
deleted once it is.** `stamp` writes `2(k+1)+role`, which equals the canonical
position only while every read is fresh; a repeat read is (correctly) not
re-stamped, leaving a hole, so every later operand overshoots — i.e. `stamp` is
wrong *exactly on the shared-source subset that is failing*
(`compressor/model.py:226-239`). `rank` stamps the running count of distinct
registers read, so `read_time` **is** the canonical position for every consumer,
with no conversion. Nothing has yet tested `rank` in a place the essence can see
it. See "Flag cleanup" below for what deleting `stamp` costs.

### 2. I-type immediate encoding

~35% of the remaining len=2 error, concentrated in bits 0–11 (the 12-bit
immediate field). Present even on single instructions with one output, so it is
independent of task 1. It shrinks with training rather than being structural
(`ADDI` 92.3% → 97.7% from 250k → 1M steps), so it may be a capacity or
optimization problem rather than a representational one.

T2 has a built but unused `imm_head`, skipped on the grounds that a chunk has
no single immediate — but it has one per instruction. Untried.

### 3. Mint a production T1

Every T1 in use is a diagnostic trained on a clean corpus (`--inject-invalid 0`),
so it has no validity task and cannot reject invalid input. Needed before
anything leaves the happy path. Fold in the `rd==x0` immediate mask.

### 4. No-write instructions are invisible to the essence

Blocks milestone 3 (~⅓ of the ISA). Content reaches the essence only through a
register write (`value[out] = res` is gated on `out != 0`), so for branches,
stores and x0-writes the op-cell result is computed and discarded. Verified:
valid-no-output and invalid windows produce the *identical* essence (cos
1.0000), so validity cannot separate them and all branches collapse to one
point.

Designed, not built: route the op-cell result into a non-renameable **operation
slot** (seeded zero, fixed under rename, like x0) for every active instruction,
so content reaches the essence regardless of register write. Open: whether a
single slot last-write or an accumulation across instructions; whether routing
the result makes equivalent x0-write no-ops spuriously distinct across
immediates; and whether branches also need a real PC-effect vp target (the SSA
oracle already computes `SELECT_PC` but does not ship it).

### 5. Flag cleanup — lands with task 1, not separately

**Project policy: options do not survive on their own.** Once a better way to do
something is settled, the worse way gets deleted rather than kept behind a flag.
Ongoing branching/versioning complexity is a cost we do not pay if we can pay a
one-time reshaping cost instead. Breaking changes — including reminting the
frozen T1 — are acceptable **on the one condition that new and old can be
compared and shown not to have regressed.**

Standing candidates:

- **`read_time_mode`: delete the option, keep `rank` only.** `stamp` is a known
  wrong encoding (open task 1), so there is nothing to preserve. Cost of doing
  it: `stamp` is baked into *every* existing checkpoint — the frozen
  `runs/t1_writers_250k` and every `t2_len2_*` run — so deleting it means
  reminting T1 and re-running the current-best len=2 config, then checking the
  new numbers against the recorded ones (vp, the `vp_slot_b_split` columns, the
  GVN rows, `state_max_err`, binding accuracies). Do that comparison; do not
  skip it because the change "should" be neutral.
- **`pool_extra`**: `order` and `event` are both recorded dead ends. Once task 1
  resolves, keep only the survivors (`write_time`, plus whatever task 1 adds).
- **`pool_mode`**: `compete` is a recorded dead end.

## Constraints — do not regress

- **Exact equivariance.** `state_max_err` must stay `0.00e+00`. Anything fed to
  the pool must be register-indexed.
- **Equivalence (GVN) collapse.** Currently 521× separation at len=2/1M. The
  constraint is on the **measured outcome**, not on any mechanism: a commutative
  operand swap must not move the essence. Feeding read-order to the pool
  *unconditionally* violates it — measured, not theoretical: `read_time` and
  `read_role` change under a commutative swap, so `pool_extra=event` doubles
  commutative distance and halves separation. `write_time` is the one pool
  channel known to be safe. A **content-conditioned** consumer (the op-cell) may
  legitimately use read-order and still collapse commutative pairs, because it
  can symmetrize per opcode — but that is a claim to be measured on these rows,
  not assumed (open task 1).
- **Magnitude-as-validity — a health check, not a live objective.** The writers
  corpus injects no invalid windows, so every row's validity target is 1 and vp
  consumes `F.normalize(vecs)` anyway; nothing we currently run should produce
  an invalid output. Essences should therefore sit on the unit sphere. A
  regression in `frac<0.9` is a symptom worth chasing, not a semantic failure.
  Real validity supervision is deferred to open task 3.
- **Decode-back / reconstruction is a NON-GOAL this phase.** Do not spend
  effort improving it. Do check it has not regressed — currently 91.5% opcode /
  95.4% immediate per instruction at len=8. (If anyone does pick it up later:
  `scripts/train_decoder.py` is stale and calls a pre-equivariant 2-arg
  `encode()`.)
- **Supervise only behaviorally-live signals.** Dead values have no target; the
  `rd==x0` immediate mask is the existing instance.

## Dead ends — do not re-run

- **Capacity and training budget** for vp at a fixed chunk length: 2× width and
  5× steps both leave it essentially unchanged.
- **`pool_mode=compete`.** No vp benefit, and with ordering enabled the essence
  becomes sensitive to the anonymous per-chunk register tags.
- **Supplying more ordering information TO THE POOL** — `pool_extra=order` and
  `pool_extra=event` both leave the shared-source deficit at ~83%; `event`
  additionally costs equivalence collapse. The pool cannot see `content`, so it
  must use any channel unconditionally; that is the reason, and it does **not**
  extend to content-conditioned consumers (open task 1). Two things this line
  does NOT cover: the op-cell, the untried `order`+`rank` cell (the `order` run
  used the buggy `stamp` encoding), and `read_time_mode=rank` (its one run had
  `pool_extra=write_time`, so the corrected read-order never reached the
  essence). See open task 1.
- **Probe methodology.** Several conclusions had to be withdrawn. Always
  calibrate a probe against a number known independently, and compare its score
  against the base rate — a probe scoring at the majority-class frequency has
  learned nothing, and one scoring below the model's own trained head is
  measuring itself.

## How to run

- Frozen T1 `runs/t1_writers_250k/encoder.pt`; corpus
  `configs/instr_alu_writers.json`.
- Model knobs on `scripts/train_t2_encoder.py`: `--op-cell-depth`,
  `--vp-head-depth`, `--n-pool-heads`, `--pool-mode`, `--pool-extra`,
  `--read-time-mode`. Current best: `op_cell=4 vp_head=1 n_pool_heads=16
  pma pool_extra=write_time`. Defaults reproduce pre-Phase-15 behaviour and old
  checkpoints load unchanged.
- One training cell, generation offloaded to odin:
  `LEN=2 STEPS=250000 MODES=pma NPOOL=16 PEXTRAS=write_time RTMODE=stamp bash scripts/run_t2.sh`
  — trains, then runs `scripts/eval.py` and the faithfulness probe.
- **Method: probe first, then train.** Diagnose on frozen checkpoints until the
  diagnosis yields a *falsifiable* hypothesis, and spend GPU time only to test
  that specific mechanism. Round 2 of this phase found in hours what Round 1's
  ten training runs did not.
- **Throughput: GPU utilisation is the primary signal**, with ms/step as a
  *floor* check against the same config's known-good figure. ms/step alone is
  too architecture-dependent to compare across configs, and it says nothing
  about how long the GPU spent waiting. A transient or periodic dip in the
  util reading is a sampling artifact, not a bottleneck — sustained low util is
  the thing to act on, and to act on *before* letting a long run ride.
- Probes (`probes/`, all read frozen checkpoints, none require training):
  `vp_slot_b_split.py` — operation vs operands, disjoint vs shared sources
    (open task 1); `vp_superposition.py` — one output vs two, and asserts slot
    values are unchanged by the other instruction; `vp_imm_share.py` — R-type vs
    I-type share and immediate bit positions (open task 2);
  `t2_reconstruct.py` and `t2_faithfulness.py` — regression checks only.

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
