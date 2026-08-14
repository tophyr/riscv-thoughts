# Roadmap

Forward-looking only. Findings and their evidence live in EXPERIMENT_LOG.md;
architecture and rationale in RISCV_DESIGN.md. Nothing here is a history.

## Mission

**T2 is done when its essence is a lossless behavioral description of its
instruction chunk AND two chunks that compute the same function land on the same
point.** Both are required; neither implies the other, and a model can score
perfectly on one while failing the other completely.

**Faithfulness — vp = 0.0.** vp measures whether the chunk's output register
values can be predicted from the essence given the input values. It is the
metric that matters because T2's essence is what a future T3 consumes: a next
tier can only recover what T2 preserved.

**Equivalence — computed equivalents collapse.** "Two token sequences mean the
same thing if and only if they compress to the same point" is the framework's
definition of meaning (WHAT_IS_A_THOUGHT.md), so an essence that separates
`ADDI r1,r1,256; ADDI r1,r1,-16` from `ADDI r1,r1,32; ADDI r1,r1,208` is not a
thought vector regardless of its vp. Rewrite equivalences — one operation
expressed another way — already collapse. Computed ones, where seeing the
equivalence requires arithmetic on the operands, do not collapse at all
(milestone 2). vp cannot deliver this: it is not a pairwise objective, so two
distinct points can both predict the same outputs correctly.

**The vp number in a training log is already a held-out number.** `datagen`
generates every batch fresh and the trainer never revisits one, so there is no
train/test split to construct and no memorization to discount — quote the
training log directly. (`scripts/eval.py` exists to score *binding* and the GVN
rows on a fixed corpus, not to de-bias vp.)

## Milestones

1. **[CURRENT] vp = 0.0 at len=2 — not reached.** Current best:
   `--d-out 512 --pool-extra write_time --read-time-mode rank --seed-mode rank`
   at 1M
   (`runs/t2_len2_d512_s1000000_dep4_vp1_pool16_pma_pewrite_time_rtrank_seedrank_behavframe_writers`,
   replicated at `…_behavframe_s7_writers`), on the criterion — exact 32-bit
   match of the predicted output value — **99.744%** and **99.753%**. The two
   seeds put the run-to-run noise floor at 1M steps at **±0.01 points** overall
   and 0.1 points on any cell above 1k triples.

   **Numbers from before the vp target was moved into the behavioral canonical
   frame are not comparable to these** (EXPERIMENT_LOG finding 14). A model is
   scored on the target it trained on and the two differ on 12.5% of rows; on
   the rows where they agree, the change took 99.7383% → 99.8347%.

   The gap is one shape, and task 1 is the work of closing it: an immediate
   survives in the essence when its instruction is read in isolation, and
   degrades when that instruction's operand is the other instruction's result.
   Every population carrying no immediate is at or above 99.88%, and composition
   without an immediate is at 99.957%.

   **Score this with `probes/vp_budget.py`**, which crosses the three properties
   of an output slot that decide whether it is predicted correctly:

   - **sources** — `cross-instr` when the chunk's two instructions read a common
     register, `distinct` otherwise.
   - **immediate** — whether any instruction in the chunk carries one. R-type
     ALU operations take two registers; I-type take a register and a 12-bit
     literal.
   - **depth** — the slot's dataflow depth: 1 when the instruction that wrote it
     read only chunk inputs, 2 when one of its operands is another instruction's
     result.

   The unit is the (row, output slot, anchor) triple that vp supervises, and
   **err mass** is the cell's share of all failing triples. A rate and a mass
   are different facts: a rare cell can be terrible and carry little of the
   error, a common one excellent and carry most of it.

   | sources | immediate | depth | triples | exact | err mass |
   |---|---|---|---|---|---|
   | distinct | yes | **2** | 17.4k | **93.01%** | **39.5%** |
   | distinct | yes | 1 | 767k | 99.909% | 22.5% |
   | distinct | no | 1 | 290k | 99.879% | 11.4% |
   | **cross-instr** | yes | 1 | 42.2k | 99.228% | 10.6% |
   | **cross-instr** | yes | **2** | 888 | **81.08%** | 5.4% |
   | cross-instr | no | 2 | 1.5k | 98.24% | 0.8% |
   | cross-instr | no | 1 | 39.2k | 99.980% | 0.3% |
   | distinct | no | 2 | 9.4k | 99.957% | 0.1% |

   Worst individual opcode cells at depth 2: SLTIU 78.8%, SLLI 80.3%, ADDI
   79.5%, ~1k triples each.

   Every probe's numbers describe the corpus it ran on, and the spread is wide
   enough to have hidden a bug for a phase: an R-only corpus excludes the
   `SLTIU rd,rs,-k` shape entirely, and reading 8.8× lower error there than on
   the mixed corpus was taken as a property of the model (EXPERIMENT_LOG finding
   19). Quote the mixed-corpus figure for statements about the model.

2. **Computed equivalents collapse at len=2.** Two chunks that compute the same
   function must land on the same point. They currently land as far apart as
   two chunks computing different functions: on the best model, additive chains
   with the same total offset sit **0.4634** apart against **0.3901** for chains
   whose totals differ by 1 — separation **0.8x** — while a register rename sits
   at 0.0057. XOR chains are 0.7x. Measured directly, with no fitted
   readout, by `probes/additive_fusion.py`.

   **Settle this before chunks get longer.** Every added instruction multiplies
   the equivalences a chunk participates in, so a len=4 model would be learning
   the same wrong geometry at four times the scale. It comes after milestone 1
   because vp at len=2 is nearly closed and its diagnosis is in hand.

   No loss, data augmentation or metric in the project has ever covered this
   class: the equivalence manifest contains only single-instruction rewrites.
   Task 2 has the evidence and the decisions to make before building.

3. **vp = 0.0 at len=4, then len=8.** Was 0.244 and 0.461 pre-rank.
   Note `rank` matches the vp target's canonical frame on only 83.7% of len=4 and
   52.9% of len=8 chunks (the residual is dead code, which no causal read-time
   counter can know), so the len=2 mechanism may not transfer intact.
4. **Beyond the happy path.** Everything above is writers-only (R/I-type ALU,
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
5. **T3.** Sequences of T2 blocks. Prerequisite: decide what a *composed*
   binding is — T3 is the first tier that must combine bindings rather than
   just emit them (WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface").
   Settle that before building, not during.

## Open tasks

Task 1 is the current focus. Task 2 is required for the MVP and is sequenced
after it, before chunks get longer. The rest are queued, not parallel work.

### 1. An immediate does not survive composition

**The single open cause of the len=2 vp residual.** The error lives in the cells
where an immediate meets a **composed operand** — the writing instruction taking
its operand from the other instruction's result rather than from a chunk input
(milestone 1's table):

| condition paired with the immediate | exact | err mass |
|---|---|---|
| composed operand (dataflow depth 2) | 93.01% | 39.5% |
| composed operand AND a shared source | 81.08% | 5.4% |

Each condition **alone** is solved: an immediate alone 99.909%, a shared source
alone 99.980%, composition alone 99.957%. What fails is the combination, and the
common factor is the immediate — the highest-entropy field an instruction
carries, and the only one that has to survive being folded into another
instruction's result.

**The shared-source half of this was a broken target, not an encoder limit**, and
is closed: canonical positions were assigned over live syntactic inputs while the
vp head was fed as though they covered only the behavioral ones, so two chunks
with a bit-identical encoder input carried different targets. That cell went
97.11% → 99.23%. See EXPERIMENT_LOG finding 14 for the mechanism and
`probes/vp_drop_control.py` for the construction that isolates it — a
manufactured collision, because no split of the corpus can see this class of
defect.

**Ruled out** (EXPERIMENT_LOG phase-15 finding 17, each against a ±0.75-point
run-to-run noise floor):

- **the vp head** — no refit, fine-tune, or deeper head beats the trained one;
- **the two-step arithmetic chain** — handing a readout the true intermediate
  value changes nothing;
- **the pool** — the gap is already present in the register slot's own state;
- **gradient share** — `--vp-depth-weight` at 8 and 32 leaves the hard cells
  flat and degrades the rest;
- **the operand scale mismatch** — `--value-norm` puts leaf and composed
  operands on one shell and scores slightly worse, though the 0.08–0.14×
  excitation gap it targets is real;
- **content crowded out of the op-cell's first layer** — `--content-skip` gives
  the content its own path into the slot and lands inside the noise;
- **capacity** — `d_out` 256 → 512 buys a fixed amount, 512 → 1024 buys nothing
  at either budget and leaves the error cells unchanged to three decimals.

**Ruled out as of the behavioral-frame model** (`probes/vp_frame.py`): the
remaining error in this cell is not a description mismatch. Restricting it to
rows where T1's emitted wiring matches the tokens reads 93.43%, and restricting
it to rows whose target is identical under either canonical frame reads 92.63% —
both at the cell's own 93.01%. So neither T1's binding prediction nor the frame
is what is failing here, unlike the shared-source cell.

**Untested levers that remain:**

- **Supervising the intermediate value**, which the SSA oracle computes and the
  RVT batch does not ship. Lower prior than it was — a readout can already
  recover the intermediate from the essence.
- **The op-cell's own composition step.** A depth-2 slot is
  `op_cell(content, op_cell(content', a, b), c)`, and every refuted lever above
  targeted what the slot is *made of* rather than the second application itself.
  Nothing yet measures whether the immediate is present in the intermediate and
  lost by the second application, or absent from the intermediate to begin
  with — which is the first thing to localise, on frozen checkpoints.

### 2. Collapse computed equivalences (milestone 2 — required for MVP)

Two chunks computing the same function must land on the same point. They do not,
and the failure is total rather than partial. Distances between essences, where
each chunk is two instructions of one opcode writing one register — so the
essence describes only the composite, the intermediate having been overwritten:

- **fusion** — the same composite reached by a different split of the two
  immediates (`ADDI r1,r1,256; ADDI r1,r1,-16` against
  `ADDI r1,r1,32; ADDI r1,r1,208`). Must be ~0.
- **different** — a different composite. The denominator; without it a small
  fusion distance would only mean the encoder collapses everything.
- **separation** — different / fusion. Must be large.

| family | fusion | different | separation |
|---|---|---|---|
| `ADDI/ADDI`, same total offset | 0.4634 | 0.3901 | **0.8x** |
| `XORI/XORI`, same composite mask | 0.5722 | 0.3727 | **0.7x** |

The same chunk with its registers relabelled sits at 0.0057, so the encoder is
intact and the measurement is sound.
`probes/additive_fusion.py` builds the chunks directly and takes a distance —
no fitted readout — and asserts through the emulator that each `fusion` pair is
behaviorally identical and each `different` pair is not. Extend it with a new
family by adding a `combine`/`resplit`/`perturb` triple to `FAMILIES`.

**Nothing in the project covers this class.** Equivalences are enumerated in
one place — the manifest in `datagen/generate.py` — and all 15 of its classes
are single-instruction **rewrites**, one operation expressed another way
(`template_instr_counts=[1]`). Everything built on that manifest inherits the
limit: the `--inject-equiv` data augmentation, `eval.py`'s `behavioral` GVN row,
and a pairwise equivalence loss that the trainer carried until commit `7797c81`
and which is recoverable from git history. A **fold** — where seeing the
equivalence requires arithmetic on the operands — appears in none of them.
Accordingly, every checkpoint spanning the current architecture fails it, the
oldest worst at 0.5x, with equivalent chunks further apart than inequivalent
ones.

**vp cannot deliver it.** Two chunks computing the same function have the same
vp target, and two distinct points can both predict it correctly. Nothing in the
loss is pairwise — that was a deliberate design choice, and undoing it is the
decision to make before building.

Open questions, in order:

- **How wide is the class?** Both families tested fail. Check whether any
  computed equivalence collapses, and whether the rewrite classes that DO
  collapse (commutative swap, 0.0309 at 42x separation) do so because they are
  trained on or because the architecture makes them free.
- **What is the objective?** A pairwise term over generated equivalent pairs is
  the obvious candidate. `--inject-equiv` is a recorded failure at buying
  collapse with data alone — it adds correlated rows without
  adding collapse pressure, since nothing ties an injected variant's essence to
  its base. The manifest would also need multi-instruction classes, which
  requires generating folds rather than rewrites.
- **What does it cost?** Every collapse gain so far has traded against
  discriminativeness (findings 11, 16), and a term that pulls equivalents
  together can pull inequivalents together too. `behav-diff` — pairs the
  transformation did NOT leave equivalent — is the guard, and must stay large.

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

### 5. Flag cleanup

**Project policy: options do not survive on their own.** Once a better way to do
something is settled, the worse way gets deleted rather than kept behind a flag.
Ongoing branching/versioning complexity is a cost we do not pay if we can pay a
one-time reshaping cost instead. Breaking changes — including reminting the
frozen T1 — are acceptable **on the one condition that new and old can be
compared and shown not to have regressed.**

**Refuted this session, and due for deletion** — kept for now only so the
evidence can be re-checked against the code that produced it, since each was
tested once. Delete them deliberately rather than letting them become vestigial:
`--vp-depth-weight` (and `slot_dataflow_depth`, which exists only to serve it),
`--value-norm`, `--content-skip`. Each is refuted at len=2/250k against a
measured noise floor; see task 1 and EXPERIMENT_LOG finding 17.

Standing candidates:

- **`seed_mode`: delete the option, keep `rank` only**, and **delete
  `op_cell_rank` outright** — it feeds the op-cell each operand's canonical read
  position as extra scalars, which `seed_mode=rank` already supplies through the
  operand values it reads, and enabling both scores worse than seeding alone.
- **`read_time_mode`: delete the option, keep `rank` only.** The no-regression
  comparison is satisfied — at the current-best len=2 config on one shared eval
  corpus, `rank` vs `stamp` is vp 0.0415/0.0405, in_slot 0.973/0.976, out_slot
  0.999 both, GVN rename 0.0014/0.0013, behavioral 0.0083/0.0084
  (`runs/t2_len2_d256_s250000_dep4_vp1_pool16_pma_pewrite_time_rtrank_rankfix_writers`).
  Remaining cost is reminting the frozen T1, which needs CLI plumbing first:
  `scripts/train_encoder.py` exposes none of the equivariant-core flags, so T1
  can only be trained at their defaults.
- **`pool_extra`**: `order` and `event` are both recorded dead ends. Keep
  `write_time`, delete the rest.
- **`pool_mode`**: `compete` is a recorded dead end.

### 6. Overlap the trainer's host-side work with the GPU step

The bottleneck is diagnosed and measured — see the "Throughput" bullet under
"How to run" — and what is missing is the fix: prefetch the next batch's
preparation on a worker thread, or build the dense grid on-device. Worth ~2×
wall-clock on every len=2 run.

## Constraints — do not regress

- **Exact equivariance.** `state_max_err` must stay `0.00e+00`. Anything fed to
  the pool must be register-indexed.
- **Equivalence (GVN) collapse.** The requirement: two instruction sequences that
  compute the same function must land on the same point. `scripts/eval.py` reports
  it as four distances between essences, and they are not the same kind of
  quantity.

  **rename** compares a chunk against the same chunk with its registers
  relabeled, encoding the relabeled copy with its anonymous tags permuted to
  match. Equivariance is exact by construction, so this reads ~0 for any model at
  all; it is an assertion that the core is intact, not a measure of training
  quality. **tag-invariance** compares a chunk against itself under a different
  draw of the anonymous per-chunk tags — a property the model must learn, and
  therefore one a change can regress. It is vacuous when the encoder seeds from
  canonical read rank rather than random tags, since then there are no tags.
  **behavioral** compares a chunk against a variant with a commutative
  instruction's operands swapped. **distinct** is the separation denominator: the
  distance between unrelated chunks.

  `scripts/eval.py` reports the behavioral distance **split** by whether the
  swap actually preserved the chunk's function, and the split is the thing to
  read. Swapping a commutative op's operands changes the chunk's first-read
  order, and if a later instruction reads one of the swapped registers the
  canonical input positions shift and the two chunks compute genuinely different
  functions. About 2% of generated pairs are non-equivalent in this way; they sit
  at the `distinct` scale and can inflate the pooled mean several-fold. Judge
  collapse on the equivalent subset (`behavioral`); the non-equivalent subset
  (`behav-diff`) should be LARGE, and a model that collapses it is failing to
  distinguish different functions. `behav-mixed` pools both and exists only for
  continuity with older records.

  Measured at `--gvn-chunk-len 2`, on genuinely-equivalent pairs. Seeding from
  canonical read rank: 0.0110, separation 119×, and non-equivalent pairs held
  apart at 0.88. Seeding from random tags with no read-order in the value
  channels: 0.0021, separation 579× — tighter, but it collapses the
  non-equivalent pairs to 0.0003, i.e. it cannot tell those functions apart at
  all. For scale, the architecture was validated in Phase 14 at 0.0238 and 28×.

  The `rename` row encodes a chunk's relabeled copy with the SAME anonymous
  per-chunk tags, permuted to match. Drawing independent tags for the two makes
  that row report tag-invariance instead of equivariance, and inflates it —
  recorded numbers only compare against each other when they share this.
- **Magnitude-as-validity — a health check, not a live objective.** The writers
  corpus injects no invalid windows, so every row's validity target is 1 and vp
  consumes `F.normalize(vecs)` anyway; nothing we currently run should produce
  an invalid output. Essences should therefore sit on the unit sphere. A
  regression in `frac<0.9` is a symptom worth chasing, not a semantic failure.
  Real validity supervision is deferred to open task 4.
- **Decode-back / reconstruction is a NON-GOAL this phase.** Do not spend
  effort improving it. Do check it has not regressed — currently 91.5% opcode /
  95.4% immediate per instruction at len=8. (If anyone does pick it up later:
  `scripts/train_decoder.py` is stale and calls a pre-equivariant 2-arg
  `encode()`.)
- **Supervise only behaviorally-live signals.** Dead values have no target; the
  `rd==x0` immediate mask is the existing instance.

## Dead ends — do not re-run

- **Capacity.** Scaling the model's dimensions and steps together leaves
  aggregate vp essentially unchanged at a fixed chunk length. `d_out` 256 → 512
  is the one exception and it does not repeat: 512 → 1024 gives +0.9 points of
  depth-2 exact match at 250k and −0.7 at 1M, both inside the noise floor, with
  GVN separation slipping 42× → 36×, and it leaves the error cells of task 1
  unchanged to three decimal places. `d_out 512` is adopted; the remaining error
  is not capacity-shaped.
- **Weighting vp by dataflow depth** (`--vp-depth-weight`). Raising composed
  slots' share of the vp gradient from 2.5% to 18% (W=8) and 46% (W=32) leaves
  depth-2 flat at both settings and degrades depth-1 monotonically (overall bit
  accuracy 99.834% → 99.751% → 98.839%). Composed slots are not gradient-starved.
- **A content bypass into the slot** (`--content-skip`, a direct linear path
  from an instruction's content to the slot it writes, so the content does not
  share the op-cell's first layer with decoding the operands). Inside the noise
  floor. The composed instruction's content is not blocked on its way into the
  slot — it is fused with the operand once there.
- **Normalizing values into the register slot** (`--value-norm`, RMSNorm on
  every value as it enters a slot, so a seeded leaf and a computed op-cell
  output share one scale). The scale gap is real and measured — a composed
  operand excites the op-cell's operand weights at 0.08–0.14× a leaf's energy —
  but closing it leaves depth-2 slightly worse (I-type 95.49% → 94.79%).
- **`pool_mode=compete`.** No vp benefit, and with ordering enabled the essence
  becomes sensitive to the anonymous per-chunk register tags.
- **Supplying more ordering information TO THE POOL.** Every `pool_extra` cell is
  spent: `order` and `event` leave the shared-source deficit at ~83% and cost
  equivalence collapse, and `order`+`rank` is the worst on vp (0.0434 vs the
  `write_time` baseline's 0.0405) while paying that cost (behavioral 0.0084 →
  0.0115, separation 310× → 224×). The pool cannot see `content`, so it must use
  any channel unconditionally; that is the reason, and it does **not** extend to
  content-conditioned consumers. The op-cell is the one consumer this line does
  not cover — see open task 1.
- **Injecting commutative equivalents (`--inject-equiv`) to train the
  invariance.** Makes it worse, not better: on genuinely-equivalent pairs 0.0110
  → 0.0235 and separation 119× → 54×, with vp 0.0082 → 0.0252. Nothing in the
  loss ties an injected variant's essence to its base (there is no pairwise term,
  by design), so it adds correlated rows without adding collapse pressure while
  reducing effective diversity. Partially confounded — the injected rows displace
  others, so that run saw 545 instructions per batch against 768 — but the
  direction is unambiguous.
- **More training as a fix for commutative collapse.** 250k → 1M moves it
  0.0426 → 0.0373, while an encoder seeded from random tags moves 0.0027 →
  0.0023 over the same 4×; the gap is untouched.
- **Treating tag-sensitivity as the cause of the commutative-collapse
  regression.** Removing the random tags entirely (`--seed-mode rank`) takes
  tag-invariance to the architectural floor and leaves collapse unchanged (0.0426
  against `op_cell_rank`'s 0.0366 and a 0.0027 baseline). Both are downstream of
  read-order reaching the essence.
- **Treating the behavioral row as contaminated.** Commute pairs share the vp
  target on 100% of len=1 and 97% of len=2 chunks, so the row measures what it
  claims and the regression is not an artifact of it.
- **Probe methodology.** Several conclusions had to be withdrawn. Always
  calibrate a probe against a number known independently, and compare its score
  against the base rate — a probe scoring at the majority-class frequency has
  learned nothing, and one scoring below the model's own trained head is
  measuring itself.

## Known defects


- **`_behaviorally_equal` (`compressor/eval.py`) is over-strict.** It compares the
  whole 32-register canonical file, including input registers that `_canonical_state`
  merely relocated, so it reports only 9.2% of genuine commutative equivalences as
  equivalent at len=1 (the vp target is identical for 100% of them). vp supervises
  the written slots only, so the comparison should be `out_regs` at `out_slot_regs`.
  It gates the `distinct` pool, where unrelated chunks are almost never equivalent,
  so the effect there is small — but the predicate is wrong wherever else it is used.
- **The test suite does not load any real checkpoint**, so it cannot catch a
  state_dict incompatibility: adding an unconditional `rank_emb` passed all 283
  tests and failed at step 0 of the next run against the frozen T1.

## How to run

- Frozen T1 `runs/t1_writers_250k/encoder.pt`; corpus
  `configs/instr_alu_writers.json`.
- Model knobs on `scripts/train_t2_encoder.py`: `--op-cell-depth`,
  `--vp-head-depth`, `--n-pool-heads`, `--pool-mode`, `--pool-extra`,
  `--read-time-mode`, `--op-cell-rank`, `--value-norm`, `--content-skip`, and
  the loss knob `--vp-depth-weight`. `scripts/run_t2.sh` exposes each as an env
  var (`DOUT`, `VALNORM`, `DEPTHW`, …) and names the run directory after them.
  Current best on vp:
  `d_out=512 op_cell=4 vp_head=1 n_pool_heads=16 pma pool_extra=write_time
  read_time_mode=rank --seed-mode rank`, i.e. the pool reads each slot's write
  step, read positions are counted as ranks over registers read before written,
  and register slots are seeded from those positions rather than random tags.
  Defaults reproduce pre-Phase-15 behaviour and old checkpoints load unchanged.
  `--op-cell-rank` widens the op-cell input and `--seed-mode rank` swaps
  `tag_proj` for `rank_emb`, so checkpoints are only interchangeable within a
  setting.
- **Never edit `scripts/run_t2.sh` while a copy of it is running.** Bash reads a
  script incrementally, so an edit shifts the file offsets under the running
  shell. Doing this mid-sweep produced a spurious `syntax error near 'fi'` and
  re-ran the eval-corpus block's redirect outside its `if`, truncating
  `/tmp/t2_eval_2.rvt` to 0 bytes — which surfaces as `binding: … rows=0` in
  every eval afterwards, GVN rows unaffected because they build their own pairs.
  Wait for the sweep, or edit a copy.
- One training cell, generation offloaded to odin:
  `LEN=2 STEPS=250000 MODES=pma NPOOL=16 PEXTRAS=write_time RTMODE=rank SEEDMODE=rank bash scripts/run_t2.sh`
- **A replicate needs `GEN_SEED`.** Both datagen pools default to `--seed 0`, so
  re-running a cell otherwise redraws the same instruction stream and measures
  only initialization variance. `GEN_SEED=7 … TAG=s7 bash scripts/run_t2.sh`
  draws a different corpus from the same distribution. It is unrelated to
  `ANCHOR_SEED`, which must stay equal to the trainer's `--anchor-seed`
  (INVARIANT 1). At len=2/1M the floor is ±0.01 points of overall exact match.
- **A run's `t2.pt` appears long before the run ends** — the trainer checkpoints
  periodically. Wait on the trainer process exiting, or on `run_t2.sh` logging
  `final:`; scoring the file as soon as it exists reads a partial model.
  — trains, then runs `scripts/eval.py` and the faithfulness probe. `TAG=<name>`
  suffixes the run dir so a re-run of a recorded cell does not overwrite it.
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

  **len=2 cells run at ~50% util and the bottleneck is the trainer's host-side
  work, not generation.** Measured during a len=2 cell: local trainer 633% CPU
  (6.3 cores) on per-step prep, odin's 98 workers at ~9.8% each (backpressured,
  not saturated), 64 MB/s over the wire. The GPU step is cheap at d256/len=2, so
  the host prep dominates. Adding remote workers does not help — re-measured at
  `REMOTE_GEN=120`, where the GPU sat at ~37% while odin's load average was ~11
  of 128 cores and the link carried 20 MB/s of a ~118 MB/s budget. A cheaper GPU
  step lowers utilisation further, so utilisation falling after an architecture
  change is not by itself evidence of a regression — and the converse holds:
  `--d-out 512` runs at ~60% against d256's ~40% because its GPU step is more
  expensive, not because anything about the pipeline improved. Fix is overlap (task 6);
  costs ~2× wall-clock on every len=2 cell until then. A run comparing against a
  recorded baseline should keep the pipeline as-is so the corpus mix the FIFO mux
  delivers is unchanged.
- Probes (`probes/`, all read frozen checkpoints, none require training):
  `vp_slot_b_split.py` — operation vs operands, disjoint vs shared sources,
    reported per-bit AND as exact 32-bit match (the acceptance criterion; per-bit
    flatters a broken subset — 81% per-bit on the overlap column was 21% exact);
    `opcell_rank_sufficiency.py` — whether slot B's state holds everything but
    the operand→position association, with a no-rank calibration arm;
    `vp_superposition.py` — one output vs two, and asserts slot values are
    unchanged by the other instruction; `vp_imm_share.py` — R-type vs I-type
    share, immediate bit positions, and error crossed with dataflow depth — the
    mixed-corpus counterpart to `vp_slot_b_split`, which is R-only and so cannot
    see an immediate failure at all; `imm_ceiling.py` — recovers one quantity at
    each stage (T1 essence → T2's written slot → the pool's view of that slot →
    T2 essence) to localize WHERE something is lost; `pool_bottleneck.py` —
    refits a bottleneck for a single task and sweeps its width, to size a
    suspected capacity limit before changing the architecture;
  `vp_budget.py` — **the one to reach for first**: the whole error budget from
    the model's own head, crossed over sources x immediate x depth plus opcode
    and bit position, each bucket reported as a rate AND as its share of total
    error. Takes a corpus config, so the same splits can be scored on the
    population a past claim was measured on;
  `vp_frame.py` — whether the essence and the vp target describe the SAME chunk.
    Splits vp by how T1's emitted wiring differs from the tokens' (a dropped
    read, an added one, a swap, a repeat T1 cannot express) and by whether the
    behavioral and syntactic canonical frames coincide. Reach for it when a cell
    resists every capacity lever, which is what a mismatch looks like;
  `vp_drop_control.py` — the same question answered causally: the second
    instruction held identical while only the first varies, so a difference
    cannot be explained by the chunks being harder. Use this shape whenever a
    defect can make two different chunks produce the same encoder input — a
    split of the corpus cannot see one, because the cost falls on the rows that
    do NOT carry the defect;
  `additive_fusion.py` — whether two chunks computing the same function land on
    the same point. Builds the chunks directly and takes a distance, with no
    fitted readout, and asserts through the emulator that the pairs really are
    equivalent;
  `depth_composition.py` — whether a vp deficit is in the essence or the head,
    by refitting and fine-tuning heads on frozen essences; `depth_stage.py` —
    the same quantity recovered at each encoder stage (slot → pool view →
    essence), split by dataflow depth; `depth_intermediate.py` — removes the
    "composition is a harder function" confound by handing the readout the true
    intermediate value;
  `t2_reconstruct.py` and `t2_faithfulness.py` — regression checks only.

  **Scoring GVN: pass `--gvn-chunk-len` matching the model's chunk length.**
  `gvn_collapse` builds its own pairs and defaults to 4, so a len=2 model scored
  without it is measured out of distribution — worth an order of magnitude
  (equivalent-pair distance 0.0712 at chunk_len=4 against 0.0287 in
  distribution, same checkpoint). `scripts/run_t2.sh` passes it.

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
