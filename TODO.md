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

1. **vp = 0.0 at len=2 — essentially reached.** `--seed-mode rank` at 1M
   (`runs/t2_len2_d256_s1000000_dep4_vp1_pool16_pma_pewrite_time_rtrank_seedrank_phv_1m_writers`):
   aggregate vp **0.0021**, and per-bit accuracy of the predicted output value
   **99.939%** on two-instruction chunks, **99.999%** on single instructions.

   Two probes score the two ways a chunk's behavior can be lost, and both are
   needed because each is blind to the other's failure:

   - `probes/vp_slot_b_split.py` scores OPERAND BINDING — exact 32-bit match of
     the predicted output, broken out by whether the chunk's two instructions
     read a common source register. It reaches 99.989% on the first written
     register, 99.948% on the second, and **99.765%** on the shared-source
     subset. Its corpus is R-type only (`configs/instr_r_only.json`), so it
     carries no immediates and cannot see an immediate-encoding failure at all.
   - `probes/vp_imm_share.py` scores the mixed training corpus and splits the
     error by operand type, which is what exposes immediates.

   The criterion is literally 100% on every binding column, so this is short by
   ~2 row-anchor pairs of 6915 — small, but a real deficit rather than sampling
   noise. `--op-cell-rank` reached exactly 100.000% on the shared-source column
   at 250k, which suggested the last 0.2% was available for the taking — but
   that was measured when all pool heads shared one value projection, and
   per-head projections changed how much of a slot the essence can carry. Whether
   it still helps, or is now redundant, has not been re-measured.

2. **[CURRENT] vp = 0.0 at len=4, then len=8.** Was 0.244 and 0.461 pre-rank.
   Note `rank` matches the vp target's canonical frame on only 83.7% of len=4 and
   52.9% of len=8 chunks (the residual is dead code, which no causal read-time
   counter can know), so the len=2 mechanism may not transfer intact.
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

Task 1 is the current focus; the rest are queued, not parallel work.

### 1. Dataflow composition — chunks whose second instruction reads the first's result

The remaining len=2 error is concentrated on output slots at dataflow depth 2,
where an instruction's operand is another instruction's result rather than a
chunk input. Depth-1 slots reach 99.95–99.97% per-bit while depth-2 slots reach
98.6% (I-type, n=397) and 99.3% (R-type, n=835) — a 30–60× higher error rate on
about 2.4% of slots. R-type and I-type are now hit about equally (60% / 40% of
total error), so this is composition, not operand encoding.

Depth-2 is where the op-cell must consume its own output: the value channels
written by instruction 1 are read back as an operand of instruction 2, so the
cell's output geometry has to be legible to its own input. That is a different
demand from the depth-1 case, where every operand is a freshly seeded leaf.

Untested levers: more residual blocks on the op-cell (`--op-cell-depth`); a
wider value channel (`--d-out`); and supervising the intermediate value, which
the SSA oracle already computes but the RVT batch does not ship.

### 2. Mint a production T1

Every T1 in use is a diagnostic trained on a clean corpus (`--inject-invalid 0`),
so it has no validity task and cannot reject invalid input. Needed before
anything leaves the happy path. Fold in the `rd==x0` immediate mask.

### 3. No-write instructions are invisible to the essence

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

### 4. Flag cleanup

**Project policy: options do not survive on their own.** Once a better way to do
something is settled, the worse way gets deleted rather than kept behind a flag.
Ongoing branching/versioning complexity is a cost we do not pay if we can pay a
one-time reshaping cost instead. Breaking changes — including reminting the
frozen T1 — are acceptable **on the one condition that new and old can be
compared and shown not to have regressed.**

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

### 5. Overlap the trainer's host-side work with the GPU step

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

  Numbers predating the correction to this metric are not comparable with these:
  it previously drew independent tags for a chunk and its relabeled copy, so its
  "rename" distance was reporting tag-invariance instead of equivariance.
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
  `--read-time-mode`, `--op-cell-rank`. Current best on vp:
  `op_cell=4 vp_head=1 n_pool_heads=16 pma pool_extra=write_time
  read_time_mode=rank --seed-mode rank`, i.e. the pool reads each slot's write
  step, read positions are counted as ranks over registers read before written,
  and register slots are seeded from those positions rather than random tags.
  Defaults reproduce pre-Phase-15 behaviour and old checkpoints load unchanged.
  `--op-cell-rank` widens the op-cell input and `--seed-mode rank` swaps
  `tag_proj` for `rank_emb`, so checkpoints are only interchangeable within a
  setting.
- One training cell, generation offloaded to odin:
  `LEN=2 STEPS=250000 MODES=pma NPOOL=16 PEXTRAS=write_time RTMODE=rank SEEDMODE=rank bash scripts/run_t2.sh`
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
  change is not by itself evidence of a regression. Fix is overlap (task 5);
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
  `t2_reconstruct.py` and `t2_faithfulness.py` — regression checks only.

  **Scoring GVN: pass `--gvn-chunk-len` matching the model's chunk length.**
  `gvn_collapse` builds its own pairs and defaults to 4, so a len=2 model scored
  without it is measured out of distribution — worth an order of magnitude
  (equivalent-pair distance 0.0712 at chunk_len=4 against 0.0287 in
  distribution, same checkpoint). `scripts/run_t2.sh` passes it.

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
