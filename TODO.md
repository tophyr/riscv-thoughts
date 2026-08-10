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

1. **[CURRENT] vp = 0.0 at len=2.** The acceptance criterion below is **met** by
   `--op-cell-rank` (`runs/t2_len2_d256_s250000_dep4_vp1_pool16_pma_pewrite_time_rtrank_ocrank_writers`,
   250k): `s1 overlap` 100.000% exact against the `stamp` baseline's 21.2% at the
   same step count, aggregate vp 0.0184 against 0.0405. What blocks adoption is
   the equivalence-collapse constraint, not vp — see open task 1.

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

### 1. [CURRENT] Tag-invariance, regressed by `--op-cell-rank`

**The operand-binding deficit this task started as is solved.** Feeding the
op-cell each operand's canonical read position (`--op-cell-rank`, requires
`--read-time-mode rank`) takes the `s1 overlap` column from 21.2% to **100.000%**
exact 32-bit match and operand-position recovery from ~4–49% to ~98%, at equal
step count. The pool could never do this because it never sees `content` and so
must use read-order unconditionally; the op-cell sees the opcode and both
operands together and can symmetrize per opcode.

**What blocks adoption is tag-invariance.** Scored at `--gvn-chunk-len 2`,
`op_cell_rank` against the `stamp` baseline:

| row | baseline | op_cell_rank |
|---|---|---|
| rename (equivariance assertion) | 0.0002 | 0.0008 |
| tag-invariance | 0.0010 | **0.0255** |
| behavioral / commutative | 0.0027 | **0.0366** |
| separation | 462× | **35.5×** |

Equivariance is intact — `state_max_err` is `0.00e+00` and the rename row reads
~0. The regression is in the *learned* property: the essence became 25× more
sensitive to the anonymous per-chunk tags, and commutative collapse degraded
with it. Commutative distance is only 1.4× its own tag floor here against 2.7×
for the baseline, so treat tag-invariance as the cause and commutative collapse
as downstream of it.

**Leading fix — delete the random tags rather than penalize them.** Tags exist
for exactly one reason (RISCV_DESIGN, "Distinct anonymous leaf tags"): inputs
must be *distinguishable* while carrying no intrinsic identity, so `add r2,r3`
differs from `add r2,r2`. Canonical read rank now supplies that same property
deterministically — two distinct inputs get ranks 1 and 2, a repeated input gets
one rank twice — which makes the random tag a redundant input the model is no
longer pressured to cancel. Seeding the state from rank instead of `tag_proj`
attacks the cause structurally. Equivariance should survive by the existing
argument: rank is assigned by read ORDER, which permutes with a rename, so it is
not a per-register embedding `f(r)`. Verify that rather than assume it.

Fallback if that fails: an explicit tag-invariance penalty (encode a chunk under
two tag draws, penalize the essence difference).

Score any variant on `s1 overlap` (must hold at 100%), the tag-invariance and
commutative rows, and `state_max_err`.

### 2. I-type immediate encoding

The remaining len=2 vp error once task 1's binding fix is in: on an R-only
corpus `op_cell_rank` scores 100% exact, while aggregate vp on the 55/45 R/I
training corpus is 0.0184. Concentrated in bits 0–11 (the 12-bit immediate
field), and present even on single instructions with one output. It shrinks with training rather than being structural
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

- **`read_time_mode`: delete the option, keep `rank` only.** The no-regression
  comparison is satisfied — at the current-best len=2 config on one shared eval
  corpus, `rank` vs `stamp` is vp 0.0415/0.0405, in_slot 0.973/0.976, out_slot
  0.999 both, GVN rename 0.0014/0.0013, behavioral 0.0083/0.0084
  (`runs/t2_len2_d256_s250000_dep4_vp1_pool16_pma_pewrite_time_rtrank_rankfix_writers`).
  Remaining cost is reminting the frozen T1, which needs CLI plumbing first:
  `scripts/train_encoder.py` exposes none of the equivariant-core flags, so T1
  can only be trained at their defaults.
- **`pool_extra`**: `order` and `event` are both recorded dead ends. Once task 1
  resolves, keep only the survivors (`write_time`, plus whatever task 1 adds).
- **`pool_mode`**: `compete` is a recorded dead end.

## Constraints — do not regress

- **Exact equivariance.** `state_max_err` must stay `0.00e+00`. Anything fed to
  the pool must be register-indexed.
- **Equivalence (GVN) collapse.** `gvn_collapse` reports four rows, and they
  measure different kinds of thing. **rename** encodes the twin with its tags
  permuted by the same relabeling, so it is an *architecture assertion* — exact
  equivariance makes it ~0 for any model, and a nonzero value means the core is
  broken, not the training. **tag-invariance** (same chunk, independent tag draw)
  and **behavioral** (commutative operand swap, tags held fixed) are the learned
  properties, and the ones a change can regress. Baseline at
  `--gvn-chunk-len 2`: rename 0.0002, tag 0.0010, behavioral 0.0027, separation
  462×. The constraint is on the measured outcome: a commutative operand swap
  must not move the essence. Feeding read-order to the pool *unconditionally*
  violates it (`pool_extra=event`). A content-conditioned consumer may use
  read-order and still collapse commutative pairs — `op_cell_rank` does keep
  rename at ~0, but regresses tag-invariance 25×, which drags commutative with
  it (open task 1).

  Numbers recorded before this metric was corrected are NOT comparable: the old
  `enc()` drew independent tags for base and twin, so its "rename" row was
  reporting tag-invariance.
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
  `--read-time-mode`, `--op-cell-rank`. Current best on vp:
  `op_cell=4 vp_head=1 n_pool_heads=16 pma pool_extra=write_time
  read_time_mode=rank --op-cell-rank`; current best on the GVN rows is the same
  without `--op-cell-rank` (open task 1). Defaults reproduce pre-Phase-15
  behaviour and old checkpoints load unchanged; `--op-cell-rank` widens the
  op-cell input, so its checkpoints are not interchangeable with the others.
- One training cell, generation offloaded to odin:
  `LEN=2 STEPS=250000 MODES=pma NPOOL=16 PEXTRAS=write_time RTMODE=rank OCRANK=1 bash scripts/run_t2.sh`
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
  the host prep dominates. Fix is overlap — prefetch the next batch's prep on a
  worker thread, or build the dense grid on-device. Costs ~2× wall-clock on every
  len=2 cell until then; a run comparing against a recorded baseline should keep
  the pipeline as-is so the corpus mix the FIFO mux delivers is unchanged.
- Probes (`probes/`, all read frozen checkpoints, none require training):
  `vp_slot_b_split.py` — operation vs operands, disjoint vs shared sources,
    reported per-bit AND as exact 32-bit match (the acceptance criterion; per-bit
    flatters a broken subset — 81% per-bit on the overlap column was 21% exact);
    `opcell_rank_sufficiency.py` — whether slot B's state holds everything but
    the operand→position association, with a no-rank calibration arm;
    `vp_superposition.py` — one output vs two, and asserts slot values are
    unchanged by the other instruction; `vp_imm_share.py` — R-type vs I-type
    share and immediate bit positions (open task 2);
  `t2_reconstruct.py` and `t2_faithfulness.py` — regression checks only.

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
