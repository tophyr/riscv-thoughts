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

## Milestones

1. **[CURRENT] vp = 0.0 at len=2.** Now at **0.0198**
   (`runs/t2_len2_d256_s1000000_dep4_vp1_pool16_pma_ord1_writers`), from 0.3525
   before the write-order fix. Single instructions are already exactly 0.000, so
   len=2 is the smallest case that is not yet solved.
2. **vp = 0.0 at len=4, then len=8.** Now 0.244 and 0.461. Do not start these
   until len=2 is solved — the len=2 case is the same mechanisms with less
   noise, and every diagnosis so far has transferred.
3. **Beyond the happy path.** Everything above is writers-only (R/I-type ALU,
   valid, non-x0). Then: no-write instructions, PC/branches, memory, longer and
   looser chunks.
4. **T3.** Sequences of T2 blocks. Prerequisite: decide what a *composed*
   binding is — T3 is the first tier that must combine bindings rather than
   just emit them (WHAT_IS_A_THOUGHT.md, "The Thought and Its Interface").
   Settle that before building, not during.

## Open tasks

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

**No fix identified.** Three attempts — all supplying more ordering information
to the pool — left it completely unmoved. Their shared assumption (that the
position information merely needs to be made available somewhere) is the most
likely thing to be wrong. Prefer further diagnosis over a fourth attempt.

One untested idea, offered as a guess: the op-cell reads only the `value`
channels of its operand slots and is the one consumer never modified; giving it
each operand's canonical position at read time might let the slot's own value
carry the binding. It shares the failed assumption, and it would route
read-order into the value channels where it may cost equivalence collapse.

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

## Constraints — do not regress

- **Exact equivariance.** `state_max_err` must stay `0.00e+00`. Anything fed to
  the pool must be register-indexed.
- **Equivalence (GVN) collapse.** Currently 521× separation at len=2/1M.
  **Keep read-order information out of the invariant essence** — this is
  measured, not theoretical: a commutative operand swap changes `read_time` and
  `read_role`, so feeding them to the pool doubles commutative distance and
  halves separation. `write_time` is the one ordering channel that is safe.
- **Magnitude-as-validity.** Essences on the unit sphere; watch `frac<0.9`.
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
- **Supplying more ordering information to the pool** — `pool_extra=order`,
  `pool_extra=event`, and `read_time_mode=rank` all leave the shared-source
  deficit at ~83%. `event` additionally costs equivalence collapse.
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
- Judge throughput by **ms/step** in the log, not the GPU-utilisation graph,
  which swings 2.5× here while the step rate holds constant.
- Probes (`probes/`, all read frozen checkpoints, none require training):
  `vp_slot_b_split.py` — operation vs operands, disjoint vs shared sources
    (open task 1); `vp_superposition.py` — one output vs two, and asserts slot
    values are unchanged by the other instruction; `vp_imm_share.py` — R-type vs
    I-type share and immediate bit positions (open task 2);
  `t2_reconstruct.py` and `t2_faithfulness.py` — regression checks only.

## Long horizon

Loosened boundary supervision; eventually natural language
(WHAT_IS_A_THOUGHT.md, "Method vs. Cognition").
