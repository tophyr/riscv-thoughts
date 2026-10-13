---
name: datagen
description: >
  Reference for the RVT data-generation pipeline — the datagen/ package (the
  `python -m datagen` generator tool) and the multinode wrappers
  (multinode_gen.sh, gen_corpus.sh, multinode_remote.sh). Load this BEFORE
  generating a corpus, editing datagen/ or any pipeline script, changing the RVT
  format, debugging worker/throughput/OOM issues, or wiring gen→train. Copy its
  canonical commands rather than improvising one from memory.
---

# RVT data-generation pipeline

The pipeline turns a random RV32I instruction stream into binary training
batches consumed over stdin by the trainers. One generator, one format.

```
python -m datagen     fork one worker per core → each: instrs → chunks → twins/aux
                      → packed Batch; interleave all workers (+ external RVT
                      inputs) → stdout (RVT bytes). --count N bounds the run;
                      -v reports throughput.
train_encoder.py      T1 trainer (reads RVT on stdin)
train_t2_encoder.py   T2 trainer (reads RVT on stdin, needs a frozen T1)
```

The generator and trainers are UNIX filters over the **RVT binary stream**;
they talk only through that stream. `print()` inside the generator is redirected
to stderr (`binary_stdout()` in `datagen/__main__.py:59`) so stdout stays
pure data. (Bounding the count and measuring throughput used to be separate
`batch_slice` / `bench_throughput` filter stages; both are now `datagen`'s own
`--count` and `-v` — there is no on-disk-corpus tooling.)

## `datagen` is parallel by default — there is no single-worker mode.

Generation+analysis (SSA, anchor execution, twins, value-prediction precompute)
is CPU-bound Python and one worker pins one core. The GPU is far faster than one
core can supply, so a single-core generator silently starves the trainer and
idles the GPU — historically the project's standard way to waste GPU-hours.

`datagen` therefore spawns **one worker per CPU core by default** (override
with `--gen-count N`), forked as an INTERNAL function with no CLI of its own:
you *cannot* run a lone worker by hand, by design. Generation is **infinite** by
default — bound the run with `--count N` (emit N batches then stop & reap), or
let the trainer close the pipe. Still confirm the GPU is the bottleneck: run
with `-v` to print windowed throughput (batch/s, MB/s) — if that rate sits above
the trainer's step rate the GPU is the bottleneck; if it tracks the trainer you
are core-bound (add cores / mux in more).

To mux an external RVT stream alongside the live workers, pass it as a
positional input — a file, a process substitution, or a piped/redirected stdin
(all auto-detected): `python -m datagen 8:<(nc host port) --rule …`, or simply
`cat corpus.rvt | python -m datagen --rule …`. A tty or `/dev/null` on stdin is
skipped, so bare generation (and a generator run with `</dev/null` in a script)
is unaffected. (To replay a pure corpus with no generation: `cat corpus.rvt |
python -m datagen --gen-count 0`.)

## Canonical commands — this file is their only copy; copy these, don't improvise

`datagen` defaults to all cores; bound the run with `--count N`.
Generation flags are first-class — there is no `--` separator.

Two cells are the whole catalogue: the T2 cell below ("Current T2 config") and
the T1 cell after it ("T1 config"). Both run generation LOCALLY, which is
correct for a smoke test or a quick run. **For a real training run on the GPU
box, front the trainer with `multinode_gen.sh` instead** so generation lands on
odin and does not starve the GPU (see "Multinode / remote generation" — same
generation flags as env vars, same `| trainer` tail): swap
`python -m datagen … --count N` for `<env-knobs> ./multinode_gen.sh N`.
`--count N` emits exactly N batches, then closes stdout and reaps the whole
worker pool.

`--n-steps` only shapes the cosine-LR schedule and the ETA display — it is NOT a
hard step cap. Training runs until stdin EOF (`scripts/_common.py:30-34`). Bound
the batch count with `datagen --count N` (`datagen` has no per-worker `-n`).

**Current T2 config** (writers-only happy path). The equivariant core has no
settings to pass: read positions are canonical ranks, register slots are seeded
from those ranks, and the pool reads each slot's write step, at every tier.

```bash
python -m datagen --rule "cap=$LEN" --twins 2 --batch-size 128 --n-states 8 \
    --anchor-seed 0 --config configs/instr_alu_writers.json --count "$STEPS" \
  | python scripts/train_t2_encoder.py --t1-model runs/t1_writers_250k_rank/encoder.pt \
    --d-out 512 --d-event 16 --op-cell-depth 4 --vp-head-depth 1 \
    --n-pool-heads 16 --route binding \
    --n-steps "$STEPS" --anchor-seed 0 --n-anchor-states 8 --out-dir "$OUT"
```

`--d-out 512` is the register-state width, and 512 is where it stays: 256 → 512
is a real gain and 512 → 1024 buys nothing at any training budget measured, at
2.6x the step time (EXPERIMENT_LOG, "T2 faithfulness", finding 17).

`scripts/run_t2.sh` runs exactly this cell with each knob as an env var
(`DOUT`, `LEN`, `STEPS`, `NPOOL`, …, `T1` for the frozen lower tier and
`CONFIG` for the corpus), names the run directory after them, offloads
generation to odin, records the corpus, and scores the result. The directory
ends in the config's name with the `instr_` prefix dropped
(`…_pool16_alu_writers`), so two corpus arms of one configuration cannot
collide; `TAG=<name>` additionally suffixes it so a re-run of an
already-recorded cell does not overwrite it. Prefer it to typing the pipeline by
hand — but **never edit it while a copy is running**: bash reads a script
incrementally, so an edit shifts the file offsets under the running shell. Doing
this mid-sweep produced a spurious `syntax error near 'fi'` and re-ran the
eval-corpus block's redirect outside its `if`, truncating `/tmp/t2_eval_2.rvt`
to 0 bytes — which surfaces as `binding: … rows=0` in every eval afterwards.
Wait for the sweep, or edit a copy.

**A replicate needs `GEN_SEED`.** Both generator pools default to `--seed 0`, so
re-running a cell otherwise redraws the same instruction stream and measures
only initialization variance. `GEN_SEED=7 … TAG=s7 bash scripts/run_t2.sh` draws
a different corpus from the same distribution. It is unrelated to `ANCHOR_SEED`,
which must stay equal to the trainer's `--anchor-seed` (INVARIANT 1). Replicating
is not optional for a sweep — see METRICS.md on null interventions.

**A run's `t2.pt` appears long before the run ends**, because the trainer
checkpoints every `--save-every` steps (default 10000; a save stalls the GPU,
so it is kept off the 100-step log cadence). Wait on the trainer process
exiting, or on `run_t2.sh` logging `final:`; scoring the file as soon as it
exists reads a partial model.

**T1 config** (the writers-only corpus T2 is trained against). T1 and T2 run
the same core at the same settings, so the command below differs from the T2
cell only in its ingestion front-end and its hparams. The frozen T1 is
`runs/t1_writers_250k_rank`.

```bash
python -m datagen --rule single --twins 2 --batch-size 128 --n-states 8 \
    --anchor-seed 0 --config configs/instr_alu_writers.json --count 250000 \
  | python scripts/train_encoder.py --d-model 128 --n-heads 4 --n-layers 2 \
    --d-out 64 --max-window 32 --imm-weight 0.3 \
    --n-steps 250000 --anchor-seed 0 --n-anchor-states 8 --out-dir "$OUT"
```

Both trainers expose the same model hparams — `--d-event`, `--op-cell-depth`,
`--vp-head-depth`, `--n-pool-heads` — since T1 and T2 run the same core
(`scripts/train_encoder.py`, `scripts/train_t2_encoder.py`). Those are
dimensions; the core's behaviour is not configurable from either trainer, so
there is no way to run the two tiers on different versions of it.

**A checkpoint predating the core's settling will not load.** The core has no
branch for an alternative read position, slot seeding or pool channel, so a
checkpoint carrying `tag_proj` instead of `rank_emb`, or a pool sized without
the write-step channel, fails at `load_state_dict`. Only checkpoints trained at
the settled configuration are usable; `runs/t1_writers_250k_rank` is the frozen
T1.
(A checkpoint trained when all pool heads shared one value projection is still
remapped on load — `_tile_shared_pool_v` repeats those weights once per head,
reproducing the old computation exactly.)

## INVARIANTS

The six invariants that silently corrupt data or training when violated live in
**CLAUDE.md** ("Data generation pipeline"), which is always loaded — you cannot
opt into a warning you do not know you need. They are kept in one place, there;
do not restate them here.

## datagen reference

`--rule` grammar (`datagen/__main__.py:89` `_parse_rule`, rule constructors at
`datagen/generate.py:704-729`). Default `--rule` is `branch+cap=8`:
- `single` — one instruction per chunk (T1). `max_len=1`. Ships canonical
  out_regs like every other rule (the T1/T2 unification).
- `cap=N` — fixed length N.
- `branch+cap=N` — terminate at a branch OR length N.
- `transform+cap=N` — terminate at a memory-op / branch / jump OR length N.
- Components join with `+` and compose via `either()` (terminate when ANY fires).
- **A bare `branch` or `transform` (no `cap=`) is rejected eagerly** — it can
  build an unbounded chunk. Error says "unbounded … add a cap". Validated in the
  PARENT before forking (`datagen/__main__.py:586-592` calls `_parse_rule`) and
  again in `collect_groups` via `rule.max_len is None` (`datagen/generate.py:742`).

Generation flags (per worker, first-class — NO `--` separator; defaults in
parens, from the argparse block at `datagen/__main__.py:505-527`): `--rule`
(`branch+cap=8`), `--twins T` (3 — relabeled equivalents per valid source
chunk), `--batch-size B` (128 — source rows; twins extend the batch beyond B),
`--n-states` (8 — anchor count), `--anchor-seed` (0), `--max-invalid-window`
(32 — the token-length cap on an injected invalid window), `--config PATH`
(unset → the built-in `DEFAULT_DISTRIBUTION`), `--seed` (0; worker i
gets seed+i). **A corpus's composition is settable only in the config file**:
the opcode mix, `exclude_x0_dest`, and the `equivalences` / `invalidity` /
`chain` rates have no CLI override, so the file a run names describes the
corpus completely. There is **no
`--n-batches`** — generation is infinite; bound the total with `--count N`.

Pool / mux flags: `--gen-count N` (default = all CPU cores; `0` = pure mux of the
external inputs), `--gen-weight`, positional `inputs` (`path` or `WEIGHT:path`,
files or pipes), `--mode` (`fifo` as-ready default / `round-robin` / `weighted`,
auto-selected if any `W:path` weight ≠ 1.0; weights bias ORDER not count),
`--shuffle-seed`, `--worker-log-every N` (per-worker stderr heartbeat; 0 =
start/end only — workers are noisy at scale). A piped/redirected **stdin** (FIFO
or regular file — not a tty or `/dev/null`) is auto-added as an input at weight
1.0; it's read lazily through a private dup of fd 0 (so an early `--count` exit
can't deadlock interpreter shutdown on `sys.stdin`'s buffer lock).

Output flags (folded in from the retired batch_slice / bench_throughput tools;
both hook the single `_Writer` chokepoint): `--count N` — emit exactly N batches
total (workers + muxed inputs) then stop and reap; `-v` — windowed throughput
(batch/s, MB/s) every `--log-every-sec` (default 5) plus a final summary.

**A malformed batch from ANY source is fatal** (`_InputParseError`): datagen
prints `FATAL — malformed batch from <source>: …`, reaps the whole worker pool
(`killpg`), and exits non-zero — fail early, fail loud, never train on a
silently-truncated corpus. A CLEAN end-of-stream (`read_bytes` → None) is not
malformed; it just drains that source. (Normal `--count` shutdown stops reading
at the cap, so it never trips on a truncated tail — including the multinode nc
stream cut by the BrokenPipe cascade.)

### Workers cannot orphan — PR_SET_PDEATHSIG (was the project's #1 hazard)
Workers are forked `multiprocessing.Process` children that arm
`PR_SET_PDEATHSIG(SIGKILL)` as their first act (`datagen` `_arm_pdeathsig`),
so the kernel SIGKILLs each worker the instant the parent dies — **even on
`kill -9` of the parent**, which the old reap-on-clean-exit path could not cover.
All workers share one process group reaped via `killpg` on normal/BrokenPipe
exit (`WorkerPool.reap`). The regression test `test_no_orphan_workers_after_kill`
SIGKILLs the parent and asserts zero survivors.

The old manual `pgrep→kill` ritual is no longer required after killing a
`datagen`/multinode pipeline — pdeathsig handles it. If you ever suspect a
straggler anyway, the backstop check is `pgrep -P <datagen_pid>` (workers are
fork children sharing the parent's argv, so identify them by parentage, not by
cmdline).

## RVT binary format (datagen/batch.py)

- **No version field.** There is exactly one RVT format; any change to it is
  immediate and global (no long-lived disk corpora, no out-of-sync remote
  generators), so there is no compatibility to negotiate and no legacy read
  path. The per-batch **dtype signature** still rejects a schema-mismatched
  stream — a body-field change shifts it and readers raise `Dtype mismatch`.
- **No stream header — every batch is self-describing** (4-byte magic `RVT\x00` +
  dtype signature, then a 5-uint32 dimension header, then the body). This is the
  load-bearing property: **RVT streams concatenate with `cat`.**
  - Combine corpora: `cat a.rvt b.rvt > both.rvt`
  - **Second epoch = `cat corpus.rvt corpus.rvt | trainer`.** There is no built-in
    repeat; epochs are shell-level `cat`.
- Header fields: `B, max_tokens, max_n_instrs, n_anchors, OB` (`RVT_FORMAT`,
  `datagen/batch.py:286-288`). `OB` is the out_regs row count — `B` in practice,
  always, since `build_twins` returns one out_regs row per output row
  (`datagen/batch.py:363-376`); the format still permits 0.
  Body field list + dtypes are the `BodyField(...)` block at
  `datagen/batch.py:289-303`; the per-field header sanity bounds are the
  `header_bounds={...}` at `datagen/batch.py:306`, enforced on every read by
  `BinaryFormat._check_header_bounds` (`datagen/batch.py:148`).
- The "no version field" statement is current and load-bearing in the code, not
  aspirational: the only per-batch schema check is magic + dtype-signature
  (`datagen/batch.py:81-88,140-145`). EXPERIMENT_LOG's "format v7→v8"
  (`EXPERIMENT_LOG/phase-14-rename-equivariant-t1-t2.md`) is history from when the format was still
  versioned — do not reintroduce a version field on its authority.
- Per-instruction segmentation is NOT preserved on the wire — relabeling/aux
  happen at generation time while the `list[Instruction]` is still in hand.

## Configs (configs/*.json)

**A config file IS a corpus definition, and nothing outside it can change
one.** It carries the opcode-category mix (weights must sum to 1.0) plus the
optional `equivalences`, `invalidity`, `chain` and `exclude_x0_dest` keys, and
no CLI flag or env var overrides any of them. So a sweep over corpus
composition is a file per arm — each arm reviewable in a diff, and each run
able to record what it was trained on (see "What a run records about its
corpus"). Validation: `datagen/generate.py:160`
`validate_distribution`; the accepted non-opcode keys are `_CONFIG_EXTRA_KEYS`
(`datagen/generate.py:155`) and an unknown key is a hard error.

Three configs exist. Add one when a run needs a corpus none of them describes,
and delete it once nothing runs it — an unused config is a corpus nobody can
attribute a result to.

- `instr_alu_writers.json` — **the corpus all current work uses.**
  R/I-type ALU only (0.55 / 0.45) + `exclude_x0_dest: true`. No equivalences,
  no invalidity → a clean, all-valid, always-register-writing corpus. Every
  chunk is pure ALU, so `cap=N` and `branch+cap=N` behave identically here and
  the scripts use plain `cap=N`.
- `instr_alu_writers_chain010.json` — the same, plus `chain: {rate: 0.10}`.
  A uniform register draw leaves only ~4.7% of two-instruction chunks with the
  second reading the first's result, and that scarcity — not any architectural
  limit — is what held the composed-slot value prediction down (EXPERIMENT_LOG,
  "T2 faithfulness", finding 30). `chain.rate` rewrites that fraction of chunks
  so each instruction reads its predecessor's result, leaving every opcode
  untouched, which lifts composed chunks to roughly `rate + (1-rate)*0.047` of
  rows (counted over 20 batches of `cap=2`: 4.8% of rows composed at rate 0,
  15.4% at rate 0.10). A no-op on single-instruction rules. The rate is in the
  filename because another rate is another file.
- `instr_r_only.json` — R-type ALU only, so no chunk carries an immediate.
  Three probes generate from it (`vp_slot_b_split.py`,
  `opcell_rank_sufficiency.py`, `vp_superposition.py`); it is a probe corpus,
  not a training one, and a category scored on it says nothing about an
  immediate (METRICS.md, scope conditions).

`exclude_x0_dest: true` narrows the **rd** pool to x1..x31 (`datagen/__main__.py:173`
→ `dest_regs=range(1,32)`, consumed at `datagen/generate.py:86`), so every
generated instruction actually writes a register — no write-to-x0 NOPs. It does
NOT touch the source pool: rs1/rs2 still draw from x0..x31 (`_SRC_REGS`,
`datagen/generate.py:41`), so x0-as-operand still occurs.

Invalid-window types (`datagen/invalidity.py`): `partial` (sub-slice of one
instr), `spanning` (crosses an instr boundary mis-aligned), `multi` (2–3 whole
instrs concatenated), `bogus` (random vocab tokens). The encoder learns to map
these near the origin (low magnitude); valid windows sit near the unit sphere.
A corpus with no `invalidity` block injects none, which is what makes the
writers corpora clean.

## What a run records about its corpus

A run directory carries `hparams.json` — `vars(args)` of the TRAINER, which
never sees the generator — so the model is recorded and the data is not.
`scripts/record_corpus.py` writes the other half as `corpus.json`:

```bash
python scripts/record_corpus.py runs/<dir> --config configs/<corpus>.json \
    --rule cap=2 --twins 2 --batch-size 128 --n-states 8 \
    --anchor-seed 0 --gen-seed 0 --count 1000000
```

It copies the config's CONTENTS, not its path, so editing that file afterwards
cannot change what the run claims to have trained on, and it records the knobs
that live outside the config — rule, twins, batch size, anchor count and seed,
instruction-stream seed, batch count. `run_t2.sh` calls it before the pipeline
starts, so a run killed mid-training is still identified. A pipeline typed by
hand should call it too; a run without a `corpus.json` can only be attributed
by its directory name.

Recording the anchor seed and state count on both sides is what makes
INVARIANT 1 (CLAUDE.md) checkable after the fact: compare `corpus.json`'s
`anchor_seed` / `n_states` against `hparams.json`'s `anchor_seed` /
`n_anchor_states`, and if they disagree the run's value-prediction targets
were wrong with no error raised at the time.

## Multinode / remote generation

Two different scripts — do not confuse them:

- **`multinode_gen.sh TOTAL_BATCHES`** — the standard front-end for a real
  training run. Ships datagen sources to `odin` over ssh (tar stream), runs a
  remote `datagen` pool that streams RVT back over `nc` (`PORT`, default 6464), and muxes
  it with a local `datagen` pool. Output to stdout/file or piped to a trainer.
  Both pools generate infinitely; the local side bounds the joined stream to
  exactly `TOTAL_BATCHES` with `--count`, and on reaching it the BrokenPipe
  cascade (local→nc→remote) reaps every worker. `REMOTE=chrissarbora@odin`.
  Worker split is `REMOTE_GEN` (default 96) / `LOCAL_GEN` (default **0**):
  local datagen contends with a local GPU trainer's per-step host work, so the
  default offloads EVERY worker to odin; bump `LOCAL_GEN>0` only for a pure
  on-disk corpus with no local trainer. Odin's pool delivers ~600 batches/s on
  its own, ~2.3x what the len=2 T2 cell consumes (`--d-out 512`, op-cell depth 4,
  16 pool heads: ~3.9 ms/step, ~95% GPU util; EXPERIMENT_LOG, "Rename-
  equivariant T1/T2", Throughput). Judge saturation by GPU utilisation sampled
  finer than the step-log cadence — `nvidia-smi --query-gpu=utilization.gpu
  --format=csv -lms 100` — because a stall at every log line beats against a
  1 s sampler and reads as an occasional dip; use ms/step against the same
  config's figure as a floor check, and re-measure both whenever the cell's
  configuration changes — a figure from one configuration says nothing about
  another.

  **Adding remote workers does not raise a rate the GPU is already setting.**
  At `REMOTE_GEN=120`, odin sat at a load average of ~11 out of its 128 cores
  with the generators backpressured, so when a cell runs slow, check odin's
  load before adding workers: if it is low, generation is not what limits you.
  `RULE` defaults to `single`; `CONFIG` is REQUIRED and the script exits
  without it, because a corpus is one named file.
  `ANCHOR_SEED` (default 0) is passed to BOTH pools and MUST equal the trainer's
  `--anchor-seed` (INVARIANT 1 — mismatch silently corrupts value_predict
  targets). Pipe it straight to a trainer:
  ```bash
  RULE="cap=2" CONFIG=configs/instr_alu_writers.json ./multinode_gen.sh 2000000 \
    | python scripts/train_t2_encoder.py --t1-model runs/t1_run/encoder.pt \
      --d-out 512 --d-event 16 --route binding --n-steps 2000000 \
      --anchor-seed 0 --n-anchor-states 8 --out-dir t2_run
  ```
- **`gen_corpus.sh TOTAL_BATCHES OUT_PATH`** — writes an on-disk corpus. **Despite
  its header comment mentioning "remote", the script itself runs `datagen`
  LOCALLY** (`gen_corpus.sh`) and writes `OUT` on the local disk via
  `datagen … --count TOTAL > OUT`. "Remote" = the *intended usage* is to ssh
  into odin and run it there; there is no auto-shipping. `RULE` defaults to
  `branch+cap=8` and `WORKERS` unset → all CPU cores; `CONFIG` is REQUIRED.

Both take `BATCH_SIZE TWINS N_STATES RULE ANCHOR_SEED` as env vars and REQUIRE
`CONFIG` (plus `GEN_SEED REMOTE_GEN LOCAL_GEN PORT` for multinode_gen, and
`WORKERS` for gen_corpus → `--gen-count`). No env var sets an augmentation
rate: those are the config's.

### Running two pipelines at once

**Give each run a distinct `PORT`.** The remote pidfile is keyed by it
(`/tmp/riscv-thoughts-multinode-remote.$PORT.pid`), and that file's stale-run
cleanup `pkill`s whatever PID it finds — so two runs sharing a port do not merely
fail to bind, the second reaps the first's healthy worker pool, and whichever
exits first tears down the other's pool and `rm -rf`s its shipped sources. The
shipped `REMOTE_DIR` is already unique (`mktemp`). Halve `REMOTE_GEN` per run so
the pools do not oversubscribe odin's 128 cores.

The scoring probes `run_t2.sh` invokes each generate their own corpus under
`/tmp`, keyed by PID so two concurrent cells cannot overwrite each other's. A
probe added without that key will fail the *other* run's scoring with
`Bad magic` while its own training completes normally — so a `/tmp` corpus path
in a new probe needs `os.getpid()` in it.

**Not worth doing for throughput while one trainer keeps the GPU busy:** two
concurrent len=2 cells ran at 15.5 ms/step each against 6.0 ms for one alone,
0.78× the aggregate, because the GPU time-slices between them. Run cells
concurrently only when one alone leaves the GPU idle.

```bash
PORT=6464 REMOTE_GEN=56 RULE="cap=2" CONFIG=configs/instr_alu_writers.json \
  ./multinode_gen.sh 1000000 | python scripts/train_t2_encoder.py … &
PORT=6465 REMOTE_GEN=56 RULE="cap=2" CONFIG=configs/instr_alu_writers.json \
  ./multinode_gen.sh 1000000 | python scripts/train_t2_encoder.py … &
```

Remote operational facts:
- **`export MALLOC_ARENA_MAX=2`** is required in the remote/local pool scripts
  (`scripts/multinode_remote.sh:41`, `gen_corpus.sh:52`). Default glibc makes `8*ncpus`
  arenas per worker; on a 64-core box that makes `clear_page_erms` page-zeroing
  the #1 hotspot. Don't remove it.
- `scripts/multinode_remote.sh` (shipped to the remote and invoked there by
  `multinode_gen.sh`) owns its lifecycle via a pidfile
  (`/tmp/riscv-thoughts-multinode-remote.$PORT.pid` — keyed by port so the
  cleanup is per-listener, not global) so a prior crashed run is cleaned up by
  the next. Its pipeline stderr persists to `/tmp/multinode-remote.log`
  (survives ssh teardown — read it for post-mortems).
- **lz4 compresses the remote nc transport** (`lz4 -1 -c` before `nc` on the
  remote, `lz4 -dc` after `nc` locally). Both ends must match. Measured **3.79×**
  on real RVT — padded numeric arrays compress well — which takes the link off
  the critical path: the odin↔thor link measures **111 MB/s** over this same nc
  transport (gigabit; ignore the 10 Gb/s the WSL2 virtual adapter reports in
  `/sys/class/net/eth0/speed`), and one trainer alone drew 64 MB/s of it
  uncompressed. It was removed once, deliberately, to make byte-count-dependent
  failures reproduce ~5× faster while they were being diagnosed; strip it again
  if that kind of hunt resumes.
- **NVML cannot enumerate GPU processes under WSL2.** `nvidia-smi
  --query-compute-apps` (and so `nvitop`) lists one or none of several live
  trainers and reports `[N/A]` for memory. Judge a run by its own log's ms/step
  and the aggregate `utilization.gpu`; confirm a process is on the GPU with
  `grep -c 'nvidia\|dxg' /proc/<pid>/maps` if needed.
- `binary_stdout()` forces the fd to blocking mode because `tee` (and other
  consumers) flip pipes non-blocking, which would raise BlockingIOError mid-batch
  and truncate the reader (`datagen/__main__.py`, `binary_stdout`).

## Constants / token budget math

- `MAX_INSTR_TOKENS = 9`, `VOCAB_SIZE = 89` (`tokenizer/tokenizer.py:127,52` —
  both are computed, not literals; verify by importing, not by reading).
- A chunk's token budget: `target_max_tokens = max_chunk_len * 9`
  (`datagen/batch.py:656`), then widened to `--max-invalid-window` if that is
  larger (`datagen/batch.py:657-658`). That widening is **unconditional** — the
  flag defaults to 32 and is applied even on a corpus with no invalid injection,
  so `--rule single` emits `max_tokens=32`, not 9. (Verified: a 3-batch
  `--rule single --config configs/instr_alu_writers.json` run ships
  `tokens (24, 32)`.)
- T1's TokenEmbedder needs `--max-window` (default 72 = 8×9) to cover this. T2's
  equivariant core has no position table — it unrolls to each batch's actual
  instruction count, so there is no T2 chunk-length flag.
- N_REGS=32, MAX_INPUT_SLOTS=32, MAX_OUTPUT_SLOTS=16, AUX_CE_IGNORE=-100
  (`datagen/compare.py:77,91,92,93`). PC_REG=32, MEM_REG=33 are pseudo-slots
  (`datagen/compare.py:37-38`).

## Where things live / what's tracked

- `corpora/` — on-disk `.rvt` corpora. Present but empty by default; **NOT in
  .gitignore**, yet not committed — `.rvt` files are large binary, keep them out
  of commits.
- `runs/<name>/` — checkpoints + hparams.json + losses.json (gitignored).
- Pipeline regression tests: `scripts/tests/test_pipeline.py` (covers rule
  parsing, format detect/validate, mux modes, the pdeathsig orphan guard, and
  end-to-end 1-step train smokes). In the default `pytest` run; it is the slow
  part of it, since every case drives a CLI through a subprocess. Narrow to it
  with `python -m pytest scripts/tests`.
- This guide is the `datagen` skill (`.claude/skills/datagen/SKILL.md`), tracked
  by the `!.claude/skills/` exception in `.gitignore`. It loads on demand —
  invoke it before touching the pipeline rather than working from memory.

## Things this skill does NOT cover (read the code, don't guess)

- Loss math / what each aux head learns, and whether a clean corpus still needs a
  trainer-side validity weight — that's `compressor/train.py` + EXPERIMENT_LOG,
  not datagen. Don't assert training-loss behavior from this file.
- Measured throughput numbers — they drift with machine/arch; measure with
  `datagen -v` rather than quoting a remembered figure
  (CLAUDE.md: no speculative numbers).
