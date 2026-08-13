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

## Canonical commands (verified against README.md — copy these, don't improvise)

`datagen` defaults to all cores; bound the run with `--count N`.
Generation flags are first-class — there is no `--` separator.

The commands below run generation LOCALLY — correct for smoke tests and quick
runs. **For a real training run on the GPU box, front the trainer with
`multinode_gen.sh` instead** so generation lands on odin and does not starve the
GPU (see "Multinode / remote generation" — same generation flags via env vars,
same `| trainer` tail). Swap `python -m datagen … --count N` for
`<env-knobs> ./multinode_gen.sh N`.

T1 (single-instruction encoder):
```bash
python -m datagen --rule single --twins 2 --batch-size 128 \
    --n-states 8 --anchor-seed 0 --config configs/instr_default.json \
    --count 250000 \
  | python scripts/train_encoder.py --d-model 128 --n-heads 4 --n-layers 2 \
    --d-out 64 --max-window 32 --n-steps 250000 --out-dir t1_run
```

T2 (chunk encoder on a frozen T1):
```bash
python -m datagen --rule "branch+cap=4" --twins 3 --batch-size 128 \
    --n-states 8 --anchor-seed 0 --config configs/instr_t2.json \
    --count 2000000 \
  | python scripts/train_t2_encoder.py --t1-model runs/t1_run/encoder.pt \
    --d-out 512 --d-event 16 --route binding \
    --n-steps 2000000 --anchor-seed 0 --n-anchor-states 8 --out-dir t2_run
```
`--count 2000000` emits exactly that many batches then closes stdout and reaps
the whole worker pool.

`--n-steps` only shapes the cosine-LR schedule and the ETA display — it is NOT a
hard step cap. Training runs until stdin EOF (`scripts/_common.py:30-34`). Bound
the batch count with `datagen --count N` (`datagen` has no per-worker `-n`).

**Current T2 config** (writers-only happy path). Three flags are load-bearing
and non-default: `--pool-extra write_time` and `--read-time-mode rank` /
`--seed-mode rank` supply the ordering the essence needs — omitting any of them
reproduces an earlier, worse model rather than erroring:

```bash
python -m datagen --rule "cap=$LEN" --twins 2 --batch-size 128 --n-states 8 \
    --anchor-seed 0 --config configs/instr_alu_writers.json --count "$STEPS" \
  | python scripts/train_t2_encoder.py --t1-model runs/t1_writers_250k/encoder.pt \
    --d-out 512 --d-event 16 --op-cell-depth 4 --vp-head-depth 1 \
    --n-pool-heads 16 --pool-mode pma --pool-extra write_time \
    --read-time-mode rank --seed-mode rank --route binding \
    --n-steps "$STEPS" --anchor-seed 0 --n-anchor-states 8 --out-dir "$OUT"
```

`--d-out 512` is the register-state width, and 512 is where it stays: 256 → 512
is a real gain and 512 → 1024 buys nothing at any training budget measured, at
2.6x the step time (EXPERIMENT_LOG phase-15, finding 17).

`scripts/run_t2.sh` runs exactly this cell with each knob as an env var
(`DOUT`, `LEN`, `STEPS`, `NPOOL`, …), names the run directory after them,
offloads generation to odin, and scores the result. Prefer it to typing the
pipeline by hand — but never edit it while a copy is running, since bash reads a
script incrementally and the running shell will misparse the shifted file.

Trainer-CLI asymmetry to know before improvising: the equivariant-core
architecture flags — `--op-cell-depth`, `--vp-head-depth`, `--n-pool-heads`,
`--pool-mode {pma,compete}`, `--pool-extra {none,write_time,order,event}`,
`--read-time-mode {stamp,rank}` — exist **only on `scripts/train_t2_encoder.py`**
(`scripts/train_t2_encoder.py:39-72`). `scripts/train_encoder.py` does not expose
them even though `train_compressor` accepts them (`compressor/train.py:514-517`),
so T1 always trains at their defaults. Passing one to `train_encoder.py` is an
argparse error, not a silent no-op. (`--pool-extra` replaced the old boolean
`--pool-reads-order`; old checkpoints are remapped on load by
`scripts/_common.py:69-74`, `pool_reads_order: true` → `pool_extra:
'write_time'`. A checkpoint trained when all pool heads shared one value
projection is likewise remapped on load — `_tile_shared_pool_v` repeats those
weights once per head, which reproduces the old computation exactly, so old
multi-head checkpoints still score as they did.)

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
  PARENT before forking (`datagen/__main__.py:582-586` calls `_parse_rule`) and
  again in `collect_groups` via `rule.max_len is None` (`datagen/generate.py:742`).

Generation flags (per worker, first-class — NO `--` separator; defaults in
parens, from the argparse block at `datagen/__main__.py:503-521`): `--rule`
(`branch+cap=8`), `--twins T` (3 — relabeled equivalents per valid source
chunk), `--batch-size B` (128 — source rows; twins extend the batch beyond B),
`--n-states` (8 — anchor count), `--anchor-seed` (0), `--inject-invalid R`
(unset → config), `--inject-equiv R` (unset → config), `--max-invalid-window`
(32), `--config PATH` (unset → `DEFAULT_DISTRIBUTION`), `--seed` (0; worker i
gets seed+i). CLI flags override the JSON config's `equivalences.rate` /
`invalidity.rate`. There is **no
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

Opcode-category mix (weights must sum to 1.0) + optional `equivalences`,
`invalidity`, and `exclude_x0_dest`. Validation: `datagen/generate.py:160`
`validate_distribution`; the accepted non-opcode keys are `_CONFIG_EXTRA_KEYS`
(`datagen/generate.py:155`) and an unknown key is a hard error.
- `instr_alu_writers.json` — **the corpus all current (Phase 15) work uses.**
  R/I-type ALU only (0.55 / 0.45) + `exclude_x0_dest: true`. No equivalences,
  no invalidity → a clean, all-valid, always-register-writing corpus. Every
  chunk is pure ALU, so `cap=N` and `branch+cap=N` behave identically here and
  the scripts use plain `cap=N`.
- `instr_default.json` — T1 mix; equivalences rate 0.05; **invalidity rate 0.2**.
- `instr_t2.json` — T2 mix (ALU-heavy, few branches/mem); equivalences only.
- `instr_no_inject.json` — opcode mix only, no augmentation.
- `instr_no_mem.json` — excludes LOAD/STORE (no mem-op chunks at all).
- `instr_branch_heavy.json` — 50% branches.

`exclude_x0_dest: true` narrows the **rd** pool to x1..x31 (`datagen/__main__.py:170`
→ `dest_regs=range(1,32)`, consumed at `datagen/generate.py:86`), so every
generated instruction actually writes a register — no write-to-x0 NOPs. It does
NOT touch the source pool: rs1/rs2 still draw from x0..x31 (`_SRC_REGS`,
`datagen/generate.py:41`), so x0-as-operand still occurs. There is no CLI flag
for it; it is config-only.

Invalid-window types (`datagen/invalidity.py`): `partial` (sub-slice of one
instr), `spanning` (crosses an instr boundary mis-aligned), `multi` (2–3 whole
instrs concatenated), `bogus` (random vocab tokens). The encoder learns to map
these near the origin (low magnitude); valid windows sit near the unit sphere.
`--inject-invalid 0` overrides the config to produce a CLEAN corpus.

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
  local datagen contends with a local GPU trainer's per-step H2D prep and
  starves the GPU — measured **~54% util / 16.6 ms/step with local gen, against
  9 ms with generation offloaded**. So the default offloads EVERY worker to
  odin; bump `LOCAL_GEN>0` only for a pure on-disk corpus with no local trainer.
  Judge saturation by GPU utilisation, with ms/step against the same config's
  known-good figure as a floor check.

  **Offloading generation does not by itself saturate the GPU, and adding remote
  workers will not either.** At len=2 the trainer's own per-step host work (RVT
  parse, `_split_to_per_instruction`, `_t2_assemble`) is the constraint — it
  does not overlap the GPU step, so utilisation sits near 40–50% with the
  generators idle and backpressured. Measured at `REMOTE_GEN=120`: ~37% util,
  odin at a load average of ~11 out of 128 cores, 20 MB/s over a ~118 MB/s link.
  **Check odin's load before adding workers** — if it is low, generation is not
  what is limiting you. TODO.md carries the diagnosis and the fix.
  Other defaults: `RULE=single`, `CONFIG=configs/instr_default.json`,
  `INJECT_INVALID=0` (clean corpus regardless of what the config says) — set
  them explicitly for any T2 / writers-only run.
  `ANCHOR_SEED` (default 0) is passed to BOTH pools and MUST equal the trainer's
  `--anchor-seed` (INVARIANT 1 — mismatch silently corrupts value_predict
  targets). Pipe it straight to a trainer:
  ```bash
  RULE="branch+cap=4" CONFIG=configs/instr_t2.json ./multinode_gen.sh 2000000 \
    | python scripts/train_t2_encoder.py --t1-model runs/t1_run/encoder.pt \
      --d-out 512 --d-event 16 --route binding --n-steps 2000000 \
      --anchor-seed 0 --n-anchor-states 8 --out-dir t2_run
  ```
- **`gen_corpus.sh TOTAL_BATCHES OUT_PATH`** — writes an on-disk corpus. **Despite
  its header comment mentioning "remote", the script itself runs `datagen`
  LOCALLY** (`gen_corpus.sh`) and writes `OUT` on the local disk via
  `datagen … --count TOTAL > OUT`. "Remote" = the *intended usage* is to ssh
  into odin and run it there; there is no auto-shipping. Defaults to T2
  settings (`branch+cap=8`, `instr_t2.json`, no invalid injection); `WORKERS`
  unset → all CPU cores.

Both override knobs via env vars: `BATCH_SIZE TWINS N_STATES RULE CONFIG
INJECT_INVALID INJECT_EQUIV` (plus `ANCHOR_SEED REMOTE_GEN LOCAL_GEN PORT` for
multinode_gen, and `WORKERS` for gen_corpus → `--gen-count`).

### Running two pipelines at once

**Give each run a distinct `PORT`.** The remote pidfile is keyed by it
(`/tmp/riscv-thoughts-multinode-remote.$PORT.pid`), and that file's stale-run
cleanup `pkill`s whatever PID it finds — so two runs sharing a port do not merely
fail to bind, the second reaps the first's healthy worker pool, and whichever
exits first tears down the other's pool and `rm -rf`s its shipped sources. The
shipped `REMOTE_DIR` is already unique (`mktemp`). Halve `REMOTE_GEN` per run so
the pools do not oversubscribe odin's 128 cores.

Worth doing when a single trainer leaves the GPU idle: two concurrent len=2 runs
hold it at 84–99% util against ~50% for one, at 11 ms/step each against 7 solo —
**~1.3× aggregate**, not 2×, because the GPU becomes the binding constraint once
host serialization is filled. Local cost is ~10 of 32 cores for two trainers.

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
  end-to-end 1-step train smokes). Run `python -m pytest scripts/tests`.
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
