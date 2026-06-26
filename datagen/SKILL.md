---
name: datagen
description: >
  Reference for the RVT data-generation pipeline — the datagen/ package (the
  `python -m datagen` generator tool) and the multinode wrappers
  (multinode_gen.sh, gen_corpus.sh, multinode_remote.sh). Load this BEFORE
  generating a corpus, editing datagen/ or any pipeline script, changing the RVT
  format, debugging worker/throughput/OOM issues, or wiring gen→train. Carries
  the invariants that are silent-corruption traps if violated (anchor-seed match,
  fixed batch shape, raw-vs-canonical anchors).
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
hard step cap. Training runs until stdin EOF (`scripts/_common.py:32`). Bound the
batch count with `datagen --count N` (`datagen` has no per-worker `-n`).

## INVARIANTS — violating these silently corrupts data or training

1. **anchor-seed AND n-states must match gen↔train.** value_predict supervision
   is computed against anchor states reconstructed from `--anchor-seed` +
   `--n-anchor-states` (`scripts/train_encoder.py:60-63`, `train_t2_encoder.py:65-67`).
   `datagen`'s `--anchor-seed`/`--n-states` must equal the trainer's
   `--anchor-seed`/`--n-anchor-states`. Mismatch → wrong targets, no error.
   `make_anchor_states` is `datagen/compare.py:369`. (Anchor seed only needs to
   match gen↔train; it is *not* needed for cross-worker comparability — distances
   are self-contained per row.)

2. **Fixed batch shape, derived from the rule's length cap.** Every emitted batch
   is padded to identical (B, max_tokens, max_n_instrs) so PyTorch's caching
   allocator doesn't fragment. Stochastically-varying batch shape is a real OOM
   trigger we hit (~step 500 of t1_cosine_full). The targets come from
   `max_chunk_len = rule.max_len` (`datagen/batch.py:697`, `collect_into_batches`).
   This is why a `--rule` with no cap is REJECTED (see Rule grammar). Don't add a
   code path that emits a differently-shaped batch mid-run.

3. **EVERY rule ships the CANONICAL out_regs oracle (`OB == B`).**
   `single` (T1) and multi-instruction (T2) rules both go through `build_twins`
   and ship the full canonical `out_regs` register file (`OB == B`, carrying
   `n_anchors == --n-states`). There is no per-rule payload split — both tiers
   feed one canonical value-prediction target.
   (`scripts/tests/test_pipeline.py::test_out_regs_mode_header`.)

4. **value_predict is CANONICAL for both tiers.** `out_regs` is the
   rename-invariant canonical baseline: `precompute_chunk` relocates the chunk's
   i-th first-read input to canonical position i+1 (`_canonical_state`) and
   executes there, so `out_regs[:, r]` depends only on operand structure. The vp
   loss (`t2_value_predict_loss`, used by BOTH T1 and T2) therefore feeds the head
   the canonical-position values `anchor[:, k+1]` for input slot k — never the raw
   value at the slot's actual register. Feeding raw input values against the
   canonical target is mathematically inconsistent — the head must see canonical
   inputs to match the canonical output.

5. **x0 is fixed under relabeling.** `random_perm`/`relabel` require `perm[0]==0`
   (`datagen/generate.py:279,293`). x0 is hardwired zero, not a register.

6. **Memory-op chunks get no twins and no aux (V1 scope).** `build_twins`
   drops chunks containing LOAD/STORE from twin+aux construction because their
   `Precomputed` is
   unavailable — `precompute_chunk` raises NotImplementedError on memory ops
   (`datagen/compare.py:674`). The chunk's tokens are still shipped (so the
   encoder sees them for magnitude/validity), but its aux row is zero-mask /
   `AUX_CE_IGNORE` and its value-prediction row is masked off (`pair_valid=False`).

## datagen reference

`--rule` grammar (`datagen/__main__.py:64` `_parse_rule`, rules in
`datagen/generate.py:628`):
- `single` — one instruction per chunk (T1). `max_len=1`. Ships canonical
  out_regs like every other rule (the T1/T2 unification).
- `cap=N` — fixed length N.
- `branch+cap=N` — terminate at a branch OR length N.
- `transform+cap=N` — terminate at a memory-op / branch / jump OR length N.
- Components join with `+` and compose via `either()` (terminate when ANY fires).
- **A bare `branch` or `transform` (no `cap=`) is rejected eagerly** — it can
  build an unbounded chunk. Error says "unbounded … add a cap". Validated in the
  PARENT before forking (`datagen/__main__.py:480` calls `_parse_rule`) and again in
  `collect_groups` via `rule.max_len is None` (`datagen/generate.py:734`).

Generation flags (per worker, first-class — NO `--` separator): `--rule`,
`--twins T` (relabeled equivalents per valid source chunk), `--batch-size B`
(source rows; twins extend the batch beyond B), `--n-states` (anchor count),
`--anchor-seed`, `--inject-invalid R`, `--inject-equiv R`, `--max-invalid-window`,
`--config PATH`, `--seed` (base; worker i gets seed+i). CLI flags override the
JSON config's `equivalences.rate` / `invalidity.rate`. There is **no
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
- Header fields: `B, max_tokens, max_n_instrs, n_anchors, OB`. `OB` is the
  out_regs row count (`B` when the oracle is shipped — always, now — else 0).
  Body field list + dtypes are in the `BodyField(...)` block in
  `datagen/batch.py`; the per-field header sanity bounds are the
  `header_bounds={...}` on `RVT_FORMAT` there, enforced on every read by
  `BinaryFormat._check_header_bounds`.
- Per-instruction segmentation is NOT preserved on the wire — relabeling/aux
  happen at generation time while the `list[Instruction]` is still in hand.

## Configs (configs/*.json)

Opcode-category mix (weights must sum to 1.0) + optional `equivalences` and
`invalidity` sub-dicts. Validation: `datagen/generate.py:154`.
- `instr_default.json` — T1 mix; equivalences rate 0.05; **invalidity rate 0.2**.
- `instr_t2.json` — T2 mix (ALU-heavy, few branches/mem); equivalences only.
- `instr_no_inject.json` — opcode mix only, no augmentation.
- `instr_no_mem.json` — excludes LOAD/STORE (no mem-op chunks at all).
- `instr_branch_heavy.json` — 50% branches.

Invalid-window types (`datagen/invalidity.py`): `partial` (sub-slice of one
instr), `spanning` (crosses an instr boundary mis-aligned), `multi` (2–3 whole
instrs concatenated), `bogus` (random vocab tokens). The encoder learns to map
these near the origin (low magnitude); valid windows sit near the unit sphere.
`--inject-invalid 0` overrides the config to produce a CLEAN corpus.

## Multinode / remote generation

Two different scripts — do not confuse them:

- **`multinode_gen.sh TOTAL_BATCHES`** — ships datagen sources to `odin` over
  ssh (tar stream), runs a remote `datagen` pool that streams RVT back over
  `nc` (port 6464), and joins it with a local `datagen` pool. Output to
  stdout/file or piped to a trainer. Both pools generate infinitely; the local
  side bounds the joined stream to exactly `TOTAL_BATCHES` with `--count`, and on
  reaching it the BrokenPipe cascade (local→nc→remote) reaps every worker.
  Worker counts (48 remote, 8 local, hardcoded) now set the gen mix ratio, not
  the total. `REMOTE=chrissarbora@odin`.
- **`gen_corpus.sh TOTAL_BATCHES OUT_PATH`** — writes an on-disk corpus. **Despite
  its header comment mentioning "remote", the script itself runs `datagen`
  LOCALLY** (`gen_corpus.sh`) and writes `OUT` on the local disk via
  `datagen … --count TOTAL > OUT`. "Remote" = the *intended usage* is to ssh
  into odin and run it there; there is no auto-shipping. Defaults to T2
  settings (`branch+cap=8`, `instr_t2.json`, no invalid injection); `WORKERS`
  unset → all CPU cores.

Both override knobs via env vars: `BATCH_SIZE TWINS N_STATES RULE CONFIG
INJECT_INVALID INJECT_EQUIV` (and `WORKERS` for gen_corpus → `--gen-count`).

Remote operational facts:
- **`export MALLOC_ARENA_MAX=2`** is required in the remote/local pool scripts
  (`multinode_remote.sh:41`, `gen_corpus.sh`). Default glibc makes `8*ncpus`
  arenas per worker; on a 64-core box that makes `clear_page_erms` page-zeroing
  the #1 hotspot. Don't remove it.
- `multinode_remote.sh` owns its lifecycle via a pidfile
  (`/tmp/riscv-thoughts-multinode-remote.pid`) so a prior crashed run is cleaned
  up by the next. Its pipeline stderr persists to `/tmp/multinode-remote.log`
  (survives ssh teardown — read it for post-mortems).
- **lz4 is currently removed** from the remote nc transport (it was stripped
  during failure-hunting because failures are byte-count-dependent and removing
  lz4 reproduces them ~5x faster). If you re-add compression, re-add it on both
  ends.
- `binary_stdout()` forces the fd to blocking mode because `tee` (and other
  consumers) flip pipes non-blocking, which would raise BlockingIOError mid-batch
  and truncate the reader (`datagen/__main__.py`, `binary_stdout`).

## Constants / token budget math

- `MAX_INSTR_TOKENS = 9`, `VOCAB_SIZE = 89` (`tokenizer/tokenizer.py`).
- A chunk's token budget: `target_max_tokens = max_chunk_len * 9`
  (`datagen/batch.py:742`), widened to fit `--max-invalid-window` if larger.
- T1's TokenEmbedder needs `--max-window` (default 72 = 8×9) to cover this. T2's
  equivariant core has no position table — it unrolls to each batch's actual
  instruction count, so there is no T2 chunk-length flag.
- N_REGS=32, MAX_INPUT_SLOTS=32, MAX_OUTPUT_SLOTS=16, AUX_CE_IGNORE=-100
  (`datagen/compare.py:92,109-111`). PC_REG=32, MEM_REG=33 are pseudo-slots.

## Where things live / what's tracked

- `corpora/` — on-disk `.rvt` corpora. Present but empty by default; **NOT in
  .gitignore** (only `runs/` and `.claude/` are), yet not committed — `.rvt` files
  are large binary, keep them out of commits.
- `runs/<name>/` — checkpoints + hparams.json + losses.json (gitignored).
- Pipeline regression tests: `scripts/tests/test_pipeline.py` (covers rule
  parsing, format detect/validate, mux modes, the pdeathsig orphan guard, and
  end-to-end 1-step train smokes). Run `python -m pytest scripts/tests`.
- This guide lives at `datagen/SKILL.md` — a tracked path, committed alongside
  the code it documents. Read it directly when touching the pipeline.

## Things this skill does NOT cover (read the code, don't guess)

- Loss math / what each aux head learns, and whether a clean corpus still needs a
  trainer-side validity weight — that's `compressor/train.py` + EXPERIMENT_LOG,
  not datagen. Don't assert training-loss behavior from this file.
- Measured throughput numbers — they drift with machine/arch; measure with
  `datagen -v` rather than quoting a remembered figure
  (CLAUDE.md: no speculative numbers).
