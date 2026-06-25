---
name: datagen
description: >
  Reference for the RVT data-generation pipeline — the datagen/ package and the
  scripts/ stream tools (gen_batches, mux_batches, batch_slice, bench_throughput,
  multinode_gen.sh, gen_corpus.sh, multinode_remote.sh). Load this BEFORE
  generating a corpus, editing datagen/ or any pipeline script, changing the RVT
  format, debugging worker/throughput/OOM issues, or wiring gen→train. Carries
  the invariants that are silent-corruption traps if violated (anchor-seed match,
  fixed batch shape, raw-vs-canonical anchors, orphan workers).
---

# RVT data-generation pipeline

The pipeline turns a random RV32I instruction stream into binary training
batches consumed over stdin by the trainers. One tool, one format.

```
gen_batches.py        random instrs → chunks → twins/aux → packed Batch → stdout (RVT bytes)
mux_batches.py        spawn N gen_batches and/or read files → interleave → stdout
batch_slice.py        slice/inspect an RVT stream (--info/--count/--skip/--tail)
bench_throughput.py   terminal stage: count batch/s + MB/s
train_encoder.py      T1 trainer (reads RVT on stdin)
train_t2_encoder.py   T2 trainer (reads RVT on stdin, needs a frozen T1)
```

Every stage is a UNIX filter over the **RVT binary stream**. Stages talk only
through that stream. `print()` inside the stream tools is redirected to stderr
(`binary_stdout()` in `scripts/_streamfmt.py:106`) so stdout stays pure data.

## ALWAYS mux. A single generator process almost never feeds the GPU.

Generation+analysis (SSA, anchor execution, twins, value-prediction precompute)
is CPU-bound Python, and `gen_batches` is single-threaded — one process pins one
core. The GPU is far faster than one core can supply, so a bare
`gen_batches | train_*` pipeline leaves the trainer starved and the GPU idle most
of the step. This is the default failure mode, not an edge case.

**Rule: for any live gen→train run, drive the trainer through `mux_batches` with
`--gen-count ≥ 8` (16 is common; scale toward core count).** Confirm the GPU is
actually the bottleneck before settling — if `bench_throughput.py` (or the
trainer's step time) shows the trainer waiting, add workers. Single-process
`gen_batches` is only appropriate for tiny smoke tests, deterministic one-batch
debugging, or writing a small fixture to disk — never to feed real training.

(For an already-materialized on-disk corpus, `cat corpus.rvt | train_*` reads at
disk speed and needs no workers — the multi-worker rule is about *live*
generation.)

## Canonical commands (verified against README.md — copy these, don't improvise)

T1 (single-instruction encoder). **The bare single-process form below is the
README's illustrative minimal example — fine for a smoke test, but for a real
run wrap generation in `mux_batches --gen-count 8/16` exactly like the T2 example
(see "ALWAYS mux" above); one core will not keep the GPU fed.**
```bash
python scripts/gen_batches.py --rule single --twins 2 --batch-size 128 \
    --n-batches 250000 --n-states 8 --anchor-seed 0 \
    --config configs/instr_default.json \
  | python scripts/train_encoder.py --d-model 128 --n-heads 4 --n-layers 2 \
    --d-out 64 --max-window 32 --n-steps 250000 --out-dir t1_run
```
Real T1 run — mux'd generation (mirror of the T2 invocation, `--rule single`):
```bash
python scripts/mux_batches.py --gen-count 16 --mode fifo --seed 0 -- \
    --rule single --twins 2 --batch-size 128 --n-batches 250000 \
    --n-states 8 --anchor-seed 0 --config configs/instr_default.json \
  | python scripts/train_encoder.py --d-model 128 --n-heads 4 --n-layers 2 \
    --d-out 64 --max-window 32 --n-steps 250000 --out-dir t1_run
```

T2 (chunk encoder on a frozen T1) — oversupply workers, cap with `batch_slice`:
```bash
python scripts/mux_batches.py --gen-count 8 --mode fifo --seed 0 -- \
    --rule "branch+cap=4" --twins 3 --batch-size 128 --n-batches 2000000 \
    --n-states 8 --anchor-seed 0 --config configs/instr_t2.json \
  | python scripts/batch_slice.py --count 2000000 \
  | python scripts/train_t2_encoder.py --t1-model runs/t1_run/encoder.pt \
    --d-out 512 --d-event 16 --route binding \
    --n-steps 2000000 --anchor-seed 0 --n-anchor-states 8 --out-dir t2_run
```
Note: with `--gen-count 8` and per-worker `--n-batches 2000000`, the pool can
emit up to 16M batches; `batch_slice --count 2000000` bounds what the trainer
sees. When slice stops, mux gets BrokenPipe and reaps the workers (see Orphan
workers below).

`--n-steps` only shapes the cosine-LR schedule and the ETA display — it is NOT a
hard step cap. Training runs until stdin EOF (`scripts/_common.py:33`). Bound the
batch count upstream with `batch_slice --count N` or the generator's `-n`.

## INVARIANTS — violating these silently corrupts data or training

1. **anchor-seed AND n-states must match gen↔train.** value_predict supervision
   is computed against anchor states reconstructed from `--anchor-seed` +
   `--n-anchor-states` (`scripts/train_encoder.py:61-66`, `train_t2_encoder.py:81-84`).
   The generator's `--anchor-seed`/`--n-states` must equal the trainer's
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

## gen_batches.py reference

`--rule` grammar (`scripts/gen_batches.py:67`, rules in `datagen/generate.py:628`):
- `single` — one instruction per chunk (T1). `max_len=1`. Ships canonical
  out_regs like every other rule (the T1/T2 unification).
- `cap=N` — fixed length N.
- `branch+cap=N` — terminate at a branch OR length N.
- `transform+cap=N` — terminate at a memory-op / branch / jump OR length N.
- Components join with `+` and compose via `either()` (terminate when ANY fires).
- **A bare `branch` or `transform` (no `cap=`) is rejected eagerly** — it can
  build an unbounded chunk. Error says "unbounded … add a cap". This is enforced
  in `_parse_rule` AND in `collect_groups` via `rule.max_len is None`.

Key flags: `--twins T` (relabeled equivalents per valid source chunk),
`--batch-size B` (source rows; twins extend the batch beyond B),
`--n-states` (anchor count), `--anchor-seed`, `--inject-invalid R`,
`--inject-equiv R`, `--config PATH`, `-n/--n-batches` (required), `--seed`.
CLI flags override the JSON config's `equivalences.rate` / `invalidity.rate`.

Per-batch stderr timing (`compute_ms`/`write_ms`) is logged per worker pid.
BrokenPipeError on shutdown is benign and swallowed (`gen_batches.py:214`).

## mux_batches.py reference

Spawns gen_batches workers and/or reads file inputs, interleaves into one stream.

- **Everything after `--` is forwarded verbatim to each spawned worker.** mux only
  appends a per-worker `--seed (base + i)` so workers diverge
  (`mux_batches.py:230`). Worker config is NOT re-declared in mux.
- `--gen-count N` spawns N workers. File inputs: `path` or `WEIGHT:path`.
- Modes (`--mode`): `fifo` (as-ready; default), `round-robin` (one batch each),
  `weighted` (auto-selected if any `W:path` weight ≠ 1.0). Weights bias ORDER,
  not total count — every input drains fully.
- Worker stderr is inherited unconditionally so worker crashes/tracebacks are
  visible (`mux_batches.py:233`).

### Orphan workers — the project's #1 recurring hazard (see CLAUDE.md)
gen_batches workers are spawned via `subprocess.Popen` and **do NOT die when their
parent dies** — they reparent to init and survive until they hit a write to a
closed pipe, which can be minutes if mid-compute. mux's normal exit path reaps
them (`terminate()` + `wait()`, `mux_batches.py:271`), and the regression test
`test_no_orphan_workers_after_exit` guards it. But after ANY manual kill of a
mux/multinode pipeline, **explicitly verify and kill survivors by PID**:
```bash
pgrep -af scripts/gen_batches.py        # list survivors
kill -9 <pid> <pid> ...                 # kill each listed PID
```
`pkill -9 -f scripts/gen_batches.py` sometimes silently fails under the
permission system — prefer the explicit pgrep→kill form. Do this on the remote
(odin) too after killing a multinode run.

## RVT binary format (datagen/batch.py)

- **Version 8.** `RVT_FORMAT` in `datagen/batch.py`. A version bump is a hard
  break — readers reject mismatched versions and tell you to regenerate (older
  streams won't read).
- **No stream header — every batch is self-describing** (4-byte magic `RVT\x00` +
  1-byte version + dtype signature, then a 5-uint32 dimension header, then the
  body). This is the load-bearing property: **RVT streams concatenate with `cat`.**
  - Combine corpora: `cat a.rvt b.rvt > both.rvt`
  - **Second epoch = `cat corpus.rvt corpus.rvt | trainer`.** There is no built-in
    repeat; epochs are shell-level `cat`.
- Header fields: `B, max_tokens, max_n_instrs, n_anchors, OB`. `OB` is the
  out_regs row count (`B` when the oracle is shipped — always, now — else 0).
  Body field list + dtypes are in the `BodyField(...)` block in
  `datagen/batch.py`; header validation bounds in `scripts/_streamfmt.py`.
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

- **`multinode_gen.sh BATCHES_PER_WORKER`** — ships datagen sources to `odin` over
  ssh (tar stream), runs a remote mux pool that streams RVT back over `nc` (port
  6464), and joins it with a local mux pool. Output to stdout/file or piped to a
  trainer. 104 total workers = 96 remote + 8 local by default (hardcoded to the
  machines). `REMOTE=chrissarbora@odin`.
- **`gen_corpus.sh TOTAL_BATCHES OUT_PATH`** — writes an on-disk corpus. **Despite
  its header comment mentioning "remote", the script itself runs `mux_batches`
  LOCALLY** (`gen_corpus.sh:64`) and writes `OUT` on the local disk. "Remote" =
  the *intended usage* is to ssh into odin and run it there; there is no
  auto-shipping. `TOTAL` must be divisible by `WORKERS` (default 100). Defaults
  to T2 settings (`branch+cap=8`, `instr_t2.json`, no invalid injection).

Both override knobs via env vars: `BATCH_SIZE TWINS N_STATES RULE CONFIG
INJECT_INVALID INJECT_EQUIV` (and `WORKERS` for gen_corpus).

Remote operational facts:
- **`export MALLOC_ARENA_MAX=2`** is required in the remote/local pool scripts
  (`multinode_remote.sh:41`, `gen_corpus.sh:57`). Default glibc makes `8*ncpus`
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
  and truncate the reader (`_streamfmt.py:106`).

## Inspecting / debugging a corpus

```bash
python scripts/batch_slice.py --info corpus.rvt          # count batches/items, batch size
python scripts/batch_slice.py --count 100 corpus.rvt > head.rvt
python scripts/batch_slice.py --skip 50 --count 10 corpus.rvt
python scripts/batch_slice.py --tail 5 corpus.rvt
cat corpus.rvt | python scripts/bench_throughput.py --log-every-sec 5
```
`--lenient` absorbs a truncated tail (useful on a partially-written file).

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
  parsing, format detect/validate, mux modes, the orphan-worker guard, and
  end-to-end 1-step train smokes). Run `python -m pytest scripts/tests`.
- This guide lives at `datagen/SKILL.md` — a tracked path, committed alongside
  the code it documents. Read it directly when touching the pipeline.

## Things this skill does NOT cover (read the code, don't guess)

- Loss math / what each aux head learns, and whether a clean corpus still needs a
  trainer-side validity weight — that's `compressor/train.py` + EXPERIMENT_LOG,
  not datagen. Don't assert training-loss behavior from this file.
- Measured throughput numbers — they drift with machine/arch; measure with
  `bench_throughput.py` rather than quoting a remembered figure
  (CLAUDE.md: no speculative numbers).
