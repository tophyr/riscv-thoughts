# Bootstrapping a session

If you cannot already state the current mission and why it is the mission, you
are not oriented yet: **run the `bootstrap` skill** before touching anything. It
walks the doc reading order, the code checks that keep the docs honest, and the
questions to bring back to me before starting. I would much rather spend time
clarifying up front than have you build on a wrong premise.

# Verification Protocol

Failure modes I (Claude) have repeatedly fallen into on this project.
These rules exist to interrupt them. They take precedence over speed
or appearing decisive.

## When asked "why" or "what" about the code

Read the code first. Cite `path:line` for any claim about what the
code does. If I can't cite a line, I haven't verified — don't assert.

"I don't know — let me check" is a complete first response. Pattern-
matched plausible answers delivered before reading code are how I
generate confident-but-wrong assertions. Stop generating, start reading.

## When the user pushes back on a claim

Pushback means *go read code*, not *construct a defense*. If my next
sentence after a challenge starts with "the reason is…" or "well, X
because Y", that's the failure mode firing in real time. Switch to
verification: name the file I need to check, read it, then respond
with what I actually found.

The first hypothesis I generate is the one I should distrust most.
Generic priors (NAT timeout, OOM, network blip, "tests need
determinism") feel reasonable without being true for *this* system.
Before committing to one: list the evidence it would predict, check
whether the evidence is actually present.

## When fixing a bug

Three steps, in order:

1. **Isolate the bug with evidence** — a test, stack trace, or
   specific input that triggers it. State what evidence I have.
2. **Make the minimal fix** that addresses the root cause.
3. **Verify the fix** on the same evidence that demonstrated the bug.

Bundling speculative additional fixes ("while I'm here, also…")
is how regressions sneak in. Each fix is a separate decision with
its own evidence. If I think something else looks broken, surface
it as a separate observation — don't silently add it to the diff.

## When something is "done"

A tool, script, or fix isn't done until it has been *executed
against real input* and the output verified. "I wrote a monitor
that should detect X" is not the same as "I confirmed the monitor
fires when X happens." If I haven't exercised the path, label it
"untested" explicitly. Today's broken-monitor incident: I shipped
a monitor with a `grep -c` bug that silently returned empty on
binary-tagged log files. Never tested. Missed every event for
hours.

## When recognizing a previously-diagnosed bug

If I diagnosed a failure mode earlier in the same session, and the same
symptoms reappear, it's the same bug. Apply the previous remediation.
Don't re-diagnose from scratch and don't forget to clean up. (The
canonical example — generation workers orphan-surviving their parent —
is now fixed at the OS level; see the next section.)

### Formerly-recurring trap: orphan generation workers (now fixed)

History: `gen_batches.py` workers spawned by `mux_batches.py` via
`subprocess.Popen` did NOT die when their parent died — they reparented
to init and ate CPU for hours. I left orphans running multiple times
even after diagnosing the pattern.

RESOLVED: both scripts are merged into the `datagen` package itself —
the tool is now `python -m datagen` (`datagen/__main__.py`), whose
generation workers are forked `multiprocessing.Process` children that
arm `PR_SET_PDEATHSIG(SIGKILL)`. The kernel now SIGKILLs every worker
the instant the parent dies — including on `kill -9` of the parent.
The `test_no_orphan_workers_after_kill` regression guards it.

So the manual pgrep→kill ritual is no longer required. Backstop only:
if you ever suspect a straggler after killing a `datagen` / multinode
pipeline, list workers by parentage (they share the parent's argv, so a
cmdline scan won't find them):
`pgrep -P <datagen_pid>`, then `kill -9 <pids>`. Check the remote (odin)
too after killing a multinode run.

## When making a refactor or vectorize change

Old and new must agree on the math (see
`feedback_refactor_means_same_math.md`). Compute a side-by-side
sample on the same inputs; confirm bit-equivalence or
correlation = 1.0 for floats. If they don't match, the refactor
silently changed semantics — stop and flag, don't ship.

## When choosing defaults / parameters

Before defending a default value, check who actually depends on it.
Today: I claimed `--seed=42` defaults were needed "for tests"
without checking whether tests use them (they don't — tests build
their own RNG inline). Anchor-seed sharing across workers I claimed
was needed "for distance comparability" — also wrong; distances are
self-contained per pair. Always grep for callers before defending a
value.

## Data generation pipeline

Nearly every task here touches datagen. The pipeline guide — canonical commands,
the flag reference, the RVT wire format, configs, multinode operation — is the
**`datagen` skill**. Invoke it before editing `datagen/`, the `scripts/` stream
tools, the RVT format, or the multinode generators, and before running any
generation or training pipeline. Do not improvise a datagen command from memory;
copy the skill's canonical commands.

The invariants below stay here, always loaded, because violating one silently
corrupts data or training with no error — you cannot opt into a warning you do
not know you need.

### The invariants

1. **anchor-seed AND n-states must match gen↔train.** value_predict supervision
   is computed against anchor states reconstructed from `--anchor-seed` +
   `--n-anchor-states` (`scripts/train_encoder.py:71-76`,
   `scripts/train_t2_encoder.py:118-121`, both funnelled into
   `make_anchor_states` at `compressor/train.py:603`).
   `datagen`'s `--anchor-seed`/`--n-states` must equal the trainer's
   `--anchor-seed`/`--n-anchor-states`. Mismatch → wrong targets, no error.
   `make_anchor_states` is `datagen/compare.py:319`. (Anchor seed only needs to
   match gen↔train; it is *not* needed for cross-worker comparability — distances
   are self-contained per row.)

2. **Fixed batch shape, derived from the rule's length cap.** Every emitted batch
   is padded to identical (B, max_tokens, max_n_instrs) so PyTorch's caching
   allocator doesn't fragment. Stochastically-varying batch shape is a real OOM
   trigger we hit (~step 500 of t1_cosine_full). The targets come from
   `max_chunk_len = rule.max_len` (`datagen/batch.py:648-658` inside
   `collect_into_batches`, `datagen/batch.py:615`).
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
   i-th live syntactic input to canonical position i+1 (`_canonical_state`,
   `datagen/compare.py:331`, positions `_CANON_POSITIONS = range(1, 32)` at
   `datagen/compare.py:84`; applied at `datagen/compare.py:610`) and
   executes there, so `out_regs[:, r]` depends only on operand structure. The vp
   loss (`t2_value_predict_loss`, `compressor/train.py:1063`, used by BOTH T1 and
   T2) therefore feeds the head
   the canonical-position values `anchor[:, k+1]` for input slot k — never the raw
   value at the slot's actual register. Feeding raw input values against the
   canonical target is mathematically inconsistent — the head must see canonical
   inputs to match the canonical output.
   (Known accepted caveat, documented at `compressor/train.py:1111-1114`:
   canonical positions are assigned over ALL live syntactic inputs while
   `in_slot_regs` lists only the *behavioral* subset, so slot k ↔ position k+1
   can drift for a degenerate op with a live-but-value-irrelevant input, e.g.
   `ANDI _,_,0`. Rare; accepted, not a bug to "fix" casually.)

5. **x0 is fixed under relabeling.** `random_perm`/`relabel` require `perm[0]==0`
   (`datagen/generate.py:287,301`; the check is `datagen/generate.py:309-310`).
   x0 is hardwired zero, not a register.

6. **Memory-op chunks get no twins and no aux (V1 scope).** `build_twins`
   (`datagen/batch.py:528`) drops chunks containing LOAD/STORE from twin+aux
   construction because their `Precomputed` is
   unavailable — `precompute_chunk` raises NotImplementedError on memory ops
   (`datagen/compare.py:585,593`). The chunk's tokens are still shipped (so the
   encoder sees them for magnitude/validity), but its aux row is zero-mask /
   `AUX_CE_IGNORE` and its value-prediction row is masked off via
   `out_regs_valid=False` (`_out_regs_from_precomputeds`, `datagen/batch.py:363`).

## Anchored memory

These prior-session memory files apply without requiring reload.
When a current question maps to one, name it explicitly — the
connection is what makes the lesson land:

- `feedback_being_wrong_is_worst.md` — verify before asserting
- `feedback_think_before_acting.md` — discuss tradeoffs first
- `feedback_no_speculative_numbers.md` — measure, don't guess
- `feedback_refactor_means_same_math.md` — math invariance
- `feedback_correct_dont_obey.md` — surface the right answer over
  restructuring
- `feedback_experiment_over_assume.md` — push back as a colleague,
  don't echo
- `feedback_commit_at_boundaries.md` — wait for natural commit points
- `feedback_no_storytime_comments.md` — comments state WHY, briefly
- `feedback_confirm_gpu_saturation.md` — never leave a long run at low
  util; fix the throughput bottleneck first, don't defer it
- `feedback_no_vestigial_options.md` — delete the worse way once a
  better one is settled; verify new-vs-old rather than keeping a flag
