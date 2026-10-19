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
Don't re-diagnose from scratch and don't forget to clean up.

Backstop for a suspected straggler worker after killing a `datagen` or
multinode pipeline: workers share the parent's argv, so a cmdline scan
will not find them — list them by parentage with `pgrep -P <datagen_pid>`,
then `kill -9`. Check the remote (odin) too. They arm
`PR_SET_PDEATHSIG(SIGKILL)` so this should never be needed (the `datagen`
skill has the mechanism).

## When making a refactor or vectorize change

Old and new must agree on the math (see
`feedback_refactor_means_same_math.md`). Compute a side-by-side
sample on the same inputs; confirm bit-equivalence or
correlation = 1.0 for floats. If they don't match, the refactor
silently changed semantics — stop and flag, don't ship.

**Verify against a real checkpoint, not just the test suite.** No
test loads one, so the suite cannot catch a `state_dict`
incompatibility: adding an unconditional `rank_emb` passed the whole
suite and failed at step 0 of the next run against the frozen T1.

Breaking changes are acceptable — including reminting the frozen
T1 — on the one condition that old and new can be compared and
shown not to have regressed. That condition is what licenses
deleting the worse way instead of keeping it behind a flag
(`feedback_no_vestigial_options.md`).

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

1. **The vp head reads its input values from the batch's own anchors, never
   from a rebuilt set.** Every row's `out_regs` was executed at anchor states
   drawn for that row (one draw per source chunk, shared by its twins) and
   shipped beside it as `batch.anchors` (`build_twins`, `datagen/batch.py:540`;
   the draw is `draw_anchor_states`, `datagen/compare.py:315`). The trainer
   stages that field and feeds it to `t2_value_predict_loss`
   (`compressor/train.py:1215`); a scorer must do the same. Building input values
   from any other set of states — a seed, a fixed table — gives wrong targets
   with no error. Score at the fixed held-out set by generating the corpus with
   `datagen --heldout-anchors` (`heldout_anchor_states`,
   `datagen/compare.py:378`), whose states no training draw produces.

2. **Fixed batch shape, derived from the rule's length cap.** Every emitted batch
   is padded to identical (B, max_tokens, max_n_instrs) so PyTorch's caching
   allocator doesn't fragment. Stochastically-varying batch shape is a real OOM
   trigger we hit (~step 500 of t1_cosine_full). The targets come from
   `max_chunk_len = rule.max_len` (`datagen/batch.py:685-693` inside
   `collect_into_batches`, `datagen/batch.py:648`).
   This is why a `--rule` with no cap is REJECTED (see Rule grammar). Don't add a
   code path that emits a differently-shaped batch mid-run.

3. **EVERY rule ships the CANONICAL out_regs oracle (`OB == B`).**
   `single` (T1) and multi-instruction (T2) rules both go through `build_twins`
   and ship the full canonical `out_regs` register file (`OB == B`, carrying
   `n_anchors == --n-states`). There is no per-rule payload split — both tiers
   feed one canonical value-prediction target.
   (`scripts/tests/test_pipeline.py::test_out_regs_mode_header`.)

4. **value_predict is CANONICAL for both tiers, in the BEHAVIORAL frame.**
   `out_regs` is the rename-invariant canonical baseline: `precompute_chunk`
   relocates the chunk's i-th **behavioral** input to canonical position i+1
   (`_canonical_state`, `datagen/compare.py:386`, positions
   `_CANON_POSITIONS = range(1, 32)` at `datagen/compare.py:80`; applied at
   `datagen/compare.py:759`) and executes there, so `out_regs[:, r]` depends only
   on operand structure. The vp loss (`t2_value_predict_loss`,
   `compressor/train.py:1215`, used by BOTH T1 and T2) therefore feeds the head
   the canonical-position values `anchor[:, k+1]` for input slot k — never the raw
   value at the slot's actual register. Feeding raw input values against the
   canonical target is mathematically inconsistent — the head must see canonical
   inputs to match the canonical output.

   **The frame must be behavioral, not syntactic, and the same one everywhere.**
   `in_slot_regs` lists the behavioral inputs, so a canonical position spent on a
   live-but-value-irrelevant read (e.g. `ANDI _,_,0`, or `XOR rd,rs,rs`) shifts
   every later input off its slot. The encoder is never given such a read — T1's
   binding heads are behavioral and correctly omit it — so the target would
   depend on something the encoder cannot see, and two chunks with a
   bit-identical encoder input would carry different targets. Which reads are
   behavioral is decided per instruction at FIXED witness states
   (`instruction_reads`, `datagen/compare.py:612`), never at the anchors: per-row
   anchors would otherwise flip a label from one row to the next. The chunk's
   inputs are those reads in first-read order, a dead instruction's included
   (`_input_order`, `datagen/compare.py:638`): T1 emits them and T2's read ranks
   number them, so skipping them would shift every later input off its rank.
   The regression guards are `test_canonical_positions_go_to_behavioral_inputs`,
   `test_a_dead_instructions_reads_are_inputs` and
   `test_labels_do_not_depend_on_the_anchor_draw`.

5. **x0 is fixed under relabeling.** `random_perm`/`relabel` require `perm[0]==0`
   (`datagen/generate.py:298,312`; the check is `datagen/generate.py:320-321`).
   x0 is hardwired zero, not a register.

6. **Memory-op chunks get no twins and no aux (V1 scope).** `build_twins`
   (`datagen/batch.py:540`) drops chunks containing LOAD/STORE from twin+aux
   construction because their `Precomputed` is
   unavailable — `precompute_chunk` raises NotImplementedError on memory ops
   (`datagen/compare.py:724-725`). The chunk's tokens are still shipped (so the
   encoder sees them for magnitude/validity), but its aux row is zero-mask /
   `AUX_CE_IGNORE` and its value-prediction row is masked off via
   `out_regs_valid=False` (`_out_regs_from_precomputeds`, `datagen/batch.py:368`).

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
- `feedback_no_storytime_comments.md` — docs are expository for a
  brand-new reader; audit against the four named categories, don't
  just intend to "write status". Comments: WHY, in 1-2 lines.
  The **`document` skill** carries this standard in full, plus the
  procedure around it — which document holds what, correcting a claim
  in place, and the mechanical checks for narrative markers,
  cross-references and scope conditions. Invoke it before any
  documentation pass and before a commit that touches a `.md` file.
  A commit is a WAYPOINT: the message states the differential from its
  parent, never the route taken to reach it (skill section 6)
- `feedback_confirm_gpu_saturation.md` — never leave a long run at low
  util; fix the throughput bottleneck first, don't defer it
- `feedback_no_vestigial_options.md` — delete the worse way once a
  better one is settled; verify new-vs-old rather than keeping a flag
