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

1. **The vp evaluator (the value-prediction head, `vp_head`) reads its input
   values from the batch's own anchors, never from a rebuilt set.** Every row's `out_regs` was executed from the
   register file shipped beside it as `batch.anchors`: one structured draw per
   chunk (`draw_anchor_states`, `datagen/compare.py`), placed on that row's own
   input registers by `precompute_chunk`. The trainer stages that field and
   feeds it to `vp_loss` (`compressor/train.py`), which gathers each output's
   input values by register through `batch.out_inputs`; a scorer must do the
   same (`vp_predict`). Re-placing shipped anchors, or building input values
   from any other set of states — a seed, a fixed table — gives wrong targets
   with no error. Score at the fixed held-out set by generating the corpus with
   `datagen --heldout-anchors` (`heldout_anchor_states`,
   `datagen/compare.py`), whose states no training draw produces.

2. **Fixed batch shape, derived from the rule's length cap.** Every emitted batch
   is padded to identical (B, max_tokens, max_n_instrs) so PyTorch's caching
   allocator doesn't fragment. Stochastically-varying batch shape is a real OOM
   trigger we hit (~step 500 of t1_cosine_full). The targets come from
   `max_chunk_len = rule.max_len` (`datagen/batch.py:702-706` inside
   `collect_into_batches`, `datagen/batch.py:653`).
   This is why a `--rule` with no cap is REJECTED (see Rule grammar). Don't add a
   code path that emits a differently-shaped batch mid-run.

3. **EVERY rule ships the out_regs oracle (`OB == B`).**
   `single` (T1) and multi-instruction (T2) rules both go through
   `build_targets` and ship the full post-execution `out_regs` register file,
   the `anchors` it was executed from and `out_inputs` (`OB == B`, carrying
   `n_anchors == --n-states`). There is no per-rule payload split — both tiers
   feed one per-output value-prediction target.
   (`scripts/tests/test_pipeline.py::test_out_regs_mode_header`.)

4. **value_predict is evaluated PER WRITTEN OUTPUT, from the ESSENCE, in the
   register frame, for both tiers.** The target for output slot k is the raw
   value of register `out_slot_regs[k]` in `out_regs`; the evaluator
   (`vp_loss`, `compressor/train.py`) is handed the unit-normalized essence;
   a pointer, which is a one-hot naming which of the chunk's distinct outputs
   is asked about, read from the output register's pointer channels in `T`
   (RISCV_DESIGN, "Value prediction reads the essence"); and, for
   each register in `out_inputs[k]`, that register's value in `anchors` and
   its colour channels from `T` — never the output's own description. No
   input or output is numbered by textual order or by register number:
   `out_inputs` lists the output's external inputs in the encoder's colour
   order, every role ordered (`datagen.signature.output_inputs`), and the
   encoder seeds each leaf from the same colouring computed from the wiring
   it is given (`RegisterStateMachine._prepass`, `compressor/model.py`). The
   oracle's colouring (strings) and the encoder's (fixed random features)
   must agree on structure: identical graph, identical colour. The one place
   they differ by design is commutativity — the oracle's SIGNATURE knows which
   operations commute (`COMMUTATIVE`); the encoder and the input order are
   told nothing of it and order every role. An input order built with the
   commutative roles merged ties the two operands of an `ADD` and breaks the
   tie by register number, so a rename moves values between evaluator
   positions: no error, only a register-dependent gradient on every chunk
   with a commutative op
   (`test_signature.py::test_a_rename_never_moves_an_input_between_gather_positions`).
   `datagen/tests/test_signature.py` is the oracle's evidence against the
   emulator.

   **Which reads count, and liveness, are decided structurally, never at the
   anchors.** Which reads are behavioral is decided per instruction at FIXED
   witness states (`instruction_reads`, `datagen/compare.py`), so a per-row
   anchor draw can never flip a label; `in_slot_regs` (the binding target)
   lists them in first-read order, a dead instruction's included
   (`_input_order`). A dead instruction — overwritten before anyone reads it —
   is pruned from the colouring by a backward pass from the final writers, in
   the oracle (`datagen.signature.live_nodes`) and in the encoder alike, so
   its reads leave no trace in any colour; they remain in the binding.

5. **x0 is fixed under relabeling.** `random_perm`/`relabel` require `perm[0]==0`
   (`datagen/generate.py:313,327`; the check is `datagen/generate.py:335-336`).
   x0 is hardwired zero, not a register.

6. **Memory-op chunks get no aux (V1 scope).** `build_targets`
   (`datagen/batch.py:578`) skips chunks containing LOAD/STORE because their
   `Precomputed` is unavailable — `precompute_chunk` raises
   NotImplementedError on memory ops (`datagen/compare.py:674-675`). The
   chunk's tokens are still shipped (so the encoder sees them for
   magnitude/validity), but its aux row is zero-mask / `AUX_CE_IGNORE` and its
   value-prediction row is masked off via `out_regs_valid=False`
   (`_out_regs_from_precomputeds`, `datagen/batch.py:387`).

7. **The behavioral distance compares two rows at the SAME inputs, placed by
   colour order.** Every row is executed at one shared set of witness states,
   its i-th input in `out_inputs` order taking the state's position i+1
   (`witness_outputs`, `compressor/train.py`), so two rows' outputs are
   compared at equal values on corresponding inputs. Comparing each row at its
   own anchor draw, or placing values by register number, gives a target that
   is nonzero for a register rename and meaningless across rows — no error,
   only a collapse objective pulling the wrong pairs. `max_dist` comes from
   the shipped known bits (`out_known_zero` / `out_known_one`, from
   `datagen/knownbits.py`), never from observed values.
   (`compressor/tests/test_train.py::test_witness_outputs_match_the_cpu_emulator`,
   `::test_pair_distance_targets_match_the_reference_distance`.)

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
