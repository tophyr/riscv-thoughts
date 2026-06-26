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

Nearly every task in this project touches datagen, so the pipeline guide
is imported in full below. It carries the silent-corruption invariants
(anchor-seed match, fixed batch shape, raw-vs-canonical anchors) and the
orphan-worker remediation — apply them when editing `datagen/`, the
`scripts/` stream tools, the RVT format, or the multinode generators.

@datagen/SKILL.md

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
