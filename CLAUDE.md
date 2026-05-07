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

If I diagnosed a failure mode earlier in the same session (e.g.,
gen_batches workers orphan-survive their parent and need explicit
cleanup), and the same symptoms reappear, it's the same bug. Apply
the previous remediation. Don't re-diagnose from scratch and don't
forget to clean up.

### Specific recurring trap: orphan gen_batches workers

`gen_batches.py` workers spawned by `mux_batches.py` (via
`subprocess.Popen`) DO NOT die when their parent dies. They reparent
to init and stay alive until they hit a write to a closed pipe — which
can be many minutes if they're mid-compute on a slow pair. Killing the
multinode_gen pipeline (or any of its descendants) leaves them as
zombies that eat CPU forever.

After ANY kill of multinode_gen / mux_batches, *always* explicitly
kill remaining gen_batches by PID. `pkill -9 -f "scripts/gen_batches.py"`
sometimes silently fails (under Claude Code's permission system or
otherwise). Reliable form: pgrep -af gen_batches, then `kill -9 <pids>`
listing each PID. Also do this on the remote via ssh.

Today (2026-05-06/07) I left orphan workers running for hours at least
three times. Each time I had already diagnosed the pattern. Re-read this
section before the next kill.

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
