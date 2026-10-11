---
name: document
description: Write or revise this project's prose documentation (README, RISCV_DESIGN, EXPERIMENT_LOG, TODO, skills) and its code comments so they read as current truth to someone who has never seen the repo. Load before any documentation pass, after a result changes what a document claims, and before a commit that touches a .md file. Covers routing content to the right document, correcting claims in place, what a commit message owes its reader, and audits A-G — deictic references, undefined comparatives, jargon, structural placement, narrative markers, cross-references, and scope conditions on numbers.
---

# Documenting

**The standard: every document introduces a COMPLETELY NEW reader to the
project.** Not a reader who followed the session, not the person who asked for
the work, not a future you with context. If a sentence only parses for someone
who watched the work happen, it is broken — even if every fact in it is true.

Narrative and context-dependent writing is the default failure mode here, and it
has had to be corrected repeatedly. **Intent is not enough.** What works is a
deliberate, methodical audit against named categories, run as a separate pass
after writing. The vague goal "write status, not narrative" has failed every
time it has been tried on its own.

Two pressures drive everything below, and they pull in opposite directions:

- The reader **does not want the history**. They did not watch the work happen
  and do not care what anyone believed last week. A correction appended beside a
  stale claim makes them reconstruct a sequence to find the current truth.
- The reader **does not know the vocabulary**. A term whose first appearance is
  a *reference* rather than an *introduction* is unreadable, even when it is
  defined correctly three sections later or in another file.

## 1. Decide whether anything needs to change

A documentation pass is a judgment, not a mandate. Editing a document that is
already correct adds churn and risks introducing narrative where none was.

Ask what changed in the world since the document was written: a measurement, a
decision, a deleted flag, a renumbered list. If nothing did, say so and stop.

## 2. Route the content

Each document has a stated purpose in its own header — re-read it and honor it.
In this project:

| document | holds | does not hold |
|---|---|---|
| `README.md` | the map, the reading order, architectural intent | measurements, flag-level detail |
| `WHAT_IS_A_THOUGHT.md` | the theory being tested | anything RISC-V-specific |
| `RISCV_DESIGN.md` | the architecture and the **durable rationale** for its shape | run results, roadmap |
| `EXPERIMENT_LOG/` | findings, the evidence for them, and dead ends | forward plans |
| `METRICS.md` | what a measurement means and certifies, its noise floor, the scope conditions on a number from it | measured values |
| `TODO.md` | the mission, the milestones, the open tasks and the decisions they need | history, evidence, metric semantics, how to run anything |
| `probes/README.md` | which diagnostic answers which question | how to read one's output (that is its docstring) |
| `.claude/skills/` | procedures, commands and operational hazards for a task | findings |
| commit messages | the **differential** — what is true at this commit that was not true at its parent | the route taken between the two |

A result that changes an architectural claim usually needs edits in two places:
the finding in the experiment log, and the rationale in the design doc. Changing
only one leaves the other asserting something false.

## 3. Write it as the truth, not as a change

Rewrite the stale claim so it is true. **Never append a correction beside it**,
and never narrate that a claim moved — no "used to be X, now Y", "has now been
run", "the required comparison is DONE", "X was a correction, not original
behaviour", "turned out", "this is not a regression", "the attempt failed", "the
surprise". A reader who never held the old belief has nothing to correct.

The test: delete every sentence that only exists because someone once believed
something else. If the document still says what is true, those sentences were
narrative.

**Numbered findings lists invite appending, and appending is how narrative gets
in even when each entry reads as status.** When a new result changes an old
finding, REWRITE the old finding. Watch for a file where entry 12 and entry 17
contradict each other and the reader must replay the sequence to work out which
one holds.

Two things that look like history but are not, and should stay:

- **Operational history** — checkpoint migrations, a hazard someone may repeat,
  a ritual another document still describes. Keep it, phrased as what to do.
- **Scope conditions** — the population a number was measured on. That is part
  of the claim, not a story about it. See audit G.

## 4. The audits

Run each as a separate pass over the finished text. Audits A–D are sentence
level; E–G are document level and are mechanically checkable.

### A. Deictic references

Sentences pointing at something established in conversation rather than on the
page. "This is the adopted seeding." "That fixes it." "The approach above."

Test: read the sentence alone. If it does not stand up, name the thing.

### B. Undefined comparatives

"the baseline", "current best", "the adopted config", "supersedes X", "better
than before". These name a session artifact the reader never saw.

Replace with a description of what the thing *is*: "an encoder seeded from random
tags with no read-order in its value channels".

### C. Jargon — identifiers, and first appearances

Two failures, both about a term the reader cannot resolve:

- **Identifiers carrying the meaning.** Flags, run directory names, code symbols
  (`content`, `--pool-extra write_time`), probe column names (`s1 overlap`),
  internal shorthand ("slot B") used as if self-explanatory. The prose must state
  the mechanism; the identifier is a pointer to it, never a substitute.
- **First appearance must be an introduction.** Find where the reader meets each
  term for the first time and check that place defines it. Expand every acronym
  there, **including in section headings** — a term can appear in a heading, a
  constraint name and a table and still be expanded nowhere.

**Tables are where undefined vocabulary hides.** Column headers and cell values
escape prose review because they are not sentences. Every table needs its
columns and its categorical values defined immediately above it, along with the
unit being counted.

**Respect the reading order.** A document read late may use terms an earlier
document introduces. It may not use terms invented in a probe's output, a run
directory name, or a code symbol.

```bash
# For each suspected term: where does it first appear, and is that an introduction?
grep -n "cross-instr\|err mass\|GVN\|canonical frame" README.md TODO.md RISCV_DESIGN.md EXPERIMENT_LOG/*.md | head
```

### D. Structural placement

Content in a document whose stated purpose excludes it — a `[DONE]` section in a
forward-only TODO, results in a design-rationale doc, method in a numbered
results list, history in a file whose header says it holds none. Re-read the
document's own header, and the routing table above, and honor them.

### E. Narrative markers

```bash
grep -nE "used to |turned out|it was believed|we (thought|missed|had)|the attempt|previously|no longer|not a regression|retract|cautionary|the surprise|now (correctly|properly)" *.md EXPERIMENT_LOG/*.md
```

Every hit is a candidate, not a defect — judge each. Operational history and
scope conditions legitimately survive this grep.

### F. Cross-references resolve

Rewrites and renumbering break references silently. Check mechanically, not by
eye — a reference to "finding 17's fourth bullet" survives a rewrite of finding
17 that leaves it with three.

```bash
for n in $(grep -ohE "finding [0-9]+" *.md EXPERIMENT_LOG/*.md | grep -oE "[0-9]+$" | sort -un); do
  grep -q "^\*\*$n\." EXPERIMENT_LOG/*.md || echo "MISSING finding $n"
done
```

Repeat per numbering scheme in play (findings, tasks, milestones). Also check
references to *parts* of a target — "the fourth bullet", "the table above" —
which resolve to a document but not to the thing named.

### G. Every number carries its scope

A number recorded without the conditions it was measured under will outlive
them, and the next reader will quote it as a property of the system. This is the
project's most expensive recurring failure: two separate claims — that a tier
boundary was lossless, and that operand binding was solved — were each true of
the corpus and configuration they were measured on, and false of the system.

For every figure in the text, state the corpus, the model, and the setting it
depends on. If a change would invalidate it, say which change.

## 5. Code comments

Same standard, tighter budget. **1–2 lines, the WHY only.** No debugging
journey, no symptom chain, no which-run-hit-it, no record that the code used to
be different. If it will not compress to two lines, it belongs in the commit
message — or, if it is a result rather than a reason, in `EXPERIMENT_LOG/`.

## 6. Commit messages

A commit is a **waypoint**. What the message owes its reader is the difference
between this waypoint and the last one: what is true now that was not true
before, stated as fact. The route between two waypoints is not the subject. If
the work looped forty-seven times and chased its tail getting here, none of that
changes the difference the commit records, and none of it belongs in the
message.

**Storytelling about the differential is legitimate** — sometimes it is the
clearest way to write one. A deletion that follows from a refutation, a number
that moved because a target changed, a defect and the fix it forced: each is a
story about *the difference between two states*, and each is fair game. Reach
for the plain current-state description first, and keep the story only when it
says something the flat description does not.

**Cut the path taken.** No sequence of what was believed en route, no
hypotheses raised and dropped along the way, no "X had been held pending Y", no
"this turned out to be backwards", no tally of what was tried. The exception is
narrow and real: keep a step of the route when the movement cannot be
understood without it. A refuted hypothesis that is itself a durable result is
such a case — but it is a *finding*, so it goes in `EXPERIMENT_LOG/`, and the
commit message states the finding rather than the chase.

The mechanical test: every sentence should be checkable against the tree at this
commit or the tree at its parent. A sentence that can only be checked against
something that existed transiently in between is the journey — cut it.

Hard cap **250 words**. Evidence and tables go in `EXPERIMENT_LOG/` or
`TODO.md`.

## 7. Verify what you cannot source

A pass will surface claims you cannot immediately support. Do not smooth the
prose around them. Read the code, or measure — the same standard as any other
assertion (`CLAUDE.md`, Verification Protocol). A documentation pass is a common
way to discover that something in the repo is wrong.

## 8. Finish

- **Shorter is the tell.** A good de-narrativizing pass usually removes more
  lines than it adds. If it added lines, check whether the additions are
  exposition or apology.
- Re-read each changed document from its top, in the reading order the README
  gives, as someone who has never seen the project.
- Write the commit message to section 6: the differential, not the route.
