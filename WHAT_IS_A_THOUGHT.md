# What Is a Thought?

## A Theory of Compression, Search, and Meaning

---

## The Core Claim

A thought is a compressed representation — a point in a continuous
vector space produced by a deterministic compression function over
tokens. Expression is the act of searching for a token sequence that
closely matches that compressed representation.

The compression defines what the thought means. The search is how the
thought becomes language.

---

## Compression Defines Meaning

A compressor is a learned deterministic function that maps a token
sequence to a point in a continuous vector space. Same tokens in, same
point out, always.

Because the compressor is deterministic, the compressed point IS the
definition of semantic equivalence. Two token sequences mean the same
thing if and only if they compress to the same point. Paraphrase
equivalence, cross-lingual equivalence, behavioral equivalence — they're
all the same statement: same compressed point.

Training the compressor is defining what meaning is. The loss function
is a specification of which surface variations should be collapsed and
which should be preserved:

- Cross-lingual pairs as positives: "language identity is surface form.
  Collapse it."
- Contradictions as negatives: "factual differences are content.
  Preserve them."
- Paraphrases as positives: "word choice and syntax are surface form.
  Collapse them."
- QA-contradiction discrimination: "the specific fact this question
  targets is content. Preserve it."

The compressor doesn't encode meaning into the point. It encodes a
*criterion for semantic equivalence* — a deterministic test that any
candidate token sequence can be scored against.

---

## Thinking Is Search

Given a thought (a compressed point), thinking is the process of finding
a token sequence whose compression matches it.

This requires two components: a proposer that generates candidate token
sequences, and the compressor that scores each candidate by compressing
it and measuring proximity to the target.

### The Proposer Must Be Conditioned

An unconditioned language model generating fluent text has no idea what
thought you're trying to express. Its first-token distribution is just
the marginal distribution of sentence-starting tokens in its training
data. Searching against an unconditioned proposer is trying every
possible sentence opening and checking which ones happen to compress
near the target. At a vocabulary of 128K tokens, even the first step
is intractable.

The proposer must be informed by the target. Practically: a model with
a trained adapter (cross-attention, prefix projection, or similar) that
takes the compressed point and injects it as conditioning. The model
provides structural competence — grammar, valid syntax, vocabulary. The
adapter provides steering toward the target.

The same search-and-propose mechanism applies at every level of the
hierarchy. Lowering from T2 to T1 uses the same architecture as
lowering from T1 to T0. The only thing that changes at the bottom
level is that the final output must discretize into a finite token
vocabulary.

### Search Need Not Be Left-to-Right

Autoregressive left-to-right generation starts with the least
informative tokens. "The Kingdom of Tethra..." — "The" carries zero
bits about the thought. "Tethra" carries a lot. The early beam steps
navigate a completely flat scoring landscape where nothing distinguishes
any candidate.

Content-first generation is more natural: produce the highest-
information tokens first, then fill in gaps. This mirrors human language
production — people don't compose sentences left to right. They have key
concepts and then find syntactic structure to arrange them.

Approaches that support this:

- **Infilling / masked prediction**: generate content words, then fill
  structural tokens around them. Requires a masked LM or bidirectional
  model as the proposer, not a causal LM.
- **Iterative refinement**: start with a full sequence of mask tokens,
  predict all positions simultaneously, compress and score against
  target, replace lowest-confidence positions, repeat. Each iteration
  improves. This sidesteps "where to start" entirely.
- **Skeleton-first**: generate nouns, verbs, key modifiers; compress and
  score; then generate function words and syntax. Content words
  determine coarse compression, function words refine it.

The compressor doesn't care about generation order — it scores whatever
complete or partial sequence it's given. Only the proposer architecture
is affected by the choice of generation strategy.

### Decompression Is Non-Deterministic

The same thought decompresses to many valid token sequences. This is
correct behavior, not a flaw. The same thought SHOULD decompress
differently depending on the proposer:

- A French-conditioned proposer produces French.
- An English-conditioned proposer produces English.
- A technical-domain adapter produces jargon.
- A simplification adapter produces plain language.

All different token sequences. All compressing to the same point. All
semantically equivalent — because the compressor says so.

The proposer controls style, language, register, length. The compressor
controls fidelity. These concerns are fully separated.

### Decoder Independence (Qualified)

The thought representation is independent of any specific decoder, but
decoders are not commodity infrastructure. Each proposer needs a trained
conditioning mechanism — an adapter that maps from compressed space to
something the frozen LM can consume. This adapter is trained against the
specific compressed space.

The independence is: train the compressor once, train cheap conditioned
proposers per use case. The compressor verifies all of them identically.
Swapping proposers (different language, different style, different base
LM) requires training a new adapter but not retraining the compressor.

---

## Verification Is Free

This architecture dissolves the "thought-lowering determinism" problem.
The original concern: how do you guarantee that going from thought back
to tokens produces the right output?

With a standard learned decoder, that's a training problem — you need
the decoder to be faithful, and you can't fully verify it. With search-
based thinking against a deterministic compressor, verification is
trivially free. Compress the candidate. Check if it matches the target.
The compressor is the oracle.

If the candidate compresses to the same point, it is semantically
equivalent by definition — because the compressor IS the definition of
semantic equivalence. There is no separate ground truth to check against.

The system has a clean three-way separation:

- **Compression is deterministic.** Same tokens → same point, always.
  This is the ground truth. This is what you train.
- **Thinking is non-deterministic.** Same thought → many valid token
  sequences. The proposer controls variation. This is the search.
- **Verification is deterministic.** Compress the candidate, check the
  distance. Pass or fail.

The compressor is simultaneously the representation, the training
objective, and the verification oracle. The proposer is a practical
necessity — it must be conditioned on the target and its adapter must
be trained — but its errors are caught by the compressor. The proposer
is allowed to be imperfect because the compressor backstops it.

---

## The Hierarchy Is Search Acceleration

Multi-level compression (T0 → T1 → T2 → ... → Tn) is not a theory of
abstraction. It is a multi-resolution scoring function that makes the
search tractable.

Each level compresses a set of points into a smaller set. T0 tokens
compress to T1 concept-level representations. T1 representations
compress to T2 proposition-level representations. The levels need not
share a vector space — each can have its own dimensionality, sized to
the complexity of relationships at that level.

### The Compressed Space Is Continuous; Valid Lowerings Are Discrete

At every level, the compressed representation lives in a continuous
vector space, but the valid token sequences that can be lowered to are
discrete points in that space. You can nudge a compressed point by an
epsilon in any direction, but there may be no valid token sequence that
compresses to exactly that new point.

This is not a problem — it is the correct framing for decoding. The
decoder's job is not to perfectly invert the compression. It is to find
the **nearest valid point** in the discrete set of lowerable sequences.
The compressor scores candidates; the decoder proposes them; validity
constraints (grammar in natural language, instruction-set rules in
machine code) filter the proposals.

This is the same constraint at every level. Natural language tokens must
form grammatical sentences. Instructions must have valid opcodes,
register numbers, and immediate ranges. Instruction sequences must
respect encoding constraints. The valid lowerings are always a discrete
subset of the continuous space, and the decoder always searches for the
nearest one.

During thinking (the search), the hierarchy provides coarse-to-fine
guidance:

- Early in the search: score against the highest (coarsest) compression
  level. "Am I in roughly the right semantic region?" Cheap to check,
  prunes wildly wrong candidates immediately.
- Midway: score against intermediate levels. "Am I capturing the right
  propositional structure?" Refines the search within the correct coarse
  region.
- Near completion: score against the finest compression level. "Am I
  preserving the specific factual details?" Final verification.

Without the hierarchy, the search must score every partial candidate
against one monolithic target. The scoring signal is weak for short
partial sequences (not enough tokens to meaningfully compress) and only
becomes informative near the end — by which point the search has wasted
most of its budget exploring unprunable branches.

With the hierarchy, bad candidates are pruned early at coarse resolution,
and the search budget is concentrated on candidates that are already in
the right ballpark. Each level of compression is a checkpoint that the
search must pass through.

### Level-Specific Dimensionality

Because levels need not share a space, each level's dimensionality is a
free parameter sized to the complexity of relationships at that level.
Token-level relationships might need 512 dimensions. Proposition-level
relationships might need 256. Argument-level might need 128.

Dimensionality could decrease with abstraction — higher levels represent
less detail, so they need fewer axes to distinguish the relationships
that matter at that granularity.

The right dimensionality per level is the one that produces the smoothest,
most informative scoring gradient for the search at that level. Too low
and many different inputs score identically — the search has no guidance.
Too high and the space is so empty that everything scores well — the
search has no discrimination.

---

## What Makes Thinking Efficient: Geometric Properties

The geometric properties of the compressed representation are not
aesthetic preferences. They are computational properties of the scoring
landscape that determine whether the search converges.

**Smoothness.** The compressor must be smooth — similar inputs produce
similar outputs. If the scoring landscape has sharp discontinuities, the
search cannot use gradients to navigate. A small change to the candidate
token sequence must produce a small, directionally informative change in
the compressed point. Without smoothness, the search degenerates to
random sampling.

**Factored structure.** If independent aspects of meaning (entities,
relations, quantities, temporal structure) are encoded on independent
axes, the search can decompose into sub-problems. Getting the entity
right shouldn't interfere with getting the quantity right. Factored
structure means the search makes independent progress on independent
aspects of meaning.

**Interpolation coherence.** Points between two valid compressed
representations should correspond to semantically intermediate content.
This means the scoring landscape has no discontinuous jumps between
valid regions — the search can move smoothly from one valid solution
toward another. Without interpolation coherence, the space between valid
points is a wasteland that traps the search.

These properties are not nice-to-haves for publication. They are
necessary conditions for the search to converge in tractable time. A
rough, entangled, discontinuous compression produces a search problem
that requires exponential beam width. A smooth, factored, interpolable
compression produces a search problem that beam search can solve.

Training the compressor to have these properties is training it to be
efficiently invertible by search. The geometric quality of the
compressed space IS the computational tractability of thinking.

---

## What Matters Now

The compressor is the research contribution. Its geometric properties
are the publishable claims. Everything about decoding — the proposer
architecture, the search strategy, the adapter training — is a
downstream engineering problem.

The compressor can be validated without generating a single token:

- **QA-contradiction discrimination**: does the compressed point
  distinguish facts?
- **Cross-lingual retrieval**: do translations compress to the same
  point?
- **Geometric properties**: interpolation coherence, analogy arithmetic,
  factored structure.

These are measurements on the compressed space itself. They do not
require a working proposer, a trained adapter, or any token generation.

Solve the compressor. Prove the geometry. The decoder is a problem for
later, solvable with whatever models and compute exist at that time. The
compressor and its geometry are what need to happen now.

---

## Alignment With Human Cognition

The search-based thinking pattern aligns with how humans experience
language production and comprehension. This is a structural analogy, not
a neuroscientific claim.

**"How do I say this?"** — The defining question of language production.
The thought exists. The search for a token sequence that faithfully
compresses to it has not yet converged. Deliberate word choice IS the
search. This IS thinking.

**"It's on the tip of my tongue."** — The thought is located. Its
relationships are felt — near this, relates to that. The proposer is
generating candidates, none have scored well enough yet. You can
describe the neighborhood without finding the word.

**"That's not quite what I mean."** — Someone else's token sequence,
compressed through your own compressor, lands near but not at your
target. The error signal is geometric — you feel the direction of
discrepancy. "More like X than Y" provides a gradient direction.

**"I understand it but I can't explain it."** — The thought exists. You
can score others' attempts. But your proposer — your vocabulary, your
fluency in the relevant domain — doesn't cover that region well enough
to generate candidates that score. Someone with different expertise
expresses the same thought easily. Different proposer, same thought,
different search success.

**"I know what I want to say but I can't say it simply."** — The search
finds candidates that are too long. You want a shorter sequence that
compresses to the same point. Length-penalized search. Harder because
shorter sequences carry less information, so fewer land in any given
region. Conciseness is a compression achievement.

**Translation.** A bilingual person has a thought and runs two
independent searches — one with a French proposer, one with English —
against the same target. "Untranslatable" words aren't untranslatable
thoughts. They're regions where one language's token distribution
provides good coverage and another's doesn't.

**Writing is iterative.** A first draft is low-beam-width search — fast,
greedy, approximately right. Editing is re-scoring against the target
thought, finding segments where compression diverges, re-searching
locally. "This paragraph doesn't say what I want" is a compression
verification failure on a subsequence.

---

## Summary

A thought is a compressed representation — a point produced by a
deterministic compressor over tokens. Its meaning is defined by
the compressor: two token sequences mean the same thing if and only
if they compress to the same point.

Thinking is search: given a thought, find tokens that express it. A
conditioned proposer generates candidates; the compressor verifies them.
The search is non-deterministic; the verification is exact.

The hierarchy accelerates thinking from intractable to feasible by
providing multi-resolution scoring. Geometric properties of the
compressed space — smoothness, factored structure, interpolation
coherence — are what make the search converge. They are computational
properties of the problem, not aesthetic properties of the
representation.

Training the compressor is defining what meaning is. Evaluating the
search is testing whether thinking works. Verifying the output is free.
