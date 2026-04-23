# What Is a Thought?

### A Theory of Compression, Thinking, and Expression

Ask any capable LLM to produce a 2000-word summary of a 50-page
document and you get something genuinely useful: the key claims,
most of the nuance, the structure of the argument. Ask the same
model to reduce that to 100 words and it will do it — but most of
the specific detail is gone, and only the headline-level points
remain. At 10 words you have a title.

This is not a limitation of any particular model. It is taken as
a property of summarization itself: aggressive compression
discards information, and you cannot meaningfully reconstruct a
document from its one-sentence summary. Nobody claims you should
be able to.

Yet the same models also demonstrate something quite different
elsewhere: the famous `king − man + woman ≈ queen` arithmetic of
token embeddings. A high-dimensional vector that stands in for a
single word carries enough structure to support meaningful
geometric operations. Relationships compose. Analogies work. The
vector isn't a summary of the word — it's a representation of the
word that you can manipulate.

What if the compression of an entire document worked that way?
What if the full text of *Romeo and Juliet* compressed not to a
summary you read, but to a vector you *manipulate* — a single
point in a high-dimensional space that still carried the play in
full, that composed with other such vectors, and that decompressed
back into nearly, or exactly, the original text on demand? And what
if `Romeo and Juliet − Renaissance Verona + 1950s New York` defined
an actual work that could be written out... or pointed you at *West
Side Story*?

That is the thesis of this project. The rest of this document
works out what a "thought" would have to be for such a system to
exist, what operations it would support, and what geometric
properties its space would need to have.

---

## The Core Claim

A thought is a compressed representation — a point in a continuous
vector space produced by a deterministic compression function over
tokens.

Thinking is the composition and transformation of thought-points
within that space: analogy, inference, abstraction, planning. These
are geometric operations on vectors, not token manipulations. The
whole point of having a compressed thought space is to make such
operations meaningful and efficient.

Expression is the act of searching for a token sequence that
compresses back to a given thought — "lowering" the thought into
something a human or another system can consume. This is the
decoder's job, separate from thinking.

The compression defines what a thought means. Operations on thoughts
are thinking. The search from a thought back to tokens is expression.

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

## Expression Is Search

Given a thought (a compressed point), expression is the process of
finding a token sequence whose compression matches it — lowering the
thought into surface form.

This requires two components: a proposer that generates candidate token
sequences, and the compressor that scores each candidate by compressing
it and measuring proximity to the target. The sections below work
through the practical constraints on each piece.

### The Proposer Must Be Conditioned

Could we just use any off-the-shelf language model as the proposer?
No — and the reason is worth making explicit, because it drives
everything else about the decoding architecture.

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

Modern language models generate autoregressively: emit token one, then
token two conditioned on token one, then token three conditioned on
the previous two, and so on. When this style of model is used as the
proposer for expression search, the search inherits this left-to-right
shape — and that turns out to be badly matched to what the search
actually needs.

The problem: left-to-right generation starts with the least informative
tokens. "The Kingdom of Tethra..." — "The" carries zero bits about the
thought. "Tethra" carries a lot. The early beam steps navigate a
completely flat scoring landscape where nothing distinguishes any
candidate.

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

A natural next question: if we can swap proposers to get different
languages or styles, are the proposers themselves freely
interchangeable? Can we just bolt any off-the-shelf language model
onto a trained compressor?

Partially yes, partially no. The thought representation itself is
independent of any specific decoder, but decoders are not commodity
infrastructure. Each proposer needs a trained conditioning mechanism —
an adapter that maps from compressed space to something the frozen LM
can consume. This adapter is trained against the specific compressed
space.

The independence is: train the compressor once, train cheap conditioned
proposers per use case. The compressor verifies all of them identically.
Swapping proposers (different language, different style, different base
LM) requires training a new adapter but not retraining the compressor.

---

## Verification Is Free

Expression is non-deterministic: the same thought can produce many
different valid token sequences. That raises an obvious worry — if
decoding is a search with multiple possible outputs, how do you know
the output you got is actually faithful to the thought you meant to
express?

With a standard learned decoder — one trained end-to-end to map from
thought-vectors to tokens — that's a training problem. You need the
decoder to be faithful in general, and you can't fully verify that it
is. Every decoding is a roll of the dice with no referee.

With search-based expression against a deterministic compressor,
verification is trivially free. Compress the candidate. Check if it
matches the target. The compressor is the referee, and it's the same
function you used to define the thought in the first place.

If the candidate compresses to the same point, it is semantically
equivalent by definition — because the compressor IS the definition of
semantic equivalence. There is no separate ground truth to check against.

The system has a clean three-way separation:

- **Compression is deterministic.** Same tokens → same point, always.
  This is the ground truth. This is what you train.
- **Expression is non-deterministic.** Same thought → many valid token
  sequences. The proposer controls variation. This is the search.
- **Verification is deterministic.** Compress the candidate, check the
  distance. Pass or fail.

The compressor is simultaneously the representation, the training
objective, and the verification oracle. The proposer is a practical
necessity for expression — it must be conditioned on the target and
its adapter must be trained — but its errors are caught by the
compressor. The proposer is allowed to be imperfect because the
compressor backstops it.

---

## The Hierarchy Is Search Acceleration

The naïve form of expression-as-search has a serious problem: for any
non-trivial thought, the space of candidate token sequences is
astronomical, and scoring a partial candidate against the final
compressor target is mostly uninformative until the sequence is nearly
complete. A search that can't prune candidates early has to explore
exponentially many branches. Single-level compression makes expression
theoretically sound but practically intractable.

The fix is to stack compressors. T0 tokens compress to T1 concept-level
representations. T1 representations compress to T2 proposition-level
representations. And so on up through however many levels the problem
demands. The levels need not share a vector space — each can have its
own dimensionality, sized to the complexity of relationships at that
level.

Multi-level compression, viewed this way, is not a theory of
abstraction. It is a multi-resolution scoring function that makes the
expression search tractable by providing cheap coarse checks before
the expensive fine checks.

### The Compressed Space Is Continuous; Valid Lowerings Are Discrete

Before getting to the coarse-to-fine mechanics, one subtlety about the
levels themselves: at every level, the compressed representation lives
in a continuous vector space, but the valid token sequences that can
be lowered to are discrete points in that space. You can nudge a
compressed point by an epsilon in any direction, but there may be no
valid token sequence that compresses to exactly that new point.

This mismatch is not a problem — it is the correct framing for
decoding. The decoder's job is not to perfectly invert the compression.
It is to find the **nearest valid point** in the discrete set of
lowerable sequences. The compressor scores candidates; the decoder
proposes them; validity constraints (grammar in natural language,
instruction-set rules in machine code) filter the proposals.

This is the same constraint at every level. Natural language tokens must
form grammatical sentences. Instructions must have valid opcodes,
register numbers, and immediate ranges. Instruction sequences must
respect encoding constraints. The valid lowerings are always a discrete
subset of the continuous space, and the decoder always searches for the
nearest one.

During expression (the search), the hierarchy provides coarse-to-fine
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

## What Makes Thinking and Expression Possible: Geometric Properties

The geometric properties of the compressed representation are not
aesthetic preferences. They are what make operations on thought-points
meaningful (thinking) and what make the token search converge
(expression).

**Smoothness.** The compressor must be smooth — similar inputs produce
similar outputs. For thinking: a small perturbation of a thought should
produce a neighboring thought, not a semantically unrelated one. For
expression: if the scoring landscape has sharp discontinuities, the
search cannot use gradients to navigate. Without smoothness, the space
supports neither meaningful vector operations nor convergent search.

**Factored structure.** If independent aspects of meaning (entities,
relations, quantities, temporal structure) are encoded on independent
axes, both thought operations and expression search decompose into
sub-problems. Adjusting the quantity of a thought shouldn't scramble its
entities. The expression search can make independent progress on
independent aspects of meaning.

**Interpolation coherence.** Points between two valid compressed
representations should correspond to semantically intermediate content.
For thinking: analogies and interpolations work — (king − man + woman)
lands near (queen). For expression: the scoring landscape has no
discontinuous jumps between valid regions; the search can move smoothly
from one valid solution toward another.

These properties are not nice-to-haves for publication. They are
necessary conditions for both thought operations to be meaningful and
for expression search to converge in tractable time. A rough, entangled,
discontinuous compression produces a space where thinking is arbitrary
and search requires exponential beam width. A smooth, factored,
interpolable compression produces a space where thinking is coherent and
expression is tractable.

Training the compressor to have these properties is defining both what
meaningful thought-operations look like AND training it to be
efficiently invertible by search. The geometric quality of the
compressed space IS the computational viability of thinking and
expression.

---

## What To Build First

The framework above divides cleanly into two layers: the compressor
(which defines meaning and provides a verification oracle) and the
decoding stack (proposers, adapters, search strategies). The split
suggests a research priority.

The compressor is the novel claim and the load-bearing component. If
the compressed space doesn't have the right geometry, no amount of
decoder cleverness will rescue it. If it does, any competent decoder
can exploit it. So the first thing to validate is the compressor
itself — and that can be done without generating a single token:

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

The search-based expression pattern aligns with how humans experience
language production and comprehension. This is a structural analogy, not
a neuroscientific claim. Each example below is about expression —
getting a thought out — not about thinking itself.

**"How do I say this?"** — The defining question of language production.
The thought exists. The search for a token sequence that faithfully
compresses to it has not yet converged. Deliberate word choice IS the
search. This IS expression.

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

Thinking is what happens to thoughts in the compressed space:
composition, transformation, analogy, inference. These are geometric
operations on vectors. Having a thought space with the right geometry
is what makes thinking possible in the first place.

Expression is search: given a thought, find tokens that lower it into
surface form. A conditioned proposer generates candidates; the
compressor verifies them. The search is non-deterministic; the
verification is exact.

The hierarchy accelerates expression from intractable to feasible by
providing multi-resolution scoring. Geometric properties of the
compressed space — smoothness, factored structure, interpolation
coherence — are what enable both meaningful thought operations and
convergent expression search. They are computational properties of the
problem, not aesthetic properties of the representation.

Training the compressor is defining what meaning is. Operations on
thought-points are thinking. Finding tokens that recompress to a thought
is expression. Verification is free.
