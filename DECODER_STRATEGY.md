# Decoder Strategy

## The Problem

Given a compressed thought vector, find a token sequence that
compresses back to it. The compressor is deterministic and frozen --
it defines what "correct" means. The decoder must search the space of
valid token sequences for one that the compressor scores as close
to the target.

Multiple valid token sequences may compress to the same (or nearby)
point. This is correct behavior, not a bug. The decoder doesn't need
to find THE answer -- it needs to find A valid answer.

---

## Architecture: Proposer + Verifier

The decoder is not a single component. It's a two-part system:

**Proposer:** generates candidate token sequences conditioned on the
target thought vector. This is the learned part -- it gets smarter
over time.

**Verifier:** compresses each candidate through the frozen compressor
and measures distance to the target. This is exact and free -- it's
just a forward pass through an already-trained model.

The proposer can be imperfect because the verifier catches errors.
The proposer's job is to get close; the verifier's job is to confirm.

---

## Phase 1: Gradient Decoder (Proof of Concept)

Validate that the T1 space is invertible -- that you can reliably
recover instructions from their vectors.

### Mechanism

The compressor is differentiable: token embeddings -> transformer ->
T1 vector. Treat the token embeddings as continuous variables and
optimize them via gradient descent:

1. Pick a target T1 vector (compress a known instruction).
2. Initialize continuous embeddings randomly (or from a heuristic).
3. Forward through the frozen compressor -> get a candidate T1 vector.
4. Loss = distance(candidate, target).
5. Backprop to the embeddings, update them.
6. Repeat until converged.
7. Snap each continuous embedding to its nearest discrete token.
8. Verify: compress the discrete token sequence, check distance.

### Diversity Through Random Restarts

Gradient descent follows one path to one local minimum. For diversity,
run multiple restarts from different random initializations. Each may
converge to a different valid token sequence -- different instructions
that happen to compress to the same point. This IS the discovery of
execution equivalence via search.

### Structural Validity Filtering

After snapping to discrete tokens, the resulting sequence may not
form a valid instruction. Filter: does the sequence start with an
opcode token, followed by the right number of register and hex tokens
for that opcode type? Invalid sequences are discarded. Valid ones
proceed to verification.

### What This Proves

If gradient search reliably recovers the original instruction (or an
execution-equivalent one), the T1 space has good geometry -- the
distance landscape is smooth enough to navigate by gradient. If it
fails, the geometry is the problem, not the search method, and no
learned decoder will do better.

### Preliminary Results (Experiment 18)

Implemented and tested on the Exp 16 T0→T1 model (S^127, MSE loss,
Spearman 0.730). Key findings:

- **Search works where the geometry is good.** ADD x5,x3,x7 decoded
  to its commutative equivalent ADD x5,x7,x3 -- the decoder
  discovered execution equivalence by navigating the T1 space.
  SLLI x5,x3,1 recovered exactly.

- **Search fails where the geometry is bad.** SUB-self (always zero)
  decoded to a branch instruction (false cluster). BEQ couldn't be
  decoded at all. The T1 space's known problems (SLT/BEQ cluster,
  rough branch geometry) directly cause decoder failures.

- **Discrete snapping adds distance.** Successful decodes land at
  distance 0.4-0.8 from the target even when the continuous optimum
  is very close. The discrete token vocabulary is sparse on S^127.

These results validate the proposer+verifier framework: the
compressor IS usable as a scoring function for search, and search
DOES find execution-equivalent instructions in the regions where
the compressor learned smooth geometry.

---

## Phase 2: Learned Decoder (Implemented)

Replace gradient search with a neural network that jumps to the
answer in one forward pass. An autoregressive transformer decoder
with cross-attention to the emission vector, trained with
cross-entropy against the original tokens on a frozen encoder.

The "compressor-as-loss" framing from Phase 1 turned out to be
unnecessary in practice. Simple CE against the original tokens
produces a decoder that generalizes to held-out instructions and
emerges equivalence tolerance automatically — the encoder's
equivalence-collapsing geometry shows through at decode time,
without needing the compressor as a verifier.

See EXPERIMENT_LOG.md for empirical results.

### Architecture

An autoregressive model conditioned on the target thought vector.
At each step it outputs a distribution over the token vocabulary,
conditioned on the target and the tokens generated so far:

    P(token_t | target, token_1, ..., token_{t-1})

The target vector is injected via cross-attention, prefix projection,
or concatenation to the input embeddings. The model provides the
"language competence" (valid instruction syntax) while the target
provides the "what to say."

### Training Signal

Cross-entropy against the original tokens is the primary signal.
Given a (tokens, T1) pair, train the decoder to reproduce the
tokens autoregressively conditioned on T1. Standard supervised
learning on automatically-generated pairs — no human labels needed.

Alternative signals were investigated and rejected as primary
training loss:

- **Compressor-as-loss (Gumbel-softmax / round-trip):** decoder
  outputs re-compress near the target, but the decoder learns to
  produce "valid-looking garbage" that happens to compress nearby
  rather than the actual instruction. Adding an instruction-
  validity guardrail would likely fix this, but validity is a
  discrete predicate — enforcing it non-differentiably pushes
  the method into REINFORCE territory, losing the "clean
  differentiable chain" that motivated the approach in the first
  place.
- **REINFORCE with execution reward:** works (23% exec-equivalence
  rate on a frozen encoder) but high variance and converges much
  slower than CE. Useful as a fine-tune on top of CE, not as
  primary training.

CE was initially suspect because it appears to fight execution
equivalence (penalizing valid alternatives). Empirically, the
encoder's equivalence-collapsing geometry means the T1 vectors
for equivalents are close, so CE-trained decoders still produce
equivalent alternatives on held-out data — the equivalence
tolerance is inherited from the encoder, not learned by the
decoder.

### Temperature and Diversity

The decoder outputs token probabilities. Temperature controls
diversity:

- **Low temperature:** the single most likely token sequence. Fast,
  deterministic, good when the decoder is well-trained.
- **High temperature:** many diverse candidates. Sample N sequences,
  compress all N, keep the best. Useful for exploring equivalence
  classes or when the target is in a region the decoder hasn't seen
  much.

The verifier (compressor) scores all candidates identically regardless
of temperature. Temperature controls exploration; the compressor
controls quality.

---

## Phase 3: Continuous Learning (Future)

Every real decoding attempt is a free training example.

### Mechanism

During use, the decoder proposes sequences and the compressor scores
them. Each (target, proposal, score) triple is a training example.
The system accumulates these and periodically fine-tunes the decoder.

No human feedback. No labeling. The compressor IS the ground truth,
and it never goes stale because it's frozen. The quality signal is
exact and automatic.

### Adaptation to Usage Patterns

The decoder naturally improves in whatever region of thought space
gets used most. If the system spends time on pointer arithmetic,
the decoder gets better at expressing pointer computations -- even
if they were underrepresented in original training.

The compressor doesn't change. The decoder adapts its proposal
distribution to match the actual usage distribution. Regions of
thought space that are never visited don't waste decoder capacity.

### Deployment Modes

- **Eval mode:** decoder weights frozen, pure inference. Fastest.
  Use when the decoder is good enough.
- **Online learning:** update decoder weights from each decoding
  attempt. Slower per-query but continuously improving.
- **Batch fine-tuning:** accumulate (target, best_proposal) pairs
  during eval-mode use, periodically retrain. Best of both worlds.

---

## How This Connects to the Streaming Architecture

The decoder strategy is orthogonal to the compressor architecture.
It works the same way whether the compressor is:

- The current single-instruction T1 model
- A variable-length sequence model
- The streaming compressor with learned gates

At every level: the compressor produces a vector, the decoder inverts
it. The compressor is the oracle, the decoder is the proposer. The
training signal is always the same: compress the proposal, measure
distance to target.

For the streaming architecture specifically, decoding becomes
sequential: the decoder produces L0 tokens, the streaming compressor
processes them and emits L1 vectors, and the system checks whether
the emitted vectors match the target sequence. The decoder gets
feedback at each emission, not just at the end -- richer training
signal, faster convergence.

---

## Open Questions

1. **Embedding initialization for gradient search.** Random init vs.
   informed init (e.g., start from the embedding of the nearest known
   instruction). Informed init converges faster but might miss
   distant equivalences.

2. **Gumbel-softmax temperature schedule.** High temperature early
   (explore), low temperature late (commit). How to anneal without
   getting stuck.

3. **Decoder architecture size.** Empirically resolved: the decoder
   needs to be substantially larger than the compressor. Rough rule
   from memorization theory (Zhang et al. 2017): decoder parameter
   count should be at least the total number of token predictions
   in the training set. A too-small decoder plateaus below full
   reconstruction even given unlimited training. A too-large decoder
   wastes compute and memorizes beyond what the encoder geometry
   actually preserves as structure.

4. **Multi-sequence decoding.** For T2 targets, the decoder needs to
   produce a sequence of instructions, not just one. Does it generate
   the whole sequence autoregressively, or does it first decide on
   a structure (how many instructions, what types) and then fill in
   details?

5. **Equivalence class discovery.** High-temperature decoding with
   multiple samples naturally discovers equivalence classes (different
   proposals that score equally well). How to systematically explore
   and catalog these?
