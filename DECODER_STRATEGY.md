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

---

## Phase 2: Learned Decoder

Replace gradient search with a neural network that jumps to the
answer in one forward pass.

### Architecture

An autoregressive model conditioned on the target thought vector.
At each step it outputs a distribution over the token vocabulary,
conditioned on the target and the tokens generated so far:

    P(token_t | target, token_1, ..., token_{t-1})

The target vector is injected via cross-attention, prefix projection,
or concatenation to the input embeddings. The model provides the
"language competence" (valid instruction syntax) while the target
provides the "what to say."

### Training Signal: The Compressor Is the Loss

The compressor provides the training signal directly. No supervised
(thought, token_sequence) pairs needed:

1. Sample a target thought vector.
2. Decoder proposes a token sequence.
3. Compress the proposal through the frozen compressor.
4. Loss = distance(compressed_proposal, target).
5. Backprop through the decoder.

The gradient chain from loss to decoder parameters requires
differentiable token selection. Options:

- **Gumbel-softmax:** replace discrete token sampling with
  differentiable soft samples. The full chain (target -> decoder ->
  soft tokens -> compressor -> loss) is differentiable end-to-end.
  Cleanest approach.

- **REINFORCE:** treat the compressor's distance as a reward signal,
  train the decoder with policy gradient. Works with discrete tokens
  but has high variance.

- **Straight-through estimator:** use argmax in forward pass, soft
  gradients in backward pass. Simple but biased.

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

## Phase 3: Continuous Learning

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

3. **Decoder architecture size.** The decoder needs to be expressive
   enough to propose valid instructions but small enough that the
   compressor forward pass (for verification) isn't the bottleneck.
   The compressor is tiny (2-layer transformer), so the decoder
   probably should be too.

4. **Multi-sequence decoding.** For T2 targets, the decoder needs to
   produce a sequence of instructions, not just one. Does it generate
   the whole sequence autoregressively, or does it first decide on
   a structure (how many instructions, what types) and then fill in
   details?

5. **Equivalence class discovery.** High-temperature decoding with
   multiple samples naturally discovers equivalence classes (different
   proposals that score equally well). How to systematically explore
   and catalog these?
