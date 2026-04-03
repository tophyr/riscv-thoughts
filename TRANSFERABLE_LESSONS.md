# Transferable Lessons from HATA

[HATA](https://github.com/tophyr/hata) was (is?) an experimental ML research
project aimed at finding a way to compress long-form prose into not just
fewer words - existing LLMs are good at that - but a *single embedding* in a
continuously-smooth "thought space". Its various attempts and experiments
offer numerous lessons in what is unlikely to work.

## Source: 16 experiments on natural language thought compression (Llama 3.2 1B)

These lessons were learned on a natural language domain but apply to any
learned compression system. They are domain-independent training dynamics,
architectural constraints, and evaluation pitfalls.

---

## Training Dynamics

### 1. Cosine similarity is scale-invariant

Any mechanism that operates through vector scaling (soft gating, dropout,
magnitude modulation) is invisible to cosine-based losses. The gradient
of cos_sim with respect to scale is zero. This caused: margin dead zones
in contrastive loss, gate drift in variable-K gating, dropout producing
fake diversity that didn't affect the loss. If using cosine-based metrics,
all trainable components must operate through *direction*, not magnitude.

### 2. Single-objective losses find shortcuts

Every single-objective loss tested (pairwise contrastive, InfoNCE)
optimized for one narrow feature while destroying broader content.
Pairwise contrastive built topic clusters but destroyed passage identity
(99.6% → 69.7% retrieval). InfoNCE preserved identity but degraded
factual encoding (+6.9% → +2.3% on probes). A compression system must
be evaluated on multiple independent metrics simultaneously. Any single
metric can be gamed.

### 3. Gradient dominance in multi-objective losses

Equal loss weights do not mean equal gradient influence. Whichever term
has the steepest gradient norm dominates training regardless of weight
settings. With multiple loss terms, gradient norms must be monitored
and weights must be swept — not assumed.

### 4. Scale learning rate with model size

A 168M-parameter model collapsed at lr=1e-3 but succeeded at lr=1e-4.
Training instability masquerades as architectural failure. When a new
architecture doesn't train, try 10x lower lr before concluding the
architecture is broken.

### 5. Training degrades what the frozen encoder already provides

A randomly-initialized compressor preserved the frozen encoder's content
better than any trained version. Training optimizes for the loss at the
expense of everything the loss doesn't test. This means: (a) the loss
must test everything you care about, and (b) measuring the randomly-
initialized baseline is mandatory — it tells you what you start with
and what training destroys.

### 6. Overcomplete output spaces enable shortcut solutions

With 9,470 training samples in 131,072 dimensions (16 slots × 8192),
any loss was trivially satisfiable through combinatorial tricks (assigning
near-orthogonal directions per sample). The optimizer found solutions
that satisfied every metric without learning any semantic structure.
Johnson-Lindenstrauss guarantees that random projections into
sufficiently overcomplete spaces preserve pairwise distances for free.
Size the output space relative to the training data, not relative to
theoretical capacity arguments.

---

## Architectural Constraints

### 7. Randomly-initialized bidirectional self-attention homogenizes

Five attempts with different data, losses, and initialization all
collapsed. Random-init bidirectional attention has a strong inductive
bias toward making all representations identical. The homogenization
happens faster than any loss can correct. This appears to be fundamental,
not a training failure. If bidirectional attention is needed, either
initialize from pretrained weights or ensure the training signal is
strong enough to overcome the bias (unproven).

### 8. Dominant bias directions propagate through all linear maps

If the input has a dominant shared direction (e.g., one component
carrying 99.5% of energy), every linear projection preserves it at
full strength regardless of output dimensionality. Compression and
expansion both preserve it. The bias must be subtracted before
processing, not trained through. This is a property of linearity,
not of the specific architecture.

### 9. Residual connections amplify shared bias

Perceiver-style residual connections around cross-attention caused
input-independence: cross-attention output (dominated by bias, norm ~300)
overwhelmed the residual stream (norm ~1.8). Since bias makes all
cross-attention outputs identical across inputs, the residual stream
carried no per-input signal. If using residual connections, either
subtract the bias first or use a gating mechanism that controls
information flow.

### 10. Partial injection into frozen models corrupts non-injected layers

Injecting learned representations at a subset of a frozen model's
layers (with zeros at other layers) produced garbage even with perfect
oracle representations. The zeros corrupt attention at non-injected
layers. Either inject at all layers or use a mechanism (like prefix
tokens) that the frozen model processes natively.

---

## Evaluation Pitfalls

### 11. Validate your evaluation metrics before trusting results

A linear probe testing 200-way classification with zero training
examples per test class produced 0% accuracy and was interpreted as
"thought vectors carry no information." The metric was fundamentally
broken. Cross-lingual retrieval (the correct test) showed 69.7-100%
accuracy. A broken evaluation metric can misdirect weeks of work.
Always sanity-check what the metric actually measures.

### 12. Discriminability does not imply content encoding

100% cross-lingual retrieval (InfoNCE) coexisted with degraded factual
probes. The thought vectors could identify WHICH passage they came from
without encoding the specific facts within that passage. Passage
identity ≠ factual content. Any compression system must be evaluated
on content-specific tests (probes, QA, reconstruction), not just
discriminability.

### 13. Test components in isolation

Full-architecture runs failed ambiguously — multiple interacting
failures masked each other. Isolated component tests (oracle K/V
injection, single-slot pooling, layer sweeps) gave clear, actionable
answers. When something doesn't work, decompose and test each part
independently before changing the whole system.

---

## Dimensionality Findings (Transfer as Methodology)

The specific numbers are Llama-specific, but the methodology transfers:

- **Measure effective dimensionality of your encoder's output.** PCA
  effective rank, fraction of variance in top components, absolute
  eigenvalue spectrum. Know how many dimensions actually carry signal
  before deciding your compressor's output size.

- **Measure bias/signal energy ratio.** If one direction dominates,
  subtract it before your compressor sees the input. Measure what
  remains.

- **The "fewer wires, higher voltage" pattern may be general.** Llama's
  middle layers had 4x fewer effective dimensions than the embedding
  layer but 90x more variance per dimension. Information capacity
  (log-volume) increased monotonically despite dimension count dropping.
  Dimensional collapse in deep networks may generally be repackaging,
  not information loss. Measure both shape and scale.

- **Size your output space to your signal, not your container.** If the
  input has 400 effective signal dimensions, d_out=512 is reasonable and
  d_out=8192 is dangerously overcomplete.
