# Phase 6: Sequential Context (Pre-Streaming Validation)

Goal: before building the full streaming compressor with gates,
test whether sequential context actually improves the T1 vector
space. If seeing a preceding instruction helps predict the current
instruction's execution effect, that validates retaining context
in the streaming compressor's window and informs the evict policy.

Architecture: same as Phase 1 (2-layer transformer, d_model=128,
4 heads, S^127 output). The only change is the input: instead of
one instruction's tokens, the model sees a *window* of K consecutive
instructions' tokens. The training target is always the LAST
instruction's single-step register delta — so window_size=1 and
window_size=2 produce vectors for the same set of instructions,
enabling direct comparison.

Execution distance metric: per-register state deltas, averaged
over input states and registers.
```
delta_i(s, r) = regs_after_i(s, r) - regs_before_i(s, r)
exec_dist(i, j) = mean_s(mean_r(log1p(|delta_i(s,r) - delta_j(s,r)|)))
```

Training data: structured basic blocks with data-flow dependencies
(random_basic_block), executed on 4 random input states per sequence.
5000 batches × 256 sequences, repeated indefinitely. 5000 training
steps with cosine LR decay.

## Experiment 19: Fixed-window context comparison

Trained two identical models:
- **window_size=1**: sees one instruction (baseline)
- **window_size=2**: sees the preceding instruction + the target

Same corpus, same architecture, same loss, same targets.

**Results:**

| Model | Pearson | Spearman | Final Loss |
|-------|---------|----------|------------|
| window_size=1 | 0.834 | 0.836 | 0.157 |
| window_size=2 | **0.875** | **0.849** | 0.125 |

Context improves Pearson from 0.834 to 0.875 (+0.041). The
training loss is ~20% lower. Seeing one preceding instruction
carries information that helps predict the current instruction's
execution effect.

**Follow-up: context vector analysis.** To test whether the
improvement comes from data-flow dependency detection or something
more subtle, embedded a fixed target instruction (ADD x5, x3, x7)
with 200 different predecessors: 100 that write to a source register
(x3 or x7, creating data dependency) and 100 that write elsewhere
(no dependency). Measured cosine similarity structure.

| Comparison | ADD | SLLI | BEQ | SUB-self | LW |
|------------|-----|------|-----|----------|-----|
| dep ↔ dep | 0.935 | 0.953 | — | 0.924 | 0.949 |
| nodep ↔ nodep | 0.930 | 0.946 | 1.000 | 0.939 | 0.941 |
| dep ↔ nodep | 0.931 | 0.948 | — | 0.920 | 0.944 |

Key findings:

1. **No combinatorial explosion.** All vectors for the same target
   instruction cluster tightly (cosine sim > 0.92) regardless of
   predecessor. Context causes a small continuous modulation, not a
   partition into distinct regions. S^127 is NOT being asked to
   accommodate 3.6×10^16 two-instruction combinations.

2. **Dep/nodep separation is weak.** The model doesn't cleanly
   separate "predecessor wrote to my source register" from
   "predecessor wrote elsewhere." Cross-group similarity is
   nearly identical to within-group.

3. **BEQ produces identical vectors (cosine 1.000) regardless
   of predecessor.** Since BEQ doesn't write a register, its
   register delta is always zero, and the model correctly learns
   that context is irrelevant for this target.

4. **SUB-self shows the strongest context effect** — the largest
   gap between dep (0.924) and nodep (0.939), and dep-vs-nodep
   (0.920). When the predecessor writes to x5, the "before"
   snapshot for SUB x5,x5,x5 changes, altering the delta even
   though the result is always zero.

5. **Effective dimensionality**: ~54 dimensions capture 90% of
   variance across 200 predecessor variants. The context modulation
   lives in a low-dimensional subspace of S^127.

**Assessment:** Sequential context provides a modest but real
improvement to T1 vector quality. The improvement is NOT from
detecting data-flow dependencies (dep/nodep separation is weak).
It's more likely from the model using predecessor tokens to better
encode the computational type — a form of implicit type context
that helps the pairwise metric even without explicit dependency
detection.

This has two implications for the streaming compressor:
- **Retain recent context in the window.** The evict policy should
  keep at least the most recent completed instruction rather than
  aggressively evicting everything.
- **Don't over-invest in dependency detection.** The model extracts
  value from context without explicitly detecting dependencies.
  The gate doesn't need to be smart about which instructions are
  "relevant" — proximity is sufficient.

---
