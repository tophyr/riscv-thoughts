# Phase 6: Sequential Context (Pre-Streaming Validation)

Goal: before building a streaming compressor with gates, test whether
seeing a preceding instruction improves the T1 vector. A positive result
argues for keeping recent context in the streaming compressor's window
and bears on its evict policy.

Architecture: Phase 1's encoder (2-layer transformer, d_model=128,
4 heads, S^127 output). The input is a *window* of K consecutive
instructions' tokens; the target is always the LAST instruction's
single-step register delta, so window sizes 1 and 2 produce vectors for
the same set of instructions and compare directly.

Execution distance, from per-register state deltas averaged over input
states and registers:
```
delta_i(s, r) = regs_after_i(s, r) - regs_before_i(s, r)
exec_dist(i, j) = mean_s(mean_r(log1p(|delta_i(s,r) - delta_j(s,r)|)))
```

Training data: random basic blocks with data-flow dependencies, each
executed on 4 random input states; 5000 batches × 256 sequences,
repeated; 5000 steps with cosine LR decay.

## Experiment 19: Window size 1 vs 2

Two identical models, same corpus, loss and targets: window_size=1
(the target instruction alone) and window_size=2 (predecessor + target).

| Model | Pearson | Spearman | Final Loss |
|-------|---------|----------|------------|
| window_size=1 | 0.834 | 0.836 | 0.157 |
| window_size=2 | **0.875** | **0.849** | 0.125 |

One preceding instruction raised Pearson by 0.041 and lowered training
loss ~20%.

**Context vector analysis.** Each fixed target instruction in the table
(ADD is ADD x5, x3, x7) was embedded after 200
predecessors: 100 writing a source register of the target ("dep", a data
dependency) and 100 writing elsewhere ("nodep"). Cosine similarity within
and across the groups:

| Comparison | ADD | SLLI | BEQ | SUB-self | LW |
|------------|-----|------|-----|----------|-----|
| dep ↔ dep | 0.935 | 0.953 | — | 0.924 | 0.949 |
| nodep ↔ nodep | 0.930 | 0.946 | 1.000 | 0.939 | 0.941 |
| dep ↔ nodep | 0.931 | 0.948 | — | 0.920 | 0.944 |

1. **No combinatorial explosion.** Every target's vectors cluster tightly
   (cosine > 0.92) across predecessors: context modulates the vector
   slightly rather than partitioning S^127 among ~3.6×10^16
   two-instruction combinations.
2. **Dependency is barely separated.** Cross-group similarity is nearly
   equal to within-group.
3. **BEQ is unaffected (cosine 1.000).** It writes no register, so its
   delta is always zero and context is irrelevant to it.
4. **SUB-self shows the largest context effect** (dep 0.924, nodep
   0.939, cross 0.920): a predecessor writing x5 changes the "before"
   snapshot of SUB x5,x5,x5 and so its delta, though the result is
   always zero.
5. **The modulation is low-dimensional:** ~54 dimensions hold 90% of the
   variance across the 200 predecessor variants.

Context gave a modest, real improvement that does not come from
detecting data-flow dependencies; the likelier source is predecessor
tokens sharpening the encoding of the computation type. For the
streaming compressor this meant keeping at least the most recent
completed instruction in the window, and that proximity alone, without
dependency-aware gating, carries the benefit.
