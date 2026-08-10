# Phase 5: Gradient Decoder (Geometry Validation)

Goal: validate that the T1 space is smooth enough for gradient-based
search to invert the compression. Optimize continuous embeddings
through the frozen compressor, snap to discrete tokens, verify via
execution.

## Experiment 18: Gradient decoder on Exp 16 model

Using the best T0→T1 model (Exp 16, grouped structured, Spearman
0.730). Gradient search: Adam on continuous embeddings in the
model's embedding space, 500 steps per restart, multiple restarts
per target, all token sequence lengths 4-8 tried.

**Results:**

| Target | Best decode | Distance | Execution |
|--------|-----------|----------|-----------|
| ADD x5,x3,x7 | ADD x5,x7,x3 | 0.40 | EQUIV (commutativity discovered!) |
| SLLI x5,x3,1 | SLLI x5,x3,1 | 0.79 | EQUIV (exact recovery) |
| ADDI x5,x3,0 | SRL x3,x3,x3 | 1.44 | not equiv |
| SUB x5,x3,x3 | BGEU x27,x5,7 | 0.75 | not equiv (false cluster) |
| BEQ x3,x7,16 | (none valid) | — | — |

**Key findings:**

1. **Commutativity discovered by search.** The gradient decoder found
   ADD x5,x7,x3 (the commutative equivalent) as the closest valid
   instruction to ADD x5,x3,x7's T1 point. The T1 space correctly
   places commutative variants nearby, and gradient search navigates
   there.

2. **Exact recovery works for some instructions.** SLLI x5,x3,1 was
   recovered exactly. The T1 space is smooth enough near this point
   for gradient search + discrete snapping to land on the right
   instruction.

3. **False cluster causes wrong decodes.** SUB x5,x3,x3 (always
   zero) decoded to BGEU (a branch) — confirming the SLT/BEQ/branch
   false cluster pulls zero-producing instructions toward branches.

4. **Some regions are not gradient-navigable.** ADDI identity and
   BEQ couldn't be decoded. The T1 landscape near these points is
   either too rough for gradient search or the discrete snapping
   step loses too much.

5. **Snap distance is significant.** Even successful decodes have
   distance 0.4-0.8 after snapping. The continuous optimum is close
   to the target but the nearest discrete token sequence is further.
   This is expected — discrete tokens are sparse points on S^127.

**Assessment:** The T1 space has smooth, navigable geometry in the
regions where the model learned well (ALU ops, shifts). It's rough
or degenerate in the regions with known problems (false clusters,
branch behavior). The gradient decoder validates both the theory
(search-based inversion works in principle) and the specific model's
limitations.

---
