# Phase 5: Gradient Decoder (Geometry Validation)

Goal: test whether the T1 space is smooth enough for gradient search to
invert the compression: optimize continuous embeddings through the
frozen compressor, snap them to discrete tokens, and check the result by
execution.

## Experiment 18: Gradient decoder on the Exp 16 encoder

Encoder: Phase 4's Experiment 16 (grouped structured inputs, Spearman
0.730). Search: Adam on continuous embeddings in the model's embedding
space, 500 steps per restart, multiple restarts per target, every token
sequence length 4-8.

| Target | Best decode | Distance | Execution |
|--------|-----------|----------|-----------|
| ADD x5,x3,x7 | ADD x5,x7,x3 | 0.40 | equivalent (commutative) |
| SLLI x5,x3,1 | SLLI x5,x3,1 | 0.79 | equivalent (exact recovery) |
| ADDI x5,x3,0 | SRL x3,x3,x3 | 1.44 | not equivalent |
| SUB x5,x3,x3 | BGEU x27,x5,7 | 0.75 | not equivalent (false cluster) |
| BEQ x3,x7,16 | (none valid) | — | — |

Findings:

1. **Search finds equivalents.** The closest valid instruction to
   ADD x5,x3,x7 was its commutative form ADD x5,x7,x3.
2. **Exact recovery is possible.** SLLI x5,x3,1 was recovered exactly.
3. **The false cluster causes wrong decodes.** SUB x5,x3,x3 (always zero)
   decoded to the branch BGEU, the SLT/BEQ/branch cluster pulling
   zero-producing instructions toward branches.
4. **Some regions are not navigable.** ADDI identity and BEQ could not be
   decoded: the landscape there is too rough for gradient search, or
   snapping loses too much.
5. **Snapping costs distance.** Successful decodes sat 0.4-0.8 from the
   target after snapping; discrete token sequences are sparse points on
   S^127.

The T1 geometry is smooth and navigable where the encoder learned well
(ALU ops, shifts) and rough or degenerate where it has known defects
(false clusters, branches). Search-based inversion works in principle,
so a decoding failure indicts the encoder's geometry, not the search.
