# Roadmap

Forward-looking work only. Completed history lives in EXPERIMENT_LOG.md; the
current architecture and rationale are in RISCV_DESIGN.md.

## Status

The rename-equivariant T1/T2 stack is validated end-to-end (EXPERIMENT_LOG.md
Phase 14): one `Compressor` (a register-state machine), T1 = T2 with
`seq_len=1`, trained on one behavioral loss set, equivalence emergent. T2 routes
off T1's **predicted** binding, so the tier recursion is a real test, not
token-fed.

- **T1**: `runs/t1_equiv_d128_canonvp_250k` — equivariance exact (`0.00e+00`),
  binding ~100%.
- **T2**: `runs/t2_equiv_d512_cap4_2M_principled` — binding parity, GVN
  behavioral collapse (commutativity ~28× vs distinct).
- **Scope**: `branch+cap=4`, no memory ops. Throughput 8 ms/step @ 95–97% util.

## Next

### Decoder on the equivariant T1
The search-based decoder (DECODER_STRATEGY.md, `scripts/train_decoder.py`) has
not been validated against the equivariant T1's geometry. Retrain and measure:
token reconstruction (up-to-equivalence, since the encoder collapses equivalent
forms) and execution-equivalent recovery rate. This closes the
compress → decompress loop on the current encoder.

### Extend the equivariant core's scope
Each is real work but additive over the stable skeleton — "register-indexed
state; the 32 GP registers are the permutable slots" (RISCV_DESIGN.md "Open
problems"):

- **PC / branches.** Promote PC from an essence read to a non-renameable slot
  (renaming permutes only the 32 GP slots).
- **Memory (loads/stores).** Add a non-renameable memory slot. The real
  expressiveness risk: can the essence capture value-dependent memory dataflow
  / aliasing? Currently out of scope (`precompute_chunk` raises on memory ops).
- **Immediates.** Enrich T1's essence with richer rename-invariant immediate
  content.
- **Longer / looser chunks.** Beyond `branch+cap=4`.

### T3 — stack another tier
Sequences of T2 blocks → function-like units. The recursion test repeats: T3
consumes only T2's emitted essence + binding (routing off T2's predicted
binding, never raw tokens). Same `Compressor` core, `seq_len` = number of T2
blocks.

### Validity / parse + streaming segmentation
The equivariant intake assumes a valid parse, so validity shifts from a learned
output magnitude toward a parse-level gate. Open: how an equivariant T1
integrates with a streaming segmenter (STREAMING_COMPRESSOR.md) that proposes
possibly-invalid windows and must *discover* thought boundaries rather than be
handed them.

## Long horizon

- Loosened boundary supervision; eventually natural language (the motivating
  target). See WHAT_IS_A_THOUGHT.md "Method vs. Cognition" for the framing.
