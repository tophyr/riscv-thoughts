# Bootstrapping a session

If you cannot already state the current mission and why it is the mission, you
are not oriented yet: **run the `bootstrap` skill** before touching anything. It
walks the doc reading order, the code checks that keep the docs honest, and the
questions to bring back to me before starting. I would much rather spend time
clarifying up front than have you build on a wrong premise.

# Documentation

Docs and code comments state current truth for a reader who has never seen the
repo. The **`document` skill** carries that standard, the routing of content
to documents, and the audits; invoke it before any documentation pass and
before a commit that touches a `.md` file.

# A refactor or vectorization keeps the math

Old and new must agree on the math. Compute a side-by-side
sample on the same inputs; confirm bit-equivalence or
correlation = 1.0 for floats. If they don't match, the refactor
silently changed semantics — stop and flag, don't ship.

**Verify against a real checkpoint, not just the test suite.** No
test loads one, so the suite cannot catch a `state_dict`
incompatibility: a new unconditional parameter passes the whole
suite and fails at step 0 of the next run against the frozen T1.

Breaking changes are acceptable — including reminting the frozen
T1 — on the one condition that old and new can be compared and
shown not to have regressed. That condition is what licenses
deleting the worse way instead of keeping it behind a flag.

# Data generation pipeline

Nearly every task here touches datagen. The pipeline guide — canonical commands,
the flag reference, the RVT wire format, configs, multinode operation — is the
**`datagen` skill**. Invoke it before editing `datagen/`, the `scripts/` stream
tools, the RVT format, or the multinode generators, and before running any
generation or training pipeline. Do not improvise a datagen command from memory;
copy the skill's canonical commands.

The invariants below stay here, always loaded, because violating one silently
corrupts data or training with no error — you cannot opt into a warning you do
not know you need.

## The invariants

1. **The vp evaluator (the value-prediction head, `vp_head`) reads its input
   values from the batch itself, never from a rebuilt set.** Each row is
   executed from one structured draw per chunk (`draw_anchor_states`,
   `datagen/compare.py`), partly rewritten to straddle that chunk's
   boundaries when the config aims it (`aim_anchor_states`, `datagen/aim.py`),
   placed on that row's own input registers by `precompute_chunk`. The batch
   ships each output's value (`out_values`) and the chunk's input values in
   canonical role order (`in_role_values`, registers `in_roles`), both from
   that one execution; the trainer feeds `in_role_values` to `vp_loss`
   (`compressor/train.py`), and a scorer must do the same (`vp_predict`). Building input values from any
   other set of states — a seed, a fixed table — gives wrong targets with no
   error. Score at the fixed held-out set by generating the corpus with
   `datagen --heldout-anchors` (`heldout_anchor_states`,
   `datagen/compare.py`), whose states no training draw produces. An output
   computed from the PC after a jump also depends on the jump's operands,
   which `in_roles` does not list; PC effects are out of scope.

2. **Fixed batch shape, derived from the rule's length cap.** Every emitted batch
   is padded to identical (B, max_tokens, max_n_instrs) so PyTorch's caching
   allocator doesn't fragment. Stochastically-varying batch shape is a real OOM
   trigger (it OOMed a T1 run at ~step 500). The targets come from
   `max_chunk_len = rule.max_len` (`collect_into_batches`,
   `datagen/batch.py:668`).
   This is why a `--rule` with no cap is REJECTED (datagen skill, `--rule`
   grammar). Don't add a
   code path that emits a differently-shaped batch mid-run.

3. **EVERY rule ships the per-output value targets (`OB == B`).**
   `single` (T1) and multi-instruction (T2) rules both go through
   `build_targets` and ship `out_values`, `in_roles` and `in_role_values`
   (`OB == B`, carrying `n_anchors == --n-states`, on the 2K input positions
   of a K-instruction corpus). There is no per-rule payload split — both
   tiers feed one value-prediction target. (`scripts/tests/test_pipeline.py::test_value_targets_header`.)

4. **value_predict is evaluated on every written output at once, from the
   ESSENCE, in the register frame, for both tiers.** The target for output
   slot k is the raw value of register `out_slot_regs[k]` after the chunk
   (`out_values`); the evaluator (`vp_loss`, `compressor/train.py`) is handed
   the unit-normalized essence and, for each register in `in_roles`, its
   value (`in_role_values`) and its colour channels from `T` — never an
   output's own description and never a name for an output. It predicts every
   output as a set, matched to the outputs by the assignment of least loss
   (`vp_assignment`). No input or output is numbered by textual order or by
   register number: `in_roles` lists the chunk's inputs in canonical role
   order — ordered by what the function does with them and by whether the
   chunk overwrites them (`datagen/roles.py`), so equivalent chunks of
   different dataflow shapes present corresponding inputs at the same
   positions — with a tie between interchangeable inputs broken by the
   encoder's colour, every role ordered. Isomorphic parts of one chunk tie on
   role and colour too, and are singled out by register number; either choice
   gives the same set of outputs, so a rename presents the evaluator another
   draw of values, not another layout
   (`test_roles.py::test_equivalent_chunks_hand_the_evaluator_the_same_values`).
   The encoder seeds each leaf from the same colouring computed from the wiring
   it is given (`RegisterStateMachine._prepass`, `compressor/model.py`). The
   oracle's colouring (strings) and the encoder's (fixed random features)
   must agree on structure: identical graph, identical colour. The one place
   they differ by design is commutativity — the oracle's SIGNATURE knows which
   operations commute (`COMMUTATIVE`); the encoder and the input order are
   told nothing of it and order every role. An input order built with the
   commutative roles merged ties the two operands of an `ADD` and breaks the
   tie by register number, so a rename moves one operation's operands between
   evaluator positions with no symmetry to absorb it: no error, only a
   register-dependent gradient on every chunk with a commutative op.
   `datagen/tests/test_signature.py` is the oracle's evidence against the
   emulator.

   **Which reads count, and liveness, are decided structurally, never at the
   anchors.** Which reads are behavioral is decided per instruction at FIXED
   witness states (`instruction_reads`, `datagen/compare.py`), so a per-row
   anchor draw can never flip a label; `in_slot_regs` (the binding target)
   lists them in first-read order, a dead instruction's included
   (`_input_order`). A dead instruction — overwritten before anyone reads it —
   is pruned from the colouring by a backward pass from the final writers, in
   the oracle (`datagen.signature.live_nodes`) and in the encoder alike, so
   its reads leave no trace in any colour; they remain in the binding.

5. **x0 is fixed under relabeling.** `random_perm`/`relabel` require `perm[0]==0`
   (`datagen/generate.py:312,326`; the check is `datagen/generate.py:334-335`).
   x0 is hardwired zero, not a register.

6. **Memory-op chunks get no aux (V1 scope).** `build_targets`
   (`datagen/batch.py:606`) skips chunks containing LOAD/STORE because their
   `Precomputed` is unavailable — `precompute_chunk` raises
   NotImplementedError on memory ops (`datagen/compare.py:719-721`). The
   chunk's tokens are still shipped (so the encoder sees them for
   magnitude/validity), but its aux row is zero-mask / `AUX_CE_IGNORE` and its
   value-prediction row is masked off via `out_regs_valid=False`
   (`_values_from_precomputeds`, `datagen/batch.py:409`).

7. **The behavioral distance compares two rows' register files at the SAME
   inputs, placed by canonical role, minimised over placements.** Every chunk
   is executed at one shared set of witness states, each state's registers
   shuffled so no position is special, its canonical inputs
   (`datagen/roles.py`: ordered by what the function does with them and by
   whether the chunk overwrites them) placed on register positions under every
   placement (`placements`, `datagen/compare.py`); it ships the final value on
   each position — the new value of a register it wrote, the placed value of
   one it did not — and of each register written but not read (`rf_pos`,
   `rf_wo`, `precompute_chunk`). The target compares one row at its canonical
   placement with the other at every placement and keeps the cheapest
   (`register_distance_targets`, `compressor/train.py`; RISCV_DESIGN,
   "Collapse"). Comparing each row at its own anchor draw, placing values by
   register number, or comparing outputs without the register's own old value
   gives a target that is nonzero for a rename, or zero between a copy and an
   identity write — no error, only a collapse objective pulling the wrong
   pairs. `max_dist` comes from the bits a column never changes across the
   fixed bound family (`bound_states`, `datagen.knownbits.observed_bits`), never from a static
   analysis of the spelling, which bounds equivalent spellings differently.
   (`datagen/tests/test_compare.py::test_register_file_fields_match_the_cpu_emulator`,
   `datagen/tests/test_roles.py`,
   `compressor/tests/test_train.py::test_register_distance_targets_track_the_reference_distance`.)
