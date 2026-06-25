"""Training data generation.

Four modules:
    datagen.generate    — instruction generation, relabeling,
                          equivalences, group-collection rules
    datagen.batch       — Chunk + Batch + RVT format, pack,
                          twin/aux construction, full pipeline
    datagen.compare     — SSA analysis, anchor execution, aux targets
    datagen.invalidity  — invalid-window generators (validity training)

This package facade is the public API: external consumers (compressor,
scripts) import from `datagen` directly. The submodules and their
private helpers are the internal surface, used by tests and by each
other. One binary format (RVT) carries the batch (chunks + row-outputs
value-prediction targets + register-identity aux); the pipeline tools
auto-detect via the format magic.
"""

from .generate import (
    random_instruction, validate_distribution, load_distribution,
    DEFAULT_DISTRIBUTION, build_opcode_table,
    MANIFEST, sample_binding, materialize,
    single, until_branch, until_transformation, length_cap, either,
)

from .batch import (
    Batch, RVT_FORMAT, padding_mask,
    generate_chunks, collect_into_batches,
)

from .compare import (
    make_anchor_states, precompute_chunk,
    N_REGS, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS, AUX_CE_IGNORE,
)

from .invalidity import (
    gen_partial, gen_spanning, gen_multi, gen_bogus,
    generate_invalid, build_type_table, DEFAULT_TYPE_WEIGHTS,
)
