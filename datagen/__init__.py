"""Training data generation.

Three modules:
    datagen.generate    — instruction generation, relabeling,
                          equivalences, group-collection rules
    datagen.batch       — Chunk + Batch + RVT format, pack/unpack,
                          pair construction, full pipeline
    datagen.compare     — GVN equivalence + behavioral chunk distance
    datagen.invalidity  — invalid-window generators (validity training)

One binary format (RVT) carries the unified batch (chunks + optional
pair structure + validity flags). The pipeline tools auto-detect via
the format magic.
"""

from .generate import (
    random_instruction, validate_distribution, load_distribution,
    DEFAULT_DISTRIBUTION, random_basic_block,
    random_perm, relabel, random_relabel,
    MANIFEST, sample_binding, materialize, sample_injection_tuples,
    single, until_branch, until_transformation, length_cap, either,
    collect_groups,
)

from .batch import (
    Chunk, Batch, RVT_FORMAT,
    pack_batch, padding_mask, unpack_chunks,
    generate_chunks, build_pairs, collect_into_batches,
)

from .compare import (
    gvn_equivalent, behavioral_distance, behavioral_distance_cached,
    precompute_chunk, make_anchor_states,
)
