"""Training data generation.

Two formats:
    datagen.seqgen — instruction sequence batches (RVS format)
    datagen.batchgen — single-instruction batches (RVB format)

Import from the specific module for I/O functions, or use the
pipeline tools (batch_slice, batch_cat, etc.) which auto-detect.
"""

# Shared instruction generation.
from .instrgen import (
    random_instruction, validate_distribution, load_distribution,
    DEFAULT_DISTRIBUTION,
)

# Sequence data.
from .seqgen import (
    SequenceBatch, SequenceBatchReader,
    random_basic_block,
    execute_sequence, produce_sequence_batch,
)

# Single-instruction data.
from .batchgen import (
    InstructionBatch, InstructionBatchReader,
    produce_instruction_batch, extract_data_val,
    dest_type, dest_reg,
)
