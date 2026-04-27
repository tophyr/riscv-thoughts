"""Run T1 encoder on RVC chunks to produce T2 input batches.

The chunker proper — boundary detection, augmentation, RVC binary I/O —
lives in `datagen/chunkgen.py` (CPU-only, in the data pipeline). This
module is the trainer-side bridge: it takes an RVC ChunkBatch (per-chunk
per-instruction token spans) and runs the frozen T1 encoder on every
instruction to produce a T2Batch of T1 emission vectors ready for T2
training.

Why split: data-prep work (boundary detection, invalidity augmentation)
benefits from running CPU-side and parallel via mux; the only piece that
needs the GPU and the model is the T1 encoder forward pass, which lives
here.
"""

from dataclasses import dataclass

import numpy as np
import torch

from datagen.chunkgen import ChunkBatch, MAX_INSTR_TOKENS


@dataclass
class T2Batch:
    """A batch of T2 input chunks expressed as T1 emission vectors.

    Built from a ChunkBatch by encoding each per-instruction token span
    through the frozen T1 encoder. Trainer consumes this directly.
    """
    chunk_emissions: torch.Tensor   # (B, max_n_instrs, d_out) float
    chunk_lens:      torch.Tensor   # (B,)                     int64
    valid_mask:      torch.Tensor   # (B,)                     bool
    reg_delta:       torch.Tensor   # (B, n_inputs, 32)        int32
    chunk_type:      torch.Tensor   # (B,)                     int8


@torch.no_grad()
def encode_chunkbatch(cb: ChunkBatch, t1_encoder, device='cpu') -> T2Batch:
    """Run T1 encoder on the per-instruction tokens of an RVC ChunkBatch.

    Reshapes (B, max_n_instrs, MAX_INSTR_TOKENS) to (B*max_n_instrs,
    MAX_INSTR_TOKENS) for one batched forward through T1, then reshapes
    back. T1 is called via its compiled_encode path so this fwd doesn't
    fall back to eager.
    """
    B, M, T = cb.token_ids.shape
    flat_tokens = torch.from_numpy(cb.token_ids).to(device).view(B * M, T)
    flat_pad = torch.from_numpy(cb.instr_pad).to(device).view(B * M, T)
    # Force position 0 non-pad on empty rows so attention doesn't see
    # an all-padding sequence; outputs for those rows are gated below.
    all_pad = flat_pad.all(dim=1)
    flat_pad[all_pad, 0] = False

    vecs = t1_encoder.compiled_encode(flat_tokens, flat_pad)  # (B*M, d_out)
    d_out = vecs.shape[-1]
    chunk_emissions = vecs.view(B, M, d_out)

    # Zero out positions beyond the chunk's actual instruction count.
    chunk_lens = torch.from_numpy(cb.chunk_lens.astype(np.int64)).to(device)
    instr_positions = torch.arange(M, device=device).unsqueeze(0)  # (1, M)
    instr_valid = instr_positions < chunk_lens.unsqueeze(1)        # (B, M)
    chunk_emissions = chunk_emissions * instr_valid.unsqueeze(-1).to(
        chunk_emissions.dtype)

    return T2Batch(
        chunk_emissions=chunk_emissions,
        chunk_lens=chunk_lens,
        valid_mask=torch.from_numpy(cb.valid_mask).to(device),
        reg_delta=torch.from_numpy(cb.reg_delta).to(device),
        chunk_type=torch.from_numpy(cb.chunk_type).to(device),
    )


def concat_t2_batches(batches) -> T2Batch:
    """Concatenate a list of T2Batches along the chunk dimension.

    All inputs must share the same chunk_emissions instruction-axis
    width; the trainer accumulates batches with matching sizes since
    they come from the same RVC stream.
    """
    nonempty = [b for b in batches if b.chunk_emissions.shape[0] > 0]
    if not nonempty:
        if not batches:
            raise ValueError('concat_t2_batches: no batches to concat')
        return batches[0]
    return T2Batch(
        chunk_emissions=torch.cat([b.chunk_emissions for b in nonempty], dim=0),
        chunk_lens=torch.cat([b.chunk_lens for b in nonempty], dim=0),
        valid_mask=torch.cat([b.valid_mask for b in nonempty], dim=0),
        reg_delta=torch.cat([b.reg_delta for b in nonempty], dim=0),
        chunk_type=torch.cat([b.chunk_type for b in nonempty], dim=0),
    )


# Re-export the type code constants so existing trainer code keeps working
# without reaching across packages.
from datagen.chunkgen import (
    TYPE_NON_TERMINATOR, TYPE_LOAD, TYPE_STORE, TYPE_BRANCH, TYPE_JUMP,
    TYPE_CAPPED, TYPE_TAIL,
    INVALID_SPANNING, INVALID_MULTI, INVALID_OVERLONG,
)
