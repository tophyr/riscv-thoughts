"""T2 compressor: encodes a sequence of T1 emission vectors into a
single T2 vector representing the chunk's register-state-transformation.

Architecture mirrors T1Compressor's window encoder, lifted one level:

- T1 took `(B, L) int64` token IDs and an embedding lookup. T2 takes
  `(B, max_chunk_len, d_in)` continuous vectors directly, with a linear
  input projection from d_in to d_model.
- Position embeddings over chunk positions, same shape as T1's
  win_pos_emb but indexed by instruction-within-chunk position
  rather than token-within-instruction position.
- Bidirectional transformer encoder over the chunk's emission
  sequence with a padding mask derived from chunk_lens.
- Pool by taking the last non-pad position's representation
  (same convention as T1's encoder).
- Linear projection to d_out. No F.normalize — T2 lives in the
  unit ball, magnitude carries validity, direction carries
  semantics (same convention as T1).

Auxiliary heads (mirrors T1's dest_type / dest_reg pattern, lifted
to chunk-level static structural properties):

- modified_regs_head: BCE over 32-D logits. For each register
  index, did the chunk modify this register's value in at least
  one probed input state? Forces T2 to encode WHICH registers
  the chunk affects.
- terminator_type_head: 5-class CE. ALU-only / LOAD / STORE /
  BRANCH / JUMP. Forces T2 to encode the chunk's terminator
  category, which determines the chunk's external interface
  (memory effect, control transfer, or pure register state).

Both heads are called from the training loop with normalized
direction `T2 / ||T2||` (magnitude-invariant classification, same
consumer-split fix as T1's dest heads).

The dynamic value information (what each modified register's new
value is, as a function of input state) is encoded implicitly via
pair-MSE distance against register-state-delta over n_inputs probed
states — same pattern T1 uses for instruction-output values.

The class is small and intentionally uncomplicated. The training
loop, magnitude-as-validity loss, pair-MSE on register-state-delta,
BCE/CE on the aux heads, and the T2 training entry point live in
compressor/train.py.
"""

import torch
import torch.nn as nn


# Terminator-type classes for the aux head. Maps the chunker's
# fine-grained TYPE_* codes to a 5-class space the head predicts.
TERM_ALU    = 0  # CAPPED + TAIL (no terminator at end of chunk)
TERM_LOAD   = 1
TERM_STORE  = 2
TERM_BRANCH = 3
TERM_JUMP   = 4

N_TERMINATOR_CLASSES = 5
N_REGS = 32


class T2Compressor(nn.Module):
    """Bidirectional transformer encoder over a sequence of T1 emissions.

    Forward pass: chunk_emissions (B, T, d_in) + chunk_lens (B,) -> T2 (B, d_out).

    No F.normalize on the output — T2 magnitude carries validity, the
    same magnitude-as-validity framing as T1. Training is responsible
    for shaping ||T2|| against the chunker's valid_mask.

    Aux heads (modified_regs_head, terminator_type_head) are members
    but are not called from forward(); the training loop calls them
    with normalized direction `T2 / ||T2||` so classification is
    magnitude-invariant.
    """

    def __init__(self, d_in: int = 64, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2,
                 d_out: int = 256, max_chunk_len: int = 24,
                 dropout: float = 0.0):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.max_chunk_len = max_chunk_len

        # Input projection: linear from d_in (= T1's d_out) up to d_model.
        # No nonlinearity — the transformer layers below handle that.
        self.input_proj = nn.Linear(d_in, d_model)

        # Position embeddings over instruction-within-chunk positions.
        self.pos_emb = nn.Embedding(max_chunk_len, d_model)

        # Bidirectional transformer encoder over the chunk's sequence.
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        # Output projection to d_out. Deliberately no F.normalize:
        # magnitude is the validity signal.
        self.proj = nn.Linear(d_model, d_out)

        # Auxiliary heads (read normalized direction during training,
        # see module docstring). Static structural properties — they
        # don't depend on dynamic register values, those come via
        # pair-MSE.
        self.modified_regs_head = nn.Linear(d_out, N_REGS)
        self.terminator_type_head = nn.Linear(d_out, N_TERMINATOR_CLASSES)

    def forward(self, chunk_emissions: torch.Tensor,
                chunk_lens: torch.Tensor) -> torch.Tensor:
        """Encode a batch of T1-emission chunks to T2 vectors.

        chunk_emissions: (B, T, d_in)
        chunk_lens:      (B,) — actual length of each chunk (1..T)
        Returns:         (B, d_out)
        """
        B, T, d_in = chunk_emissions.shape
        if d_in != self.d_in:
            raise ValueError(
                f'expected d_in={self.d_in}, got {d_in}')
        if T > self.max_chunk_len:
            raise ValueError(
                f'sequence length {T} exceeds max_chunk_len={self.max_chunk_len}')

        device = chunk_emissions.device

        # Padding mask: True at positions ≥ chunk_lens.
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        pad_mask = positions >= chunk_lens.unsqueeze(1)          # (B, T)
        # Defensive: if any row is all-padding (chunk_lens == 0), force
        # position 0 to non-pad so the transformer's attention doesn't
        # see an all-padding sequence. The result for those rows is
        # zeroed below anyway.
        pad_mask[:, 0] = False

        # Linear projection + position embedding.
        x = self.input_proj(chunk_emissions) + self.pos_emb(
            torch.arange(T, device=device))

        # Bidirectional transformer encoding.
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Pool by the last non-pad position's representation.
        last_idx = (chunk_lens - 1).clamp(min=0)
        last_repr = x[torch.arange(B, device=device), last_idx]

        # Output projection — raw, no F.normalize. Magnitude is validity.
        out = self.proj(last_repr)

        # Zero rows whose original chunk_lens was 0 (defensive — these
        # shouldn't be produced by the chunker, but if they slip through
        # they shouldn't contribute spurious signal).
        nonempty = (chunk_lens > 0).unsqueeze(1).float()
        return out * nonempty
