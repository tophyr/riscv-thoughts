"""T0→T1 compressor model.

Takes an instruction's tokens (4-7 of them) and produces a single T1
vector plus destination classification logits. Trained jointly — no
frozen encoder, no pre-training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class T1Compressor(nn.Module):
    """Compresses a single RV32I instruction's tokens to a T1 vector.

    Architecture: token embeddings + learned positional embeddings →
    transformer encoder → mean pooling → linear projection → T1 vector.

    Classification heads read from the T1 vector to predict the
    instruction's destination type (register/memory) and destination
    register number.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_out: int,
        max_seq_len: int = 10,
        dropout: float = 0.0,
        n_dest_types: int = 2,
        n_regs: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out

        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                              enable_nested_tensor=False)
        self.output_proj = nn.Linear(d_model, d_out)

        self.dest_type_head = nn.Linear(d_out, n_dest_types)
        self.dest_reg_head = nn.Linear(d_out, n_regs)

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode tokens to T1 vectors and destination predictions.

        Args:
            token_ids: (batch, seq_len) int tensor of token IDs.
            padding_mask: (batch, seq_len) bool tensor, True = padding.

        Returns:
            (t1_vecs, dest_type_logits, dest_reg_logits):
                t1_vecs: (batch, d_out) T1 vectors.
                dest_type_logits: (batch, n_dest_types) classification logits.
                dest_reg_logits: (batch, n_regs) classification logits.
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.tok_embedding(token_ids) + self.pos_embedding(positions)

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Mean pooling over non-padding positions.
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()  # (B, L, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        t1 = F.normalize(self.output_proj(x), dim=-1)
        return t1, self.dest_type_head(t1), self.dest_reg_head(t1)
