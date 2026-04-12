"""Fixed-window compressor model.

Takes a window of instruction tokens, produces a vector on S^(d_out-1).
"""

import torch
import torch.nn as nn


class Compressor(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_out,
                 max_seq_len=24, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d_model, d_out)

    def forward(self, token_ids, padding_mask=None):
        """
        token_ids: (B, L) int64
        padding_mask: (B, L) bool, True = padding
        Returns: (B, d_out) L2-normalized vectors
        """
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Mean pool over non-padding positions.
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return nn.functional.normalize(self.proj(x), dim=-1)
