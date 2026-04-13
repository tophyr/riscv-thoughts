"""Compressor models.

Compressor: fixed-window model for context experiments.
StreamingCompressor: streaming encoder with learnable emit gate.
Decoder: autoregressive decoder conditioned on emission vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Compressor(nn.Module):
    """Fixed-window compressor. Takes a window of instruction tokens,
    produces a vector on S^(d_out-1)."""

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
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return F.normalize(self.proj(x), dim=-1)


def gumbel_sigmoid(logits, tau=1.0, hard=True):
    """Differentiable binary gate via Gumbel-sigmoid."""
    if not logits.requires_grad:
        return (logits > 0).float()
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logits + gumbels) / tau)
    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class StreamingCompressor(nn.Module):
    """Streaming compressor with learnable emit gate.

    Processes a full token sequence through a bidirectional transformer,
    then applies a per-position emit gate. At each emit point, pools
    the current window's token representations and projects to S^(d_out-1).

    Phase 1: accept is fixed (always-on), evict is structural (evict
    completed instructions on emit), only emit is learned.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_out,
                 max_seq_len=48, dropout=0.0, gate_tau=1.0):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.gate_tau = gate_tau

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.emit_gate = nn.Linear(d_model, 1)
        self.proj = nn.Linear(d_model, d_out)

    def forward(self, token_ids, padding_mask, token_instr_idx,
                n_instructions):
        """Process sequences and produce emissions.

        Returns:
            emissions: (N_total, d_out) L2-normalized vectors
            emission_info: list of dicts per emission with keys:
                'batch_idx', 'instr_start', 'instr_end',
                'target_tokens' (list[int] — the instruction tokens
                    covered by this emission, no BOS/EOS)
            emit_counts: (B,) number of emissions per sequence
            gate_logits: (B, L) raw gate logits
        """
        B, L = token_ids.shape
        device = token_ids.device

        # Encode full sequence.
        pos = torch.arange(L, device=device)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Emit gate.
        gate_logits = self.emit_gate(x).squeeze(-1)  # (B, L)
        gate_logits = gate_logits.masked_fill(padding_mask, -1e9)

        if self.training:
            emit_decisions = gumbel_sigmoid(gate_logits, tau=self.gate_tau)
        else:
            emit_decisions = (gate_logits > 0).float()

        # Force emit at the last non-padding position.
        seq_lens = (~padding_mask).sum(dim=1)  # (B,)
        for b in range(B):
            emit_decisions[b, seq_lens[b] - 1] = 1.0

        # Precompute on CPU for the window walk.
        instr_idx_np = token_instr_idx.cpu().numpy()
        tok_np = token_ids.cpu().numpy()

        # Simulate windows and collect emissions.
        all_vecs = []
        all_info = []
        emit_counts = []

        for b in range(B):
            slen = int(seq_lens[b])
            n_instr = int(n_instructions[b])

            # Find first/last token for each instruction.
            instr_first = {}
            instr_last = {}
            for t in range(slen):
                idx = int(instr_idx_np[b, t])
                if idx < 0:
                    continue
                if idx not in instr_first:
                    instr_first[idx] = t
                instr_last[idx] = t

            window_start = 0
            n_emits = 0

            for t in range(slen):
                if emit_decisions[b, t] < 0.5:
                    continue

                # Pool window [window_start, t] inclusive.
                window_repr = x[b, window_start:t + 1].mean(dim=0)
                vec = F.normalize(self.proj(window_repr), dim=-1)

                # Which instructions are FULLY in window?
                complete = []
                for k in range(n_instr):
                    if k in instr_first:
                        if (instr_first[k] >= window_start
                                and instr_last[k] <= t):
                            complete.append(k)

                if complete:
                    i_start = min(complete)
                    i_end = max(complete)

                    # Collect the raw instruction tokens (no BOS/EOS).
                    target_toks = []
                    for tt in range(slen):
                        idx = int(instr_idx_np[b, tt])
                        if i_start <= idx <= i_end:
                            target_toks.append(int(tok_np[b, tt]))

                    all_vecs.append(vec)
                    all_info.append({
                        'batch_idx': b,
                        'instr_start': i_start,
                        'instr_end': i_end,
                        'target_tokens': target_toks,
                        'window_size': t - window_start + 1,
                    })
                    n_emits += 1

                # Evict completed instructions.
                if complete:
                    last_evicted_tok = max(instr_last[k] for k in complete)
                    window_start = last_evicted_tok + 1
                    while (window_start < slen
                           and int(instr_idx_np[b, window_start]) < 0
                           and window_start <= t):
                        window_start += 1

            emit_counts.append(n_emits)

        if all_vecs:
            emissions = torch.stack(all_vecs)
        else:
            emissions = torch.zeros(0, self.d_out, device=device)

        return (emissions, all_info,
                torch.tensor(emit_counts, device=device),
                gate_logits)

    def encode_soft(self, soft_emb, padding_mask=None):
        """Encode from soft embeddings (for round-trip loss).

        Shares weights with the main encoder. Takes pre-computed
        embeddings instead of token IDs.

        Args:
            soft_emb: (N, T, d_model) — differentiable token embeddings
            padding_mask: (N, T) bool — True = padding
        Returns:
            (N, d_out) L2-normalized vectors
        """
        B, T, _ = soft_emb.shape
        pos = torch.arange(T, device=soft_emb.device)
        x = soft_emb + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return F.normalize(self.proj(x), dim=-1)


class Decoder(nn.Module):
    """Autoregressive decoder conditioned on an emission vector.

    Given an emission vector from the compressor, generates the
    token sequence that the emission covers. Uses cross-attention
    to the emission vector (projected to a single key-value pair)
    and causal self-attention over generated tokens.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_emb,
                 max_seq_len=64, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Project emission vector to decoder's d_model.
        self.emission_proj = nn.Linear(d_emb, d_model)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, emission_vecs, target_tokens, target_padding_mask):
        """Teacher-forced forward pass.

        Args:
            emission_vecs: (N, d_emb) — one vector per emission
            target_tokens: (N, T) int64 — target token sequences
                (shifted right: first token is BOS, predict next)
            target_padding_mask: (N, T) bool — True = padding

        Returns:
            logits: (N, T, vocab_size) — next-token predictions
        """
        N, T = target_tokens.shape
        device = target_tokens.device

        # Emission as a single memory token for cross-attention.
        memory = self.emission_proj(emission_vecs).unsqueeze(1)  # (N, 1, d_model)

        # Embed target tokens.
        pos = torch.arange(T, device=device)
        tgt = self.tok_emb(target_tokens) + self.pos_emb(pos)

        # Causal mask for autoregressive decoding.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device)

        out = self.decoder(
            tgt, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask)

        return self.head(out)
