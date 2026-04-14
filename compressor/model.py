"""Compressor models.

Compressor: fixed-window model for context experiments.
ShiftReduceCompressor: neural shift-reduce parser with per-iteration
    gate decisions and bidirectional window encoder.
Decoder: autoregressive decoder conditioned on emission vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Compressor(nn.Module):
    """Fixed-window compressor for context experiments."""

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


class ShiftReduceCompressor(nn.Module):
    """Neural shift-reduce parser for thought compression.

    Per iteration (batched across all active sequences):
    1. Window encoder runs on each sequence's window → T1 candidates
    2. GRU processes T1 candidates → updates hidden states
    3. All three gates sample decisions (Bernoulli from learned logits)
    4. Execute gate decisions, update window states
    5. Record T1 candidate + window tokens (for decoder evaluation)

    Gate training: gates are trained via REINFORCE using per-iteration
    reconstruction quality (decoder evaluates T1 candidate against
    current window tokens at every iteration).
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_out,
                 max_window=64, dropout=0.0,
                 max_iterations_per_token=4):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.max_iters_per_token = max_iterations_per_token

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.win_pos_emb = nn.Embedding(max_window, d_model)
        win_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.win_encoder = nn.TransformerEncoder(win_layer, n_layers)
        self.proj = nn.Linear(d_model, d_out)

        self.gru = nn.GRUCell(d_out, d_model)

        self.accept_head = nn.Sequential(
            nn.Linear(d_model + d_out + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1))
        self.emit_head = nn.Sequential(
            nn.Linear(d_out + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1))
        self.evict_head = nn.Sequential(
            nn.Linear(d_out + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1))

        nn.init.constant_(self.accept_head[-1].bias, 2.0)
        nn.init.constant_(self.emit_head[-1].bias, -2.0)
        nn.init.constant_(self.evict_head[-1].bias, -2.0)

    def _encode_windows_batched(self, all_window_tokens, device):
        """Encode multiple windows in one batched pass.

        Returns: (B, d_out) T1 candidates. Zero vector for empty windows.
        """
        B = len(all_window_tokens)
        if B == 0:
            return torch.zeros(0, self.d_out, device=device)

        nonempty = [(i, toks) for i, toks in enumerate(all_window_tokens)
                    if toks]

        result = torch.zeros(B, self.d_out, device=device)
        if not nonempty:
            return result

        indices, tok_lists = zip(*nonempty)
        max_len = max(len(t) for t in tok_lists)
        N = len(tok_lists)

        win_ids = torch.zeros(N, max_len, dtype=torch.long, device=device)
        win_pad = torch.ones(N, max_len, dtype=torch.bool, device=device)
        for j, toks in enumerate(tok_lists):
            L = len(toks)
            win_ids[j, :L] = torch.tensor(toks, dtype=torch.long,
                                          device=device)
            win_pad[j, :L] = False

        pos = torch.arange(max_len, device=device)
        x = self.tok_emb(win_ids) + self.win_pos_emb(pos)
        x = self.win_encoder(x, src_key_padding_mask=win_pad)

        lengths = (~win_pad).sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        last_repr = x[torch.arange(N, device=device), last_idx]
        vecs = F.normalize(self.proj(last_repr), dim=-1)

        for j, idx in enumerate(indices):
            result[idx] = vecs[j]

        return result

    def forward(self, token_ids, padding_mask, token_instr_idx,
                n_instructions):
        """Run batched shift-reduce loop.

        Returns:
            iter_t1s: list of (B, d_out) T1 candidates, one per iteration
            iter_window_tokens: list of [list of token IDs per seq],
                one per iteration
            iter_gate_log_probs: list of (B, 3) log-probabilities of
                the sampled gate decisions [accept, emit, evict]
            iter_gate_decisions: list of (B, 3) bool decisions
            emission_info: list of dicts for actual emissions
                (with 'batch_idx', 'instr_start', 'instr_end',
                 'target_tokens', 'window_size', 'has_complete',
                 'iteration', 't1_index')
            emit_counts: (B,) emissions per sequence
        """
        B, L = token_ids.shape
        device = token_ids.device
        tok_np = token_ids.cpu().numpy()
        idx_np = token_instr_idx.cpu().numpy()
        pad_np = padding_mask.cpu().numpy()
        n_instr_np = n_instructions.cpu().numpy()

        all_tok_embs = self.tok_emb(token_ids)  # (B, L, d_model)

        # Per-sequence state.
        seq_lens = [int((~pad_np[b]).sum()) for b in range(B)]
        windows = [[] for _ in range(B)]
        input_pos = [0] * B
        gru_h = torch.zeros(B, self.d_model, device=device)
        done = [False] * B

        # Precompute instruction boundaries.
        instr_first = [{} for _ in range(B)]
        instr_last = [{} for _ in range(B)]
        token_to_instr = [{} for _ in range(B)]
        for b in range(B):
            for t in range(seq_lens[b]):
                ii = int(idx_np[b, t])
                if ii >= 0:
                    if ii not in instr_first[b]:
                        instr_first[b][ii] = t
                    instr_last[b][ii] = t
                token_to_instr[b][t] = ii

        # Per-iteration outputs.
        iter_t1s = []
        iter_window_tokens = []
        iter_gate_log_probs = []
        iter_gate_decisions = []

        # Emission tracking.
        emission_info = []
        emit_counts = [0] * B

        max_total_iters = max(seq_lens) * self.max_iters_per_token
        iteration = 0

        while not all(done) and iteration < max_total_iters:
            # 1. Batch-encode windows.
            win_tok_lists = [[wt[0] for wt in windows[b]]
                             if not done[b] else []
                             for b in range(B)]
            t1 = self._encode_windows_batched(win_tok_lists, device)

            # 2. GRU update.
            active = torch.tensor([not d for d in done],
                                  dtype=torch.bool, device=device)
            if active.any():
                new_h = self.gru(t1, gru_h)
                gru_h = torch.where(active.unsqueeze(1), new_h, gru_h)

            # 3. Gate logits (batched).
            next_embs = torch.zeros(B, self.d_model, device=device)
            has_input = [False] * B
            for b in range(B):
                if not done[b] and input_pos[b] < seq_lens[b]:
                    next_embs[b] = all_tok_embs[b, input_pos[b]]
                    has_input[b] = True

            accept_logits = self.accept_head(
                torch.cat([next_embs, t1, gru_h], dim=1)).squeeze(-1)
            emit_logits = self.emit_head(
                torch.cat([t1, gru_h], dim=1)).squeeze(-1)
            evict_logits = self.evict_head(
                torch.cat([t1, gru_h], dim=1)).squeeze(-1)

            # Sample gate decisions (Bernoulli) and compute log-probs.
            accept_probs = torch.sigmoid(accept_logits)
            emit_probs = torch.sigmoid(emit_logits)
            evict_probs = torch.sigmoid(evict_logits)

            accept_sample = torch.bernoulli(accept_probs)
            emit_sample = torch.bernoulli(emit_probs)
            evict_sample = torch.bernoulli(evict_probs)

            # Log-probabilities of the sampled decisions.
            # Recompute from logits (not from probs used for sampling)
            # to get a fresh computation graph for REINFORCE backward.
            accept_lp = F.logsigmoid(
                torch.where(accept_sample > 0.5,
                            accept_logits, -accept_logits))
            emit_lp = F.logsigmoid(
                torch.where(emit_sample > 0.5,
                            emit_logits, -emit_logits))
            evict_lp = F.logsigmoid(
                torch.where(evict_sample > 0.5,
                            evict_logits, -evict_logits))

            # Record per-iteration data.
            iter_t1s.append(t1)
            iter_window_tokens.append(
                [list(wt) for wt in win_tok_lists])  # deep copy
            iter_gate_log_probs.append(
                torch.stack([accept_lp, emit_lp, evict_lp], dim=1))
            iter_gate_decisions.append(
                torch.stack([accept_sample, emit_sample, evict_sample],
                            dim=1))

            # 4. Execute decisions.
            for b in range(B):
                if done[b]:
                    continue

                a = accept_sample[b].item() > 0.5 and has_input[b]
                e = emit_sample[b].item() > 0.5 and len(windows[b]) > 0
                v = evict_sample[b].item() > 0.5 and len(windows[b]) > 0

                if not any([a, e, v]):
                    if has_input[b]:
                        a = True
                    elif windows[b]:
                        e = True

                if a and input_pos[b] < seq_lens[b]:
                    windows[b].append(
                        (int(tok_np[b, input_pos[b]]), input_pos[b]))
                    input_pos[b] += 1

                if e and windows[b]:
                    target_toks = [wp[0] for wp in windows[b]]
                    win_orig = [wp[1] for wp in windows[b]]
                    n_instr = int(n_instr_np[b])
                    complete = []
                    for k in range(n_instr):
                        if (k in instr_first[b]
                                and instr_first[b][k] in win_orig
                                and instr_last[b][k] in win_orig):
                            complete.append(k)

                    emission_info.append({
                        'batch_idx': b,
                        'instr_start': min(complete) if complete else -1,
                        'instr_end': max(complete) if complete else -1,
                        'target_tokens': target_toks,
                        'window_size': len(windows[b]),
                        'has_complete': len(complete) > 0,
                        'iteration': iteration,
                    })
                    emit_counts[b] += 1

                if v and windows[b]:
                    windows[b].pop(0)

                if input_pos[b] >= seq_lens[b] and not windows[b]:
                    done[b] = True

            iteration += 1

        return (iter_t1s, iter_window_tokens,
                iter_gate_log_probs, iter_gate_decisions,
                emission_info,
                torch.tensor(emit_counts, device=device))

    def encode_soft(self, soft_emb, padding_mask=None):
        """Encode from soft embeddings (for round-trip loss)."""
        B, T, _ = soft_emb.shape
        pos = torch.arange(T, device=soft_emb.device)
        x = soft_emb + self.win_pos_emb(pos)
        x = self.win_encoder(x, src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)
            last_repr = x[torch.arange(B, device=soft_emb.device), last_idx]
        else:
            last_repr = x[:, -1]
        return F.normalize(self.proj(last_repr), dim=-1)


class Decoder(nn.Module):
    """Autoregressive decoder conditioned on an emission vector."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_emb,
                 max_seq_len=64, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.emission_proj = nn.Linear(d_emb, d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, emission_vecs, target_tokens, target_padding_mask):
        N, T = target_tokens.shape
        device = target_tokens.device
        memory = self.emission_proj(emission_vecs).unsqueeze(1)
        pos = torch.arange(T, device=device)
        tgt = self.tok_emb(target_tokens) + self.pos_emb(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device)
        out = self.decoder(
            tgt, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask)
        return self.head(out)
