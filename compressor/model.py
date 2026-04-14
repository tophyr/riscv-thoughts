"""T1 Compressor and Decoder.

T1Compressor: neural shift-reduce parser with per-iteration gate
    decisions and bidirectional window encoder. Can also operate in
    fixed-window mode (no gates) for encoder-only training.
Decoder: autoregressive decoder conditioned on emission vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class T1Compressor(nn.Module):
    """Neural shift-reduce parser for thought compression.

    Operates in two modes:

    **Streaming mode** (forward): runs the full shift-reduce loop
    with gates, GRU, and per-iteration state tracking. Used for
    gate training and inference.

    **Fixed-window mode** (encode): encodes a batch of token
    sequences directly through the window encoder. No gates, no
    GRU. Used for encoder-only training (step 1) and evaluation.

    Both modes share the same window encoder and projection, so
    weights trained in fixed-window mode transfer to streaming mode.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_out,
                 max_window=64, dropout=0.0,
                 max_iterations_per_token=4,
                 n_dest_types=2, n_regs=32):
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

        # Destination classification heads. Read from the normalized
        # T1 vector to force the encoder to carry destination info
        # that the scalar exec_distance metric is blind to.
        self.dest_type_head = nn.Linear(d_out, n_dest_types)
        self.dest_reg_head = nn.Linear(d_out, n_regs)

        self.gru = nn.GRUCell(d_out, d_model)

        # Accept: next_token_emb + t1 + gru_hidden
        self.accept_head = nn.Sequential(
            nn.Linear(d_model + d_out + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1))
        # Emit: t1 + gru_hidden
        self.emit_head = nn.Sequential(
            nn.Linear(d_out + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1))
        # Evict: t1 + gru_hidden
        self.evict_head = nn.Sequential(
            nn.Linear(d_out + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1))

        nn.init.constant_(self.accept_head[-1].bias, 2.0)
        nn.init.constant_(self.emit_head[-1].bias, -2.0)
        nn.init.constant_(self.evict_head[-1].bias, -2.0)

        # Compiled encode: a standalone function capturing only the
        # encoder submodules, so torch.compile doesn't trace through
        # the gate/GRU submodules unused during encoding.
        tok_emb = self.tok_emb
        pos_emb = self.win_pos_emb
        win_encoder = self.win_encoder
        proj = self.proj

        def _encode_fn(token_ids, padding_mask):
            B, L = token_ids.shape
            x = tok_emb(token_ids) + pos_emb(
                torch.arange(L, device=token_ids.device))
            x = win_encoder(x, src_key_padding_mask=padding_mask)
            if padding_mask is not None:
                lengths = (~padding_mask).sum(dim=1)
                last_idx = (lengths - 1).clamp(min=0)
                last_repr = x[torch.arange(B, device=token_ids.device),
                              last_idx]
            else:
                last_repr = x[:, -1]
            return F.normalize(proj(last_repr), dim=-1)

        self.encode = _encode_fn
        self.compiled_encode = torch.compile(_encode_fn)

    def encode_soft(self, soft_emb, padding_mask=None):
        """Encode from soft embeddings (for round-trip loss).

        Same as encode() but takes pre-computed embeddings instead
        of token IDs. Shares all weights.
        """
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

    def _encode_windows_batched(self, all_window_tokens, device):
        """Encode multiple windows in one batched pass.

        all_window_tokens: list of lists of token IDs, one per sequence
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

        vecs = self.encode(win_ids, win_pad)

        for j, idx in enumerate(indices):
            result[idx] = vecs[j]

        return result

    def forward(self, token_ids, padding_mask, token_instr_idx,
                n_instructions):
        """Run batched shift-reduce loop (streaming mode).

        Returns:
            iter_t1s: list of (B, d_out) T1 candidates per iteration
            iter_window_tokens: list of window token lists per iteration
            iter_gate_log_probs: list of (B, 3) gate log-probs
            iter_gate_decisions: list of (B, 3) gate decisions
            emission_info: list of dicts for actual emissions
            emit_counts: (B,) emissions per sequence
        """
        B, L = token_ids.shape
        device = token_ids.device
        tok_np = token_ids.cpu().numpy()
        idx_np = token_instr_idx.cpu().numpy()
        pad_np = padding_mask.cpu().numpy()
        n_instr_np = n_instructions.cpu().numpy()

        all_tok_embs = self.tok_emb(token_ids)

        seq_lens = [int((~pad_np[b]).sum()) for b in range(B)]
        windows = [[] for _ in range(B)]
        input_pos = [0] * B
        gru_h = torch.zeros(B, self.d_model, device=device)
        done = [False] * B

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

        iter_t1s = []
        iter_window_tokens = []
        iter_gate_log_probs = []
        iter_gate_decisions = []
        emission_info = []
        emit_counts = [0] * B

        max_total_iters = max(seq_lens) * self.max_iters_per_token
        iteration = 0

        while not all(done) and iteration < max_total_iters:
            win_tok_lists = [[wt[0] for wt in windows[b]]
                             if not done[b] else []
                             for b in range(B)]
            t1 = self._encode_windows_batched(win_tok_lists, device)

            active = torch.tensor([not d for d in done],
                                  dtype=torch.bool, device=device)
            if active.any():
                new_h = self.gru(t1, gru_h)
                gru_h = torch.where(active.unsqueeze(1), new_h, gru_h)

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

            accept_probs = torch.sigmoid(accept_logits)
            emit_probs = torch.sigmoid(emit_logits)
            evict_probs = torch.sigmoid(evict_logits)

            accept_sample = torch.bernoulli(accept_probs)
            emit_sample = torch.bernoulli(emit_probs)
            evict_sample = torch.bernoulli(evict_probs)

            accept_lp = F.logsigmoid(
                torch.where(accept_sample > 0.5,
                            accept_logits, -accept_logits))
            emit_lp = F.logsigmoid(
                torch.where(emit_sample > 0.5,
                            emit_logits, -emit_logits))
            evict_lp = F.logsigmoid(
                torch.where(evict_sample > 0.5,
                            evict_logits, -evict_logits))

            iter_t1s.append(t1)
            iter_window_tokens.append(
                [list(wt) for wt in win_tok_lists])
            iter_gate_log_probs.append(
                torch.stack([accept_lp, emit_lp, evict_lp], dim=1))
            iter_gate_decisions.append(
                torch.stack([accept_sample, emit_sample, evict_sample],
                            dim=1))

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

        if not iter_t1s:
            iter_t1s = [torch.zeros(B, self.d_out, device=device)]

        return (iter_t1s, iter_window_tokens,
                iter_gate_log_probs, iter_gate_decisions,
                emission_info,
                torch.tensor(emit_counts, device=device))


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
