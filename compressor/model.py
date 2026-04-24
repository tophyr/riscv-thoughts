"""T1 Compressor and Decoder.

T1Compressor: neural shift-reduce parser with per-iteration gate
    decisions and bidirectional window encoder. Can also operate in
    fixed-window mode (no gates) for encoder-only training.
Decoder: autoregressive decoder conditioned on emission vectors.
"""

import numpy as np
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
        # Runtime cap on window size during streaming forward().
        # Defaults to the full pos-embedding table; callers can lower
        # it after loading a checkpoint to skip wasted compute on
        # padded positions that will never hold meaningful tokens.
        self.runtime_max_window = max_window

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

        # The per-iteration shift-reduce body is always compiled.
        # dynamic=True so batch size, sequence length, and max_k can
        # vary without triggering a fresh compile each time.
        self._shift_reduce_step = torch.compile(
            self._shift_reduce_step, dynamic=True)

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

    def _encode_window_buf(self, window_buf, window_lens):
        """Encode windows represented as a padded tensor.

        window_buf: (B, MAX_WINDOW) long
        window_lens: (B,) long — valid length of each window
        Returns: (B, d_out) T1 candidates. Zero vector for empty windows.
        """
        B, MAX_WINDOW = window_buf.shape
        device = window_buf.device

        nonempty = window_lens > 0  # (B,)

        # Padding mask: True where padding (position >= length).
        positions = torch.arange(MAX_WINDOW, device=device).unsqueeze(0)
        pad_mask = positions >= window_lens.unsqueeze(1)  # (B, MAX_WINDOW)
        # Force position 0 non-pad even for empty rows so the encoder
        # never sees an all-padding sequence (torch's attention complains).
        # We zero the output of those rows after.
        pad_mask[:, 0] = False

        vecs = self.encode(window_buf, pad_mask)  # (B, d_out)
        return vecs * nonempty.unsqueeze(1).float()

    def _shift_reduce_step(self, window_buf, window_orig, window_lens,
                           input_pos, gru_h, done, emit_counts,
                           token_ids, seq_lens,
                           instr_first, instr_last,
                           slot_range, k_range, max_k):
        """One iteration of the shift-reduce loop. Pure tensor ops — no
        syncs, no Python control flow, no list ops. Fused/compiled as a
        whole.

        Returns the updated loop state plus the per-iteration outputs
        (t1, gate log-probs/decisions, emit/completeness tensors, and a
        snapshot of the window contents).
        """
        B = window_buf.shape[0]
        device = window_buf.device
        MAX_WINDOW = window_buf.shape[1]
        L = token_ids.shape[1]

        # Snapshot window state before mutation.
        snap_window_buf = window_buf.clone()
        snap_window_lens = window_lens.clone()

        t1 = self._encode_window_buf(window_buf, window_lens)
        active = ~done

        new_h = self.gru(t1, gru_h)
        gru_h = torch.where(active.unsqueeze(1), new_h, gru_h)

        has_input = (input_pos < seq_lens) & active
        safe_pos = input_pos.clamp(max=L - 1)
        gather_idx = safe_pos.view(B, 1, 1).expand(-1, 1, self.d_model)
        next_embs = self.tok_emb(token_ids).gather(
            1, gather_idx).squeeze(1)
        next_embs = next_embs * has_input.unsqueeze(1).float()

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

        accept_lp = F.logsigmoid(torch.where(
            accept_sample > 0.5, accept_logits, -accept_logits))
        emit_lp = F.logsigmoid(torch.where(
            emit_sample > 0.5, emit_logits, -emit_logits))
        evict_lp = F.logsigmoid(torch.where(
            evict_sample > 0.5, evict_logits, -evict_logits))

        # Gate decisions — no fallback. The gate must learn to fire
        # actions itself; iterations where (accept, emit, evict) all
        # sample zero (or are masked invalid) simply pass with no
        # state change. The max_total_iters budget caps how long a
        # stalled sequence can go. A force-accept / force-emit
        # fallback would be a hardcoded policy that the gate would
        # learn to exploit (sample nothing, let the fallback do the
        # work) — which is the failure mode we observed.
        has_window = window_lens > 0
        a = (accept_sample > 0.5) & has_input & active
        e = (emit_sample > 0.5) & has_window & active
        v = (evict_sample > 0.5) & has_window & active

        pos_mask = slot_range < window_lens.unsqueeze(1)
        has_start = (
            (window_orig.unsqueeze(1) == instr_first.unsqueeze(2))
            & pos_mask.unsqueeze(1)
        ).any(dim=2)
        has_end = (
            (window_orig.unsqueeze(1) == instr_last.unsqueeze(2))
            & pos_mask.unsqueeze(1)
        ).any(dim=2)
        valid_instr = instr_first >= 0
        complete = has_start & has_end & valid_instr
        has_complete = complete.any(dim=1) & e

        k_expand = k_range.expand(B, max_k)
        k_min = torch.where(
            complete, k_expand,
            torch.full_like(k_expand, max_k)).min(dim=1).values
        k_max = torch.where(
            complete, k_expand,
            torch.full_like(k_expand, -1)).max(dim=1).values
        instr_start_out = torch.where(
            has_complete, k_min, torch.full_like(k_min, -1))
        instr_end_out = torch.where(
            has_complete, k_max, torch.full_like(k_max, -1))

        gate_log_probs = torch.stack([accept_lp, emit_lp, evict_lp], dim=1)
        gate_decisions = torch.stack(
            [accept_sample, emit_sample, evict_sample], dim=1)
        gate_logits = torch.stack(
            [accept_logits, emit_logits, evict_logits], dim=1)

        # Apply accept: write next token into window at slot window_lens.
        next_tok = token_ids.gather(
            1, safe_pos.unsqueeze(1)).squeeze(1)
        slot = window_lens.clamp(max=MAX_WINDOW - 1)
        slot_mask = (slot_range == slot.unsqueeze(1))
        write_mask = slot_mask & a.unsqueeze(1)
        window_buf = torch.where(
            write_mask,
            next_tok.unsqueeze(1).expand(-1, MAX_WINDOW),
            window_buf)
        window_orig = torch.where(
            write_mask,
            input_pos.unsqueeze(1).expand(-1, MAX_WINDOW),
            window_orig)
        window_lens = torch.where(a, window_lens + 1, window_lens)
        input_pos = torch.where(a, input_pos + 1, input_pos)

        # Apply evict: left-shift window by one slot.
        shifted_buf = torch.cat(
            [window_buf[:, 1:],
             torch.zeros(B, 1, dtype=torch.long, device=device)],
            dim=1)
        shifted_orig = torch.cat(
            [window_orig[:, 1:],
             torch.full((B, 1), -1, dtype=torch.long, device=device)],
            dim=1)
        window_buf = torch.where(
            v.unsqueeze(1), shifted_buf, window_buf)
        window_orig = torch.where(
            v.unsqueeze(1), shifted_orig, window_orig)
        window_lens = torch.where(
            v, (window_lens - 1).clamp(min=0), window_lens)

        emit_counts = emit_counts + e.long()
        done = done | ((input_pos >= seq_lens) & (window_lens == 0))

        return (window_buf, window_orig, window_lens, input_pos, gru_h,
                done, emit_counts,
                t1, gate_log_probs, gate_decisions, gate_logits,
                e, has_complete, instr_start_out, instr_end_out,
                snap_window_buf, snap_window_lens)

    def forward(self, token_ids, padding_mask, token_instr_idx,
                n_instructions):
        """Run batched shift-reduce loop (streaming mode).

        Tensor-backed state: window_buf/window_orig/window_lens,
        input_pos, done all live on GPU, no per-sequence Python loops
        in the hot path. The per-iteration body is extracted into
        ``_shift_reduce_step`` so it can be torch.compile'd cleanly.
        Bookkeeping for emission_info is deferred to the end as a
        single bulk CPU transfer.

        Returns:
            iter_t1s: list of (B, d_out) T1 candidates per iteration
            iter_window_tokens: list of window token lists per iteration
            iter_gate_log_probs: list of (B, 3) log-probs of sampled
                gate decisions
            iter_gate_decisions: list of (B, 3) gate decisions
            iter_gate_logits: list of (B, 3) raw gate logits before
                sigmoid/sampling (needed for supervised BCE training)
            iter_window_buf: (n_iters, B, MAX_WINDOW) long tensor of
                window token IDs per iteration (GPU-resident; for
                callers that want batched parsing)
            iter_window_lens: (n_iters, B) long tensor of valid
                window lengths per iteration
            emission_info: list of dicts for actual emissions
            emit_counts: (B,) emissions per sequence
        """
        B, L = token_ids.shape
        device = token_ids.device
        MAX_WINDOW = self.runtime_max_window

        # --- Precompute on GPU (once per batch) ---
        not_pad = ~padding_mask
        seq_lens = not_pad.sum(dim=1).long()  # (B,)

        # Per-instruction first/last token positions (B, max_k).
        max_k = int(n_instructions.max().item()) if B > 0 else 0
        if max_k == 0:
            max_k = 1  # avoid empty tensors
        instr_first = torch.full(
            (B, max_k), -1, dtype=torch.long, device=device)
        instr_last = torch.full(
            (B, max_k), -1, dtype=torch.long, device=device)
        if B > 0:
            positions = (torch.arange(L, device=device)
                         .unsqueeze(0).expand(B, L))
            valid = (token_instr_idx >= 0) & not_pad
            idx_clamped = token_instr_idx.clamp(min=0).long()
            # scatter_reduce uses the minimum / maximum position for each
            # (batch, instr_idx) bucket. Invalid rows (idx < 0 or padding)
            # are masked by replacing positions with +inf / -inf sentinels
            # so they never win min/max.
            positions_for_min = torch.where(
                valid, positions,
                torch.full_like(positions, L))
            positions_for_max = torch.where(
                valid, positions,
                torch.full_like(positions, -1))
            instr_first = instr_first.scatter_reduce(
                dim=1, index=idx_clamped, src=positions_for_min,
                reduce='amin', include_self=False)
            instr_last = instr_last.scatter_reduce(
                dim=1, index=idx_clamped, src=positions_for_max,
                reduce='amax', include_self=False)
            # Clean up sentinel values from buckets that received no
            # real contributions.
            instr_first = torch.where(
                instr_first == L,
                torch.full_like(instr_first, -1), instr_first)
            # instr_last is already -1 for empty buckets.

        # --- Tensor-backed loop state ---
        window_buf = torch.zeros(
            B, MAX_WINDOW, dtype=torch.long, device=device)
        window_orig = torch.full(
            (B, MAX_WINDOW), -1, dtype=torch.long, device=device)
        window_lens = torch.zeros(B, dtype=torch.long, device=device)
        input_pos = torch.zeros(B, dtype=torch.long, device=device)
        gru_h = torch.zeros(B, self.d_model, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        emit_counts = torch.zeros(B, dtype=torch.long, device=device)

        iter_t1s = []
        iter_window_buf = []
        iter_window_lens = []
        iter_gate_log_probs = []
        iter_gate_decisions = []
        iter_gate_logits = []
        iter_emit_mask = []
        iter_has_complete = []
        iter_instr_start = []
        iter_instr_end = []

        max_seq_len = int(seq_lens.max().item()) if B > 0 else 0
        max_total_iters = max_seq_len * self.max_iters_per_token

        slot_range = torch.arange(MAX_WINDOW, device=device).unsqueeze(0)
        k_range = torch.arange(max_k, device=device).unsqueeze(0)

        for iteration in range(max_total_iters):
            (window_buf, window_orig, window_lens, input_pos, gru_h,
             done, emit_counts,
             t1, gate_log_probs, gate_decisions, gate_logits,
             e_mask, hc_mask, kmin_out, kmax_out,
             snap_buf, snap_lens) = self._shift_reduce_step(
                window_buf, window_orig, window_lens,
                input_pos, gru_h, done, emit_counts,
                token_ids, seq_lens,
                instr_first, instr_last,
                slot_range, k_range, max_k)

            iter_t1s.append(t1)
            iter_window_buf.append(snap_buf)
            iter_window_lens.append(snap_lens)
            iter_gate_log_probs.append(gate_log_probs)
            iter_gate_decisions.append(gate_decisions)
            iter_gate_logits.append(gate_logits)
            iter_emit_mask.append(e_mask)
            iter_has_complete.append(hc_mask)
            iter_instr_start.append(kmin_out)
            iter_instr_end.append(kmax_out)

        if not iter_t1s:
            iter_t1s = [torch.zeros(B, self.d_out, device=device)]
            empty_buf = torch.zeros(
                0, B, MAX_WINDOW, dtype=torch.long, device=device)
            empty_lens = torch.zeros(
                0, B, dtype=torch.long, device=device)
            return (iter_t1s, [[[] for _ in range(B)]],
                    [torch.zeros(B, 3, device=device)],
                    [torch.zeros(B, 3, device=device)],
                    [torch.zeros(B, 3, device=device)],
                    empty_buf, empty_lens,
                    [], emit_counts)

        # Keep window-state tensors on GPU for callers that need
        # efficient batched processing (e.g. parser-based gate
        # training). The list-of-lists form below is still built for
        # compatibility with consumers that expect per-iteration token
        # lists (log output, emission_info construction).
        iter_window_buf_t = torch.stack(iter_window_buf)  # (n_iters, B, MAX_WINDOW)
        iter_window_lens_t = torch.stack(iter_window_lens)  # (n_iters, B)

        # --- Post-loop: convert to the expected list-of-lists / dict
        # formats. One bulk CPU transfer per saved tensor is fine
        # because it only happens once per batch. ---
        n_iters = len(iter_t1s)
        all_wbufs = iter_window_buf_t.cpu().numpy()
        all_wlens = iter_window_lens_t.cpu().numpy()
        all_emit = torch.stack(iter_emit_mask).cpu().numpy()
        all_hcomp = torch.stack(iter_has_complete).cpu().numpy()
        all_kmin = torch.stack(iter_instr_start).cpu().numpy()
        all_kmax = torch.stack(iter_instr_end).cpu().numpy()

        iter_window_tokens = []
        emission_info = []
        for it in range(n_iters):
            wbuf = all_wbufs[it]
            wlen = all_wlens[it]
            win_list = [
                wbuf[b, :wlen[b]].tolist() for b in range(B)
            ]
            iter_window_tokens.append(win_list)
            emitters = np.nonzero(all_emit[it])[0]
            for b in emitters:
                target_toks = win_list[b]
                emission_info.append({
                    'batch_idx': int(b),
                    'instr_start': int(all_kmin[it, b]),
                    'instr_end': int(all_kmax[it, b]),
                    'target_tokens': target_toks,
                    'window_size': len(target_toks),
                    'has_complete': bool(all_hcomp[it, b]),
                    'iteration': it,
                })

        return (iter_t1s, iter_window_tokens,
                iter_gate_log_probs, iter_gate_decisions,
                iter_gate_logits,
                iter_window_buf_t, iter_window_lens_t,
                emission_info, emit_counts)


class Decoder(nn.Module):
    """Autoregressive decoder conditioned on an emission vector."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_emb,
                 max_seq_len=64, dropout=0.0, n_memory_tokens=1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_memory_tokens = n_memory_tokens
        self.emission_proj = nn.Linear(d_emb, n_memory_tokens * d_model)
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
        memory = self.emission_proj(emission_vecs).view(
            N, self.n_memory_tokens, self.d_model)
        pos = torch.arange(T, device=device)
        tgt = self.tok_emb(target_tokens) + self.pos_emb(pos)
        causal_mask = torch.ones(
            T, T, dtype=torch.bool, device=device).triu(diagonal=1)
        out = self.decoder(
            tgt, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask)
        return self.head(out)
