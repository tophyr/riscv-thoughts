"""Training loops for the compressor.

- train(): fixed-window training for context experiments
- streaming_train(): shift-reduce compressor with REINFORCE gate training
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import BOS, EOS, PAD, VOCAB_SIZE
from .model import Compressor, ShiftReduceCompressor, Decoder


# ---------------------------------------------------------------------------
# Shared: execution distance from per-register deltas
# ---------------------------------------------------------------------------

def exec_distance(deltas, device):
    """Pairwise execution distance from per-register deltas."""
    d = deltas.to(device=device, dtype=torch.float32)
    diff = (d.unsqueeze(1) - d.unsqueeze(0)).abs()
    return diff.log1p_().mean(dim=(-1, -2))


# ---------------------------------------------------------------------------
# Fixed-window training (context experiment)
# ---------------------------------------------------------------------------

def extract_windows(batch, window_size=1):
    """Extract per-instruction training examples from a sequence batch."""
    B = batch.token_ids.shape[0]
    all_tokens = []
    all_deltas = []

    for b in range(B):
        n = int(batch.n_instructions[b])
        tok = batch.token_ids[b]
        idx = batch.token_instr_idx[b]
        mask = ~batch.padding_mask[b]

        for i in range(window_size - 1, n):
            start_instr = i - window_size + 1
            sel = mask & (idx >= start_instr) & (idx <= i)
            instr_tokens = tok[sel].tolist()
            all_tokens.append([BOS] + instr_tokens + [EOS])
            delta = (batch.per_instr_regs[b, i + 1, :, :]
                     - batch.per_instr_regs[b, i, :, :])
            all_deltas.append(delta)

    if not all_tokens:
        return None, None, None

    max_len = max(len(t) for t in all_tokens)
    N = len(all_tokens)
    token_ids = np.full((N, max_len), PAD, dtype=np.int64)
    padding_mask = np.ones((N, max_len), dtype=np.bool_)
    for j, t in enumerate(all_tokens):
        token_ids[j, :len(t)] = t
        padding_mask[j, :len(t)] = False

    deltas = np.stack(all_deltas, axis=0)
    return token_ids, padding_mask, deltas


def train(batch_iter, window_size=1, d_model=128, n_heads=4, n_layers=2,
          d_out=128, lr=3e-4, device='auto', n_steps=None, log_every=100,
          lr_schedule=None, lr_min=1e-6):
    """Train a fixed-window compressor. Returns (model, losses)."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Compressor(VOCAB_SIZE, d_model, n_heads, n_layers, d_out)
    model = model.to(device)
    compiled = torch.compile(model)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_schedule and n_steps:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr_min)

    losses = []
    step = 0
    t0 = time.time()

    for batch in batch_iter:
        token_ids, padding_mask, deltas = extract_windows(batch, window_size)
        if token_ids is None:
            continue

        tok = torch.from_numpy(token_ids).to(device)
        pad = torch.from_numpy(padding_mask).to(device)
        delta_t = torch.from_numpy(deltas)

        vecs = compiled(tok, pad)
        model_dists = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0),
                                  p=2).squeeze(0)
        exec_dists = exec_distance(delta_t, device)
        scale = 2.0 / max(exec_dists.max().item(), 1e-6)
        exec_dists = exec_dists * min(scale, 4.0)

        N = vecs.shape[0]
        tri = torch.triu_indices(N, N, offset=1, device=device)
        loss = F.mse_loss(model_dists[tri[0], tri[1]],
                          exec_dists[tri[0], tri[1]])

        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler:
            scheduler.step()

        losses.append(loss.item())
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t0
            ms_per = elapsed / step * 1000
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            print(f'step {step:>6d}  loss {losses[-1]:.4f}  '
                  f'lr {current_lr:.1e}  {ms_per:.0f}ms/step  '
                  f'N={N}')

        if n_steps and step >= n_steps:
            break

    raw = getattr(compiled, '_orig_mod', compiled)
    return raw, losses


# ---------------------------------------------------------------------------
# Decoder target preparation
# ---------------------------------------------------------------------------

def _prepare_decoder_targets(token_lists, device):
    """Build decoder input/target tensors from lists of token IDs.

    Each token list gets wrapped: [BOS] + tokens + [EOS].
    Returns dec_input (shifted right), dec_target, dec_padding.
    Returns None tuple if token_lists is empty.
    """
    if not token_lists:
        return None, None, None

    seqs = [[BOS] + toks + [EOS] for toks in token_lists]
    max_len = max(len(s) - 1 for s in seqs)
    N = len(seqs)
    dec_input = np.full((N, max_len), PAD, dtype=np.int64)
    dec_target = np.full((N, max_len), PAD, dtype=np.int64)
    dec_padding = np.ones((N, max_len), dtype=np.bool_)

    for j, seq in enumerate(seqs):
        L = len(seq) - 1
        dec_input[j, :L] = seq[:-1]
        dec_target[j, :L] = seq[1:]
        dec_padding[j, :L] = False

    return (torch.from_numpy(dec_input).to(device),
            torch.from_numpy(dec_target).to(device),
            torch.from_numpy(dec_padding).to(device))


def _compute_emission_deltas(batch, emission_info, device):
    """Compute cumulative register deltas for complete emissions."""
    deltas = []
    for info in emission_info:
        b = info['batch_idx']
        i_start = info['instr_start']
        i_end = info['instr_end']
        delta = (batch.per_instr_regs[b, i_end + 1, :, :]
                 - batch.per_instr_regs[b, i_start, :, :])
        deltas.append(torch.from_numpy(delta.astype(np.float32)))

    if not deltas:
        return torch.zeros(0, 1, 32, device=device)
    return torch.stack(deltas).to(device)


# ---------------------------------------------------------------------------
# Streaming compressor training with REINFORCE gate training
# ---------------------------------------------------------------------------

def streaming_train(batch_iter, d_model=128, n_heads=4, n_layers=2,
                    d_out=128, dec_d_model=128, dec_n_heads=4,
                    dec_n_layers=2, lr=3e-4, device='auto',
                    n_steps=None, log_every=100, lr_schedule=None,
                    lr_min=1e-6, pairwise_weight=1.0,
                    reinforce_lr=1e-3, baseline_decay=0.99):
    """Train shift-reduce compressor + decoder.

    The encoder (window encoder + GRU + projector) and decoder are
    trained via normal backprop through reconstruction loss.

    The gates are trained via REINFORCE: at every iteration, the
    decoder evaluates the T1 candidate against the current window
    contents. The per-iteration reconstruction loss serves as the
    reward signal (negated: lower loss = higher reward).

    Returns (encoder, decoder, losses, gate_stats).
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = ShiftReduceCompressor(
        VOCAB_SIZE, d_model, n_heads, n_layers, d_out)
    encoder = encoder.to(device)

    decoder = Decoder(
        VOCAB_SIZE, dec_d_model, dec_n_heads, dec_n_layers, d_emb=d_out)
    decoder = decoder.to(device)

    # Separate optimizers: normal backprop for encoder+decoder,
    # REINFORCE for gates.
    encoder_params = (
        list(encoder.tok_emb.parameters())
        + list(encoder.win_pos_emb.parameters())
        + list(encoder.win_encoder.parameters())
        + list(encoder.proj.parameters())
        + list(encoder.gru.parameters()))
    gate_params = (
        list(encoder.accept_head.parameters())
        + list(encoder.emit_head.parameters())
        + list(encoder.evict_head.parameters()))

    opt_main = torch.optim.Adam(
        encoder_params + list(decoder.parameters()), lr=lr)
    opt_gates = torch.optim.Adam(gate_params, lr=reinforce_lr)

    scheduler = None
    if lr_schedule and n_steps:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_main, T_max=n_steps, eta_min=lr_min)

    # REINFORCE baseline (moving average of per-iteration loss).
    baseline = 0.0

    losses = []
    gate_stats = []
    step = 0
    t0 = time.time()

    for batch in batch_iter:
        tok = torch.from_numpy(batch.token_ids).to(device)
        pad = torch.from_numpy(batch.padding_mask).to(device)
        idx = torch.from_numpy(batch.token_instr_idx).to(device)
        n_instr = torch.from_numpy(batch.n_instructions).to(device)

        B = tok.shape[0]

        # --- Forward pass (shift-reduce loop) ---
        (iter_t1s, iter_window_tokens,
         iter_gate_log_probs, iter_gate_decisions,
         emission_info, emit_counts) = encoder(tok, pad, idx, n_instr)

        n_iters = len(iter_t1s)
        if n_iters == 0:
            continue

        # --- Per-iteration decoder evaluation ---
        # For each iteration, run the decoder on the T1 candidate
        # against the current window tokens. This gives a per-iteration
        # reconstruction loss used as REINFORCE reward.
        iter_recon_losses = []  # per-iteration, shape (B,) each

        for it in range(n_iters):
            t1 = iter_t1s[it]  # (B, d_out)
            win_toks = iter_window_tokens[it]  # list of B token lists

            # Find sequences with non-empty windows.
            nonempty = [b for b in range(B) if win_toks[b]]
            if not nonempty:
                iter_recon_losses.append(
                    torch.zeros(B, device=device))
                continue

            # Prepare decoder targets for non-empty windows.
            nonempty_toks = [win_toks[b] for b in nonempty]
            dec_in, dec_tgt, dec_pad = _prepare_decoder_targets(
                nonempty_toks, device)

            # Decoder forward.
            ne_t1 = t1[nonempty]
            logits = decoder(ne_t1, dec_in, dec_pad)

            # Per-sequence reconstruction loss.
            per_tok = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                dec_tgt.reshape(-1),
                ignore_index=PAD, reduction='none')
            per_tok = per_tok.reshape(len(nonempty), -1)
            non_pad_counts = (~dec_pad).sum(dim=1).clamp(min=1).float()
            per_seq_loss = per_tok.sum(dim=1) / non_pad_counts

            # Window-size weighting.
            win_sizes = torch.tensor(
                [len(win_toks[b]) for b in nonempty],
                dtype=torch.float32, device=device)
            per_seq_loss = per_seq_loss * win_sizes

            # Scatter back to full batch.
            full_loss = torch.zeros(B, device=device)
            for j, b in enumerate(nonempty):
                full_loss[b] = per_seq_loss[j]

            iter_recon_losses.append(full_loss)

        # --- Reconstruction loss for encoder+decoder (normal backprop) ---
        # Average across all iterations and sequences.
        all_recon = torch.stack(iter_recon_losses)  # (n_iters, B)
        # Mask out done sequences (zero loss).
        recon_mask = all_recon > 0
        if recon_mask.any():
            recon_loss = all_recon[recon_mask].mean()
        else:
            recon_loss = torch.tensor(0.0, device=device)

        # --- Pairwise MSE loss (on actual emissions with complete instrs) ---
        complete_info = [info for info in emission_info
                         if info['has_complete']]

        if len(complete_info) >= 2:
            # Gather emission T1 vectors.
            emission_vecs = []
            for info in complete_info:
                it = info['iteration']
                b = info['batch_idx']
                emission_vecs.append(iter_t1s[it][b])
            emission_vecs = torch.stack(emission_vecs)

            emission_deltas = _compute_emission_deltas(
                batch, complete_info, device)
            Nc = emission_vecs.shape[0]
            model_dists = torch.cdist(
                emission_vecs.unsqueeze(0),
                emission_vecs.unsqueeze(0),
                p=2).squeeze(0)
            exec_dists = exec_distance(emission_deltas, device)
            scale = 2.0 / max(exec_dists.max().item(), 1e-6)
            exec_dists = exec_dists * min(scale, 4.0)
            tri = torch.triu_indices(Nc, Nc, offset=1, device=device)
            pairwise_loss = F.mse_loss(
                model_dists[tri[0], tri[1]],
                exec_dists[tri[0], tri[1]])
        else:
            pairwise_loss = torch.tensor(0.0, device=device)

        # --- REINFORCE for gates ---
        # Compute REINFORCE loss BEFORE main backward (shares graph).
        with torch.no_grad():
            iter_rewards = torch.stack(iter_recon_losses)
            mean_reward = iter_rewards[iter_rewards > 0].mean().item() \
                if (iter_rewards > 0).any() else 0.0
            baseline = baseline_decay * baseline + (1 - baseline_decay) * mean_reward

        reinforce_loss = torch.tensor(0.0, device=device)
        n_terms = 0
        for it in range(n_iters):
            log_probs = iter_gate_log_probs[it]  # (B, 3)
            rewards = iter_recon_losses[it].detach()  # (B,)
            advantages = rewards - baseline

            total_lp = log_probs.sum(dim=1)
            active = rewards > 0
            if active.any():
                reinforce_loss = reinforce_loss + \
                    (total_lp[active] * advantages[active]).mean()
                n_terms += 1

        if n_terms > 0:
            reinforce_loss = reinforce_loss / n_terms

        # --- Combined backward ---
        main_loss = recon_loss + pairwise_weight * pairwise_loss
        total_loss = main_loss + reinforce_loss
        opt_main.zero_grad()
        opt_gates.zero_grad()
        total_loss.backward()
        opt_main.step()
        opt_gates.step()
        if scheduler:
            scheduler.step()

        # --- Logging ---
        recon_val = recon_loss.item()
        pair_val = pairwise_loss.item()
        loss_val = main_loss.item()
        reinforce_val = reinforce_loss.item() if n_terms > 0 else 0.0

        with torch.no_grad():
            # Gate decision rates.
            total_a = total_e = total_v = total_steps = 0
            total_window = 0
            for it in range(n_iters):
                decs = iter_gate_decisions[it]  # (B, 3)
                for b in range(B):
                    wt = iter_window_tokens[it][b]
                    if wt or (it == 0):  # active sequence
                        total_a += decs[b, 0].item() > 0.5
                        total_e += decs[b, 1].item() > 0.5
                        total_v += decs[b, 2].item() > 0.5
                        total_steps += 1
                        total_window += len(wt)

            accept_rate = total_a / max(total_steps, 1)
            emit_rate = total_e / max(total_steps, 1)
            evict_rate = total_v / max(total_steps, 1)
            mean_emits = emit_counts.float().mean().item()
            mean_instrs = n_instr.float().mean().item()
            mean_window = total_window / max(total_steps, 1)
            complete_frac = len(complete_info) / max(len(emission_info), 1)

            # Reconstruction accuracy on actual emissions.
            if emission_info:
                all_tgt_toks = [info['target_tokens'] for info in emission_info]
                dec_in, dec_tgt, dec_pad = _prepare_decoder_targets(
                    all_tgt_toks, device)
                if dec_in is not None:
                    # Find emission T1 vectors.
                    em_vecs = []
                    for info in emission_info:
                        em_vecs.append(
                            iter_t1s[info['iteration']][info['batch_idx']])
                    em_vecs = torch.stack(em_vecs)
                    em_logits = decoder(em_vecs, dec_in, dec_pad)
                    pred = em_logits.argmax(dim=-1)
                    non_pad_dec = ~dec_pad
                    correct = (pred == dec_tgt) & non_pad_dec
                    recon_acc = correct.sum().item() / non_pad_dec.sum().item()
                else:
                    recon_acc = 0.0
            else:
                recon_acc = 0.0

        losses.append({
            'total': loss_val,
            'recon': recon_val,
            'pairwise': pair_val,
            'reinforce': reinforce_val,
            'mean_window': mean_window,
        })
        gs = {
            'accept_rate': accept_rate,
            'emit_rate': emit_rate,
            'evict_rate': evict_rate,
            'emits_per_seq': mean_emits,
            'instrs_per_seq': mean_instrs,
            'n_emissions': len(emission_info),
            'complete_frac': complete_frac,
            'recon_acc': recon_acc,
            'iters_per_seq': n_iters,
        }
        gate_stats.append(gs)

        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t0
            ms_per = elapsed / step * 1000
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            print(f'step {step:>6d}  '
                  f'loss {loss_val:.4f} '
                  f'(recon {recon_val:.3f} pair {pair_val:.4f} '
                  f'rl {reinforce_val:.3f})  '
                  f'acc {recon_acc:.1%}  '
                  f'lr {current_lr:.1e}  {ms_per:.0f}ms/step  '
                  f'a/e/v={accept_rate:.0%}/{emit_rate:.0%}/'
                  f'{evict_rate:.0%}  '
                  f'{gs["emits_per_seq"]:.1f}/'
                  f'{gs["instrs_per_seq"]:.1f}  '
                  f'win {mean_window:.1f}  '
                  f'ok {complete_frac:.0%}  '
                  f'iters {n_iters}')

        if n_steps and step >= n_steps:
            break

    return encoder, decoder, losses, gate_stats
