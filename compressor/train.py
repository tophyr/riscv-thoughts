"""Training loops for the compressor.

- train(): fixed-window training for context experiments
- streaming_train(): streaming compressor with emit gate + decoder
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import BOS, EOS, PAD, VOCAB_SIZE
from .model import Compressor, StreamingCompressor, Decoder


# ---------------------------------------------------------------------------
# Shared: execution distance from per-register deltas
# ---------------------------------------------------------------------------

def exec_distance(deltas, device):
    """Pairwise execution distance from per-register deltas.

    deltas: (N, n_inputs, 32) tensor
    Returns: (N, N) float32 distance matrix
    """
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
# Streaming compressor training with decoder
# ---------------------------------------------------------------------------

def _prepare_decoder_targets(emission_info, device):
    """Build decoder input/target tensors from emission info.

    For each emission, the target sequence is:
        [BOS] + instruction_tokens + [EOS]

    Decoder input (teacher forcing): target[:-1]
    Decoder target labels: target[1:]

    Returns:
        dec_input: (N, T) int64 — decoder input tokens
        dec_target: (N, T) int64 — target labels (PAD = ignore)
        dec_padding: (N, T) bool — True = padding
    """
    seqs = []
    for info in emission_info:
        seq = [BOS] + info['target_tokens'] + [EOS]
        seqs.append(seq)

    if not seqs:
        return None, None, None

    # Input is seq[:-1], target is seq[1:].
    max_len = max(len(s) - 1 for s in seqs)
    N = len(seqs)
    dec_input = np.full((N, max_len), PAD, dtype=np.int64)
    dec_target = np.full((N, max_len), PAD, dtype=np.int64)
    dec_padding = np.ones((N, max_len), dtype=np.bool_)

    for j, seq in enumerate(seqs):
        L = len(seq) - 1  # input/target length
        dec_input[j, :L] = seq[:-1]
        dec_target[j, :L] = seq[1:]
        dec_padding[j, :L] = False

    return (torch.from_numpy(dec_input).to(device),
            torch.from_numpy(dec_target).to(device),
            torch.from_numpy(dec_padding).to(device))


def _compute_emission_deltas(batch, emission_info, device):
    """Compute cumulative register deltas for each emission."""
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


def streaming_train(batch_iter, d_model=128, n_heads=4, n_layers=2,
                    d_out=128, dec_d_model=128, dec_n_heads=4,
                    dec_n_layers=2, lr=3e-4, device='auto',
                    n_steps=None, log_every=100, lr_schedule=None,
                    lr_min=1e-6, gate_tau=1.0, recon_weight=1.0,
                    pairwise_weight=1.0, roundtrip_weight=0.0,
                    gumbel_tau=1.0):
    """Train streaming compressor + decoder jointly.

    Losses:
        - Reconstruction: cross-entropy on decoder output vs original tokens.
          Trains encoder, gate, and decoder.
        - Pairwise MSE: distance matching on emission vectors.
          Trains encoder and gate (not decoder).

    Returns (encoder, decoder, losses, gate_stats).
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = StreamingCompressor(
        VOCAB_SIZE, d_model, n_heads, n_layers, d_out,
        gate_tau=gate_tau)
    encoder = encoder.to(device)

    decoder = Decoder(
        VOCAB_SIZE, dec_d_model, dec_n_heads, dec_n_layers, d_emb=d_out)
    decoder = decoder.to(device)

    all_params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(all_params, lr=lr)
    scheduler = None
    if lr_schedule and n_steps:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr_min)

    losses = []
    gate_stats = []
    step = 0
    t0 = time.time()

    for batch in batch_iter:
        tok = torch.from_numpy(batch.token_ids).to(device)
        pad = torch.from_numpy(batch.padding_mask).to(device)
        idx = torch.from_numpy(batch.token_instr_idx).to(device)
        n_instr = torch.from_numpy(batch.n_instructions).to(device)

        # Encoder forward.
        emissions, emission_info, emit_counts, gate_logits = encoder(
            tok, pad, idx, n_instr)

        N = emissions.shape[0]
        if N < 2:
            continue

        # --- Reconstruction loss ---
        dec_input, dec_target, dec_padding = _prepare_decoder_targets(
            emission_info, device)

        logits = decoder(emissions, dec_input, dec_padding)

        # Cross-entropy ignoring PAD positions.
        recon_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            dec_target.reshape(-1),
            ignore_index=PAD)

        # --- Pairwise MSE loss ---
        emission_deltas = _compute_emission_deltas(
            batch, emission_info, device)

        model_dists = torch.cdist(
            emissions.unsqueeze(0), emissions.unsqueeze(0),
            p=2).squeeze(0)
        exec_dists = exec_distance(emission_deltas, device)

        scale = 2.0 / max(exec_dists.max().item(), 1e-6)
        exec_dists = exec_dists * min(scale, 4.0)

        tri = torch.triu_indices(N, N, offset=1, device=device)
        pairwise_loss = F.mse_loss(
            model_dists[tri[0], tri[1]],
            exec_dists[tri[0], tri[1]])

        # --- Round-trip loss ---
        rt_loss_val = 0.0
        if roundtrip_weight > 0:
            # Gumbel-softmax on decoder output for differentiable tokens.
            soft = F.gumbel_softmax(logits, tau=gumbel_tau,
                                    hard=True, dim=-1)  # (N, T, V)

            # Soft embeddings using encoder's token embedding matrix.
            soft_emb = soft @ encoder.tok_emb.weight  # (N, T, d_model)

            # Prepend BOS embedding.
            bos_emb = encoder.tok_emb(
                torch.full((N, 1), BOS, dtype=torch.long, device=device))
            full_emb = torch.cat([bos_emb, soft_emb], dim=1)  # (N, T+1, d_model)
            full_pad = torch.cat([
                torch.zeros(N, 1, dtype=torch.bool, device=device),
                dec_padding,
            ], dim=1)

            # Re-encode through the encoder.
            reencoded = encoder.encode_soft(full_emb, full_pad)

            # Cosine distance (vectors are L2-normalized).
            rt_loss = (1 - (emissions * reencoded).sum(dim=-1)).mean()
            rt_loss_val = rt_loss.item()
        else:
            rt_loss = 0.0

        # --- Combined loss ---
        loss = (recon_weight * recon_loss
                + pairwise_weight * pairwise_loss
                + roundtrip_weight * rt_loss)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler:
            scheduler.step()

        loss_val = loss.item()
        recon_val = recon_loss.item()
        pair_val = pairwise_loss.item()
        losses.append({
            'total': loss_val,
            'recon': recon_val,
            'pairwise': pair_val,
            'roundtrip': rt_loss_val,
        })

        # Gate statistics.
        with torch.no_grad():
            non_pad = ~pad
            gate_probs = torch.sigmoid(gate_logits)
            mean_prob = gate_probs[non_pad].mean().item()
            mean_emits = emit_counts.float().mean().item()
            mean_instrs = n_instr.float().mean().item()

            # Reconstruction accuracy: fraction of non-PAD tokens
            # where argmax(logits) == target.
            pred = logits.argmax(dim=-1)
            non_pad_dec = ~dec_padding
            correct = (pred == dec_target) & non_pad_dec
            recon_acc = correct.sum().item() / non_pad_dec.sum().item()

        gs = {
            'emit_prob': mean_prob,
            'emits_per_seq': mean_emits,
            'instrs_per_seq': mean_instrs,
            'n_emissions': N,
            'recon_acc': recon_acc,
        }
        gate_stats.append(gs)

        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t0
            ms_per = elapsed / step * 1000
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            rt_str = f' rt {rt_loss_val:.4f}' if roundtrip_weight > 0 else ''
            print(f'step {step:>6d}  '
                  f'loss {loss_val:.4f} '
                  f'(recon {recon_val:.3f} pair {pair_val:.4f}{rt_str})  '
                  f'acc {recon_acc:.1%}  '
                  f'lr {current_lr:.1e}  {ms_per:.0f}ms/step  '
                  f'emit {gs["emits_per_seq"]:.1f}/'
                  f'{gs["instrs_per_seq"]:.1f}')

        if n_steps and step >= n_steps:
            break

    return encoder, decoder, losses, gate_stats
