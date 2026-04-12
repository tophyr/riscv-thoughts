"""Training loop for the fixed-window compressor experiment.

Extracts per-instruction windows from sequence batches, trains with
pairwise MSE loss against per-register execution deltas.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from tokenizer import BOS, EOS, PAD
from .model import Compressor


# ---------------------------------------------------------------------------
# Window extraction from sequence batches
# ---------------------------------------------------------------------------

def extract_windows(batch, window_size=1):
    """Extract per-instruction training examples from a sequence batch.

    For each instruction i (where i >= window_size - 1), extracts:
    - tokens: the tokens of instructions [i - window_size + 1, ..., i]
      with BOS/EOS wrapping
    - delta: register change from executing instruction i alone
      shape (n_inputs, 32)

    The model sees a context window but the target is always the
    LAST instruction's single-step register delta. This lets us
    directly compare window_size=1 (no context) vs window_size=2
    (one instruction of context).

    Returns (token_ids, padding_mask, deltas):
        token_ids: (N, max_len) int64
        padding_mask: (N, max_len) bool
        deltas: (N, n_inputs, 32) int32
    """
    B = batch.token_ids.shape[0]
    n_inputs = batch.per_instr_regs.shape[2]

    all_tokens = []
    all_deltas = []

    for b in range(B):
        n = int(batch.n_instructions[b])
        tok = batch.token_ids[b]
        idx = batch.token_instr_idx[b]
        mask = ~batch.padding_mask[b]

        for i in range(window_size - 1, n):
            # Token window: instructions [i - window_size + 1, i]
            start_instr = i - window_size + 1
            sel = mask & (idx >= start_instr) & (idx <= i)
            instr_tokens = tok[sel].tolist()
            all_tokens.append([BOS] + instr_tokens + [EOS])

            # Register delta for instruction i only.
            delta = (batch.per_instr_regs[b, i + 1, :, :]
                     - batch.per_instr_regs[b, i, :, :])
            all_deltas.append(delta)

    if not all_tokens:
        return None, None, None

    # Pad tokens.
    max_len = max(len(t) for t in all_tokens)
    N = len(all_tokens)
    token_ids = np.full((N, max_len), PAD, dtype=np.int64)
    padding_mask = np.ones((N, max_len), dtype=np.bool_)
    for j, t in enumerate(all_tokens):
        token_ids[j, :len(t)] = t
        padding_mask[j, :len(t)] = False

    deltas = np.stack(all_deltas, axis=0)  # (N, n_inputs, 32)

    return token_ids, padding_mask, deltas


# ---------------------------------------------------------------------------
# Execution distance from per-register deltas
# ---------------------------------------------------------------------------

def exec_distance(deltas, device):
    """Pairwise execution distance from per-register deltas.

    deltas: (N, n_inputs, 32) int32 tensor
    Returns: (N, N) float32 distance matrix

    Distance = mean over inputs and registers of log1p(|diff|).
    """
    d = deltas.to(device=device, dtype=torch.float32)
    # (N, 1, S, R) - (1, N, S, R) -> (N, N, S, R)
    diff = (d.unsqueeze(1) - d.unsqueeze(0)).abs()
    return diff.log1p_().mean(dim=(-1, -2))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(batch_iter, window_size=1, d_model=128, n_heads=4, n_layers=2,
          d_out=128, lr=3e-4, device='auto', n_steps=None, log_every=100,
          lr_schedule=None, lr_min=1e-6):
    """Train a fixed-window compressor.

    Args:
        batch_iter: yields Batch objects
        window_size: number of instructions per window (1=baseline, 2=context)
        n_steps: stop after this many steps (None = exhaust iterator)

    Returns (model, losses).
    """
    from tokenizer import VOCAB_SIZE

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

        # Model forward.
        vecs = compiled(tok, pad)

        # Pairwise model distances (upper triangle).
        model_dists = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0),
                                  p=2).squeeze(0)

        # Pairwise execution distances.
        exec_dists = exec_distance(delta_t, device)

        # Scale exec dists to roughly match model distance range [0, 2].
        # Max log1p(2^32) ≈ 22, but mean over 32 registers (most zero)
        # gives typical values ~0.5-2. Light scaling.
        scale = 2.0 / max(exec_dists.max().item(), 1e-6)
        exec_dists = exec_dists * min(scale, 4.0)

        # MSE on upper triangle.
        N = vecs.shape[0]
        tri = torch.triu_indices(N, N, offset=1, device=device)
        loss = nn.functional.mse_loss(
            model_dists[tri[0], tri[1]],
            exec_dists[tri[0], tri[1]])

        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler:
            scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t0
            ms_per = elapsed / step * 1000
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            print(f'step {step:>6d}  loss {loss_val:.4f}  '
                  f'lr {current_lr:.1e}  {ms_per:.0f}ms/step  '
                  f'N={N}')

        if n_steps and step >= n_steps:
            break

    # Unwrap compiled model.
    raw = getattr(compiled, '_orig_mod', compiled)
    return raw, losses
