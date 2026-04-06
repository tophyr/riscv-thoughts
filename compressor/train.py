"""Training loop and loss functions for the T0→T1 compressor."""

import numpy as np
import torch
import torch.nn.functional as F

from emulator import Instruction
from tokenizer import encode_instruction, PAD, VOCAB_SIZE

from .model import T1Compressor
from datagen import ParallelProducer


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_batch(instructions, device=torch.device('cpu')):
    """Tokenize a batch of instructions into padded tensors."""
    encoded = [encode_instruction(instr) for instr in instructions]
    max_len = max(len(e) for e in encoded)
    token_ids = torch.full((len(encoded), max_len), PAD, dtype=torch.long,
                           device=device)
    padding_mask = torch.ones(len(encoded), max_len, dtype=torch.bool,
                              device=device)
    for i, enc in enumerate(encoded):
        token_ids[i, :len(enc)] = torch.tensor(enc, dtype=torch.long)
        padding_mask[i, :len(enc)] = False
    return token_ids, padding_mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def correlation_loss(t1_vecs, exec_dist_matrix):
    """1 - Pearson correlation between T1 and execution pairwise distances."""
    t1_dists = torch.cdist(t1_vecs, t1_vecs)

    B = t1_vecs.shape[0]
    idx = torch.triu_indices(B, B, offset=1, device=t1_vecs.device)
    t1_flat = t1_dists[idx[0], idx[1]]
    exec_flat = exec_dist_matrix[idx[0], idx[1]]

    t1_centered = t1_flat - t1_flat.mean()
    exec_centered = exec_flat - exec_flat.mean()

    num = (t1_centered * exec_centered).sum()
    denom = (t1_centered.norm() * exec_centered.norm()).clamp(min=1e-8)
    return 1.0 - num / denom


def equiv_loss(t1_vecs, exec_dists, data_ranges):
    """Weighted distance-matching loss: push T1 distances toward
    being proportional to exec distances, weighted by output range.

    Pairs of high-range instructions that agree get strong signal.
    Pairs of low-range instructions get weak signal.

    Both attractive (collapse equivalences) and repulsive (maintain
    separation for non-equivalences). The weight emphasizes pairs
    where agreement is most surprising given the output range.
    """
    t1_dists = torch.cdist(t1_vecs, t1_vecs)

    B = t1_vecs.shape[0]
    idx = torch.triu_indices(B, B, offset=1, device=t1_vecs.device)
    t1_flat = t1_dists[idx[0], idx[1]]
    exec_flat = exec_dists[idx[0], idx[1]]

    # Per-pair range: max of the two instructions' output ranges.
    # +1 ensures constant-output equivalences (range=0) still get
    # collapse signal, while preserving ordering (range=0 gets half
    # the weight of range=1).
    pair_range = torch.maximum(data_ranges[idx[0]], data_ranges[idx[1]]) + 1.0

    weight = pair_range / (1.0 + exec_flat)
    weight = weight / weight.sum().clamp(min=1e-8)

    # Scale exec distances to match T1 distance scale.
    # Use the weighted ratio of T1 to exec distances as the scale factor.
    with torch.no_grad():
        scale = (t1_flat * weight).sum() / (exec_flat * weight).sum().clamp(min=1e-8)

    target = exec_flat * scale
    return (weight * (t1_flat - target).square()).sum()


def combined_loss(t1_vecs, dt_logits, dr_logits,
                  exec_dists, dt_targets, dr_targets, data_ranges):
    """Correlation + equivalence + destination classification loss."""
    corr = correlation_loss(t1_vecs, exec_dists)
    eq = equiv_loss(t1_vecs, exec_dists, data_ranges)
    type_loss = F.cross_entropy(dt_logits, dt_targets)

    reg_mask = (dt_targets == 0)
    if reg_mask.any():
        reg_loss = F.cross_entropy(dr_logits[reg_mask], dr_targets[reg_mask])
    else:
        reg_loss = torch.tensor(0.0, device=t1_vecs.device)

    return corr + eq + type_loss + reg_loss


# ---------------------------------------------------------------------------
# GPU distance computation
# ---------------------------------------------------------------------------

def exec_distance(data_vals, pc_vals, device):
    """Two-component pairwise distance on GPU.

    Distance = mean_over_inputs(|data_diff| + |pc_diff|).
    Vectorized across input states — one kernel launch instead of 32.
    """
    dv = torch.tensor(data_vals, dtype=torch.bfloat16, device=device)
    pv = torch.tensor(pc_vals, dtype=torch.bfloat16, device=device)

    # (B, 1, S) - (1, B, S) → (B, B, S) pairwise diffs across all inputs.
    d_all = (dv.unsqueeze(1) - dv.unsqueeze(0)).abs()
    p_all = (pv.unsqueeze(1) - pv.unsqueeze(0)).abs()
    return (d_all + p_all).mean(dim=2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    batch_size: int,
    n_steps: int,
    n_inputs: int,
    n_producers: int,
    prefetch: int,
    torch_threads: int,
    lr: float,
    lr_min: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_out: int,
    device: str,
    log_every: int,
    seed: int,
):
    """Train the T0→T1 compressor on all RV32I instruction types."""
    rng = np.random.default_rng(seed)

    # Fork producers BEFORE initializing CUDA or torch threads.
    with ParallelProducer(
        batch_size=batch_size, n_inputs=n_inputs,
        n_batches=n_steps, seed=int(rng.integers(0, 2**63)),
        n_workers=n_producers, prefetch=prefetch,
    ) as producer:

        torch.set_num_threads(torch_threads)

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        model = T1Compressor(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_out=d_out,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=lr_min)
        import time
        losses = []
        t_start = time.monotonic()

        for step, batch in enumerate(producer):
            token_ids = torch.from_numpy(batch.token_ids).to(device)
            padding_mask = torch.from_numpy(batch.padding_mask).to(device)
            exec_dists = exec_distance(batch.data_vals, batch.pc_vals, device)
            dt_targets = torch.from_numpy(batch.dest_types).to(device)
            dr_targets = torch.from_numpy(batch.dest_regs).to(device)

            # Per-instruction output range across input states.
            dv_range = batch.data_vals.max(axis=1) - batch.data_vals.min(axis=1)
            data_ranges = torch.tensor(dv_range, dtype=torch.float32, device=device)

            t1_vecs, dt_logits, dr_logits = model(token_ids, padding_mask)
            loss = combined_loss(t1_vecs, dt_logits, dr_logits,
                                 exec_dists, dt_targets, dr_targets,
                                 data_ranges)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            losses.append(loss_val)

            if step % log_every == 0:
                elapsed = time.monotonic() - t_start
                current_lr = optimizer.param_groups[0]['lr']
                if step > 0:
                    remaining = elapsed / step * (n_steps - step)
                    m, s = divmod(int(remaining), 60)
                    h, m = divmod(m, 60)
                    eta = f'{h}h{m:02d}m' if h else f'{m}m{s:02d}s'
                else:
                    eta = '...'
                print(f'step {step:5d}  loss={loss_val:.4f}  '
                      f'lr={current_lr:.2e}  eta={eta}')

    return model, losses


def save_run(model, losses, hparams=None, out_dir='runs'):
    """Save model checkpoint, loss history, and hyperparameters."""
    import json
    from pathlib import Path
    from datetime import datetime

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = out / stamp
    run_dir.mkdir()

    torch.save(model.state_dict(), run_dir / 'model.pt')
    with open(run_dir / 'losses.json', 'w') as f:
        json.dump(losses, f)
    if hparams is not None:
        with open(run_dir / 'hparams.json', 'w') as f:
            json.dump(hparams, f, indent=2)

    print(f'Saved to {run_dir}')
    return run_dir
