"""Training loop and loss functions for the T0→T1 compressor."""

import numpy as np
import torch
import torch.nn.functional as F

from emulator import Instruction
from tokenizer import encode_instruction, PAD, VOCAB_SIZE

from .model import T1Compressor


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

_EXEC_DIST_SCALE = 2.0 / 22.0  # map log-scaled exec dist [0, ~22] → [0, ~2]


@torch.compile
def combined_loss(t1_vecs, dt_logits, dr_logits,
                  exec_dists, dt_targets, dr_targets):
    """MSE distance matching + destination classification loss."""
    t1_dists = torch.cdist(t1_vecs, t1_vecs)

    B = t1_vecs.shape[0]
    idx = torch.triu_indices(B, B, offset=1, device=t1_vecs.device)
    t1_flat = t1_dists[idx[0], idx[1]]
    exec_flat = exec_dists[idx[0], idx[1]]

    # MSE: T1 distances should match scaled exec distances.
    target = exec_flat * _EXEC_DIST_SCALE
    dist_loss = (t1_flat - target).square().mean()

    type_loss = F.cross_entropy(dt_logits, dt_targets)

    reg_mask = (dt_targets == 0)
    reg_loss = F.cross_entropy(dr_logits[reg_mask], dr_targets[reg_mask])

    return dist_loss + type_loss + reg_loss


# ---------------------------------------------------------------------------
# GPU distance computation
# ---------------------------------------------------------------------------

def _exec_distance_impl(dv, pv):
    B, S = dv.shape
    acc = torch.zeros(B, B, dtype=torch.float32, device=dv.device)
    for s in range(S):
        d = (dv[:, s].unsqueeze(1) - dv[:, s].unsqueeze(0)).abs()
        p = (pv[:, s].unsqueeze(1) - pv[:, s].unsqueeze(0)).abs()
        acc += torch.log1p(d + p)
    return acc / S

_exec_distance_compiled = torch.compile(_exec_distance_impl)


def exec_distance(data_vals, pc_vals, device, compiled=True):
    """Pairwise execution distance: mean log1p(|data_diff| + |pc_diff|)."""
    dv = torch.tensor(data_vals, dtype=torch.bfloat16, device=device)
    pv = torch.tensor(pc_vals, dtype=torch.bfloat16, device=device)
    fn = _exec_distance_compiled if compiled else _exec_distance_impl
    return fn(dv, pv)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    torch_threads: int,
    lr: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_out: int,
    device: str,
    log_every: int,
    batch_iter,
    n_steps: int | None = None,
    lr_schedule: int | None = None,
    lr_min: float = 1e-6,
):
    """Train the T0→T1 compressor on all RV32I instruction types.

    batch_iter: any iterable yielding Batch objects.
    n_steps: expected batch count. Used for ETA display only.
    lr_schedule: if set, number of steps for cosine LR decay to lr_min.
        If None, LR is constant. Implies n_steps for ETA.
    """
    import time

    torch.set_num_threads(torch_threads)
    torch.set_float32_matmul_precision('high')

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
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_schedule is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=lr_schedule, eta_min=lr_min)
    expected = lr_schedule or n_steps
    losses = []
    t_start = time.monotonic()

    for step, batch in enumerate(batch_iter):
        token_ids = torch.from_numpy(batch.token_ids).to(device)
        padding_mask = torch.from_numpy(batch.padding_mask).to(device)
        exec_dists = exec_distance(batch.data_vals, batch.pc_vals, device)
        dt_targets = torch.from_numpy(batch.dest_types).to(device)
        dr_targets = torch.from_numpy(batch.dest_regs).to(device)

        t1_vecs, dt_logits, dr_logits = model(token_ids, padding_mask)
        loss = combined_loss(t1_vecs, dt_logits, dr_logits,
                             exec_dists, dt_targets, dr_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % log_every == 0:
            loss_val = loss.item()
            losses.append(loss_val)
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.monotonic() - t_start
            ms_per_step = elapsed / step * 1000 if step > 0 else 0
            eta = ''
            if expected is not None and step > 0:
                remaining = elapsed / step * (expected - step)
                m, s = divmod(int(remaining), 60)
                h, m = divmod(m, 60)
                eta = f'  eta={h}h{m:02d}m' if h else f'  eta={m}m{s:02d}s'
            print(f'step {step:5d}  loss={loss_val:.4f}  '
                  f'lr={current_lr:.2e}  {ms_per_step:.0f}ms/step{eta}')

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

    # torch.compile wraps the model, prefixing state_dict keys with
    # _orig_mod. Save the underlying model so checkpoints are portable.
    raw_model = getattr(model, '_orig_mod', model)
    torch.save(raw_model.state_dict(), run_dir / 'model.pt')
    with open(run_dir / 'losses.json', 'w') as f:
        json.dump(losses, f)
    if hparams is not None:
        with open(run_dir / 'hparams.json', 'w') as f:
            json.dump(hparams, f, indent=2)

    print(f'Saved to {run_dir}')
    return run_dir
