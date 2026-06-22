"""Shared support for the model-running CLIs (train_* and eval_*).

The compute-side counterpart to _streamfmt (which is the stream-format
wire layer): device selection, the common training arg spine, run-dir IO,
and checkpoint loading.

  resolve_device:                       'auto'/'cuda'/'cpu' → concrete device.
  add_common_train_args, open_run_dir:  shared training CLI + run-dir IO.
  load_frozen_encoder, load_t2:         build a model from a run dir/ckpt.
"""

import json
from datetime import datetime
from pathlib import Path

import torch

from compressor.model import T1Compressor, T2Compressor
from compressor.train import load_checkpoint, resolve_device
from tokenizer import VOCAB_SIZE


# ---------------------------------------------------------------------------
# Training CLI + run-directory IO — shared spine across all trainers
# ---------------------------------------------------------------------------
def add_common_train_args(p, *, lr=3e-4):
    """Add the training CLI spine shared by all trainers: --lr, --n-steps,
    --log-every, --out-dir. Each trainer adds its own model/loss args."""
    p.add_argument('--lr', type=float, default=lr)
    p.add_argument('--n-steps', type=int, required=True,
                   help='Cosine-LR T_max and ETA display. NOT a hard cap on '
                        'duration — training reads until stdin EOF. Bound the '
                        'pipeline upstream with `batch_slice --count N` to '
                        'control how many batches the trainer sees.')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--out-dir', type=str, default=None,
                   help='Run-directory NAME (created under runs/). Default: a '
                        'timestamp. Always saves; the run-dir structure '
                        '(<model>.pt + hparams.json + losses.json) is '
                        'identical across all trainers.')


def open_run_dir(args, model_name, *, suffix=''):
    """Create runs/<--out-dir or timestamp+suffix>/, write hparams.json up
    front, and return (save_dir, save_fn). save_fn(model, losses,
    train_state=None) writes <model_name>.pt + losses.json (+ train_state.pt
    when given), giving every trainer an identical run-dir structure."""
    save_dir = Path('runs') / (
        args.out_dir or datetime.now().strftime('%Y%m%d_%H%M%S') + suffix)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'hparams.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f'Run dir: {save_dir}')

    def save_fn(model, losses, train_state=None):
        torch.save(model.state_dict(), save_dir / f'{model_name}.pt')
        with open(save_dir / 'losses.json', 'w') as f:
            json.dump(losses, f)
        if train_state is not None:
            torch.save(train_state, save_dir / 'train_state.pt')

    return save_dir, save_fn


# ---------------------------------------------------------------------------
# Model loading — same pattern across train_*, eval suites, etc.
# ---------------------------------------------------------------------------

def load_frozen_encoder(model_path, device):
    """Build a T1Compressor from the encoder's companion hparams.json
    (next to model_path), strict-load the checkpoint, eval(), freeze
    parameters.

    model_path: path to encoder.pt; its sibling hparams.json supplies the
                arch (d_model, n_heads, n_layers, d_out, max_window).
    """
    hp = json.loads((Path(model_path).parent / 'hparams.json').read_text())
    encoder = T1Compressor(
        VOCAB_SIZE,
        d_model=hp.get('d_model', 128),
        n_heads=hp.get('n_heads', 4),
        n_layers=hp.get('n_layers', 2),
        d_out=hp.get('d_out', 64),
        max_window=hp.get('max_window', 32),
    ).to(device)
    encoder.load_state_dict(load_checkpoint(model_path, device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_t2(t2_dir, t1, device):
    """Build a T2Compressor from t2_dir/hparams.json, strict-load t2.pt,
    eval(). Returns (t2, hparams). NOT gradient-frozen — callers needing
    grads (measure_loss_dims) backprop through it; eval-only callers wrap
    in torch.no_grad()."""
    t2_dir = Path(t2_dir)
    hp = json.loads((t2_dir / 'hparams.json').read_text())
    t2 = T2Compressor(
        d_t1=t1.d_out, d_model=hp['d_model'], n_heads=hp['n_heads'],
        n_layers=hp['n_layers'], d_out=hp['d_out'],
        max_chunk_len=hp['max_chunk_len'],
    ).to(device)
    t2.load_state_dict(load_checkpoint(t2_dir / 't2.pt', device))
    t2.eval()
    return t2, hp
