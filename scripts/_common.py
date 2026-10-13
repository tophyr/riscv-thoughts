"""Shared support for the model-running CLIs (train_* and eval_*).

The compute-side counterpart to the RVT wire layer (datagen.batch's
BinaryFormat): device selection, the common training arg spine, run-dir IO,
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
from compressor.train import load_checkpoint, resolve_device, _split_t1_state
from tokenizer import VOCAB_SIZE


# ---------------------------------------------------------------------------
# Training CLI + run-directory IO — shared spine across all trainers
# ---------------------------------------------------------------------------
def add_common_train_args(p, *, lr=3e-4):
    """Add the training CLI spine shared by all trainers: --lr, --n-steps,
    --log-every, --save-every, --out-dir. Each trainer adds its own model/loss
    args."""
    p.add_argument('--lr', type=float, default=lr)
    p.add_argument('--n-steps', type=int, required=True,
                   help='Cosine-LR T_max and ETA display. NOT a hard cap on '
                        'duration — training reads until stdin EOF. Bound the '
                        'pipeline upstream with `datagen --count N` to '
                        'control how many batches the trainer sees.')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--save-every', type=int, default=10000,
                   help='Checkpoint (model, losses.json, train_state) at the '
                        'first log boundary at least this many steps after the '
                        'last. A save stalls the GPU, so it is kept off the '
                        'log cadence. The final model is always saved.')
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


def _tile_shared_pool_v(state, model):
    """Widen a checkpoint's single shared pool value projection into the
    per-head projections the pool now uses.

    The pool used to give every head one shared projection of width
    d_value/n_pool_heads; it now gives each head its own slice of a d_value-wide
    projection. Repeating the shared weights once per head reproduces the old
    computation exactly, so a checkpoint trained under the shared projection
    keeps its behaviour and only gains the freedom to differentiate the heads if
    it is trained further. Checkpoints with one pool head already match.
    """
    for prefix in ('', 'compressor.'):
        key = f'{prefix}core.pool_v.weight'
        if key not in state:
            continue
        want = model.state_dict()[key].shape[0]
        have = state[key].shape[0]
        if have != want:
            reps = want // have
            for suffix in ('weight', 'bias'):
                k = f'{prefix}core.pool_v.{suffix}'
                state[k] = state[k].repeat(reps, *([1] * (state[k].dim() - 1)))
    return state


def load_frozen_encoder(model_path, device):
    """Build a T1Compressor (TokenEmbedder + Compressor) from the encoder's
    companion hparams.json (next to model_path), load the checkpoint (old flat
    state dicts are split-remapped to embedder./compressor.), eval(), freeze.

    model_path: path to encoder.pt; its sibling hparams.json supplies the arch
                (d_model, n_heads, n_layers, d_out, max_window, d_event).
    """
    hp = json.loads((Path(model_path).parent / 'hparams.json').read_text())
    encoder = T1Compressor(
        vocab_size=VOCAB_SIZE,
        d_model=hp.get('d_model', 128),
        n_heads=hp.get('n_heads', 4),
        n_layers=hp.get('n_layers', 2),
        max_window=hp.get('max_window', 32),
        d_out=hp.get('d_out', 64),
        d_event=hp.get('d_event', 16),
        op_cell_depth=hp.get('op_cell_depth', 1),
        vp_head_depth=hp.get('vp_head_depth', 4),
        n_pool_heads=hp.get('n_pool_heads', 1),
    ).to(device)
    encoder.load_state_dict(_tile_shared_pool_v(
        _split_t1_state(load_checkpoint(model_path, device)), encoder))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_t2(t2_dir, t1, device):
    """Build a T2Compressor (a Compressor fed t1's essence) from
    t2_dir/hparams.json, load t2.pt, eval(). Returns (t2, hparams)."""
    t2_dir = Path(t2_dir)
    hp = json.loads((t2_dir / 'hparams.json').read_text())
    t2 = T2Compressor(
        d_t1=t1.d_out, d_out=hp['d_out'], d_event=hp.get('d_event', 16),
        op_cell_depth=hp.get('op_cell_depth', 1),
        vp_head_depth=hp.get('vp_head_depth', 4),
        n_pool_heads=hp.get('n_pool_heads', 1),
    ).to(device)
    t2.load_state_dict(_tile_shared_pool_v(
        load_checkpoint(t2_dir / 't2.pt', device), t2))
    t2.eval()
    return t2, hp
