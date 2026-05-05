#!/usr/bin/env python3
"""Train a T2 encoder on top of a frozen T1 encoder via pair-MSE on
multi-instruction RVT chunks.

For each chunk, T1 encodes each instruction individually; T2 attention-
pools the resulting sequence into a single chunk vector. Loss is
||t2(a) - t2(b)|| ≈ behavioral_distance(a, b) over the batch's pair structure.

Usage:
    gen_batches.py --rule 'branch+cap=8' --twins 3 --partners 20 \\
        --batch-size 256 -n 5000 -v |
            batch_repeat.py --forever |
            train_t2_encoder.py --t1-model runs/t1_good/encoder.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from compressor.train import train_t2_encoder, load_checkpoint
from compressor.model import T1Compressor
from datagen.batch import RVT_FORMAT, Batch
from scripts._common import resolve_device
from tokenizer import VOCAB_SIZE


def _default_save_dir():
    return f'runs/{datetime.now().strftime("%Y%m%d_%H%M%S")}_t2'


def _load_t1(args, device):
    """Build T1Compressor from --t1-* hparams and load the checkpoint
    (with companion hparams.json next to the .pt for sanity)."""
    hp_path = Path(args.t1_model).parent / 'hparams.json'
    if hp_path.exists():
        hp = json.loads(hp_path.read_text())
        # Pull T1 dims from companion if present (so caller doesn't
        # have to re-specify), but allow CLI overrides.
        d_model = args.t1_d_model or hp.get('d_model', 128)
        n_heads = args.t1_n_heads or hp.get('n_heads', 4)
        n_layers = args.t1_n_layers or hp.get('n_layers', 2)
        d_out = args.t1_d_out or hp.get('d_out', 64)
        max_window = args.t1_max_window or hp.get('max_window', 32)
    else:
        d_model = args.t1_d_model or 128
        n_heads = args.t1_n_heads or 4
        n_layers = args.t1_n_layers or 2
        d_out = args.t1_d_out or 64
        max_window = args.t1_max_window or 32

    t1 = T1Compressor(
        VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_out=d_out, max_window=max_window,
    ).to(device)
    t1.load_state_dict(load_checkpoint(args.t1_model, device), strict=False)
    t1.eval()
    return t1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t1-model', required=True,
                   help='Path to T1 encoder.pt. Companion hparams.json '
                        'in the same dir is read automatically.')
    p.add_argument('--t1-d-model', type=int, default=None)
    p.add_argument('--t1-n-heads', type=int, default=None)
    p.add_argument('--t1-n-layers', type=int, default=None)
    p.add_argument('--t1-d-out', type=int, default=None)
    p.add_argument('--t1-max-window', type=int, default=None)

    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=256)
    p.add_argument('--max-chunk-len', type=int, default=32,
                   help='Position-embedding table size for the T2 sequence '
                        '(max instructions per chunk).')

    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-steps', type=int, required=True)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--behavioral-weight', type=float, default=1.0)
    p.add_argument('--behavioral-scale', type=float, default=10.0,
                   help='Behavioral_distance compression scale.')
    p.add_argument('--valid-weight', type=float, default=0.0,
                   help='Magnitude-validity loss weight. 0 (default) for '
                        'corpora without invalid windows.')

    p.add_argument('--device', default='auto')
    p.add_argument('--save', type=str, default=None,
                   help='Run directory. Default: runs/<timestamp>_t2.')
    p.add_argument('--no-save', action='store_true')
    args = p.parse_args()

    device = resolve_device(args.device)

    save_dir = None
    if not args.no_save:
        save_dir = Path(args.save or _default_save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save hparams up front so the run dir is self-describing as
        # soon as it's created (helpful if the run dies before the
        # first log point).
        with open(save_dir / 'hparams.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f'Run dir: {save_dir}')

    t1 = _load_t1(args, device)

    reader = RVT_FORMAT.reader(sys.stdin.buffer, Batch)

    def _on_log(step, t2, losses_so_far):
        if save_dir is None:
            return
        torch.save(t2.state_dict(), save_dir / 't2.pt')
        with open(save_dir / 'losses.json', 'w') as f:
            json.dump(losses_so_far, f)

    t2, losses = train_t2_encoder(
        reader, t1,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_out=args.d_out, max_chunk_len=args.max_chunk_len,
        lr=args.lr, n_steps=args.n_steps, log_every=args.log_every,
        behavioral_weight=args.behavioral_weight,
        behavioral_scale=args.behavioral_scale,
        valid_weight=args.valid_weight,
        device=device,
        on_log=_on_log,
    )
    print(f'\nDone: {len(losses)} steps, '
          f'final loss {losses[-1]["total"]:.4f}')

    if save_dir:
        torch.save(t2.state_dict(), save_dir / 't2.pt')
        with open(save_dir / 'losses.json', 'w') as f:
            json.dump(losses, f)
        print(f'Saved to {save_dir}')


if __name__ == '__main__':
    main()
