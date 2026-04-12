#!/usr/bin/env python3
"""Train a fixed-window compressor to test whether sequential context
improves the T1 vector space.

Usage:
    # Baseline (single instruction):
    gen_seq_batches.py --n-batches 5000 | train_compressor.py --window-size 1

    # With one instruction of context:
    gen_seq_batches.py --n-batches 5000 | train_compressor.py --window-size 2

    # From file:
    train_compressor.py --window-size 2 --n-steps 10000 < corpus.bin
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from datagen import BatchReader
from compressor.train import train


def main():
    p = argparse.ArgumentParser(
        description='Train fixed-window compressor.')
    p.add_argument('--window-size', type=int, default=1,
                   help='Instructions per window (1=baseline, 2=context)')
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-steps', type=int, default=None)
    p.add_argument('--lr-schedule', type=int, default=None,
                   help='Cosine decay over this many steps')
    p.add_argument('--device', default='auto')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--save', type=str, default=None,
                   help='Directory to save model and losses')
    args = p.parse_args()

    reader = BatchReader(sys.stdin.buffer)

    model, losses = train(
        batch_iter=reader,
        window_size=args.window_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_out=args.d_out,
        lr=args.lr,
        device=args.device,
        n_steps=args.n_steps,
        log_every=args.log_every,
        lr_schedule=args.lr_schedule,
    )

    print(f'\nDone: {len(losses)} steps, '
          f'final loss {losses[-1]:.4f}')

    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / 'model.pt')
        with open(save_dir / 'losses.json', 'w') as f:
            json.dump(losses, f)
        with open(save_dir / 'hparams.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f'Saved to {save_dir}')


if __name__ == '__main__':
    main()
