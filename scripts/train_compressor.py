#!/usr/bin/env python3
"""Train the T0→T1 compressor.

Reads batches from stdin. Use gen_batches.py to produce them.

Usage:
    gen_batches.py --n-batches 10000 | train_compressor.py
    gen_batches.py --n-batches 100000 > corpus.bin
    train_compressor.py --lr-schedule 100000 < corpus.bin
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compressor.train import train, save_run
from datagen import BatchReader

DEFAULTS = {
    'torch_threads': 8,
    'lr': 3e-4,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 2,
    'd_out': 128,
    'device': 'auto',
    'log_every': 100,
}


def parse_args():
    p = argparse.ArgumentParser(description='Train the T0→T1 compressor.')
    p.add_argument('--config', type=str)
    p.add_argument('--lr', type=float)
    p.add_argument('--n-steps', type=int, default=None,
                   help='Expected batch count (for ETA display)')
    p.add_argument('--lr-schedule', type=int, default=None,
                   help='Cosine decay over this many steps (implies --n-steps)')
    p.add_argument('--lr-min', type=float, default=1e-6,
                   help='Minimum LR for cosine decay (default: 1e-6)')
    p.add_argument('--d-model', type=int)
    p.add_argument('--n-heads', type=int)
    p.add_argument('--n-layers', type=int)
    p.add_argument('--d-out', type=int)
    p.add_argument('--torch-threads', type=int)
    p.add_argument('--device', type=str)
    p.add_argument('--log-every', type=int)
    p.add_argument('--out-dir', type=str, default='runs')
    return p.parse_args()


def build_hparams(args):
    hparams = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            hparams.update(json.load(f))
    for k, v in vars(args).items():
        if v is not None and k in hparams:
            hparams[k] = v
    return hparams


def main():
    args = parse_args()
    hparams = build_hparams(args)

    print('Hyperparameters:')
    for k, v in sorted(hparams.items()):
        print(f'  {k}: {v}')
    if args.lr_schedule:
        print(f'  lr_schedule: {args.lr_schedule}')
        print(f'  lr_min: {args.lr_min}')
    else:
        print('  lr_schedule: none (constant LR)')
    if args.n_steps:
        print(f'  n_steps: {args.n_steps}')
    print()

    batch_iter = BatchReader(sys.stdin.buffer)
    model, losses = train(
        **hparams,
        batch_iter=batch_iter,
        n_steps=args.n_steps,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
    )
    save_run(model, losses, hparams={
        **hparams,
        'n_steps': args.n_steps,
        'lr_schedule': args.lr_schedule,
        'lr_min': args.lr_min,
        'steps_completed': len(losses) * hparams['log_every'],
    }, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
