#!/usr/bin/env python3
"""Train the T0→T1 compressor.

Usage:
    python scripts/train_compressor.py
    python scripts/train_compressor.py --config hparams.json
    python scripts/train_compressor.py --batch-size 512 --lr 3e-4 --n-steps 10000
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compressor.train import train, save_run


DEFAULTS = {
    'batch_size': 4096,
    'n_steps': 100_000,
    'n_inputs': 32,
    'n_producers': 24,
    'prefetch': 16,
    'torch_threads': 8,
    'lr': 3e-4,
    'lr_min': 1e-6,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 2,
    'd_out': 128,
    'device': 'auto',
    'log_every': 1000,
    'seed': 42,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train the T0→T1 compressor.')

    parser.add_argument('--config', type=str, default=None,
                        help='JSON file with hyperparameters (overrides defaults, '
                             'CLI args override config file)')

    # Training
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--n-steps', type=int)
    parser.add_argument('--n-inputs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr-min', type=float)
    parser.add_argument('--seed', type=int)

    # Model
    parser.add_argument('--d-model', type=int)
    parser.add_argument('--n-heads', type=int)
    parser.add_argument('--n-layers', type=int)
    parser.add_argument('--d-out', type=int)

    # Infrastructure
    parser.add_argument('--n-producers', type=int)
    parser.add_argument('--prefetch', type=int)
    parser.add_argument('--torch-threads', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--log-every', type=int)

    # Output
    parser.add_argument('--out-dir', type=str, default='runs')

    return parser.parse_args()


def build_hparams(args):
    """Merge defaults ← config file ← CLI args."""
    hparams = dict(DEFAULTS)

    # Layer 2: config file
    if args.config:
        with open(args.config) as f:
            hparams.update(json.load(f))

    # Layer 3: CLI args (only non-None values)
    cli_overrides = {
        'batch_size': args.batch_size,
        'n_steps': args.n_steps,
        'n_inputs': args.n_inputs,
        'lr': args.lr,
        'lr_min': args.lr_min,
        'seed': args.seed,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_out': args.d_out,
        'n_producers': args.n_producers,
        'prefetch': args.prefetch,
        'torch_threads': args.torch_threads,
        'device': args.device,
        'log_every': args.log_every,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            hparams[k] = v

    return hparams


def main():
    args = parse_args()
    hparams = build_hparams(args)

    print('Hyperparameters:')
    for k, v in sorted(hparams.items()):
        print(f'  {k}: {v}')
    print()

    # Separate out_dir from train() params.
    out_dir = args.out_dir
    train_params = {k: v for k, v in hparams.items() if k != 'out_dir'}

    model, losses = train(**train_params)
    save_run(model, losses, hparams=hparams, out_dir=out_dir)


if __name__ == '__main__':
    main()
