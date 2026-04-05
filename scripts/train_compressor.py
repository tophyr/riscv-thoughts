#!/usr/bin/env python3
"""Train and evaluate the T0→T1 compressor."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compressor.train import train, save_run
from compressor.eval import evaluate

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
    'log_every': 100,
    'seed': 42,
}


def parse_args():
    p = argparse.ArgumentParser(description='Train the T0→T1 compressor.')
    p.add_argument('--config', type=str)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--n-steps', type=int)
    p.add_argument('--n-inputs', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--lr-min', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--d-model', type=int)
    p.add_argument('--n-heads', type=int)
    p.add_argument('--n-layers', type=int)
    p.add_argument('--d-out', type=int)
    p.add_argument('--n-producers', type=int)
    p.add_argument('--prefetch', type=int)
    p.add_argument('--torch-threads', type=int)
    p.add_argument('--device', type=str)
    p.add_argument('--log-every', type=int)
    p.add_argument('--out-dir', type=str, default='runs')
    p.add_argument('--skip-eval', action='store_true')
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
    print()

    model, losses = train(**hparams)
    run_dir = save_run(model, losses, hparams=hparams, out_dir=args.out_dir)

    if not args.skip_eval:
        print()
        evaluate(str(run_dir))


if __name__ == '__main__':
    main()
