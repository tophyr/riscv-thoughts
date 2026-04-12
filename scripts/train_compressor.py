#!/usr/bin/env python3
"""Train a compressor model.

Modes:
    fixed:     Fixed-window model for context experiments.
    streaming: Streaming encoder + decoder with learnable emit gate.

Usage:
    # Context experiment (fixed window):
    gen_seq_batches.py --n-batches 5000 | \
        train_compressor.py --mode fixed --window-size 2

    # Streaming compressor with decoder:
    gen_seq_batches.py --n-batches 5000 | \
        train_compressor.py --mode streaming --n-steps 10000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from datagen import BatchReader
from compressor.train import train, streaming_train


def main():
    p = argparse.ArgumentParser(description='Train compressor.')
    p.add_argument('--mode', choices=['fixed', 'streaming'],
                   default='streaming')

    # Encoder architecture.
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=128)

    # Decoder architecture (streaming mode).
    p.add_argument('--dec-d-model', type=int, default=128)
    p.add_argument('--dec-n-heads', type=int, default=4)
    p.add_argument('--dec-n-layers', type=int, default=2)

    # Training.
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-steps', type=int, default=None)
    p.add_argument('--lr-schedule', type=int, default=None,
                   help='Cosine decay over this many steps')
    p.add_argument('--device', default='auto')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--save', type=str, default=None)

    # Fixed-window mode.
    p.add_argument('--window-size', type=int, default=1,
                   help='Instructions per window (fixed mode only)')

    # Streaming mode.
    p.add_argument('--gate-tau', type=float, default=1.0,
                   help='Gumbel-sigmoid temperature')
    p.add_argument('--recon-weight', type=float, default=1.0,
                   help='Weight for reconstruction loss')
    p.add_argument('--pairwise-weight', type=float, default=1.0,
                   help='Weight for pairwise MSE loss')

    args = p.parse_args()

    reader = BatchReader(sys.stdin.buffer)

    if args.mode == 'fixed':
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
        extra = {}
    else:
        encoder, decoder, losses, gate_stats = streaming_train(
            batch_iter=reader,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_out=args.d_out,
            dec_d_model=args.dec_d_model,
            dec_n_heads=args.dec_n_heads,
            dec_n_layers=args.dec_n_layers,
            lr=args.lr,
            device=args.device,
            n_steps=args.n_steps,
            log_every=args.log_every,
            lr_schedule=args.lr_schedule,
            gate_tau=args.gate_tau,
            recon_weight=args.recon_weight,
            pairwise_weight=args.pairwise_weight,
        )
        last = losses[-1]
        print(f'\nDone: {len(losses)} steps, '
              f'loss {last["total"]:.4f} '
              f'(recon {last["recon"]:.3f} pair {last["pairwise"]:.4f})')
        model = encoder
        extra = {'gate_stats': gate_stats}

    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / 'encoder.pt')
        if args.mode == 'streaming':
            torch.save(decoder.state_dict(), save_dir / 'decoder.pt')
        with open(save_dir / 'losses.json', 'w') as f:
            json.dump(losses, f)
        with open(save_dir / 'hparams.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        if extra:
            with open(save_dir / 'gate_stats.json', 'w') as f:
                json.dump(extra.get('gate_stats', []), f)
        print(f'Saved to {save_dir}')


if __name__ == '__main__':
    main()
