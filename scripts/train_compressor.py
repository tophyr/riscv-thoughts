#!/usr/bin/env python3
"""Train a compressor model.

Modes:
    instr:     N×N pairwise MSE on single-instruction batches (step 1).
    streaming: Shift-reduce compressor with REINFORCE gate training.

Usage:
    # Encoder training on RVB batches (step 1):
    gen_instr_batches.py --n-batches 10000 | \
        train_compressor.py --mode instr --n-steps 100000

    # Streaming compressor on RVS batches:
    gen_seq_batches.py --n-batches 5000 | \
        train_compressor.py --mode streaming --n-steps 10000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from compressor.train import train_batches, streaming_train


def _default_save_dir():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'runs/{stamp}'


def main():
    p = argparse.ArgumentParser(description='Train compressor.')
    p.add_argument('--mode', choices=['instr', 'streaming'],
                   default='instr')

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
    p.add_argument('--device', default='auto')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--save', type=str, default=None,
                   help='Save directory (default: runs/<timestamp>)')
    p.add_argument('--no-save', action='store_true',
                   help='Disable auto-save')

    # Streaming mode.
    p.add_argument('--pairwise-weight', type=float, default=1.0)
    p.add_argument('--reinforce-lr', type=float, default=1e-3)

    # Explicit equivalence loss (instr mode).
    p.add_argument('--equiv-weight', type=float, default=0.0,
                   help='Weight for explicit equivalence collapse loss. '
                        '0 disables.')

    args = p.parse_args()

    save_dir = None
    if not args.no_save:
        save_dir = Path(args.save or _default_save_dir())

    if args.mode == 'instr':
        from datagen import InstructionBatchReader
        reader = InstructionBatchReader(sys.stdin.buffer)
        model, losses = train_batches(
            batch_iter=reader,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_out=args.d_out,
            lr=args.lr,
            device=args.device,
            n_steps=args.n_steps,
            log_every=args.log_every,
            equiv_weight=args.equiv_weight,
        )
        print(f'\nDone: {len(losses)} steps, '
              f'final loss {losses[-1]:.4f}')
        extra = {}

    else:  # streaming
        from datagen import SequenceBatchReader
        reader = SequenceBatchReader(sys.stdin.buffer)
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
            pairwise_weight=args.pairwise_weight,
            reinforce_lr=args.reinforce_lr,
        )
        last = losses[-1]
        print(f'\nDone: {len(losses)} steps, '
              f'loss {last["total"]:.4f} '
              f'(recon {last["recon"]:.3f} pair {last["pairwise"]:.4f})')
        model = encoder
        extra = {'gate_stats': gate_stats}

    if save_dir:
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
