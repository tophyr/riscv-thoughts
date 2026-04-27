#!/usr/bin/env python3
"""Train the T2 compressor over chunked T1-emission sequences.

Reads RVS sequence batches from stdin, runs them through a frozen T1
encoder + the T2 chunker (structurally splits at memory ops, control
flow changes, or a max-length cap), augments with invalid chunks
(spanning / multi / overlong), and trains T2Compressor with:
- magnitude-as-validity loss
- per-register pairwise MSE on loglog'd register-state-delta differences
- modified-regs BCE
- terminator-type CE

Usage:
    gen_seq_batches.py --n-batches 5000 --config configs/instr_t2.json |
        train_t2.py --t1-encoder runs/<stamp>/encoder.pt \
                    --n-steps 5000 --d-out 256
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from compressor.model import T1Compressor
from compressor.train import train_t2, load_checkpoint
from datagen import SequenceBatchReader
from tokenizer import VOCAB_SIZE
from scripts._common import resolve_device


def _default_save_dir():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'runs/{stamp}_t2'


def main():
    p = argparse.ArgumentParser(description='Train T2 compressor.')

    # Frozen T1 encoder.
    p.add_argument('--t1-encoder', required=True,
                   help='Path to a trained T1 encoder.pt')
    p.add_argument('--t1-d-model', type=int, default=128)
    p.add_argument('--t1-n-heads', type=int, default=4)
    p.add_argument('--t1-n-layers', type=int, default=2)
    p.add_argument('--t1-d-out', type=int, default=64)
    p.add_argument('--t1-max-window', type=int, default=32)

    # T2 architecture.
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=256)
    p.add_argument('--max-chunk-len', type=int, default=24,
                   help='Storage cap for chunk_emissions; ≥ overlong size')
    p.add_argument('--validity-max-chunk-len', type=int, default=16,
                   help='Cap above which a chunk is structurally invalid')
    p.add_argument('--invalidity-rate', type=float, default=0.2)

    # Training.
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-steps', type=int, default=None)
    p.add_argument('--device', default='auto')
    p.add_argument('--log-every', type=int, default=100)

    # Loss weights.
    p.add_argument('--mag-weight', type=float, default=1.0)
    p.add_argument('--reg-effect-weight', type=float, default=1.0)
    p.add_argument('--modified-regs-weight', type=float, default=1.0)
    p.add_argument('--term-type-weight', type=float, default=1.0)

    # Save/RNG.
    p.add_argument('--chunker-seed', type=int, default=0)
    p.add_argument('--save', type=str, default=None,
                   help='Save directory (default: runs/<timestamp>_t2)')
    p.add_argument('--no-save', action='store_true',
                   help='Disable auto-save')

    args = p.parse_args()

    device = resolve_device(args.device)
    save_dir = None
    if not args.no_save:
        save_dir = Path(args.save or _default_save_dir())

    # Load frozen T1 encoder.
    t1 = T1Compressor(
        VOCAB_SIZE,
        d_model=args.t1_d_model, n_heads=args.t1_n_heads,
        n_layers=args.t1_n_layers, d_out=args.t1_d_out,
        max_window=args.t1_max_window).to(device)
    t1.load_state_dict(load_checkpoint(args.t1_encoder, device), strict=False)

    reader = SequenceBatchReader(sys.stdin.buffer)
    t2_model, losses = train_t2(
        batch_iter=reader,
        t1_encoder=t1,
        d_in=args.t1_d_out,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_out=args.d_out,
        max_chunk_len=args.max_chunk_len,
        validity_max_chunk_len=args.validity_max_chunk_len,
        invalidity_rate=args.invalidity_rate,
        lr=args.lr,
        device=args.device,
        n_steps=args.n_steps,
        log_every=args.log_every,
        mag_weight=args.mag_weight,
        reg_effect_weight=args.reg_effect_weight,
        modified_regs_weight=args.modified_regs_weight,
        term_type_weight=args.term_type_weight,
        chunker_seed=args.chunker_seed,
    )

    print(f'\nDone: {len(losses)} steps, '
          f'final loss {losses[-1]:.4f}')

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(t2_model.state_dict(), save_dir / 't2.pt')
        with open(save_dir / 'losses.json', 'w') as f:
            json.dump(losses, f)
        with open(save_dir / 'hparams.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f'Saved to {save_dir}')


if __name__ == '__main__':
    main()
