#!/usr/bin/env python3
"""Train a T1 encoder via pair-MSE + magnitude-validity on RVT batches.

Step 1 of the piecemeal-assembly plan. CLI shell over
compressor.train.train_encoder; saves encoder.pt + hparams.json
+ losses.json into a run directory.

Usage:
    gen_batches.py --rule branch+cap=8 --twins 3 --partners 20 \\
        --inject-invalid 0.1 -n 5000 -v |
            batch_repeat.py --forever |
            train_encoder.py --n-steps 50000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from compressor.train import train_encoder
from datagen.batch import RVT_FORMAT, Batch
from scripts._common import resolve_device


def _default_save_dir():
    return f'runs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=128)
    p.add_argument('--max-window', type=int, default=72,
                   help='Position-embedding table size. Must accommodate '
                        'the longest chunk (9 tokens × max instructions).')

    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-steps', type=int, required=True,
                   help='Cosine-LR T_max and ETA display. NOT a hard cap on '
                        'duration — training reads until stdin EOF. Bound the '
                        'pipeline upstream with `batch_slice --count N` to '
                        'control how many batches the trainer sees.')
    p.add_argument('--log-every', type=int, default=100)

    p.add_argument('--behavioral-weight', type=float, default=1.0)
    p.add_argument('--behavioral-scale', type=float, default=10.0,
                   help='Behavioral_distance compression scale. '
                        'd_target_compressed = 2 * tanh(d_target / scale). '
                        '10.0 puts mean (~5) at 1.5 and max (~55) at ~2.')
    p.add_argument('--valid-weight', type=float, default=0.1)
    p.add_argument('--equiv-weight', type=float, default=0.0,
                   help='Auxiliary MANIFEST equivalence loss. 0 disables.')
    p.add_argument('--dest-type-weight', type=float, default=0.1,
                   help='CE loss on dest_type_head (reg vs mem write).')
    p.add_argument('--dest-reg-weight', type=float, default=0.1,
                   help='CE loss on dest_reg_head (which register is dest).')
    p.add_argument('--src-reg-weight', type=float, default=0.1,
                   help='CE loss on src_reg_head_0 + src_reg_head_1 '
                        '(applied to each slot, summed).')

    p.add_argument('--device', default='auto')
    p.add_argument('--save', type=str, default=None,
                   help='Run directory. Default: runs/<timestamp>.')
    p.add_argument('--no-save', action='store_true')
    args = p.parse_args()

    device = resolve_device(args.device)

    save_dir = None
    if not args.no_save:
        save_dir = Path(args.save or _default_save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save hparams up front so the run dir is self-describing
        # immediately after launch.
        with open(save_dir / 'hparams.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f'Run dir: {save_dir}')

    reader = RVT_FORMAT.reader(sys.stdin.buffer, Batch)

    def _on_log(step, encoder, losses_so_far):
        if save_dir is None:
            return
        torch.save(encoder.state_dict(), save_dir / 'encoder.pt')
        with open(save_dir / 'losses.json', 'w') as f:
            json.dump(losses_so_far, f)

    encoder, losses = train_encoder(
        reader,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_out=args.d_out, max_window=args.max_window,
        lr=args.lr, n_steps=args.n_steps, log_every=args.log_every,
        behavioral_weight=args.behavioral_weight,
        behavioral_scale=args.behavioral_scale,
        valid_weight=args.valid_weight,
        equiv_weight=args.equiv_weight,
        dest_type_weight=args.dest_type_weight,
        dest_reg_weight=args.dest_reg_weight,
        src_reg_weight=args.src_reg_weight,
        device=device,
        on_log=_on_log,
    )
    print(f'\nDone: {len(losses)} steps, '
          f'final loss {losses[-1]["total"]:.4f}')

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), save_dir / 'encoder.pt')
        with open(save_dir / 'losses.json', 'w') as f:
            json.dump(losses, f)
        with open(save_dir / 'hparams.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f'Saved to {save_dir}')


if __name__ == '__main__':
    main()
