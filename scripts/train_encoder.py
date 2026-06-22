#!/usr/bin/env python3
"""Train a T1 encoder via magnitude-validity + register-identity aux +
value-prediction on RVT batches.

Step 1 of the piecemeal-assembly plan. CLI shell over
compressor.train.train_encoder; saves encoder.pt + hparams.json
+ losses.json into a run directory.

Usage:
    gen_batches.py --rule branch+cap=8 --twins 3 \\
        --inject-invalid 0.1 -n 5000 -v > corpus.rvt
    cat corpus.rvt |
        train_encoder.py --n-steps 50000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compressor.train import train_encoder
from datagen import RVT_FORMAT, Batch
from scripts._common import (
    resolve_device, add_common_train_args, open_run_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=128)
    p.add_argument('--max-window', type=int, default=72,
                   help='Position-embedding table size. Must accommodate '
                        'the longest chunk (9 tokens × max instructions).')

    add_common_train_args(p)

    p.add_argument('--valid-weight', type=float, default=0.1)
    p.add_argument('--live-in-weight', type=float, default=0.1,
                   help='BCE on the behavioral input-register mask.')
    p.add_argument('--live-out-weight', type=float, default=0.1,
                   help='BCE on the behavioral output-register mask.')
    p.add_argument('--pc-writes-weight', type=float, default=0.1,
                   help='BCE on the explicit-PC-write flag.')
    p.add_argument('--in-slot-weight', type=float, default=0.1,
                   help='ListMLE on the input-slot register ordering.')
    p.add_argument('--out-slot-weight', type=float, default=0.1,
                   help='ListMLE on the output-slot register ordering.')
    p.add_argument('--value-predict-weight', type=float, default=1.0,
                   help='V1 experiment: MSE on rd-value prediction '
                        'from (shape, input register values). Tests '
                        'whether shape encodes the I/O function of '
                        'the operator. Requires a --rule single corpus. '
                        'Uses anchor_states '
                        'reconstructed from --anchor-seed and '
                        '--n-anchor-states — these MUST match the '
                        'values gen_batches used or the targets and '
                        'inputs will be inconsistent.')
    p.add_argument('--anchor-seed', type=int, default=0,
                   help='Anchor-states seed for value-prediction. Must '
                        'match gen_batches --anchor-seed.')
    p.add_argument('--n-anchor-states', type=int, default=8,
                   help='Anchor states count. Must match gen_batches '
                        '--n-states.')
    p.add_argument('--no-compile', action='store_true',
                   help='Disable torch.compile (CUDA-graph) of the train '
                        'step. Compile is on by default (~2x faster, ~80%% '
                        'GPU util); disable for a bit-faithful eager run or '
                        'fast-start short runs.')

    args = p.parse_args()

    device = resolve_device('auto')
    save_dir, save = open_run_dir(args, 'encoder')

    reader = RVT_FORMAT.reader(sys.stdin.buffer, Batch)

    encoder, losses = train_encoder(
        reader,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_out=args.d_out, max_window=args.max_window,
        lr=args.lr, n_steps=args.n_steps, log_every=args.log_every,
        valid_weight=args.valid_weight,
        live_in_weight=args.live_in_weight,
        live_out_weight=args.live_out_weight,
        pc_writes_weight=args.pc_writes_weight,
        in_slot_weight=args.in_slot_weight,
        out_slot_weight=args.out_slot_weight,
        value_predict_weight=args.value_predict_weight,
        anchor_seed=args.anchor_seed,
        n_anchor_states=args.n_anchor_states,
        device=device,
        on_log=save,
        compile_step=not args.no_compile,
    )
    print(f'\nDone: {losses[-1]["step"]} steps, '
          f'final loss {losses[-1]["total"]:.4f}')

    save(encoder, losses)
    print(f'Saved to {save_dir}')


if __name__ == '__main__':
    main()
