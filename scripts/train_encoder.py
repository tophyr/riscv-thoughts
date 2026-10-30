#!/usr/bin/env python3
"""Train a T1 encoder via magnitude-validity + register-identity aux +
value-prediction on RVT batches.

Step 1 of the piecemeal-assembly plan. CLI shell over
compressor.train.train_encoder; saves encoder.pt + hparams.json
+ losses.json into a run directory.

Usage:
    python -m datagen --rule single --batch-size 384 --n-states 8 \\
        --config configs/instr_alu_writers.json --count 50000 \\
        | train_encoder.py --n-steps 50000
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

    # The equivariant core's settings. T1 and T2 run the SAME core, so these
    # have to be reachable from both entry points; defaults reproduce the
    # pre-Phase-15 T1 exactly. See scripts/train_t2_encoder.py for the
    # counterparts.
    p.add_argument('--d-event', type=int, default=16,
                   help='Per-slot event-channel width (read/write trace).')
    p.add_argument('--op-cell-depth', type=int, default=1,
                   help='Number of op-cell blocks (1 = shallow baseline; '
                        '>1 stacks residual FFN blocks for deeper composition).')
    p.add_argument('--vp-head-depth', type=int, default=3,
                   help='vp value-prediction head hidden layers (residual '
                        'blocks with LayerNorm). 3 is the smallest head with '
                        'which T1 reaches exact vp at held-out anchors (500k '
                        'steps); a shallower head puts more of the work on the '
                        'encoder but stops short.')
    p.add_argument('--n-pool-heads', type=int, default=1,
                   help='essence pool seed queries (1 = single-query sum; '
                        '>1 = multi-head, keeps multiple outputs separable). '
                        'Must divide d_out.')
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
                   help='value prediction read from the essence: each '
                        "output's value from the essence, the output's "
                        "pointer and its inputs' values. The anchor states "
                        'ride in each batch.')
    p.add_argument('--distance-weight', type=float, default=0.0,
                   help='MSE between every two rows\' essence distance and '
                        'their behavioral distance at shared witness states '
                        '(log-log of the output difference over the largest '
                        'difference their known bits allow); T1 only.')
    p.add_argument('--vp-smooth-weight', type=float, default=0.2,
                   help='weight on the vp smooth-magnitude (ordering) term. The '
                        'per-bit target carries resolution; this keeps '
                        'numerically-close values ordered.')
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
        d_event=args.d_event, op_cell_depth=args.op_cell_depth,
        vp_head_depth=args.vp_head_depth, n_pool_heads=args.n_pool_heads,
        lr=args.lr, n_steps=args.n_steps, log_every=args.log_every,
        valid_weight=args.valid_weight,
        live_in_weight=args.live_in_weight,
        live_out_weight=args.live_out_weight,
        pc_writes_weight=args.pc_writes_weight,
        in_slot_weight=args.in_slot_weight,
        out_slot_weight=args.out_slot_weight,
        value_predict_weight=args.value_predict_weight,
        distance_weight=args.distance_weight,
        vp_smooth_weight=args.vp_smooth_weight,
        device=device,
        on_log=save, save_every=args.save_every,
        compile_step=not args.no_compile,
    )
    print(f'\nDone: {losses[-1]["step"]} steps, '
          f'final loss {losses[-1]["total"]:.4f}')

    save(encoder, losses)
    print(f'Saved to {save_dir}')


if __name__ == '__main__':
    main()
