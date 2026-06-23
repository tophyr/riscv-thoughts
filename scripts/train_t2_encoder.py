#!/usr/bin/env python3
"""Train a T2 encoder on top of a frozen T1 encoder over
multi-instruction RVT chunks.

For each chunk, T1 encodes each instruction individually; T2 attention-
pools the resulting sequence into a single chunk vector. Supervision is
value-prediction and aux register-identity heads.

Usage:
    gen_batches.py --rule 'branch+cap=8' --twins 3 \\
        --batch-size 256 -n 5000 -v > corpus.rvt
    cat corpus.rvt |
        train_t2_encoder.py --t1-model runs/t1_good/encoder.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compressor.train import train_t2_encoder
from datagen import RVT_FORMAT, Batch
from scripts._common import (
    resolve_device, add_common_train_args, open_run_dir, load_frozen_encoder)




def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t1-model', required=True,
                   help='Path to T1 encoder.pt. Companion hparams.json '
                        'in the same dir is read automatically.')

    p.add_argument('--d-model', type=int, default=512)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--d-out', type=int, default=512)
    p.add_argument('--max-chunk-len', type=int, default=32,
                   help='Position-embedding table size for the T2 sequence '
                        '(max instructions per chunk).')

    add_common_train_args(p, lr=1e-4)
    p.add_argument('--warmup-steps', type=int, default=10000,
                   help='Linear LR warmup from ~0 to --lr over this many '
                        'steps before the cosine schedule kicks in. Useful '
                        'for deeper encoders that stall at init under a high '
                        'starting LR. 0 = no warmup.')
    p.add_argument('--valid-weight', type=float, default=0.1,
                   help='Magnitude-validity loss weight. 0 for '
                        'corpora without invalid windows.')
    p.add_argument('--live-in-weight', type=float, default=0.1,
                   help='BCE on live_in_head (which regs are read).')
    p.add_argument('--live-out-weight', type=float, default=0.1,
                   help='BCE on live_out_head (which regs are written).')
    p.add_argument('--pc-writes-weight', type=float, default=0.1,
                   help='BCE on pc_writes_head (does chunk write PC).')
    p.add_argument('--in-slot-weight', type=float, default=0.1,
                   help='Mean CE over per-input-slot register heads.')
    p.add_argument('--out-slot-weight', type=float, default=0.1,
                   help='Mean CE over per-output-slot register heads.')
    p.add_argument('--value-predict-weight', type=float, default=1.0,
                   help='Per-anchor per-out-slot value prediction MSE '
                        'on shape (chunk I/O function supervision). '
                        'Reconstructs '
                        'anchor_states from --anchor-seed and '
                        '--n-anchor-states — must match gen_batches.')
    p.add_argument('--value-predict-every', type=int, default=1,
                   help='Run value-prediction loss every Nth step '
                        '(default 1 = every step). Amortizes the '
                        'Python-side per-row precompute_chunk loop '
                        'that dominates per-step cost otherwise.')
    p.add_argument('--pair-weight', type=float, default=1.0,
                   help='Oracle behavioral pairwise loss (rename-sensitive): '
                        'match chunk-vector cosine distance to a behavioral '
                        'distance from their out_regs. 0 = off.')
    p.add_argument('--pair-scale', type=float, default=0.5,
                   help='Maps behavioral distance (loglog ~[0,3]) to target '
                        'cosine distance ([0,2]).')
    p.add_argument('--anchor-seed', type=int, default=0,
                   help='Anchor-states seed (must match gen_batches).')
    p.add_argument('--n-anchor-states', type=int, default=8,
                   help='Anchor states count (must match gen_batches).')

    p.add_argument('--resume', type=str, default=None,
                   help='Path to a t2.pt checkpoint to resume from. '
                        'Loads weights into a freshly-built T2 (model '
                        'hparams must match). Optimizer state is reset.')
    p.add_argument('--no-compile', action='store_true',
                   help='Disable torch.compile (CUDA-graph) of the train '
                        'step. Compile is on by default (saturates the GPU); '
                        'disable for a bit-faithful eager run or fast-start '
                        'short runs.')
    args = p.parse_args()

    device = resolve_device('auto')
    save_dir, save = open_run_dir(args, 't2', suffix='_t2')

    t1 = load_frozen_encoder(args.t1_model, device)

    reader = RVT_FORMAT.reader(sys.stdin.buffer, Batch)

    t2, losses = train_t2_encoder(
        reader, t1,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_out=args.d_out,
        max_chunk_len=args.max_chunk_len,
        lr=args.lr, n_steps=args.n_steps, log_every=args.log_every,
        warmup_steps=args.warmup_steps,
        valid_weight=args.valid_weight,
        live_in_weight=args.live_in_weight,
        live_out_weight=args.live_out_weight,
        pc_writes_weight=args.pc_writes_weight,
        in_slot_weight=args.in_slot_weight,
        out_slot_weight=args.out_slot_weight,
        value_predict_weight=args.value_predict_weight,
        value_predict_every=args.value_predict_every,
        pair_weight=args.pair_weight,
        pair_scale=args.pair_scale,
        anchor_seed=args.anchor_seed,
        n_anchor_states=args.n_anchor_states,
        t2_checkpoint=args.resume,
        device=device,
        on_log=save,
        compile_step=not args.no_compile,
    )
    print(f'\nDone: {losses[-1]["step"]} steps, '
          f'final loss {losses[-1]["total"]:.4f}')

    save(t2, losses)
    print(f'Saved to {save_dir}')


if __name__ == '__main__':
    main()
