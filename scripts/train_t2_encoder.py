#!/usr/bin/env python3
"""Train a T2 encoder on top of a frozen T1 encoder over
multi-instruction RVT chunks.

For each chunk, T1 encodes each instruction individually; T2 attention-
pools the resulting sequence into a single chunk vector. Supervision is
value-prediction and aux register-identity heads.

Usage:
    python -m datagen --rule 'branch+cap=8' --twins 3 --batch-size 256 \\
        --count 5000 \\
        | train_t2_encoder.py --t1-model runs/t1_good/encoder.pt
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

    p.add_argument('--d-out', type=int, default=512,
                   help='T2 register-state width (d_value).')
    p.add_argument('--d-event', type=int, default=16,
                   help='Per-slot event-channel width (read/write trace).')
    p.add_argument('--op-cell-depth', type=int, default=1,
                   help='Number of op-cell blocks (1 = shallow baseline; '
                        '>1 stacks residual FFN blocks for deeper composition).')
    p.add_argument('--vp-head-depth', type=int, default=4,
                   help='vp value-prediction head hidden layers (4 = default; '
                        '0 = linear — weakens the head to force the essence to '
                        'expose value structure geometrically).')
    p.add_argument('--n-pool-heads', type=int, default=1,
                   help='essence pool seed queries (1 = single-query sum; '
                        '>1 = multi-head, keeps multiple outputs separable). '
                        'Must divide d_out.')
    p.add_argument('--pool-mode', choices=('pma', 'compete'), default='pma',
                   help='pma = each head independently pools slots (heads can '
                        'collapse); compete = Slot-Attention, slots divide among '
                        'heads so they cover the register file (needs >1 head).')

    p.add_argument('--pool-extra',
                   choices=('none', 'write_time', 'order', 'event'),
                   default='none',
                   help='what the essence pool reads per slot beyond the '
                        'value channels: write_time = output order; order = '
                        '+read_time (canonical input positions); event = the '
                        'whole slot state. All register-indexed, so the pool '
                        'stays exactly rename-invariant.')

    p.add_argument('--read-time-mode', choices=('stamp', 'rank'),
                   default='stamp',
                   help="how a slot's first-read is stamped. 'stamp' (legacy) "
                        'writes 2(k+1)+role, which equals the canonical input '
                        'position only while every read is fresh — a repeated '
                        'source register leaves a hole and every later operand '
                        "overshoots. 'rank' stamps the running count of "
                        'DISTINCT registers read, so read_time IS the canonical '
                        'position.')

    p.add_argument('--op-cell-rank', action='store_true',
                   help="feed the op-cell each operand's canonical read "
                        'position. The pool cannot use read-order (it never '
                        'sees content, so it must apply it unconditionally, '
                        'which breaks commutative collapse); the op-cell can, '
                        'since it sees the opcode and both operands together. '
                        'Requires --read-time-mode rank; widens op_cell, so '
                        'checkpoints trained with it are not interchangeable.')

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
                        '--n-anchor-states — must match datagen.')
    p.add_argument('--vp-smooth-weight', type=float, default=0.2,
                   help='weight on the vp smooth-magnitude (ordering) term; the '
                        'per-bit target carries resolution.')
    p.add_argument('--contrastive-weight', type=float, default=0.0,
                   help='Weight on the behavioral-distance contrastive loss on '
                        'the essence (0 = off). Pulls behaviorally-equivalent '
                        'chunks together / pushes distinct ones apart using the '
                        'canonical out_regs oracle as the pairwise target. When '
                        '> 0, out_regs ships even with --value-predict-weight 0 '
                        '(demote vp to a diagnostic).')
    p.add_argument('--contrastive-loss', choices=('regression', 'infonce'),
                   default='regression',
                   help="Contrastive head: 'regression' (cosine→1−d_beh MSE, "
                        "primary) or 'infonce' (soft-weighted NT-Xent, fallback).")
    p.add_argument('--contrastive-s', type=float, default=0.0,
                   help='φ saturation scale on the d_beh target (THE swept '
                        'knob). 0 = exact-match indicator (binary, magnitude-'
                        'agnostic control); rising s adds magnitude sensitivity '
                        '(s≈7 ⇒ saturation knee at |Δ|~1e5–1e6).')
    p.add_argument('--anchor-seed', type=int, default=0,
                   help='Anchor-states seed (must match datagen).')
    p.add_argument('--n-anchor-states', type=int, default=8,
                   help='Anchor states count (must match datagen).')

    p.add_argument('--route', choices=('binding', 'tokens'), default='binding',
                   help="How T2 gets per-instruction wiring. 'binding' "
                        "(default): from the frozen T1's PREDICTED "
                        'binding — the real tier-recursion test. '
                        "'tokens': decoded-token wiring (diagnostic A/B, not "
                        'a valid recursion result).')
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
        d_out=args.d_out, d_event=args.d_event, op_cell_depth=args.op_cell_depth,
        vp_head_depth=args.vp_head_depth, n_pool_heads=args.n_pool_heads,
        pool_mode=args.pool_mode, pool_extra=args.pool_extra,
        read_time_mode=args.read_time_mode, op_cell_rank=args.op_cell_rank,
        lr=args.lr, n_steps=args.n_steps, log_every=args.log_every,
        warmup_steps=args.warmup_steps,
        valid_weight=args.valid_weight,
        live_in_weight=args.live_in_weight,
        live_out_weight=args.live_out_weight,
        pc_writes_weight=args.pc_writes_weight,
        in_slot_weight=args.in_slot_weight,
        out_slot_weight=args.out_slot_weight,
        value_predict_weight=args.value_predict_weight,
        vp_smooth_weight=args.vp_smooth_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_loss=args.contrastive_loss,
        contrastive_s=args.contrastive_s,
        anchor_seed=args.anchor_seed,
        n_anchor_states=args.n_anchor_states,
        t2_checkpoint=args.resume,
        t2_route=args.route,
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
