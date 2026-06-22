#!/usr/bin/env python3
"""Per-loss gradient-subspace diagnostic for a trained T2.

A debugging instrument, not a quality score: it asks whether the T2 loss
terms cooperate or compete for capacity in the shared pre-projection
`pooled` vector (B, d_model) — the representation every output head reads
through `proj`.

For each loss L, backprop g = dL/d(pooled) is (B, d) — one gradient row per
sample. With C = g^T g (d x d):
  effective_dims(L) = trace(C)^2 / ||C||_F^2     (participation ratio)
  overlap(L1, L2)   = <C1, C2>_F / (||C1||_F ||C2||_F)   (matrix cosine)

effective_dims is scale-invariant (independent of the loss weight), so it
measures the dimensionality of the *directions* a loss pushes, not its
magnitude. overlap measures how much two losses fight over shared
directions (the tool behind the in_slot-vs-vp tension findings).

`loss_grad_dims(...)` is importable but is NOT on the training path — it is
run on demand against a checkpoint, here.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compressor.train import loss_grad_dims, _GRAD_DIM_LOSSES
from datagen import RVT_FORMAT, Batch, make_anchor_states
from scripts._common import resolve_device, load_frozen_encoder, load_t2

_LOSS_ORDER = _GRAD_DIM_LOSSES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('t2_run_dir', type=Path)
    p.add_argument('--t1-model', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--n-batches', type=int, default=4)
    p.add_argument('--anchor-seed', type=int, default=0)
    p.add_argument('--n-anchor-states', type=int, default=8)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    device = resolve_device(args.device)

    t1 = load_frozen_encoder(args.t1_model, device)
    t2, hp = load_t2(args.t2_run_dir, t1, device)
    anchor = torch.from_numpy(
        make_anchor_states(args.n_anchor_states, args.anchor_seed)).to(device)

    # Average dims/overlaps over a few batches for a stable estimate.
    unc_acc, cen_acc, ov_acc, n = {}, {}, {}, 0
    with open(args.corpus, 'rb') as f:
        reader = RVT_FORMAT.reader(f, Batch)
        for i, batch in enumerate(reader):
            if i >= args.n_batches:
                break
            dims, ov = loss_grad_dims(t2, t1, batch, anchor, device)
            if dims is None:
                continue
            for k, v in dims.items():
                unc_acc[k] = unc_acc.get(k, 0.0) + v['unc']
                cen_acc[k] = cen_acc.get(k, 0.0) + v['cen']
            for k, v in ov.items():
                ov_acc[k] = ov_acc.get(k, 0.0) + v
            n += 1
    unc = {k: v / n for k, v in unc_acc.items()}
    cen = {k: v / n for k, v in cen_acc.items()}
    ov = {k: v / n for k, v in ov_acc.items()}

    d_model = hp['d_model']
    print(f'\n=== Per-loss effective dimensions (of {d_model} pooled dims) '
          f'— {n} batches ===')
    print(f'  {"loss":10s} {"uncentered":>11s}  {"centered":>10s}')
    for k in _LOSS_ORDER:
        print(f'  {k:10s} {unc[k]:11.1f}  {cen[k]:10.1f}   '
              f'(centered = {100*cen[k]/d_model:.1f}% of space)')
    print(f'\n=== Pairwise overlap on CENTERED gradient variation '
          f'(0=orthogonal, 1=same) ===')
    for k in sorted(ov):
        print(f'  {k:22s}: {ov[k]:.3f}')


if __name__ == '__main__':
    main()
