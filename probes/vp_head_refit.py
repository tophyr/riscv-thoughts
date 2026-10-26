#!/usr/bin/env python3
"""Is a T2's residual vp error in its essence or in its head?

Freezes T1 and T2 and trains only the vp head further, on the live training
stream read from stdin (anchors drawn per row, as in training), then writes a
copy of the T2 run directory carrying the refit head, to be scored with
`vp_budget.py` or `vp_miss_kind.py`. Error the refit removes was the head's;
error it leaves is in the essence, or beyond the head's capacity.

  --mode continue  keeps the trained head and fine-tunes it
  --mode fresh     re-initialises the head (same depth unless --depth)

Usage: datagen ... | vp_head_refit.py <t2_dir> <t1_path> <out_dir>
                     [--mode continue|fresh] [--depth N] [--steps N] [--lr X]
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.model import VPHead, _VP_VALUE_BITS
from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              t2_value_predict_loss)
from datagen import RVT_FORMAT, Batch, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS
from scripts._common import resolve_device, load_frozen_encoder, load_t2

p = argparse.ArgumentParser()
p.add_argument('t2_dir'); p.add_argument('t1_path'); p.add_argument('out_dir')
p.add_argument('--mode', choices=('continue', 'fresh'), default='continue')
p.add_argument('--depth', type=int, default=None)
p.add_argument('--steps', type=int, default=200000)
p.add_argument('--lr', type=float, default=1e-4)
args = p.parse_args()

dev = resolve_device('auto')
t1 = load_frozen_encoder(args.t1_path, dev)
t2, hp = load_t2(args.t2_dir, t1, dev)
for q in t2.parameters():
    q.requires_grad_(False)
depth = args.depth if args.depth is not None else hp['vp_head_depth']
if args.mode == 'fresh' or depth != hp['vp_head_depth']:
    t2.vp_head = VPHead(
        t2.d_out + MAX_INPUT_SLOTS * (_VP_VALUE_BITS + 1),
        MAX_OUTPUT_SLOTS * (_VP_VALUE_BITS + 1), depth).to(dev)
for q in t2.vp_head.parameters():
    q.requires_grad_(True)
t2.vp_head.train()
opt = torch.optim.Adam(t2.vp_head.parameters(), lr=args.lr)
warm = 1000 if args.mode == 'continue' else 2000


def lr_at(s):
    if s < warm:
        return args.lr * (s + 1) / warm
    return args.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm)))


step = 0
ema = None
for batch in RVT_FORMAT.reader(sys.stdin.buffer, Batch):
    if step >= args.steps:
        break
    split = _split_to_per_instruction(batch)
    if split[0].shape[0] == 0:
        continue
    nch, mni = batch.instr_lens.shape
    with torch.no_grad():
        ct1, c0, c1, co, cact = _t2_assemble(t1, split, nch, mni, dev,
                                             route='binding')
        ess = t2.encode(ct1, c0, c1, co, cact)
    t = lambda a: torch.from_numpy(a).to(dev)
    loss = t2_value_predict_loss(
        t2, ess, t(batch.anchors), t(batch.in_slot_regs.astype(np.int64)),
        t(batch.out_slot_regs.astype(np.int64)), t(batch.out_regs),
        t(batch.out_regs_valid))
    for g in opt.param_groups:
        g['lr'] = lr_at(step)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(t2.vp_head.parameters(), 1.0)
    opt.step()
    ema = float(loss) if ema is None else 0.99 * ema + 0.01 * float(loss)
    step += 1
    if step % 5000 == 0:
        print(f'step {step} vp {ema:.5f} lr {lr_at(step):.2e}', flush=True)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
hp = dict(hp, vp_head_depth=depth)
(out / 'hparams.json').write_text(json.dumps(hp, indent=2))
torch.save(t2.state_dict(), out / 't2.pt')
print(f'saved {out} after {step} steps', flush=True)
