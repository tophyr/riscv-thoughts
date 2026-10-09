#!/usr/bin/env python3
"""Is the essence pool's per-head value width too narrow to carry an immediate?

The pool builds the essence as `n_pool_heads` attention-weighted sums over the
32 register slots, and every head draws from ONE shared value projection
`pool_v` whose output width is `d_value // n_pool_heads` — 16 dimensions at the
current d_value=256, n_pool_heads=16. Everything a slot contributes to the
essence passes through that projection (plus the `n_pool_heads` attention
logits). Measured on one-instruction chunks, a written slot's 256-dimensional
value carries its 12-bit immediate at 99.4% exact recovery, and the pool's view
of the same slot carries it at 77.5% — the projection is where it is lost.

That leaves two explanations, which this probe separates by REFITTING the
bottleneck from scratch for the value-prediction task alone:

- WIDTH. 16 dimensions cannot carry the immediate alongside the operation
  identity, and accuracy rises with the bottleneck width.
- ALLOCATION. 16 dimensions suffice, and the trained projection spends them on
  something else. Then a freshly fitted 16-wide bottleneck reaches ~100% and
  widening it changes nothing.

Fits `slot value -> W dims -> predict the output register value` for a sweep of
W, scored as exact 32-bit match on held-out rows, with W = d_value as the
no-bottleneck control. The fitted bottleneck faces only this one task, so it is
an upper bound on what a width-W projection can deliver.

Usage: pool_bottleneck.py <t2_dir> <t1_path> [n_batches]
"""
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              _value_bits, _value_smooth)
from datagen import (RVT_FORMAT, Batch, AUX_CE_IGNORE, MAX_INPUT_SLOTS,
                     N_REGS, make_anchor_states)
from emulator import I_TYPE
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 200
N_ANCHORS, ANCHOR_SEED, NBITS = 8, 0, 32
HID, STEPS, MB = 1024, 12000, 2048
WIDTHS = (8, 16, 32, 64, 256)

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)
D_VALUE = t2.core.d_value

CORPUS = '/tmp/imm_ceiling_len1.rvt'
subprocess.run(
    f'python -m datagen --rule "cap=1" --twins 0 --batch-size 128 '
    f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
    f'--config configs/instr_alu_writers.json --count {NB} > {CORPUS}',
    shell=True, check=True, stderr=subprocess.DEVNULL)

anchors = torch.from_numpy(
    np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)

slots, tgts, ifeats, is_i = [], [], [], []
with open(CORPUS, 'rb') as f:
    for batch in RVT_FORMAT.reader(f, Batch):
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        (_, _, _, _, n_per_chunk, _, _, wo) = split
        nch, mni = batch.instr_lens.shape
        with torch.no_grad():
            ct1, c0, c1, co, cact = _t2_assemble(t1, split, nch, mni, dev,
                                                 route='binding')
            T2_state, _ = t2.core(ct1, c0, c1, co, cact)
            w2 = T2_state.gather(
                1, co[:, :1].unsqueeze(-1).expand(-1, -1, T2_state.shape[-1])
            ).squeeze(1)[:, :D_VALUE]

        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        keep, ityp = [], []
        pos = 0
        for b in range(nch):
            n = int(n_per_chunk[b])
            if (n == 1 and batch.out_regs_valid[b]
                    and int(out_slot[b, 0]) != AUX_CE_IGNORE
                    and int(wo[pos]) != 0):
                try:
                    ins, _ = decode_instruction(batch.tokens[b].tolist(), 0)
                except Exception:
                    ins = None
                if ins is not None:
                    keep.append(b)
                    ityp.append(ins.opcode in I_TYPE)
            pos += n
        if not keep:
            continue
        kb = torch.tensor(keep, device=dev)
        fill = (in_slot[kb] != AUX_CE_IGNORE).to(torch.int64)
        iv = ANCH_IN.unsqueeze(0) * fill.unsqueeze(1)          # (b, A, max_in)
        ifeats.append(torch.cat(
            [_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
            dim=-1).reshape(len(keep), N_ANCHORS, -1))
        tgts.append(out_regs[kb].gather(
            -1, out_slot[kb, :1].clamp(min=0).unsqueeze(1)
            .expand(-1, N_ANCHORS, -1)).squeeze(-1))           # (b, A)
        slots.append(w2[kb])
        is_i.append(torch.tensor(ityp, device=dev))

SLOT, TGT = torch.cat(slots), torch.cat(tgts)
IFEAT, ISI = torch.cat(ifeats), torch.cat(is_i)
N = SLOT.shape[0]
cut = int(0.8 * N)
TR = torch.arange(cut, device=dev)
TE = torch.arange(cut, N, device=dev)


def fit(width):
    """slot value -> `width` dims -> output-value bits. Returns exact 32-bit
    match on held-out rows, split R-type / I-type."""
    torch.manual_seed(0)
    neck = nn.Linear(D_VALUE, width).to(dev)
    head = nn.Sequential(
        nn.Linear(width + IFEAT.shape[-1], HID), nn.ReLU(),
        nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, NBITS)).to(dev)
    opt = torch.optim.Adam(list(neck.parameters()) + list(head.parameters()), 1e-3)
    rng = np.random.default_rng(0)
    for _ in range(STEPS):
        i = TR[torch.from_numpy(rng.integers(0, cut, MB)).to(dev)]
        a = int(rng.integers(0, N_ANCHORS))
        loss = F.binary_cross_entropy_with_logits(
            head(torch.cat([neck(SLOT[i]), IFEAT[i, a]], -1)),
            _value_bits(TGT[i, a]))
        opt.zero_grad(); loss.backward(); opt.step()
    exact = torch.zeros(TE.numel(), device=dev)
    with torch.no_grad():
        for a in range(N_ANCHORS):
            hit = ((head(torch.cat([neck(SLOT[TE]), IFEAT[TE, a]], -1)) > 0)
                   .float() == _value_bits(TGT[TE, a]))
            exact += hit.all(-1).float()
    exact /= N_ANCHORS
    sel = ISI[TE]
    return exact[~sel].mean(), exact[sel].mean()


print()
print(f'  one-instruction chunks: {N} rows, {int(TE.numel())} held out')
print(f'  the deployed pool gives each head {D_VALUE // t2.core.n_pool_heads} '
      f'dimensions ({D_VALUE} value dims / {t2.core.n_pool_heads} heads)')
print()
print(f'  {"bottleneck":>12s} {"R-type":>10s} {"I-type":>10s}   '
      f'(exact 32-bit match of the output value)')
for w in WIDTHS:
    r, i = fit(w)
    note = '  <- deployed width' if w == D_VALUE // t2.core.n_pool_heads else (
        '  <- no bottleneck' if w == D_VALUE else '')
    print(f'  {w:12d} {100 * r:9.3f}% {100 * i:9.3f}%{note}')
print()
