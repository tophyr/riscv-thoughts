#!/usr/bin/env python3
"""Probe whether a trained encoder's T1 carries validity signal.

For a trained encoder checkpoint, generate windows of five types —
one valid class (complete single instruction) and four invalid
classes (partial / spanning / multi / bogus) from the invalidity
module — encode each through the frozen encoder, and ask:

  1. Do magnitudes separate valid from invalid?
  2. Does a linear probe on raw T1 recover is_complete?
  3. Would a simple magnitude threshold do the same job?

Intended as a go/no-go check before gate training: if T1 does not
carry the validity signal, no gate architecture reading T1
linearly will discriminate emit points.

Usage:
    probe_validity.py --encoder runs/<stamp>/encoder.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from compressor.model import T1Compressor
from compressor.train import load_checkpoint
from datagen.instrgen import DEFAULT_DISTRIBUTION, _build_opcode_table, random_instruction
from datagen.invalidity import (
    gen_partial, gen_spanning, gen_multi, gen_bogus,
)
from tokenizer import VOCAB_SIZE, PAD, encode_instruction
from emulator import batch_is_complete_instruction
from scripts._common import resolve_device


def build_windows(rng, opcode_table, n_per_class, max_window):
    """Return (tokens_list, class_ids) across five balanced classes."""
    gens = [
        ('valid',    lambda: encode_instruction(
            random_instruction(rng, opcode_table=opcode_table))),
        ('partial',  lambda: gen_partial(rng, opcode_table, max_window)),
        ('spanning', lambda: gen_spanning(rng, opcode_table, max_window)),
        ('multi',    lambda: gen_multi(rng, opcode_table, max_window)),
        ('bogus',    lambda: gen_bogus(rng, max_window)),
    ]
    tokens_list, class_ids = [], []
    for cid, (_, fn) in enumerate(gens):
        for _ in range(n_per_class):
            toks = fn()
            if len(toks) > max_window:
                toks = toks[:max_window]
            tokens_list.append(toks)
            class_ids.append(cid)
    names = [g[0] for g in gens]
    return tokens_list, class_ids, names


def pad_batch(tokens_list, max_window):
    B = len(tokens_list)
    tok = np.full((B, max_window), PAD, dtype=np.int64)
    pad = np.ones((B, max_window), dtype=bool)
    lens = np.zeros(B, dtype=np.int64)
    for i, seq in enumerate(tokens_list):
        L = len(seq)
        tok[i, :L] = seq
        pad[i, :L] = False
        lens[i] = L
    return tok, pad, lens


def encode_all(model, tokens_list, device, max_window, batch_size=512):
    N = len(tokens_list)
    chunks = []
    for i in range(0, N, batch_size):
        tok, pd, _ = pad_batch(tokens_list[i:i + batch_size], max_window)
        with torch.no_grad():
            v = model.encode(
                torch.from_numpy(tok).to(device),
                torch.from_numpy(pd).to(device))
        chunks.append(v.cpu())
    return torch.cat(chunks, dim=0)


def labels_complete(tokens_list, lens, device, max_window,
                    batch_size=512):
    N = len(tokens_list)
    chunks = []
    for i in range(0, N, batch_size):
        tok, _, _ = pad_batch(tokens_list[i:i + batch_size], max_window)
        len_slice = np.array(lens[i:i + batch_size], dtype=np.int64)
        with torch.no_grad():
            c = batch_is_complete_instruction(
                torch.from_numpy(tok).to(device),
                torch.from_numpy(len_slice).to(device),
                device)
        chunks.append(c.cpu())
    return torch.cat(chunks, dim=0)


def train_linear_probe(X, y, device, n_epochs=300, lr=0.05):
    X = X.to(device).float()
    y = y.to(device).float()
    N = X.shape[0]
    perm = torch.randperm(N, device=device)
    n_tr = int(0.8 * N)
    idx_tr, idx_va = perm[:n_tr], perm[n_tr:]
    probe = torch.nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_epochs):
        probe.train()
        logits = probe(X[idx_tr]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y[idx_tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        pred = (probe(X[idx_va]).squeeze(-1) > 0).float()
        acc = (pred == y[idx_va]).float().mean().item()
    return acc, idx_va, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--n-per-class', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--max-window', type=int, default=32)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--d-out', type=int, default=64)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--n-layers', type=int, default=2)
    args = ap.parse_args()

    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    model = T1Compressor(
        VOCAB_SIZE, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_out=args.d_out,
        max_window=args.max_window).to(device)
    model.load_state_dict(load_checkpoint(args.encoder, device), strict=False)
    model.eval()

    opcode_table = _build_opcode_table(DEFAULT_DISTRIBUTION)
    tokens_list, class_ids, names = build_windows(
        rng, opcode_table, args.n_per_class, args.max_window)
    N = len(tokens_list)
    lens = [len(t) for t in tokens_list]
    class_ids_t = torch.tensor(class_ids)

    print(f'device={device} n_per_class={args.n_per_class} '
          f'max_window={args.max_window}')
    print(f'Built {N} windows across {len(names)} classes: {names}')

    # Ground-truth labels.
    y = labels_complete(tokens_list, lens, device, args.max_window).cpu()

    print('\n--- Per-class is_complete rate ---')
    for cid, name in enumerate(names):
        mask = class_ids_t == cid
        rate = y[mask].float().mean().item()
        print(f'  {name:9s}  n={int(mask.sum())}  '
              f'is_complete rate={rate:.3f}')
    pos_rate = y.float().mean().item()
    print(f'\nOverall is_complete rate: {pos_rate:.3f}')
    print(f'Majority baseline accuracy: {max(pos_rate, 1-pos_rate):.3f}')

    # Encode.
    print('\nEncoding windows...')
    X = encode_all(model, tokens_list, device, args.max_window)
    X_norms = X.norm(dim=-1)

    print('\n--- Per-class ||T1|| statistics ---')
    for cid, name in enumerate(names):
        mask = class_ids_t == cid
        if not mask.any():
            continue
        mags = X_norms[mask]
        print(f'  {name:9s}  mean={mags.mean():.3f} '
              f'std={mags.std():.3f} '
              f'min={mags.min():.3f} max={mags.max():.3f}')

    # Magnitude-only probe: threshold ||T1|| at 0.5.
    print('\n--- Magnitude-only probe (threshold ||T1|| > 0.5) ---')
    mag_pred = (X_norms > 0.5).float()
    mag_acc = (mag_pred == y.float()).float().mean().item()
    print(f'Magnitude-threshold accuracy: {mag_acc:.3f}')
    for cid, name in enumerate(names):
        mask = class_ids_t == cid
        if not mask.any():
            continue
        acc = (mag_pred[mask] == y[mask].float()).float().mean().item()
        print(f'  {name:9s}  acc={acc:.3f}  '
              f'pred_positive={mag_pred[mask].mean().item():.3f}')

    # Linear probe on raw T1 vector.
    print('\n--- Linear probe on T1 (sanity check) ---')
    acc, idx_va, pred = train_linear_probe(X, y, device)
    print(f'Linear probe val acc: {acc:.3f}')
    cls_va = class_ids_t[idx_va.cpu()]
    y_va = y[idx_va.cpu()]
    pred = pred.cpu()
    for cid, name in enumerate(names):
        mask = cls_va == cid
        if not mask.any():
            continue
        a = (pred[mask] == y_va[mask]).float().mean().item()
        print(f'  {name:9s}  acc={a:.3f}  '
              f'n={int(mask.sum())}')


if __name__ == '__main__':
    main()
