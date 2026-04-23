#!/usr/bin/env python3
"""Evaluate a trained T1 encoder against the equivalence manifest.

For each class: sample N bindings, materialize canonical + contrast
instances per binding, encode, and measure
  intra: mean pairwise distance among canonical members
  inter: mean distance from canonical to contrast
  ratio: intra / inter  (lower is better; <0.5 = collapsing)

Usage:
    eval_equivalences.py --model runs/<stamp>/encoder.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from datagen.equivalences import MANIFEST, sample_binding, materialize
from compressor.model import T1Compressor
from compressor.train import load_checkpoint
from tokenizer import VOCAB_SIZE, encode_instruction, PAD
from scripts._common import resolve_device


def encode_instructions(model, instrs, device):
    """Tokenize + pad + encode a list of Instructions."""
    encoded = [encode_instruction(i) for i in instrs]
    max_len = max(len(e) for e in encoded)
    tok = np.full((len(encoded), max_len), PAD, dtype=np.int64)
    pad = np.ones((len(encoded), max_len), dtype=np.bool_)
    for i, e in enumerate(encoded):
        tok[i, :len(e)] = e
        pad[i, :len(e)] = False
    tok_t = torch.from_numpy(tok).to(device)
    pad_t = torch.from_numpy(pad).to(device)
    with torch.no_grad():
        return model.encode(tok_t, pad_t)


def evaluate_class(klass, model, device, n_samples, rng):
    """Return (intra, inter) means across n_samples bindings.

    intra or inter may be None if the class doesn't support the metric
    (<2 canonical members, or no contrast).
    """
    intras = []
    inters = []
    for _ in range(n_samples):
        binding = sample_binding(klass, rng)
        canon_instrs = [materialize(t, binding) for t in klass.canonical]
        contrast_instrs = [materialize(t, binding) for t in klass.contrast]
        all_instrs = canon_instrs + contrast_instrs
        vecs = encode_instructions(model, all_instrs, device)
        nc = len(canon_instrs)
        canon_vecs = vecs[:nc]
        contrast_vecs = vecs[nc:]

        if nc >= 2:
            dists = torch.cdist(canon_vecs.unsqueeze(0),
                                canon_vecs.unsqueeze(0)).squeeze(0)
            idx = torch.triu_indices(nc, nc, offset=1, device=device)
            intras.append(dists[idx[0], idx[1]].mean().item())

        if contrast_vecs.shape[0] > 0:
            dists = torch.cdist(canon_vecs.unsqueeze(0),
                                contrast_vecs.unsqueeze(0)).squeeze(0)
            inters.append(dists.mean().item())

    intra = float(np.mean(intras)) if intras else None
    inter = float(np.mean(inters)) if inters else None
    return intra, inter


def verdict(intra, inter):
    if intra is None and inter is None:
        return '-'
    if intra is None:
        return f'(single-canonical; inter only)'
    if inter is None:
        return f'(no contrast; intra only)'
    if inter < 1e-6:
        return 'FAIL (inter≈0)'
    ratio = intra / inter
    if ratio < 0.3:
        return 'PASS'
    if ratio < 0.7:
        return 'WEAK'
    return 'FAIL'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Path to encoder.pt')
    p.add_argument('--n-samples', type=int, default=50)
    p.add_argument('--device', default='auto')
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    device = resolve_device(args.device)

    model = T1Compressor(VOCAB_SIZE, args.d_model, args.n_heads,
                         args.n_layers, args.d_out).to(device)
    # strict=False so pre-dest-heads checkpoints still load.
    missing, unexpected = model.load_state_dict(
        load_checkpoint(args.model, device), strict=False)
    if missing:
        print(f'# note: missing keys (random-init): {missing}',
              file=sys.stderr)
    if unexpected:
        print(f'# note: unexpected keys ignored: {unexpected}',
              file=sys.stderr)
    model.eval()

    rng = np.random.default_rng(args.seed)

    print(f'# model: {args.model}')
    print(f'# n_samples per class: {args.n_samples}')
    print()
    print(f'{"class":<23} {"intra":>8} {"inter":>8} {"ratio":>7}  verdict')
    print('-' * 72)

    pass_count = weak_count = fail_count = 0
    for klass in MANIFEST:
        intra, inter = evaluate_class(klass, model, device,
                                      args.n_samples, rng)
        v = verdict(intra, inter)
        intra_s = f'{intra:8.4f}' if intra is not None else '       -'
        inter_s = f'{inter:8.4f}' if inter is not None else '       -'
        ratio_s = f'{intra/inter:7.3f}' if (
            intra is not None and inter is not None and inter > 1e-6
        ) else '      -'
        print(f'{klass.name:<23} {intra_s} {inter_s} {ratio_s}  {v}')
        if v == 'PASS':
            pass_count += 1
        elif v == 'WEAK':
            weak_count += 1
        elif v == 'FAIL':
            fail_count += 1

    print()
    print(f'Summary: {pass_count} PASS, {weak_count} WEAK, '
          f'{fail_count} FAIL, '
          f'{len(MANIFEST) - pass_count - weak_count - fail_count} other')


if __name__ == '__main__':
    main()
