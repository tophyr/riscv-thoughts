#!/usr/bin/env python3
"""Evaluate a trained compressor: Pearson/Spearman correlation and
qualitative inspection of specific instruction patterns.

Usage:
    gen_seq_batches.py --n-batches 100 --batch-size 64 | \
        eval_compressor.py --model runs/w1/model.pt --window-size 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy import stats

from datagen import BatchReader
from compressor.model import Compressor
from compressor.train import extract_windows, exec_distance
from tokenizer import VOCAB_SIZE, VOCAB


def evaluate(model, batch_iter, window_size, device, max_batches=100):
    """Compute correlation between model distances and execution distances."""
    model.eval()
    all_model_dists = []
    all_exec_dists = []

    with torch.no_grad():
        for i, batch in enumerate(batch_iter):
            if i >= max_batches:
                break
            token_ids, padding_mask, deltas = extract_windows(
                batch, window_size)
            if token_ids is None:
                continue

            tok = torch.from_numpy(token_ids).to(device)
            pad = torch.from_numpy(padding_mask).to(device)
            delta_t = torch.from_numpy(deltas)

            vecs = model(tok, pad)
            md = torch.cdist(vecs.unsqueeze(0), vecs.unsqueeze(0),
                             p=2).squeeze(0)
            ed = exec_distance(delta_t, device)

            N = vecs.shape[0]
            tri = torch.triu_indices(N, N, offset=1, device=device)
            all_model_dists.append(md[tri[0], tri[1]].cpu().numpy())
            all_exec_dists.append(ed[tri[0], tri[1]].cpu().numpy())

    md_flat = np.concatenate(all_model_dists)
    ed_flat = np.concatenate(all_exec_dists)

    pearson_r, pearson_p = stats.pearsonr(md_flat, ed_flat)
    spearman_r, spearman_p = stats.spearmanr(md_flat, ed_flat)

    print(f'Pairs:    {len(md_flat):,}')
    print(f'Pearson:  {pearson_r:.4f} (p={pearson_p:.2e})')
    print(f'Spearman: {spearman_r:.4f} (p={spearman_p:.2e})')

    return pearson_r, spearman_r


def main():
    p = argparse.ArgumentParser(description='Evaluate compressor.')
    p.add_argument('--model', required=True, help='Path to model.pt')
    p.add_argument('--window-size', type=int, default=1)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=128)
    p.add_argument('--device', default='auto')
    p.add_argument('--max-batches', type=int, default=100)
    args = p.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    model = Compressor(VOCAB_SIZE, args.d_model, args.n_heads,
                       args.n_layers, args.d_out)
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model = model.to(device)

    reader = BatchReader(sys.stdin.buffer)
    evaluate(model, reader, args.window_size, device,
             max_batches=args.max_batches)


if __name__ == '__main__':
    main()
