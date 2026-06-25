#!/usr/bin/env python3
"""Held-out scorecard for the equivariant T1 (and optional T2) encoders.

Drives compressor.eval. Reports the
load-bearing properties:

  - equivariance: relabel -> state permutes EXACTLY, essence invariant (~0)
  - tag-invariance: essence cosine across anonymous-tag resamples (~1)
  - binding accuracy: live_in/out, pc, in/out-slot order, dup rate
  - (T2 only) GVN collapse split by pair type: rename-twin (free) vs
    behavioral-non-rename (trained) vs distinct

T1 metrics need a SINGLE-instruction corpus (--rule single); T2 metrics need
a multi-instruction corpus (e.g. branch+cap=4). GVN collapse self-generates
its chunks.

Usage:
    eval.py --t1-model runs/<t1>/encoder.pt --t1-corpus t1_head.rvt
    eval.py --t1-model runs/<t1>/encoder.pt --t2 runs/<t2> \\
        --t1-corpus t1_head.rvt --t2-corpus t2_head.rvt --route binding
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compressor import eval as E
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from datagen import RVT_FORMAT, Batch


def _read_batches(path, n):
    with open(path, 'rb') as f:
        reader = RVT_FORMAT.reader(f, Batch)
        out = []
        for i, b in enumerate(reader):
            if i >= n:
                break
            out.append(b)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--t1-model', required=True)
    p.add_argument('--t2', default=None, help='T2 run dir (optional).')
    p.add_argument('--t1-corpus', default=None,
                   help='SINGLE-instruction corpus for T1 metrics.')
    p.add_argument('--t2-corpus', default=None,
                   help='Multi-instruction corpus for T2 binding metrics.')
    p.add_argument('--max-batches', type=int, default=20)
    p.add_argument('--route', choices=('binding', 'tokens'), default='binding')
    p.add_argument('--gvn-pairs', type=int, default=200)
    p.add_argument('--gvn-chunk-len', type=int, default=4)
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    device = resolve_device(args.device)
    t1 = load_frozen_encoder(args.t1_model, device)

    if args.t1_corpus:
        batches = _read_batches(args.t1_corpus, args.max_batches)
        print('== T1 (single-instruction corpus) ==')
        eq = E.equivariance_error(t1, batches[0], device=device)
        print(f'equivariance: state_max_err={eq["state_max_err"]:.2e} '
              f'essence_max_err={eq["essence_max_err"]:.2e}')
        ti = E.tag_invariance(t1, batches[0], device=device)
        print(f'tag-invariance: cos_mean={ti["tag_cos_mean"]:.4f} '
              f'cos_min={ti["tag_cos_min"]:.4f}')
        b1 = E.t1_binding_accuracy(t1, batches, device=device,
                                   max_batches=args.max_batches)
        print(f'binding: live_in={b1["live_in_acc"]:.3f} '
              f'live_out={b1["live_out_acc"]:.3f} pc={b1["pc_acc"]:.3f} '
              f'in_slot={b1["in_slot_acc"]:.3f} out_slot={b1["out_slot_acc"]:.3f} '
              f'dup={b1["dup_rate"]:.3f} (rows={b1["n_rows"]})')

    if args.t2:
        t2, _ = load_t2(args.t2, t1, device)
        print(f'== T2 (route={args.route}) ==')
        if args.t2_corpus:
            t2_batches = _read_batches(args.t2_corpus, args.max_batches)
            b2 = E.t2_binding_accuracy(t1, t2, t2_batches, device=device,
                                       route=args.route,
                                       max_batches=args.max_batches)
            print(f'binding: live_in={b2["live_in_acc"]:.3f} '
                  f'live_out={b2["live_out_acc"]:.3f} pc={b2["pc_acc"]:.3f} '
                  f'in_slot={b2["in_slot_acc"]:.3f} '
                  f'out_slot={b2["out_slot_acc"]:.3f} dup={b2["dup_rate"]:.3f} '
                  f'(rows={b2["n_rows"]})')
        g = E.gvn_collapse(t1, t2, device=device, route=args.route,
                           n=args.gvn_pairs, chunk_len=args.gvn_chunk_len)
        rr = g['rename_ratio']; br = g['behavioral_ratio']
        print(f'GVN collapse (n={g["n_pairs"]}, distinct={g["n_distinct"]}):')
        print(f'  rename     mean={g["rename_mean"]:.4f} '
              f'ratio={rr:.4f}' if rr is not None else
              f'  rename     mean={g["rename_mean"]:.4f}')
        print(f'  behavioral mean={g["behavioral_mean"]:.4f} '
              f'ratio={br:.4f}' if br is not None else
              f'  behavioral mean={g["behavioral_mean"]:.4f}')
        print(f'  distinct   mean={g["distinct_mean"]:.4f}')


if __name__ == '__main__':
    main()
