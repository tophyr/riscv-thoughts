#!/usr/bin/env python3
"""Held-out scorecard for the equivariant T1 (and optional T2) encoders.

Drives compressor.eval. Reports the
load-bearing properties:

  - equivariance: relabel -> state permutes EXACTLY, essence invariant (~0)
  - binding: the emitted wiring exact per instruction, by operand form (T1);
    the emitted input and output lists exact per chunk, by chunk shape (T2)
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
        b1 = E.t1_binding_accuracy(t1, batches, device=device,
                                   max_batches=args.max_batches)
        print(f'binding: emitted wiring exact {100 * b1["exact"]:.4f}% '
              f'of {b1["n_rows"]} instructions (target 100%)')
        misses = sorted((acc, k, n) for k, (acc, n) in b1['by_form'].items()
                        if acc < 1.0)
        for acc, k, n in misses:
            print(f'  {k:<28} {100 * acc:7.2f}%  of {n}')

    if args.t2:
        t2, _ = load_t2(args.t2, t1, device)
        print(f'== T2 (route={args.route}) ==')
        if args.t2_corpus:
            t2_batches = _read_batches(args.t2_corpus, args.max_batches)
            b2 = E.t2_exact_binding(t1, t2, t2_batches, device=device,
                                    route=args.route,
                                    max_batches=args.max_batches)
            print(f'binding: emitted inputs and outputs exact '
                  f'{100 * b2["exact"]:.4f}% of {b2["n_rows"]} chunks '
                  f'(target 100%; inputs {100 * b2["exact_in"]:.4f}%, '
                  f'outputs {100 * b2["exact_out"]:.4f}%)')
            for k, (acc, n) in sorted(b2['by_shape'].items(),
                                      key=lambda kv: kv[1][0]):
                print(f'  {k:<48} {100 * acc:7.2f}%  of {n}')
            misses = sorted(((1 - acc) * n, k, acc, n)
                            for k, (acc, n) in b2['by_form'].items()
                            if acc < 1.0)[::-1][:20]
            if misses:
                print('  forms with the most failing chunks:')
            for _, k, acc, n in misses:
                print(f'    {k:<60} {100 * acc:7.2f}%  of {n}')
        g = E.gvn_collapse(t1, t2, device=device, route=args.route,
                           n=args.gvn_pairs, chunk_len=args.gvn_chunk_len)
        rr = g['rename_ratio']; br = g['behavioral_ratio']
        print(f'GVN collapse (n={g["n_pairs"]}, distinct={g["n_distinct"]}):')
        print(f'  rename     mean={g["rename_mean"]:.4f} '
              f'ratio={rr:.4f}' if rr is not None else
              f'  rename     mean={g["rename_mean"]:.4f}')
        if g['behavioral_eq_mean'] is not None:
            print(f'  behavioral mean={g["behavioral_eq_mean"]:.4f}   '
                  f'(commutative swap, pairs that ARE equivalent; n='
                  f'{g["n_behavioral_eq"]}) <- judge collapse on this')
        if g['behavioral_ne_mean'] is not None:
            print(f'  behav-diff mean={g["behavioral_ne_mean"]:.4f}   '
                  f'(swap changed the function; n={g["n_behavioral_ne"]} — '
                  f'SHOULD be large)')
        print(f'  behav-mixed mean={g["behavioral_mean"]:.4f} '
              f'ratio={br:.4f}   (both kinds pooled — kept for continuity '
              f'with older records; not interpretable on its own)'
              if br is not None else
              f'  behav-mixed mean={g["behavioral_mean"]:.4f}')
        print(f'  distinct   mean={g["distinct_mean"]:.4f}')


if __name__ == '__main__':
    main()
