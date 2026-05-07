#!/usr/bin/env python3
"""Generate unified RVT training batches.

Single tool, single output format. Subsumes the previous instr / seq /
chunk / pair generators by parameterizing on:

  --rule {single, branch, transform, cap=N, branch+cap=N, transform+cap=N}
       How to group the random instruction stream into chunks.
       single        — one instruction per chunk (T1 encoder pair-MSE)
       branch        — until_branch (basic blocks)
       transform     — until_transformation (T2 chunks)
       cap=N         — fixed length N
       branch+cap=N  — until_branch OR length cap
       transform+cap=N — until_transformation OR length cap

  --twins T          # relabeled equivalents per source chunk
  --partners K       # total partners per source row (>= T); K-T are random
  --batch-size B     # batch size in source rows (twins extend each batch)
  --inject-invalid R # mix in invalid windows at fraction R of batch-size
                       (requires the --config invalidity sub-dict, or just
                       uses defaults if R is set without config)
  --inject-equiv R   # post-chunk MANIFEST-equivalence injection rate
  --config PATH      # JSON: opcode mix + equivalences + invalidity
                       sub-dicts. CLI flags override config.

Pipeline shape:
    gen_batches.py --rule branch+cap=8 --twins 3 --partners 20 \\
        --inject-invalid 0.2 -v -n 1000 |
            mux_batches.py | batch_repeat.py | train_encoder.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from datagen.batch import (
    RVT_FORMAT, generate_chunks, collect_into_batches,
)
from datagen.compare import make_anchor_states
from datagen.generate import (
    DEFAULT_DISTRIBUTION, _build_opcode_table, validate_distribution,
    load_distribution,
    single, until_branch, until_transformation, length_cap, either,
)
from datagen.invalidity import (
    DEFAULT_TYPE_WEIGHTS, build_type_table, generate_invalid,
)
from scripts._batch_util import binary_stdout


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

def _parse_rule(spec):
    """Parse the --rule argument. Returns a TerminationRule."""
    s = spec.strip()
    if s == 'single':
        return single()
    if s == 'branch':
        return until_branch()
    if s == 'transform':
        return until_transformation()
    if s.startswith('cap='):
        return length_cap(int(s[4:]))
    if s.startswith('branch+cap='):
        return either(until_branch(), length_cap(int(s[len('branch+cap='):])))
    if s.startswith('transform+cap='):
        return either(until_transformation(),
                      length_cap(int(s[len('transform+cap='):])))
    raise ValueError(f'Unknown rule: {spec!r}')


def _max_chunk_len_for_rule(spec):
    """Return the upper bound on instructions-per-chunk implied by
    `spec`. None for cap-less rules where there's no fixed bound; that
    case keeps batches data-shaped (legacy variable-shape behavior)."""
    s = spec.strip()
    if s == 'single':
        return 1
    if s.startswith('cap='):
        return int(s[4:])
    if s.startswith('branch+cap='):
        return int(s[len('branch+cap='):])
    if s.startswith('transform+cap='):
        return int(s[len('transform+cap='):])
    # 'branch' and 'transform' (no cap) — variable max chunk length.
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Generate unified training batches (RVT).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Pipeline shape:')[1] if 'Pipeline shape:' in __doc__ else '')

    p.add_argument('--rule', default='branch+cap=8',
                   help='Chunk-termination rule. Default: branch+cap=8.')
    p.add_argument('--batch-size', type=int, default=128,
                   help='Source rows per batch (twins extend the batch).')
    p.add_argument('--twins', type=int, default=3,
                   help='Relabeled equivalents per valid source chunk.')
    p.add_argument('--partners', type=int, default=20,
                   help='Partners per source (>= --twins; '
                        'K-T are random non-cluster).')
    p.add_argument('--n-states', type=int, default=8,
                   help='Anchor states for behavioral_distance.')
    p.add_argument('--anchor-seed', type=int, default=0,
                   help='Anchor-states RNG seed (must match across workers).')

    p.add_argument('--inject-invalid', type=float, default=None,
                   help='Invalid-window injection rate. Overrides config.')
    p.add_argument('--inject-equiv', type=float, default=None,
                   help='Post-chunk MANIFEST equivalence injection rate. '
                        'Overrides config.')
    p.add_argument('--max-invalid-window', type=int, default=32,
                   help='Token-length cap on injected invalid windows.')

    p.add_argument('--config', type=str, default=None,
                   help='JSON config: opcode mix + equivalences + invalidity.')

    p.add_argument('-n', '--n-batches', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('-v', '--verbose', action='store_true')

    args = p.parse_args()

    if args.partners < args.twins:
        sys.exit('--partners must be >= --twins')

    rule = _parse_rule(args.rule)
    max_chunk_len = _max_chunk_len_for_rule(args.rule)
    rng = np.random.default_rng(args.seed)
    anchor_states = make_anchor_states(args.n_states, args.anchor_seed)

    # Resolve config: load file (if any), apply CLI overrides.
    if args.config:
        dist = load_distribution(args.config)
    else:
        dist = dict(DEFAULT_DISTRIBUTION)
        validate_distribution(dist)

    opcode_table = _build_opcode_table(dist)

    eq_cfg = dist.get('equivalences', {})
    eq_rate = (args.inject_equiv if args.inject_equiv is not None
               else eq_cfg.get('rate', 0.0))
    eq_max_per_class = eq_cfg.get('max_per_class', 8)
    eq_min_per_class = eq_cfg.get('min_per_class', 0)
    eq_boost = eq_cfg.get('boost', None)

    inv_cfg = dist.get('invalidity', {})
    inv_rate = (args.inject_invalid if args.inject_invalid is not None
                else inv_cfg.get('rate', 0.0))
    inv_types = inv_cfg.get('types', DEFAULT_TYPE_WEIGHTS)

    # Invalid-window provider closure.
    invalid_provider = None
    if inv_rate > 0:
        type_table = build_type_table(inv_types)
        def invalid_provider():
            tokens, _ = generate_invalid(
                rng, opcode_table, args.max_invalid_window, type_table)
            return tokens

    # Build the chunk and batch streams.
    chunks_iter = generate_chunks(
        rule, rng, opcode_table=opcode_table,
        eq_rate=eq_rate,
        eq_max_per_class=eq_max_per_class,
        eq_min_per_class=eq_min_per_class,
        eq_boost=eq_boost,
    )
    # Single-instruction rule → row-outputs mode (per-row canonical
    # outputs shipped, training forms (B,B) target distance on-GPU).
    # Multi-instruction rules keep the per-pair CPU distance path.
    row_outputs_mode = (args.rule.strip() == 'single')

    batches_iter = collect_into_batches(
        chunks_iter,
        batch_size=args.batch_size,
        twins=args.twins, partners=args.partners,
        anchor_states=anchor_states, rng=rng,
        invalid_rate=inv_rate, invalid_provider=invalid_provider,
        max_invalid_window=args.max_invalid_window,
        max_chunk_len=max_chunk_len,
        row_outputs_mode=row_outputs_mode,
    )

    out = binary_stdout()
    RVT_FORMAT.write_stream_header(out)
    written = 0
    pid = os.getpid()
    bi = iter(batches_iter)
    print(f'[pid={pid}] worker_start seed={args.seed} target={args.n_batches}',
          file=sys.stderr, flush=True)
    try:
        while written < args.n_batches:
            t0 = time.monotonic()
            try:
                batch = next(bi)
            except StopIteration:
                break
            t1 = time.monotonic()
            RVT_FORMAT.write_batch(out, batch)
            t2 = time.monotonic()
            written += 1
            compute_ms = (t1 - t0) * 1000
            write_ms = (t2 - t1) * 1000
            n_pairs = int(batch.pair_indices.shape[0])
            print(f'[pid={pid}] batch={written} '
                  f'compute_ms={compute_ms:.0f} write_ms={write_ms:.0f} '
                  f'n_pairs={n_pairs}',
                  file=sys.stderr, flush=True)
    except BrokenPipeError:
        pass

    out.close()
    print(f'[pid={pid}] worker_end written={written}',
          file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
