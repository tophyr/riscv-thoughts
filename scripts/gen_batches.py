#!/usr/bin/env python3
"""Generate unified RVT training batches.

Single tool, single output format. Subsumes the previous instr / seq /
chunk / pair generators by parameterizing on:

  --rule {single, cap=N, branch+cap=N, transform+cap=N}
       How to group the random instruction stream into chunks. Every
       rule imposes a finite length cap (bare 'branch'/'transform' are
       rejected — they can build an unbounded chunk).
       single        — one instruction per chunk (T1 encoder, row-outputs)
       cap=N         — fixed length N
       branch+cap=N  — until_branch OR length cap
       transform+cap=N — until_transformation OR length cap

  --twins T          # relabeled equivalents per source chunk
  --batch-size B     # batch size in source rows (twins extend each batch)
  --inject-invalid R # mix in invalid windows at fraction R of batch-size
                       (requires the --config invalidity sub-dict, or just
                       uses defaults if R is set without config)
  --inject-equiv R   # post-chunk MANIFEST-equivalence injection rate
  --config PATH      # JSON: opcode mix + equivalences + invalidity
                       sub-dicts. CLI flags override config.

Pipeline shape:
    gen_batches.py --rule branch+cap=8 --twins 3 \\
        --inject-invalid 0.2 -v -n 1000 |
            mux_batches.py | train_encoder.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from datagen import (
    RVT_FORMAT, generate_chunks, collect_into_batches,
    make_anchor_states,
    DEFAULT_DISTRIBUTION, build_opcode_table, validate_distribution,
    load_distribution,
    single, until_branch, until_transformation, length_cap, either,
    DEFAULT_TYPE_WEIGHTS, build_type_table, generate_invalid,
)
from scripts._streamfmt import binary_stdout


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

# --rule is a '+'-joined set of components, composed via either() so chunk
# termination fires when ANY matches. 'single' is a standalone shorthand.
# Every rule must stay length-bounded (the set must include a cap=N), but
# that requirement is enforced structurally downstream by collect_groups
# via rule.max_len — not re-checked here.
_TERMINATOR_COMPONENTS = {
    'branch': until_branch,
    'transform': until_transformation,
}


def _parse_rule(spec):
    """Parse a '+'-joined --rule into a TerminationRule. 'single' is a
    standalone shorthand. Unknown components and unbounded rules (no cap)
    raise here, eagerly — gen_batches reads rule.max_len to size batch
    arrays before the lazy chunk generator would reach collect_groups'
    own bound check, so a bare 'branch'/'transform' must be caught here or
    it surfaces downstream as an opaque None-arithmetic TypeError."""
    s = spec.strip()
    if s == 'single':
        return single()
    rules = []
    for part in s.split('+'):
        part = part.strip()
        if part.startswith('cap='):
            rules.append(length_cap(int(part[len('cap='):])))
        elif part in _TERMINATOR_COMPONENTS:
            rules.append(_TERMINATOR_COMPONENTS[part]())
        else:
            raise ValueError(f'Unknown rule component {part!r} in {spec!r}')
    rule = rules[0] if len(rules) == 1 else either(*rules)
    if rule.max_len is None:
        raise ValueError(
            f'Rule {spec!r} is unbounded (no length cap); it can build an '
            f'infinite chunk from the instruction stream. Add a cap, '
            f'e.g. {spec.strip()}+cap=8.')
    return rule


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
    p.add_argument('--n-states', type=int, default=8,
                   help='Anchor states for value-prediction / aux precompute.')
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

    rule = _parse_rule(args.rule)
    max_chunk_len = rule.max_len
    rng = np.random.default_rng(args.seed)
    anchor_states = make_anchor_states(args.n_states, args.anchor_seed)

    # Resolve config: load file (if any), apply CLI overrides.
    if args.config:
        dist = load_distribution(args.config)
    else:
        dist = dict(DEFAULT_DISTRIBUTION)
        validate_distribution(dist)

    opcode_table = build_opcode_table(dist)

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
        twins=args.twins,
        anchor_states=anchor_states, rng=rng,
        invalid_rate=inv_rate, invalid_provider=invalid_provider,
        max_invalid_window=args.max_invalid_window,
        max_chunk_len=max_chunk_len,
        row_outputs_mode=row_outputs_mode,
    )

    out = binary_stdout()
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
            print(f'[pid={pid}] batch={written} '
                  f'compute_ms={compute_ms:.0f} write_ms={write_ms:.0f}',
                  file=sys.stderr, flush=True)
    except BrokenPipeError:
        pass

    out.close()
    print(f'[pid={pid}] worker_end written={written}',
          file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
