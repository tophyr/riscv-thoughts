#!/usr/bin/env python3
"""Chunk RVS sequences into RVC chunks for T2 training.

Reads RVS batches from stdin, splits each sequence into T2 chunks at
register-state-transformation-block boundaries (terminator instructions
or a max-length cap), augments with invalid chunks (spanning / multi /
overlong), and writes RVC batches to stdout.

CPU-only and pipe-friendly. Multiple instances can run in parallel via
mux_batches.py for throughput.

Usage:
    gen_seq_batches.py --n-batches 5000 --batch-size 256 |
        chunk_t2.py --invalidity-rate 0.2 --storage-max-chunk-len 24 |
        train_t2.py --t1-encoder runs/<stamp>/encoder.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from datagen.seqgen import SequenceBatchReader
from datagen.chunkgen import (
    chunk_rvs, augment_chunkbatch_with_invalid,
    DEFAULT_INVALIDITY_WEIGHTS, DEFAULT_MAX_CHUNK_LEN,
    write_stream_header, write_batch,
)
from scripts._batch_util import binary_stdout


def main():
    p = argparse.ArgumentParser(
        description='Chunk RVS sequences into RVC chunks for T2 training.')
    p.add_argument('--max-chunk-len', type=int, default=DEFAULT_MAX_CHUNK_LEN,
                   help='Cap above which a chunk forces a split. Valid '
                        'chunks have length 1..max_chunk_len.')
    p.add_argument('--storage-max-chunk-len', type=int, default=24,
                   help='Padding cap for the instruction axis. Must '
                        'accommodate overlong augmentation chunks. '
                        '24 fits up to 8 instructions over the 16-cap.')
    p.add_argument('--invalidity-rate', type=float, default=0.2)
    p.add_argument(
        '--invalidity-types', type=str, default=None,
        help='JSON-encoded {type: weight} dict; overrides defaults. '
             'Types: spanning, multi, overlong.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    if args.invalidity_types:
        import json
        type_weights = json.loads(args.invalidity_types)
    else:
        type_weights = DEFAULT_INVALIDITY_WEIGHTS

    rng = np.random.default_rng(args.seed)
    out = binary_stdout()
    write_stream_header(out)

    reader = SequenceBatchReader(sys.stdin.buffer)
    batches_written = 0
    try:
        for rvs_batch in reader:
            cb = chunk_rvs(
                rvs_batch,
                max_chunk_len=args.max_chunk_len,
                storage_max_chunk_len=args.storage_max_chunk_len)
            if args.invalidity_rate > 0:
                cb = augment_chunkbatch_with_invalid(
                    cb,
                    invalidity_rate=args.invalidity_rate,
                    type_weights=type_weights,
                    storage_max_chunk_len=args.storage_max_chunk_len,
                    rng=rng)
            if cb.token_ids.shape[0] == 0:
                continue
            write_batch(out, cb)
            batches_written += 1
            if args.verbose and batches_written % 100 == 0:
                print(f'{batches_written} chunked batches written',
                      file=sys.stderr)
    except BrokenPipeError:
        pass

    out.close()
    if args.verbose:
        print(f'Done: {batches_written} chunked batches', file=sys.stderr)


if __name__ == '__main__':
    main()
