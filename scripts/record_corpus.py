#!/usr/bin/env python3
"""Write `corpus.json` into a run directory: the corpus a run was trained on.

A run's `hparams.json` is `vars(args)` of the TRAINER, which never sees the
generator — so a run records its model and nothing about the data that shaped
it, and the only remaining identification is the run directory's name. This
script is what closes that gap, and a launcher calls it before starting the
pipeline so an interrupted run is still identified.

The config file's CONTENTS are copied in, not its path: a config edited after
the run would otherwise silently rewrite what the run claims to have trained
on. Everything else a corpus depends on — the chunk rule, batch size,
anchors per batch, the instruction-stream seed, and how many batches were
drawn — is a CLI argument here because it lives outside the config file.

    python scripts/record_corpus.py runs/<dir> --config configs/<c>.json \
        --rule cap=2 --batch-size 384 --n-states 8 \
        --gen-seed 0 --count 1000000
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', type=Path,
                   help='Run directory (created if absent).')
    p.add_argument('--config', type=Path, required=True,
                   help='Corpus config whose CONTENTS are copied in.')
    p.add_argument('--rule', required=True, help='datagen --rule.')
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--n-states', type=int, required=True)
    p.add_argument('--gen-seed', type=int, required=True,
                   help='Instruction-stream and anchor seed (datagen '
                        '--seed); worker i draws seed+i. Two runs differing '
                        'only here are replicates on separately drawn '
                        'corpora.')
    p.add_argument('--count', type=int, required=True,
                   help='Batches drawn (datagen --count).')
    p.add_argument('--generator', default='multinode_gen.sh',
                   help='Which front-end produced the stream.')
    args = p.parse_args()

    record = {
        'config_path': str(args.config),
        'config': json.loads(args.config.read_text()),
        'rule': args.rule,
        'batch_size': args.batch_size,
        'n_states': args.n_states,
        'gen_seed': args.gen_seed,
        'count': args.count,
        'generator': args.generator,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    out = args.run_dir / 'corpus.json'
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Corpus recorded: {out}')


if __name__ == '__main__':
    main()
