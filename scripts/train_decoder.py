#!/usr/bin/env python3
"""Train a decoder on a frozen encoder with teacher-forced CE.

CLI shell over compressor.train.train_decoder. Reads RVT batches from
stdin; only valid rows contribute to the reconstruction loss (the decoder
has no target for non-decodable token sequences). Reports per-token and
per-instruction accuracy at each log point.

Pre-generate a corpus for fair comparison across encoder checkpoints:
    gen_batches.py --rule single --twins 0 \\
        --batch-size 256 -n 5000 > /tmp/decoder_corpus.rvt

Then pipe a bounded stream to each checkpoint (reads until stdin EOF like
the encoders; --n-steps only shapes the LR schedule. Encoder arch is read
from its companion hparams.json):
    batch_slice.py --count 5000 /tmp/decoder_corpus.rvt | \\
        train_decoder.py --t1-model runs/XXX/encoder.pt --n-steps 5000
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', message='.*nested tensors.*prototype stage.*')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compressor.train import train_decoder
from datagen import RVT_FORMAT, Batch
from scripts._common import (
    resolve_device, add_common_train_args, open_run_dir, load_frozen_encoder)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t1-model', required=True,
                   help='Path to T1 encoder.pt. Companion hparams.json in '
                        'the same dir is read automatically.')

    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--n-memory', type=int, default=1)

    add_common_train_args(p)
    p.add_argument('--no-compile', action='store_true',
                   help='Disable torch.compile (CUDA-graph) of the train '
                        'step. Compile is on by default (saturates the GPU); '
                        'disable for a bit-faithful eager run or fast-start '
                        'short runs.')
    args = p.parse_args()

    device = resolve_device('auto')
    save_dir, save = open_run_dir(args, 'decoder', suffix='_decoder')

    encoder = load_frozen_encoder(args.t1_model, device)
    reader = RVT_FORMAT.reader(sys.stdin.buffer, Batch)

    decoder, losses = train_decoder(
        reader, encoder,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        n_memory=args.n_memory,
        lr=args.lr, n_steps=args.n_steps, log_every=args.log_every,
        device=device, on_log=save,
        compile_step=not args.no_compile,
    )

    save(decoder, losses)
    print(f'Saved to {save_dir}')


if __name__ == '__main__':
    main()
