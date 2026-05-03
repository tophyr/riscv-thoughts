"""Shared utilities for tools (scripts/).

resolve_device, format_eta:           generic helpers.
load_frozen_encoder, load_decoder:    consume CLI args + a checkpoint
                                       path, return a ready-to-use module.
                                       Encoder is .eval() and gradient-frozen.
"""

import sys

import torch

from compressor.model import T1Compressor, Decoder
from compressor.train import load_checkpoint
from tokenizer import VOCAB_SIZE


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------

def resolve_device(spec):
    """Map an 'auto'/'cuda'/'cpu' spec to a concrete device string."""
    if spec == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return spec


def format_eta(secs):
    """Format a duration in seconds as a compact human string."""
    secs = int(secs)
    if secs < 60:
        return f'{secs}s'
    if secs < 3600:
        return f'{secs // 60}m'
    if secs < 86400:
        return f'{secs // 3600}h{(secs % 3600) // 60}m'
    return f'{secs // 86400}d{(secs % 86400) // 3600}h'


# ---------------------------------------------------------------------------
# Model loading — same pattern across train_*, eval suites, etc.
# ---------------------------------------------------------------------------

def load_frozen_encoder(args, device):
    """Build a T1Compressor from CLI hparams, load the checkpoint at
    args.model, eval(), freeze parameters. strict=False so older
    checkpoints (e.g., pre-dest-heads) still load.

    Required CLI args: --model, --d-model, --n-heads, --n-layers, --d-out,
    optionally --max-window.
    """
    encoder = T1Compressor(
        VOCAB_SIZE, args.d_model, args.n_heads, args.n_layers, args.d_out,
        max_window=getattr(args, 'max_window', 32),
    ).to(device)
    encoder.load_state_dict(
        load_checkpoint(args.model, device), strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_decoder(args, device, *, ckpt_path=None):
    """Build a Decoder from CLI hparams (--dec-d-model, --dec-n-heads,
    --dec-n-layers, --dec-n-memory) and optionally load weights.

    ckpt_path: if given, load these weights. If None, returns
               freshly-initialized decoder (training will fill it in).
    """
    decoder = Decoder(
        VOCAB_SIZE, args.dec_d_model, args.dec_n_heads, args.dec_n_layers,
        d_emb=args.d_out,
        n_memory_tokens=getattr(args, 'dec_n_memory', 1),
    ).to(device)
    if ckpt_path:
        decoder.load_state_dict(load_checkpoint(ckpt_path, device))
        print(f'Loaded decoder from {ckpt_path}', file=sys.stderr)
    return decoder
