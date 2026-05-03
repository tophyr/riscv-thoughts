#!/usr/bin/env python3
"""Train a decoder on a frozen encoder with teacher-forced CE.

Reads RVT batches from stdin. Only valid rows contribute to the
reconstruction loss; invalid rows are skipped (the decoder has no
target for non-decodable token sequences).

Pre-generate a corpus for fair comparison across encoder checkpoints:
    gen_batches.py --rule single --twins 0 --partners 0 \\
        --batch-size 256 -n 5000 > /tmp/decoder_corpus.rvt

Then pipe to each checkpoint:
    batch_repeat.py --forever /tmp/decoder_corpus.rvt | \\
        train_decoder.py --model runs/XXX/encoder.pt --d-out 64

Reports per-token and per-instruction accuracy at each log point.
"""

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', message='.*nested tensors.*prototype stage.*')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from compressor.train import prepare_decoder_targets
from datagen.batch import RVT_FORMAT, Batch, padding_mask
from tokenizer import VOCAB_SIZE, PAD
from scripts._common import (
    resolve_device, format_eta, load_frozen_encoder, load_decoder,
)


def _batch_to_decoder_inputs(batch, encoder, device):
    """Encode a Batch and return (vecs, dec_in, dec_tgt, dec_pad) for
    the valid rows only. Returns (None, None, None, None) if the batch
    has no valid rows."""
    valid = batch.valid
    if not valid.any():
        return None, None, None, None

    tok = torch.from_numpy(batch.tokens).to(device)
    pad = torch.from_numpy(padding_mask(batch)).to(device)
    with torch.no_grad():
        vecs_all = encoder.encode(tok, pad)

    valid_idx = np.flatnonzero(valid)
    vecs = vecs_all[valid_idx]
    token_lists = []
    for i in valid_idx:
        n = int(batch.token_lens[i])
        token_lists.append(batch.tokens[i, :n].tolist())
    dec_in, dec_tgt, dec_pad = prepare_decoder_targets(token_lists, device)
    return vecs, dec_in, dec_tgt, dec_pad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Path to encoder.pt')
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=128)
    p.add_argument('--max-window', type=int, default=72)

    p.add_argument('--dec-d-model', type=int, default=128)
    p.add_argument('--dec-n-heads', type=int, default=4)
    p.add_argument('--dec-n-layers', type=int, default=2)
    p.add_argument('--dec-n-memory', type=int, default=1)

    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-steps', type=int, default=5000)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--device', default='auto')
    p.add_argument('--save-decoder', type=str, default=None)
    p.add_argument('--load-decoder', type=str, default=None)
    p.add_argument('--constant-lr', action='store_true')
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--decay-steps', type=int, default=0)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--cache', action='store_true',
                   help='Read all batches into memory, pre-compute '
                        'encoder outputs and decoder inputs.')
    p.add_argument('--micro-batch', type=int, default=None)
    args = p.parse_args()

    device = resolve_device(args.device)

    encoder = load_frozen_encoder(args, device)
    decoder = load_decoder(args, device, ckpt_path=args.load_decoder)
    decoder = torch.compile(decoder)
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr,
                           fused=(device == 'cuda'))

    if args.constant_lr:
        scheduler = None
    elif args.warmup_steps > 0 or args.decay_steps > 0:
        warmup = args.warmup_steps
        decay = args.decay_steps
        stable_end = args.n_steps - decay
        if warmup + decay > args.n_steps:
            raise ValueError(
                f'warmup ({warmup}) + decay ({decay}) > n_steps '
                f'({args.n_steps})')

        def wsd_lambda(step):
            if step < warmup:
                return math.sin(math.pi / 2 * step / warmup)
            if step < stable_end:
                return 1.0
            t = (step - stable_end) / decay
            return 0.5 * (1 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, wsd_lambda)
        print(f'WSD schedule: warmup={warmup}, stable={stable_end - warmup}, '
              f'decay={decay}', file=sys.stderr)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.n_steps, eta_min=1e-6)

    reader = RVT_FORMAT.reader(sys.stdin.buffer, Batch)

    # Pre-compute everything if --cache.
    if args.cache:
        print('Caching encoder outputs + decoder inputs...', file=sys.stderr)
        cached = []
        for batch in reader:
            vecs, dec_in, dec_tgt, dec_pad = _batch_to_decoder_inputs(
                batch, encoder, device)
            if dec_in is not None:
                cached.append((vecs, dec_in, dec_tgt, dec_pad))
        print(f'Cached {len(cached)} batches on GPU', file=sys.stderr)

        if args.micro_batch:
            chunked = []
            for vecs, dec_in, dec_tgt, dec_pad in cached:
                B = vecs.shape[0]
                for s in range(0, B, args.micro_batch):
                    e = min(s + args.micro_batch, B)
                    chunked.append((vecs[s:e], dec_in[s:e],
                                    dec_tgt[s:e], dec_pad[s:e]))
            print(f'Re-chunked into {len(chunked)} micro-batches '
                  f'of ≤{args.micro_batch}', file=sys.stderr)
            cached = chunked
        del encoder
        if device == 'cuda':
            torch.cuda.empty_cache()

    step = 0
    t0 = time.time()
    t_last_log = t0
    batch_idx = 0
    reader_iter = None if args.cache else iter(reader)

    def next_batch():
        nonlocal batch_idx
        if args.cache:
            item = cached[batch_idx % len(cached)]
            batch_idx += 1
            return item
        while True:
            batch = next(reader_iter)
            v, di, dt, dp = _batch_to_decoder_inputs(batch, encoder, device)
            if di is not None:
                return v, di, dt, dp

    while step < args.n_steps:
        vecs, dec_in, dec_tgt, dec_pad = next_batch()
        B = dec_in.shape[0]

        if device == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = decoder(vecs, dec_in, dec_pad)
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    dec_tgt.reshape(-1), ignore_index=PAD)
        else:
            logits = decoder(vecs, dec_in, dec_pad)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                dec_tgt.reshape(-1), ignore_index=PAD)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), args.grad_clip)
        opt.step()
        if scheduler:
            scheduler.step()
        step += 1

        if step % args.log_every == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                non_pad = ~dec_pad
                tok_correct = ((pred == dec_tgt) & non_pad).sum().item()
                tok_total = non_pad.sum().item()
                tok_acc = tok_correct / tok_total if tok_total > 0 else 0
                instr_correct = 0
                for b in range(B):
                    n = non_pad[b].sum().item()
                    if n > 0 and (pred[b, :n] == dec_tgt[b, :n]).all():
                        instr_correct += 1
                instr_acc = instr_correct / B

            now = time.time()
            ms_per_recent = (now - t_last_log) / args.log_every * 1000
            t_last_log = now
            eta_secs = (args.n_steps - step) * ms_per_recent / 1000
            lr_now = scheduler.get_last_lr()[0] if scheduler else args.lr
            print(f'step {step:>5d}  loss {loss.item():.4f}  '
                  f'tok_acc {tok_acc:.1%}  instr_acc {instr_acc:.1%}  '
                  f'lr {lr_now:.1e}  {ms_per_recent:.0f}ms/step  '
                  f'eta {format_eta(eta_secs)}')
            if args.save_decoder:
                torch.save(decoder.state_dict(), args.save_decoder)

    # Final eval on last batch.
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        non_pad = ~dec_pad
        tok_correct = ((pred == dec_tgt) & non_pad).sum().item()
        tok_total = non_pad.sum().item()
        final_tok_acc = tok_correct / tok_total if tok_total > 0 else 0
        instr_correct = 0
        for b in range(B):
            n = non_pad[b].sum().item()
            if n > 0 and (pred[b, :n] == dec_tgt[b, :n]).all():
                instr_correct += 1
        final_instr_acc = instr_correct / B

    print(f'\nFinal ({step} steps):  '
          f'tok_acc={final_tok_acc:.1%}  instr_acc={final_instr_acc:.1%}')

    if args.save_decoder:
        torch.save(decoder.state_dict(), args.save_decoder)
        hp_path = Path(args.save_decoder).with_suffix('.hparams.json')
        with open(hp_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f'Saved decoder to {args.save_decoder}')


if __name__ == '__main__':
    main()
