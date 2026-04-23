#!/usr/bin/env python3
"""Train a decoder on a frozen encoder with teacher-forced CE.

Pre-generate a corpus for fair comparison across encoder checkpoints:
    gen_instr_batches.py --n-batches 5000 --batch-size 256 --seed 12345 \
        > /tmp/decoder_corpus.bin

Then pipe to each checkpoint:
    batch_repeat.py --forever /tmp/decoder_corpus.bin | \
        train_decoder.py --model runs/XXX/encoder.pt --d-out 64

Reports per-token accuracy (fraction of tokens correct) and
per-instruction accuracy (fraction of fully-correct reconstructions)
at each log point. For REINFORCE-based decoder training (autoregressive
sampling with execution-equivalence reward), see train_reinforce_decoder.py.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', message='.*nested tensors.*prototype stage.*')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from datagen import InstructionBatchReader
from compressor.model import T1Compressor, Decoder
from compressor.train import _prepare_decoder_targets, load_checkpoint
from tokenizer import VOCAB_SIZE, PAD
from scripts._common import resolve_device, format_eta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Path to encoder.pt')
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2,
                   help='Encoder layers (must match checkpoint)')
    p.add_argument('--d-out', type=int, default=128)

    p.add_argument('--dec-d-model', type=int, default=128)
    p.add_argument('--dec-n-heads', type=int, default=4)
    p.add_argument('--dec-n-layers', type=int, default=2)
    p.add_argument('--dec-n-memory', type=int, default=1,
                   help='Number of memory tokens for cross-attention')

    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-steps', type=int, default=5000)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--device', default='auto')
    p.add_argument('--save-decoder', type=str, default=None,
                   help='Save decoder weights on completion')
    p.add_argument('--load-decoder', type=str, default=None,
                   help='Resume from pre-trained decoder weights')
    p.add_argument('--scheduled-sampling', action='store_true',
                   help='Mix model predictions into decoder input, '
                        'ramping from 0%% to 50%% over training')
    p.add_argument('--constant-lr', action='store_true',
                   help='Use constant LR instead of cosine decay')
    p.add_argument('--warmup-steps', type=int, default=0,
                   help='WSD: sine warmup from 0 to lr over N steps')
    p.add_argument('--decay-steps', type=int, default=0,
                   help='WSD: cosine decay to 0 over last N steps')
    p.add_argument('--grad-clip', type=float, default=1.0,
                   help='Max gradient norm (0 to disable)')
    p.add_argument('--cache', action='store_true',
                   help='Read all batches into memory, pre-compute '
                        'encoder outputs and decoder inputs. '
                        'Pipe the corpus directly (no batch_repeat).')
    p.add_argument('--micro-batch', type=int, default=None,
                   help='Re-chunk cached batches into this size '
                        '(reduces activation memory)')
    args = p.parse_args()

    device = resolve_device(args.device)

    # Load frozen encoder.
    encoder = T1Compressor(
        VOCAB_SIZE, args.d_model, args.n_heads, args.n_layers, args.d_out
    ).to(device)
    encoder.load_state_dict(
        load_checkpoint(args.model, device), strict=False)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    decoder = Decoder(
        VOCAB_SIZE, args.dec_d_model, args.dec_n_heads,
        args.dec_n_layers, d_emb=args.d_out,
        n_memory_tokens=args.dec_n_memory,
    ).to(device)
    if args.load_decoder:
        decoder.load_state_dict(load_checkpoint(args.load_decoder, device))
        print(f'Loaded decoder from {args.load_decoder}',
              file=sys.stderr)

    decoder = torch.compile(decoder)
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr, fused=True)
    if args.constant_lr:
        scheduler = None
    elif args.warmup_steps > 0 or args.decay_steps > 0:
        # WSD: sine warmup → constant → cosine decay.
        import math
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

    reader = InstructionBatchReader(sys.stdin.buffer)

    # Pre-compute encoder outputs and decoder inputs if --cache.
    if args.cache:
        print('Caching encoder outputs + decoder inputs...',
              file=sys.stderr)
        cached = []
        for batch in reader:
            tok = torch.from_numpy(batch.token_ids).to(device)
            pad = torch.from_numpy(batch.padding_mask).to(device)
            with torch.no_grad():
                vecs = encoder.compiled_encode(tok, pad)
            token_lists = []
            for b in range(tok.shape[0]):
                mask = ~batch.padding_mask[b]
                token_lists.append(batch.token_ids[b][mask].tolist())
            dec_in, dec_tgt, dec_pad = _prepare_decoder_targets(
                token_lists, device)
            if dec_in is not None:
                cached.append((vecs, dec_in, dec_tgt, dec_pad))
        print(f'Cached {len(cached)} batches on GPU', file=sys.stderr)

        if args.micro_batch:
            chunked = []
            for vecs, dec_in, dec_tgt, dec_pad in cached:
                B = vecs.shape[0]
                for start in range(0, B, args.micro_batch):
                    end = min(start + args.micro_batch, B)
                    chunked.append((
                        vecs[start:end],
                        dec_in[start:end],
                        dec_tgt[start:end],
                        dec_pad[start:end],
                    ))
            print(f'Re-chunked into {len(chunked)} micro-batches '
                  f'of ≤{args.micro_batch}', file=sys.stderr)
            cached = chunked

        del encoder
        torch.cuda.empty_cache()

    step = 0
    t0 = time.time()
    t_last_log = t0
    batch_idx = 0

    def next_batch():
        nonlocal batch_idx
        if args.cache:
            item = cached[batch_idx % len(cached)]
            batch_idx += 1
            return item
        batch = next(reader_iter)
        tok = torch.from_numpy(batch.token_ids).to(device)
        pad = torch.from_numpy(batch.padding_mask).to(device)
        with torch.no_grad():
            vecs = encoder.encode(tok, pad)
        token_lists = []
        for b in range(tok.shape[0]):
            mask = ~batch.padding_mask[b]
            token_lists.append(batch.token_ids[b][mask].tolist())
        dec_in, dec_tgt, dec_pad = _prepare_decoder_targets(
            token_lists, device)
        return vecs, dec_in, dec_tgt, dec_pad

    if not args.cache:
        reader_iter = iter(reader)

    while step < args.n_steps:
        vecs, dec_in, dec_tgt, dec_pad = next_batch()
        if dec_in is None:
            continue
        B = dec_in.shape[0]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = decoder(vecs, dec_in, dec_pad)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                dec_tgt.reshape(-1),
                ignore_index=PAD)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), args.grad_clip)
        opt.step()
        if scheduler:
            scheduler.step()

        step += 1

        # Phase-transition saves for WSD.
        if args.save_decoder and args.warmup_steps > 0:
            if step == args.warmup_steps:
                torch.save(decoder.state_dict(),
                           args.save_decoder + '.warmup_end.pt')
            if step == args.n_steps - args.decay_steps:
                torch.save(decoder.state_dict(),
                           args.save_decoder + '.stable_end.pt')

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
            remaining = args.n_steps - step
            eta_secs = remaining * ms_per_recent / 1000
            lr_now = scheduler.get_last_lr()[0] if scheduler else args.lr
            print(f'step {step:>5d}  loss {loss.item():.4f}  '
                  f'tok_acc {tok_acc:.1%}  instr_acc {instr_acc:.1%}  '
                  f'lr {lr_now:.1e}  {ms_per_recent:.0f}ms/step  '
                  f'eta {format_eta(eta_secs)}')

            if args.save_decoder:
                torch.save(decoder.state_dict(), args.save_decoder)

        if step >= args.n_steps:
            break

    # Final evaluation on last batch.
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
          f'tok_acc={final_tok_acc:.1%}  '
          f'instr_acc={final_instr_acc:.1%}')

    if args.save_decoder:
        torch.save(decoder.state_dict(), args.save_decoder)
        print(f'Saved decoder to {args.save_decoder}')


if __name__ == '__main__':
    main()
