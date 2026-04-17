#!/usr/bin/env python3
"""Train a decoder on a frozen encoder and report reconstruction accuracy.

Pre-generate a corpus for fair comparison across encoder checkpoints:
    gen_instr_batches.py --n-batches 5000 --batch-size 256 --seed 12345 \
        > /tmp/decoder_corpus.bin

Then pipe to each checkpoint:
    batch_repeat.py --forever /tmp/decoder_corpus.bin | \
        eval_decoder.py --model runs/XXX/encoder.pt --d-out 64

Reports per-token accuracy (fraction of tokens correct) and
per-instruction accuracy (fraction of fully-correct reconstructions).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from datagen import InstructionBatchReader
from compressor.model import T1Compressor, Decoder
from compressor.train import _prepare_decoder_targets
from tokenizer import VOCAB_SIZE, PAD


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

    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-steps', type=int, default=5000)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Load frozen encoder.
    encoder = T1Compressor(
        VOCAB_SIZE, args.d_model, args.n_heads, args.n_layers, args.d_out
    ).to(device)
    state = torch.load(args.model, map_location=device, weights_only=True)
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Fresh decoder.
    decoder = Decoder(
        VOCAB_SIZE, args.dec_d_model, args.dec_n_heads,
        args.dec_n_layers, d_emb=args.d_out
    ).to(device)

    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.n_steps, eta_min=1e-6)

    reader = InstructionBatchReader(sys.stdin.buffer)
    step = 0
    t0 = time.time()

    for batch in reader:
        tok = torch.from_numpy(batch.token_ids).to(device)
        pad = torch.from_numpy(batch.padding_mask).to(device)
        B = tok.shape[0]

        # Encode (frozen).
        with torch.no_grad():
            vecs = encoder.encode(tok, pad)

        # Extract raw token lists (no padding).
        token_lists = []
        for b in range(B):
            mask = ~batch.padding_mask[b]
            tokens = batch.token_ids[b][mask].tolist()
            token_lists.append(tokens)

        dec_in, dec_tgt, dec_pad = _prepare_decoder_targets(
            token_lists, device)
        if dec_in is None:
            continue

        logits = decoder(vecs, dec_in, dec_pad)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            dec_tgt.reshape(-1),
            ignore_index=PAD)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
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

            elapsed = time.time() - t0
            ms_per = elapsed / step * 1000
            lr_now = scheduler.get_last_lr()[0]
            print(f'step {step:>5d}  loss {loss.item():.4f}  '
                  f'tok_acc {tok_acc:.1%}  instr_acc {instr_acc:.1%}  '
                  f'lr {lr_now:.1e}  {ms_per:.0f}ms/step')

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


if __name__ == '__main__':
    main()
