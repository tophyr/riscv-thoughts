#!/usr/bin/env python3
"""Decode T1 vectors back to tokens via gradient-based encoder inversion.

No trained decoder needed. For each instruction:
1. Encode it to get target T1 vector (frozen encoder)
2. Initialize random soft embeddings at the known sequence length
3. Forward through encode_soft, backprop distance to target
4. Iterate until soft embeddings converge
5. Snap each position to nearest token embedding

Uses ground-truth sequence lengths (cheating for eval purposes —
in production, length would need a separate prediction head).

Usage:
    eval_gradient_decode.py --model runs/XXX/encoder.pt --d-out 64
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from datagen import produce_instruction_batch
from compressor.model import T1Compressor
from tokenizer import VOCAB_SIZE, PAD, encode_instruction


def gradient_decode_batch(encoder, target_vecs, lengths, max_len,
                          n_iters=500, lr=0.1):
    """Decode a batch of T1 vectors via gradient optimization.

    Returns list of token ID lists (one per instruction).
    """
    B = target_vecs.shape[0]
    d_model = encoder.d_model
    device = target_vecs.device

    soft_emb = torch.randn(B, max_len, d_model, device=device) * 0.1
    soft_emb = torch.nn.Parameter(soft_emb)

    padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
    for b in range(B):
        padding_mask[b, :lengths[b]] = False

    opt = torch.optim.Adam([soft_emb], lr=lr)
    target = target_vecs.detach()

    for i in range(n_iters):
        v = encoder.encode_soft(soft_emb, padding_mask)
        loss = (v - target).square().sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Snap to nearest token embeddings.
    with torch.no_grad():
        tok_weights = encoder.tok_emb.weight  # (VOCAB_SIZE, d_model)
        decoded = []
        for b in range(B):
            tokens = []
            for pos in range(lengths[b]):
                emb = soft_emb[b, pos]
                dists = (tok_weights - emb.unsqueeze(0)).square().sum(dim=1)
                tokens.append(dists.argmin().item())
            decoded.append(tokens)

    final_loss = loss.item()
    return decoded, final_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Path to encoder.pt')
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=128)
    p.add_argument('--n-iters', type=int, default=500)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--n-instrs', type=int, default=256,
                   help='Number of instructions to test')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=54321)
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    encoder = T1Compressor(
        VOCAB_SIZE, args.d_model, args.n_heads, args.n_layers, args.d_out
    ).to(device)
    state = torch.load(args.model, map_location=device, weights_only=True)
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    rng = np.random.default_rng(args.seed)
    batch = produce_instruction_batch(args.n_instrs, 1, rng)

    # Encode all instructions.
    tok = torch.from_numpy(batch.token_ids).to(device)
    pad = torch.from_numpy(batch.padding_mask).to(device)
    with torch.no_grad():
        target_vecs = encoder.encode(tok, pad)

    # Ground-truth token lists and lengths.
    gt_tokens = []
    lengths = []
    for b in range(args.n_instrs):
        mask = ~batch.padding_mask[b]
        tokens = batch.token_ids[b][mask].tolist()
        gt_tokens.append(tokens)
        lengths.append(len(tokens))

    max_len = max(lengths)
    lengths_t = torch.tensor(lengths, device=device)

    print(f'Decoding {args.n_instrs} instructions '
          f'(d_out={args.d_out}, n_layers={args.n_layers}, '
          f'n_iters={args.n_iters}, lr={args.lr})')

    t0 = time.time()

    # Process in batches.
    all_decoded = []
    for start in range(0, args.n_instrs, args.batch_size):
        end = min(start + args.batch_size, args.n_instrs)
        batch_vecs = target_vecs[start:end]
        batch_lens = lengths[start:end]

        decoded, final_loss = gradient_decode_batch(
            encoder, batch_vecs, batch_lens, max_len,
            n_iters=args.n_iters, lr=args.lr)
        all_decoded.extend(decoded)

        print(f'  batch {start}-{end}: '
              f'final_loss={final_loss:.6f}')

    elapsed = time.time() - t0

    # Compute accuracy.
    tok_correct = tok_total = 0
    instr_correct = 0
    for b in range(args.n_instrs):
        gt = gt_tokens[b]
        dec = all_decoded[b]
        for i in range(len(gt)):
            tok_total += 1
            if i < len(dec) and dec[i] == gt[i]:
                tok_correct += 1
        if dec == gt:
            instr_correct += 1

    tok_acc = tok_correct / tok_total if tok_total > 0 else 0
    instr_acc = instr_correct / args.n_instrs

    print(f'\nResult ({elapsed:.1f}s):  '
          f'tok_acc={tok_acc:.1%}  '
          f'instr_acc={instr_acc:.1%}')


if __name__ == '__main__':
    main()
