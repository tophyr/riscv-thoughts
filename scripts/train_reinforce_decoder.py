#!/usr/bin/env python3
"""Train a decoder via REINFORCE with execution-equivalence reward.

Frozen encoder. Decoder samples discrete tokens, decoded
instructions are executed via the GPU batch emulator, and compared
to the original instruction's execution output. Reward = fraction
of matching outputs across random register states.

Usage:
    gen_instr_batches.py --n-batches 5000 --batch-size 4096 | \
        train_reinforce_decoder.py --model runs/XXX/encoder.pt --d-out 64
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
from compressor.train import (
    _prepare_decoder_targets,
    autoregressive_sample,
    compute_log_probs,
    gpu_reward,
    load_checkpoint,
)
from tokenizer import VOCAB_SIZE, PAD
from scripts._common import resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=64)
    p.add_argument('--dec-d-model', type=int, default=128)
    p.add_argument('--dec-n-heads', type=int, default=4)
    p.add_argument('--dec-n-layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n-steps', type=int, default=1000)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--n-reward-inputs', type=int, default=4)
    p.add_argument('--k-samples', type=int, default=1,
                   help='Samples per instruction for variance reduction')
    p.add_argument('--load-decoder', type=str, default=None,
                   help='Load pre-trained decoder weights (e.g., from CE phase)')
    p.add_argument('--save-decoder', type=str, default=None,
                   help='Save decoder weights to this path on completion')
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    device = resolve_device(args.device)

    torch.set_float32_matmul_precision('high')

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
        args.dec_n_layers, d_emb=args.d_out
    ).to(device)
    if args.load_decoder:
        decoder.load_state_dict(load_checkpoint(args.load_decoder, device))
        print(f'Loaded decoder from {args.load_decoder}',
              file=sys.stderr)
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    baseline = 0.0
    baseline_decay = 0.99

    reader = InstructionBatchReader(sys.stdin.buffer)
    step = 0
    t0 = time.time()

    for batch in reader:
        tok = torch.from_numpy(batch.token_ids).to(device)
        pad = torch.from_numpy(batch.padding_mask).to(device)
        B = tok.shape[0]

        if device != 'cpu':
            torch.cuda.synchronize()
        t_enc = time.time()

        with torch.no_grad():
            vecs = encoder.compiled_encode(tok, pad)

        # Extract ground-truth token lists.
        token_lists = []
        for b in range(B):
            mask = ~batch.padding_mask[b]
            token_lists.append(batch.token_ids[b][mask].tolist())

        dec_in, dec_tgt, dec_pad = _prepare_decoder_targets(
            token_lists, device)
        if dec_in is None:
            continue

        if device != 'cpu':
            torch.cuda.synchronize()
        t_dec = time.time()

        non_pad = ~dec_pad
        orig_lengths = non_pad.sum(dim=1)
        max_len = int(orig_lengths.max().item())

        if device != 'cpu':
            torch.cuda.synchronize()

        # K-sample autoregressive REINFORCE (two-phase).
        # Phase 1: sample K sequences autoregressively (no gradient),
        #          evaluate with GPU emulator, keep the best.
        # Phase 2: compute log_probs for the best sample WITH gradient
        #          in one parallel forward pass.
        t_reward = time.time()
        K = args.k_samples

        best_rewards = torch.full((B,), -1.0, device=device)
        best_sampled = torch.zeros(B, max_len, dtype=torch.long,
                                   device=device)
        best_valid = torch.zeros(B, dtype=torch.bool, device=device)
        best_equiv = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(K):
            sampled = autoregressive_sample(
                decoder, vecs, max_len, device)

            rewards_k, valid_k, equiv_k = gpu_reward(
                tok, sampled, orig_lengths, orig_lengths,
                device, n_inputs=args.n_reward_inputs)

            better = rewards_k > best_rewards
            best_rewards = torch.where(better, rewards_k, best_rewards)
            best_sampled = torch.where(
                better.unsqueeze(1).expand_as(sampled),
                sampled, best_sampled)
            best_valid = torch.where(better, valid_k, best_valid)
            best_equiv = torch.where(better, equiv_k, best_equiv)

        rewards = best_rewards.clamp(min=0)
        valid_mask = best_valid
        equiv_mask = best_equiv

        # Phase 2: log_probs with gradient for the best sample.
        seq_log_prob = compute_log_probs(
            decoder, vecs, best_sampled, orig_lengths, device)

        if device != 'cpu':
            torch.cuda.synchronize()
        reward_ms = (time.time() - t_reward) * 1000

        # REINFORCE.
        advantage = rewards - baseline
        reinforce_loss = -(seq_log_prob * advantage).mean()

        if device != 'cpu':
            torch.cuda.synchronize()
        t_back = time.time()

        opt.zero_grad()
        reinforce_loss.backward()
        opt.step()

        if device != 'cpu':
            torch.cuda.synchronize()
        t_done = time.time()

        baseline = (baseline_decay * baseline
                    + (1 - baseline_decay) * rewards.mean().item())

        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            ms_per = elapsed / step * 1000
            mean_reward = rewards.mean().item()
            valid_frac = valid_mask.float().mean().item()
            equiv_frac = equiv_mask.float().mean().item()

            enc_ms = (t_dec - t_enc) * 1000
            dec_ms = (t_reward - t_dec) * 1000
            back_ms = (t_done - t_back) * 1000
            print(f'step {step:>5d}  '
                  f'reward {mean_reward:.3f}  '
                  f'valid {valid_frac:.0%}  '
                  f'equiv {equiv_frac:.0%}  '
                  f'enc {enc_ms:.0f} dec+rew {dec_ms + reward_ms:.0f} '
                  f'back {back_ms:.0f}  '
                  f'{ms_per:.0f}ms/step')

        if step >= args.n_steps:
            break

    print(f'\nDone: {step} steps')
    if args.save_decoder:
        torch.save(decoder.state_dict(), args.save_decoder)
        print(f'Saved decoder to {args.save_decoder}')


if __name__ == '__main__':
    main()
