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
from compressor.train import _prepare_decoder_targets
from tokenizer import VOCAB_SIZE, PAD
from emulator.batch_emulator import (
    batch_execute, batch_parse_tokens, random_regs_gpu,
)


def gpu_reward(orig_tokens, sampled_tokens, orig_lengths,
               sampled_lengths, device, n_inputs=4):
    """Fully-GPU shaped execution-equivalence reward.

    Decomposes the reward into per-field signals so the decoder
    gets credit for partially-correct decodings:
      0.25 × opcode match
      0.25 × dest_reg match
      0.25 × data_val match (fraction across n_inputs)
      0.25 × pc match (fraction across n_inputs)

    Returns (rewards, valid_mask, equiv_mask) as (B,) tensors.
    """
    o_op, o_rd, o_rs1, o_rs2, o_imm, o_valid = batch_parse_tokens(
        orig_tokens, orig_lengths, device)
    d_op, d_rd, d_rs1, d_rs2, d_imm, d_valid = batch_parse_tokens(
        sampled_tokens, sampled_lengths, device)

    both_valid = o_valid & d_valid
    B = orig_tokens.shape[0]

    # Field-level rewards.
    op_match = (o_op == d_op).float()
    rd_match = (o_rd == d_rd).float()

    dv_match_count = torch.zeros(B, dtype=torch.float32, device=device)
    pc_match_count = torch.zeros(B, dtype=torch.float32, device=device)

    for _ in range(n_inputs):
        regs = random_regs_gpu(B, device=device)
        pc = torch.randint(0, 256, (B,), dtype=torch.int32,
                           device=device) * 4

        o_dv, o_pc = batch_execute(o_op, o_rd, o_rs1, o_rs2, o_imm,
                                   regs, pc)
        d_dv, d_pc = batch_execute(d_op, d_rd, d_rs1, d_rs2, d_imm,
                                   regs, pc)

        dv_match_count += (o_dv == d_dv).float()
        pc_match_count += (o_pc == d_pc).float()

    dv_match = dv_match_count / n_inputs
    pc_match = pc_match_count / n_inputs

    shaped_reward = 0.25 * op_match + 0.25 * rd_match \
                  + 0.25 * dv_match + 0.25 * pc_match

    rewards = torch.where(both_valid, shaped_reward,
                          torch.zeros(B, dtype=torch.float32,
                                      device=device))

    equiv_mask = both_valid & (dv_match_count == n_inputs) \
                            & (pc_match_count == n_inputs)

    return rewards, d_valid, equiv_mask


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
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    torch.set_float32_matmul_precision('high')

    # Load frozen encoder.
    encoder = T1Compressor(
        VOCAB_SIZE, args.d_model, args.n_heads, args.n_layers, args.d_out
    ).to(device)
    state = torch.load(args.model, map_location=device, weights_only=True)
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    decoder = Decoder(
        VOCAB_SIZE, args.dec_d_model, args.dec_n_heads,
        args.dec_n_layers, d_emb=args.d_out
    ).to(device)
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

        logits = decoder(vecs, dec_in, dec_pad)
        non_pad = ~dec_pad
        orig_lengths = non_pad.sum(dim=1)

        if device != 'cpu':
            torch.cuda.synchronize()

        # K-sample REINFORCE: sample K times, evaluate each,
        # use the best reward per instruction for lower variance.
        t_reward = time.time()
        K = args.k_samples
        dist = torch.distributions.Categorical(logits=logits)

        best_rewards = torch.full((B,), -1.0, device=device)
        best_log_probs = torch.zeros(B, device=device)
        best_valid = torch.zeros(B, dtype=torch.bool, device=device)
        best_equiv = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(K):
            sampled = dist.sample()
            log_probs = (dist.log_prob(sampled) * non_pad.float()).sum(dim=1)

            rewards_k, valid_k, equiv_k = gpu_reward(
                tok, sampled, orig_lengths, orig_lengths,
                device, n_inputs=args.n_reward_inputs)

            # Keep the sample with highest reward per instruction.
            better = rewards_k > best_rewards
            best_rewards = torch.where(better, rewards_k, best_rewards)
            best_log_probs = torch.where(better, log_probs, best_log_probs)
            best_valid = torch.where(better, valid_k, best_valid)
            best_equiv = torch.where(better, equiv_k, best_equiv)

        rewards = best_rewards.clamp(min=0)
        seq_log_prob = best_log_probs
        valid_mask = best_valid
        equiv_mask = best_equiv

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

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                tok_correct = ((pred == dec_tgt) & non_pad).sum().item()
                tok_total = non_pad.sum().item()
                tok_acc = tok_correct / tok_total

            enc_ms = (t_dec - t_enc) * 1000
            dec_ms = (t_reward - t_dec) * 1000
            back_ms = (t_done - t_back) * 1000
            print(f'step {step:>5d}  '
                  f'reward {mean_reward:.3f}  '
                  f'valid {valid_frac:.0%}  '
                  f'equiv {equiv_frac:.0%}  '
                  f'tok_acc {tok_acc:.0%}  '
                  f'enc {enc_ms:.0f} dec {dec_ms:.0f} '
                  f'rew {reward_ms:.0f} back {back_ms:.0f}  '
                  f'{ms_per:.0f}ms/step')

        if step >= args.n_steps:
            break

    print(f'\nDone: {step} steps')


if __name__ == '__main__':
    main()
