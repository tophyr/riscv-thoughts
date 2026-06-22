#!/usr/bin/env python3
"""Held-out scorecard for a trained T2 checkpoint.

Falsification probes for the unified T2 vector — does it encode what it
should?

  - **API-signature accuracy**: live_in / live_out / pc_writes (BCE,
    threshold 0.5) and in/out_slot register-IDs decoded by argsort top-K of
    the score heads, plus a duplicate-prediction diagnostic. Tests that the
    vector recovers the chunk's interface (which regs read/written, does it
    branch, which registers in which slots). This decode is what the tool
    adds — the training loss uses ListMLE ranking, not a decode.
  - **Value-prediction RMSE**: the sqrt of the training vp MSE
    (t2_value_predict_loss) on held-out chunks, in compressed-space units
    (T1-mature reached ~0.33). Tests that the vector carries the chunk's
    I/O function.

Usage:
    eval_t2.py <t2_run_dir> --t1-model <t1_run>/encoder.pt --corpus <file>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from compressor.train import (
    _decode_chunk_instructions, _compute_chunk_out_regs,
    _split_to_per_instruction, t2_chunk_forward, t2_value_predict_loss,
)
from datagen import (
    RVT_FORMAT, Batch, padding_mask, N_REGS,
    make_anchor_states, MAX_OUTPUT_SLOTS, AUX_CE_IGNORE,
)
from emulator import Instruction
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import VOCAB_SIZE, PAD, encode_instruction


@torch.no_grad()
def _t2_encode_chunks(t1, t2, chunks, device):
    """Encode a list of chunks (each a list of Instructions) through
    T1 then T2. Returns (n_chunks, d_out) T2 vectors."""
    flat_tokens = []
    flat_chunk_idx = []
    flat_slot_idx = []
    chunk_lens = []
    for c_idx, instrs in enumerate(chunks):
        chunk_lens.append(len(instrs))
        for j, instr in enumerate(instrs):
            flat_tokens.append(encode_instruction(instr))
            flat_chunk_idx.append(c_idx)
            flat_slot_idx.append(j)

    N_instr = len(flat_tokens)
    if N_instr == 0:
        return torch.zeros(len(chunks), t2.d_out, device=device)
    max_tok = max(len(t) for t in flat_tokens)
    instr_tokens = np.full((N_instr, max_tok), PAD, dtype=np.int64)
    instr_pad = np.ones((N_instr, max_tok), dtype=bool)
    for i, t in enumerate(flat_tokens):
        instr_tokens[i, :len(t)] = t
        instr_pad[i, :len(t)] = False

    instr_tok_t = torch.from_numpy(instr_tokens).to(device)
    instr_pad_t = torch.from_numpy(instr_pad).to(device)
    t1_vecs = t1.encode(instr_tok_t, instr_pad_t)

    n_chunks = len(chunks)
    max_n_instrs = max(chunk_lens) if chunk_lens else 1
    chunk_t1 = torch.zeros(n_chunks, max_n_instrs, t1.d_out, device=device)
    chunk_pad = torch.ones(n_chunks, max_n_instrs, dtype=torch.bool, device=device)
    chunk_idx_t = torch.tensor(flat_chunk_idx, dtype=torch.long, device=device)
    slot_idx_t = torch.tensor(flat_slot_idx, dtype=torch.long, device=device)
    chunk_t1[chunk_idx_t, slot_idx_t] = t1_vecs
    for c_idx, L in enumerate(chunk_lens):
        chunk_pad[c_idx, :L] = False
    chunk_pad[:, 0] = False

    return t2.encode(chunk_t1, chunk_pad)


@torch.no_grad()
def measure_api_accuracy(t1, t2, batches, device):
    """BCE-style accuracy on live_in / live_out / pc_writes (threshold
    at 0.5), and CE-style top-1 accuracy on in/out_slot register IDs.
    """
    live_in_correct = 0
    live_in_total = 0
    live_out_correct = 0
    live_out_total = 0
    pc_correct = 0
    pc_total = 0
    in_slot_correct = 0
    in_slot_total = 0
    out_slot_correct = 0
    out_slot_total = 0
    # Duplicate-prediction diagnostic: independent per-slot heads can
    # predict the same register at multiple slots even when ground-truth
    # registers are all distinct. Count rows where the model's predicted
    # set of out_slot registers has duplicates, over rows where ground
    # truth has >=2 valid out_slots.
    out_slot_dup_rows = 0
    out_slot_multislot_rows = 0
    in_slot_dup_rows = 0
    in_slot_multislot_rows = 0

    for batch in batches:
        B, max_n_instrs = batch.instr_lens.shape
        t2_out = t2_chunk_forward(
            t1, t2, _split_to_per_instruction(batch), B, max_n_instrs, device)
        if t2_out is None:
            continue

        live_in = torch.from_numpy(batch.live_in_mask).to(device)
        live_out = torch.from_numpy(batch.live_out_mask).to(device)
        pc_writes = torch.from_numpy(batch.pc_writes).to(device)
        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(device)
        out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(device)

        active = live_in.any(dim=-1) | live_out.any(dim=-1)
        if not active.any():
            continue

        binding_dir = F.normalize(t2_out, dim=-1)

        li_pred = (torch.sigmoid(t2.live_in_head(binding_dir)) > 0.5)
        lo_pred = (torch.sigmoid(t2.live_out_head(binding_dir)) > 0.5)
        pc_pred = (torch.sigmoid(t2.pc_writes_head(binding_dir).squeeze(-1)) > 0.5)

        # Score-heads decode: argsort scores descending, take top-K as the
        # predicted slot ordering. Distinctness by construction (sort is a
        # total order on floats). ListMLE includes (active, inactive)
        # pairs so actives outscore inactives in expectation, making pure
        # sort decode work without coupling to live heads.
        in_scores = t2.in_score_head(binding_dir)   # (B, n_regs)
        out_scores = t2.out_score_head(binding_dir)
        in_sorted = in_scores.argsort(dim=-1, descending=True)
        out_sorted = out_scores.argsort(dim=-1, descending=True)
        in_preds_full = in_sorted[:, :t2.max_input_slots]   # (B, K_in)
        out_preds_full = out_sorted[:, :t2.max_output_slots]
        in_preds = [in_preds_full[:, k]
                    for k in range(t2.max_input_slots)]
        out_preds = [out_preds_full[:, k]
                     for k in range(t2.max_output_slots)]
        in_valids = [in_slot[:, k] != AUX_CE_IGNORE
                     for k in range(t2.max_input_slots)]
        out_valids = [out_slot[:, k] != AUX_CE_IGNORE
                      for k in range(t2.max_output_slots)]
        for k in range(t2.max_input_slots):
            if in_valids[k].any():
                in_slot_correct += int(
                    (in_preds[k][in_valids[k]]
                     == in_slot[:, k][in_valids[k]]).sum())
                in_slot_total += int(in_valids[k].sum())
        for k in range(t2.max_output_slots):
            if out_valids[k].any():
                out_slot_correct += int(
                    (out_preds[k][out_valids[k]]
                     == out_slot[:, k][out_valids[k]]).sum())
                out_slot_total += int(out_valids[k].sum())

        in_stack_pred = torch.stack(in_preds, dim=1)
        in_stack_valid = torch.stack(in_valids, dim=1)
        for b in range(in_stack_pred.shape[0]):
            v = in_stack_valid[b]
            if v.sum() >= 2:
                in_slot_multislot_rows += 1
                preds_b = in_stack_pred[b][v]
                if len(set(preds_b.tolist())) < len(preds_b):
                    in_slot_dup_rows += 1
        out_stack_pred = torch.stack(out_preds, dim=1)
        out_stack_valid = torch.stack(out_valids, dim=1)
        for b in range(out_stack_pred.shape[0]):
            v = out_stack_valid[b]
            if v.sum() >= 2:
                out_slot_multislot_rows += 1
                preds_b = out_stack_pred[b][v]
                if len(set(preds_b.tolist())) < len(preds_b):
                    out_slot_dup_rows += 1

        live_in_correct += int((li_pred[active] == live_in[active]).sum())
        live_in_total += int(active.sum() * N_REGS)
        live_out_correct += int((lo_pred[active] == live_out[active]).sum())
        live_out_total += int(active.sum() * N_REGS)
        pc_correct += int((pc_pred[active] == pc_writes[active]).sum())
        pc_total += int(active.sum())

    return {
        'live_in_acc': live_in_correct / max(1, live_in_total),
        'live_out_acc': live_out_correct / max(1, live_out_total),
        'pc_writes_acc': pc_correct / max(1, pc_total),
        'in_slot_acc': in_slot_correct / max(1, in_slot_total),
        'out_slot_acc': out_slot_correct / max(1, out_slot_total),
        'in_slot_dup_rate': in_slot_dup_rows / max(1, in_slot_multislot_rows),
        'out_slot_dup_rate': out_slot_dup_rows / max(1, out_slot_multislot_rows),
        'in_slot_multislot_rows': in_slot_multislot_rows,
        'out_slot_multislot_rows': out_slot_multislot_rows,
    }


@torch.no_grad()
def measure_value_prediction(t1, t2, batches, device, anchor_seed, n_anchors):
    """Held-out value-prediction RMSE — the sqrt of the training vp MSE
    (reuses t2_value_predict_loss). Compressed-space units; T1-mature
    reached ~0.33 at convergence."""
    anchor_np = make_anchor_states(n_anchors, anchor_seed)
    anchor_states_t = torch.from_numpy(anchor_np).to(device)

    sq_err_total = 0.0
    n_total = 0
    for batch in batches:
        B, max_n_instrs = batch.instr_lens.shape
        t2_out = t2_chunk_forward(
            t1, t2, _split_to_per_instruction(batch), B, max_n_instrs, device)
        if t2_out is None:
            continue
        in_slot_t = torch.from_numpy(
            batch.in_slot_regs.astype(np.int64)).to(device)
        out_slot_t = torch.from_numpy(
            batch.out_slot_regs.astype(np.int64)).to(device)
        out_regs_np, vp_mask_np = _compute_chunk_out_regs(batch, anchor_np)
        out_regs_t = torch.from_numpy(out_regs_np).to(device)
        mask_t = torch.from_numpy(vp_mask_np).to(device)

        mse, n = t2_value_predict_loss(
            t2, t2_out, anchor_states_t, in_slot_t, out_slot_t,
            out_regs_t, mask_t, return_count=True)
        sq_err_total += float(mse) * n
        n_total += n

    if n_total == 0:
        return {}
    mse = sq_err_total / n_total
    return {'value_pred_mse': mse, 'value_pred_rmse': float(np.sqrt(mse)),
            'n_slot_anchor': n_total}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('t2_run_dir', type=Path)
    p.add_argument('--t1-model', type=Path, required=True,
                   help='Path to T1 encoder.pt')
    p.add_argument('--corpus', type=Path, required=True,
                   help='RVT corpus file for eval batches')
    p.add_argument('--max-batches', type=int, default=20)
    p.add_argument('--anchor-seed', type=int, default=0)
    p.add_argument('--n-anchor-states', type=int, default=8)
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    device = resolve_device(args.device)
    t1 = load_frozen_encoder(args.t1_model, device)
    t2, t2_hp = load_t2(args.t2_run_dir, t1, device)

    # Load batches
    with open(args.corpus, 'rb') as f:
        reader = RVT_FORMAT.reader(f, Batch)
        batches = []
        for i, b in enumerate(reader):
            if i >= args.max_batches:
                break
            batches.append(b)
    print(f'Loaded {len(batches)} batches')

    print('\n== API signature accuracy ==')
    api = measure_api_accuracy(t1, t2, batches, device)
    for k, v in api.items():
        print(f'  {k:>15s}: {v:.4f}')

    print('\n== Value-prediction ==')
    vp = measure_value_prediction(
        t1, t2, batches, device, args.anchor_seed, args.n_anchor_states)
    for k, v in vp.items():
        if isinstance(v, float):
            print(f'  {k:>20s}: {v:.4f}')
        else:
            print(f'  {k:>20s}: {v}')


if __name__ == '__main__':
    main()
