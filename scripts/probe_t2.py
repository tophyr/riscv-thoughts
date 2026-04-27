#!/usr/bin/env python3
"""Probes for the trained T2 encoder.

Three probes:

  validity:    does ||T2|| separate valid from invalid chunks?
               (analog of probe_validity.py for T1)
  equivalence: do execution-equivalent chunk pairs collapse in T2
               direction space, while non-equivalent pairs stay apart?
  reg_effect:  is the per-register pair-MSE training signal
               actually informative? Compares the distribution of
               exec_dist_per_reg targets against the reg_effect_head's
               predictions per pair, to distinguish "encoder
               underfitting" from "loss target itself degenerate."

Intended as a sanity check after T2 training. If validity is bad,
the magnitude-as-validity geometry didn't transfer to T2 — T2 gates
won't work. If equivalence is bad, T2 is encoding instruction-set
specifics rather than functional equivalence — composition won't work.
If reg_effect is bad, the diagnostic narrows: target-degenerate means
the loss formulation can't teach what we want; flat-prediction means
the encoder is undercooked and more / better training would help.

Usage:
    probe_t2.py validity --t1-encoder <T1.pt> --t2-encoder <T2.pt>
    probe_t2.py equivalence --t1-encoder <T1.pt> --t2-encoder <T2.pt>
    probe_t2.py reg_effect --t1-encoder <T1.pt> --t2-encoder <T2.pt>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from compressor.model import T1Compressor
from compressor.t2_model import T2Compressor
from compressor.train import load_checkpoint
from compressor.chunker import encode_chunkbatch
from datagen.seqgen import (
    SequenceBatch, _encode_with_instr_idx, execute_sequence,
    produce_sequence_batch,
)
from datagen.chunkgen import (
    chunk_rvs, augment_chunkbatch_with_invalid, ChunkBatch,
    INVALID_SPANNING, INVALID_MULTI, INVALID_OVERLONG,
)
from emulator import Instruction, make_ctx, random_regs
from tokenizer import VOCAB_SIZE, PAD
from scripts._common import resolve_device


def _load_t1(path, device):
    t1 = T1Compressor(VOCAB_SIZE, d_model=128, n_heads=4, n_layers=2,
                     d_out=64, max_window=32).to(device)
    t1.load_state_dict(load_checkpoint(path, device), strict=False)
    t1.eval()
    return t1


def _load_t2(path, device, args):
    t2 = T2Compressor(d_in=args.t1_d_out, d_model=args.d_model,
                      n_heads=args.n_heads, n_layers=args.n_layers,
                      d_out=args.d_out,
                      max_chunk_len=args.max_chunk_len).to(device)
    t2.load_state_dict(load_checkpoint(path, device), strict=False)
    t2.eval()
    return t2


def _hand_rvs(sequences, n_inputs=4, seed=0):
    """Build an RVS batch from a list of instruction lists."""
    rng = np.random.default_rng(seed)
    encoded = [_encode_with_instr_idx(instrs) for instrs in sequences]
    max_tokens = max(len(toks) for toks, _ in encoded)
    max_instrs = max(len(s) for s in sequences)
    B = len(sequences)

    token_ids = np.full((B, max_tokens), PAD, dtype=np.int64)
    padding_mask = np.ones((B, max_tokens), dtype=np.bool_)
    token_instr_idx = np.full((B, max_tokens), -1, dtype=np.int32)
    n_instructions = np.array([len(s) for s in sequences], dtype=np.int32)

    for b, (toks, idx) in enumerate(encoded):
        n = len(toks)
        token_ids[b, :n] = toks
        padding_mask[b, :n] = False
        token_instr_idx[b, :n] = idx

    per_instr_regs = np.zeros((B, max_instrs + 1, n_inputs, 32),
                              dtype=np.int32)
    per_instr_pcs = np.zeros((B, max_instrs + 1, n_inputs),
                             dtype=np.int64)
    ctx = make_ctx()
    for s in range(n_inputs):
        regs = random_regs(rng)
        pc = int(rng.integers(0, 1024)) * 4
        for b, instrs in enumerate(sequences):
            regs_snap, pc_snap = execute_sequence(instrs, regs, pc, rng, ctx)
            n = len(instrs)
            per_instr_regs[b, :n + 1, s, :] = regs_snap
            per_instr_pcs[b, :n + 1, s] = pc_snap

    return SequenceBatch(
        token_ids=token_ids,
        padding_mask=padding_mask,
        token_instr_idx=token_instr_idx,
        n_instructions=n_instructions,
        per_instr_regs=per_instr_regs,
        per_instr_pcs=per_instr_pcs,
    )


def _encode_chunks(chunkbatch, t1, t2, device):
    """Run a ChunkBatch through T1 then T2 to get T2 vectors."""
    t2_in = encode_chunkbatch(chunkbatch, t1, device=device)
    with torch.no_grad():
        t2_vecs = t2(t2_in.chunk_emissions, t2_in.chunk_lens)
    return t2_vecs, t2_in.valid_mask, t2_in.chunk_type


# ---------------------------------------------------------------------------
# Validity probe
# ---------------------------------------------------------------------------

def run_validity_probe(args):
    device = resolve_device(args.device)
    t1 = _load_t1(args.t1_encoder, device)
    t2 = _load_t2(args.t2_encoder, device, args)

    rng = np.random.default_rng(args.seed)
    print(f'device={device} n_per_class={args.n_per_class}')

    # Generate a large RVS batch, chunk, then augment.
    rvs = produce_sequence_batch(
        batch_size=args.n_per_class * 2,
        n_inputs=4, max_block_len=10, rng=rng)
    valid_cb = chunk_rvs(rvs, max_chunk_len=16,
                        storage_max_chunk_len=args.max_chunk_len)
    n_valid_avail = valid_cb.token_ids.shape[0]
    if n_valid_avail < args.n_per_class:
        print(f'WARN: only {n_valid_avail} valid chunks generated; '
              f'requested {args.n_per_class}')

    # Cap valid set to n_per_class.
    n_valid = min(args.n_per_class, n_valid_avail)
    valid_cb = ChunkBatch(
        token_ids=valid_cb.token_ids[:n_valid],
        instr_pad=valid_cb.instr_pad[:n_valid],
        chunk_lens=valid_cb.chunk_lens[:n_valid],
        valid_mask=valid_cb.valid_mask[:n_valid],
        chunk_type=valid_cb.chunk_type[:n_valid],
        reg_delta=valid_cb.reg_delta[:n_valid],
    )

    # Generate invalid chunks via augmentation, separately for each type.
    type_classes = {
        'spanning': {'spanning': 1.0},
        'multi':    {'multi': 1.0},
        'overlong': {'overlong': 1.0},
    }
    aug_per_type = {}
    for type_name, weights in type_classes.items():
        # invalidity_rate r: n_invalid = round(n_valid * r/(1-r)).
        # Want n_invalid = n_per_class → r = n_per_class/(n_valid + n_per_class)
        r = args.n_per_class / (n_valid + args.n_per_class)
        aug = augment_chunkbatch_with_invalid(
            valid_cb, invalidity_rate=r,
            type_weights=weights,
            storage_max_chunk_len=args.max_chunk_len,
            rng=np.random.default_rng(args.seed + hash(type_name) % 10000))
        # Take only the invalid rows.
        inv_mask = ~aug.valid_mask
        aug_per_type[type_name] = ChunkBatch(
            token_ids=aug.token_ids[inv_mask],
            instr_pad=aug.instr_pad[inv_mask],
            chunk_lens=aug.chunk_lens[inv_mask],
            valid_mask=aug.valid_mask[inv_mask],
            chunk_type=aug.chunk_type[inv_mask],
            reg_delta=aug.reg_delta[inv_mask],
        )

    # Concat all chunks; track class labels.
    all_classes = ['valid'] + list(type_classes.keys())
    chunk_batches = [valid_cb] + [aug_per_type[c] for c in type_classes]
    class_ids = []
    for cid, cb in enumerate(chunk_batches):
        class_ids.extend([cid] * cb.token_ids.shape[0])
    class_ids = np.array(class_ids)

    # Concat into one ChunkBatch.
    combined = ChunkBatch(
        token_ids=np.concatenate([cb.token_ids for cb in chunk_batches]),
        instr_pad=np.concatenate([cb.instr_pad for cb in chunk_batches]),
        chunk_lens=np.concatenate([cb.chunk_lens for cb in chunk_batches]),
        valid_mask=np.concatenate([cb.valid_mask for cb in chunk_batches]),
        chunk_type=np.concatenate([cb.chunk_type for cb in chunk_batches]),
        reg_delta=np.concatenate([cb.reg_delta for cb in chunk_batches]),
    )
    N = combined.token_ids.shape[0]
    print(f'Built {N} chunks across {len(all_classes)} classes: '
          f'{all_classes}')

    # Encode through T1 then T2.
    t2_vecs, valid_mask_t, chunk_type_t = _encode_chunks(
        combined, t1, t2, device)
    norms = t2_vecs.norm(dim=-1).cpu()
    is_valid = combined.valid_mask  # True for valid_cb rows

    # Per-class magnitude statistics.
    print('\n--- Per-class ||T2|| ---')
    for cid, name in enumerate(all_classes):
        mask = class_ids == cid
        if not mask.any():
            continue
        m = norms[torch.from_numpy(mask)]
        print(f'  {name:9s}  n={int(mask.sum())}  '
              f'mean={m.mean():.3f}  std={m.std():.3f}  '
              f'min={m.min():.3f}  max={m.max():.3f}')

    # Magnitude-threshold accuracy.
    print('\n--- Magnitude-threshold (||T2|| > 0.5) ---')
    pred_valid = (norms > 0.5).numpy()
    acc = (pred_valid == is_valid).mean()
    print(f'Overall accuracy: {acc:.3f}')
    print(f'Majority baseline: {max(is_valid.mean(), 1 - is_valid.mean()):.3f}')
    for cid, name in enumerate(all_classes):
        mask = class_ids == cid
        if not mask.any():
            continue
        true_v = is_valid[mask]
        pred_v = pred_valid[mask]
        per_acc = (pred_v == true_v).mean()
        pred_pos = pred_v.mean()
        print(f'  {name:9s}  acc={per_acc:.3f}  '
              f'pred_positive={pred_pos:.3f}')

    # Linear probe on raw T2.
    print('\n--- Linear probe on raw T2 ---')
    X = t2_vecs.cpu()
    y = torch.from_numpy(is_valid).float()
    perm = torch.randperm(N)
    n_tr = int(0.8 * N)
    idx_tr, idx_va = perm[:n_tr], perm[n_tr:]
    probe = torch.nn.Linear(args.d_out, 1)
    opt = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(300):
        probe.train()
        logits = probe(X[idx_tr]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y[idx_tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        pred = (probe(X[idx_va]).squeeze(-1) > 0).float()
        lp_acc = (pred == y[idx_va]).float().mean().item()
    print(f'Linear probe val acc: {lp_acc:.3f}')


# ---------------------------------------------------------------------------
# Equivalence probe
# ---------------------------------------------------------------------------

# Pairs of chunks that are execution-equivalent. Each entry yields two
# instruction lists that compute the same register-state delta.
# All pairs are length-preserving so they fit T2's chunk shape.

def _pick_distinct(rng, n, lo=1, hi=32):
    """Pick n distinct register indices from [lo, hi)."""
    return list(rng.choice(np.arange(lo, hi), size=n, replace=False))


def _equiv_commutative_swap(rng):
    """Commutative operand swap on a 2-instruction-then-branch chunk."""
    rd1, rs1, rs2, rs3, rd2, rs_b = _pick_distinct(rng, 6)
    op1 = str(rng.choice(['ADD', 'AND', 'OR', 'XOR']))   # all commutative
    op2 = str(rng.choice(['ADD', 'AND', 'OR', 'XOR']))
    chunk_a = [
        Instruction(op1, rd1, rs1, rs2),
        Instruction(op2, rd2, rd1, rs3),
        Instruction('BEQ', rd2, rs_b, 8),
    ]
    chunk_b = [
        Instruction(op1, rd1, rs2, rs1),       # swap rs1<->rs2
        Instruction(op2, rd2, rs3, rd1),       # swap rs1<->rs2
        Instruction('BEQ', rd2, rs_b, 8),
    ]
    return chunk_a, chunk_b, 'commutative_swap'


def _equiv_independent_reorder(rng):
    """Two ALU ops with disjoint dst/src sets → reorderable."""
    rd1, rs1a, rs1b, rd2, rs2a, rs2b = _pick_distinct(rng, 6)
    op1 = str(rng.choice(['ADD', 'SUB', 'XOR', 'AND', 'OR']))
    op2 = str(rng.choice(['ADD', 'SUB', 'XOR', 'AND', 'OR']))
    chunk_a = [
        Instruction(op1, rd1, rs1a, rs1b),
        Instruction(op2, rd2, rs2a, rs2b),
        Instruction('BEQ', rd1, rd2, 8),
    ]
    chunk_b = [
        Instruction(op2, rd2, rs2a, rs2b),
        Instruction(op1, rd1, rs1a, rs1b),
        Instruction('BEQ', rd1, rd2, 8),
    ]
    return chunk_a, chunk_b, 'indep_reorder'


def _equiv_x0_writes_nop(rng):
    """First instruction writes to x0 (no-op). Different sources
    on the two chunks; same final state."""
    rs1a, rs1b, rs2a, rs2b, rd2, rs_b = _pick_distinct(rng, 6)
    chunk_a = [
        Instruction('ADD', 0, rs1a, rs1b),
        Instruction('ADD', rd2, rs2a, rs2b),
        Instruction('BEQ', rd2, rs_b, 8),
    ]
    chunk_b = [
        Instruction('OR', 0, rs1a, rs1b),     # different op, same nop
        Instruction('ADD', rd2, rs2a, rs2b),
        Instruction('BEQ', rd2, rs_b, 8),
    ]
    return chunk_a, chunk_b, 'x0_writes_nop'


def _equiv_double_to_shl1(rng):
    """ADD x,y,y == SLLI x,y,1 (double a value)."""
    rd1, rs, rd2, rs_other = _pick_distinct(rng, 4)
    rs_store = int(rng.integers(1, 32))
    chunk_a = [
        Instruction('ADD', rd1, rs, rs),
        Instruction('ADD', rd2, rd1, rs_other),
        Instruction('SW', rs_store, 0, rd2),
    ]
    chunk_b = [
        Instruction('SLLI', rd1, rs, 1),
        Instruction('ADD', rd2, rd1, rs_other),
        Instruction('SW', rs_store, 0, rd2),
    ]
    return chunk_a, chunk_b, 'double_to_shl1'


def _non_equivalent(rng):
    """Negative control: differ in one source register only.
    Should NOT collapse — the two chunks compute different functions."""
    rd1, rs1a, rs1b, rs1c, rd2, rs2c, rs_b = _pick_distinct(rng, 7)
    chunk_a = [
        Instruction('ADD', rd1, rs1a, rs1b),
        Instruction('XOR', rd2, rd1, rs2c),
        Instruction('BEQ', rd2, rs_b, 8),
    ]
    chunk_b = [
        Instruction('ADD', rd1, rs1a, rs1c),  # different rs2 in first instr
        Instruction('XOR', rd2, rd1, rs2c),
        Instruction('BEQ', rd2, rs_b, 8),
    ]
    return chunk_a, chunk_b, 'NON_EQUIV_(control)'


_EQUIVALENCE_GENERATORS = [
    _equiv_commutative_swap,
    _equiv_independent_reorder,
    _equiv_x0_writes_nop,
    _equiv_double_to_shl1,
    _non_equivalent,  # Negative control — should NOT collapse.
]


def run_equivalence_probe(args):
    device = resolve_device(args.device)
    t1 = _load_t1(args.t1_encoder, device)
    t2 = _load_t2(args.t2_encoder, device, args)

    rng = np.random.default_rng(args.seed)

    # Build pairs.
    pairs = []  # list of (chunk_a, chunk_b, name)
    for gen in _EQUIVALENCE_GENERATORS:
        for _ in range(args.n_per_class):
            pairs.append(gen(rng))

    # Each pair contributes two sequences. Build one big RVS.
    sequences = []
    pair_idx = []
    side = []  # 'a' or 'b'
    names = []
    for i, (a, b, name) in enumerate(pairs):
        sequences.append(a)
        pair_idx.append(i)
        side.append('a')
        names.append(name)
        sequences.append(b)
        pair_idx.append(i)
        side.append('b')
        names.append(name)

    rvs = _hand_rvs(sequences, n_inputs=4, seed=args.seed)
    cb = chunk_rvs(rvs, max_chunk_len=16,
                   storage_max_chunk_len=args.max_chunk_len)
    # Each sequence should produce exactly one chunk (terminated by
    # the last instruction). Defensive: drop sequences that produced
    # multiple chunks.
    chunks_per_seq = np.zeros(len(sequences), dtype=int)

    # The chunker emits chunks in source-sequence order, so we can
    # verify by checking chunk count matches sequence count.
    if cb.token_ids.shape[0] != len(sequences):
        print(f'WARN: {cb.token_ids.shape[0]} chunks for '
              f'{len(sequences)} sequences — some sequences chunked '
              f'into multiple. Equivalence probe assumes 1:1.')

    t2_in = encode_chunkbatch(cb, t1, device=device)
    with torch.no_grad():
        t2_vecs = t2(t2_in.chunk_emissions, t2_in.chunk_lens)
    # Normalize for direction-only comparison.
    norms = t2_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    t2_dir = (t2_vecs / norms).cpu()

    # Aggregate per-class within-pair distances.
    print('\n--- Within-pair directional distances (lower = more equivalent) ---')
    print(f'{"class":24s} {"n":>4s}  {"mean":>6s}  {"std":>6s}  '
          f'{"min":>6s}  {"max":>6s}')
    print('-' * 60)
    by_name = {}
    for i in range(len(pairs)):
        a_idx = i * 2
        b_idx = i * 2 + 1
        if a_idx >= t2_dir.shape[0] or b_idx >= t2_dir.shape[0]:
            continue
        d = (t2_dir[a_idx] - t2_dir[b_idx]).norm().item()
        name = pairs[i][2]
        by_name.setdefault(name, []).append(d)
    for name, ds in by_name.items():
        a = np.array(ds)
        print(f'{name:24s} {len(a):4d}  {a.mean():6.3f}  {a.std():6.3f}  '
              f'{a.min():6.3f}  {a.max():6.3f}')

    # Inter-class baseline: distance between random pairs across the
    # whole batch, regardless of equivalence class.
    n_random = min(500, t2_dir.shape[0] * (t2_dir.shape[0] - 1) // 2)
    rrng = np.random.default_rng(args.seed + 1)
    randomized = []
    for _ in range(n_random):
        i, j = rrng.integers(0, t2_dir.shape[0], size=2)
        if i == j:
            continue
        d = (t2_dir[i] - t2_dir[j]).norm().item()
        randomized.append(d)
    rs = np.array(randomized)
    print('\n--- Random pair baseline ---')
    print(f'{len(rs)} random pairs across batch: '
          f'mean={rs.mean():.3f}  std={rs.std():.3f}')


# ---------------------------------------------------------------------------
# reg_effect probe — diagnose the per-register pair-MSE loss
# ---------------------------------------------------------------------------

def run_reg_effect_probe(args):
    """Compare exec_dist_per_reg targets vs reg_effect_head predictions.

    For a batch of valid chunks, computes:
    - target_pair[a, b, r] = mean_s log1p(log1p(|delta_r[a,s] - delta_r[b,s]|))
      (the loss target the trainer optimizes against)
    - pred_pair[a, b, r] = |reg_effect_head(T2_dir[a])[r]
                          - reg_effect_head(T2_dir[b])[r]|
      (what the encoder actually produces)

    Reports:
    - Distribution of target_pair values per register (across all pairs).
      If most are ~0 with a long tail, the loss is dominated by trivial
      "predict zero" outcomes.
    - Distribution of pred_pair.
    - Per-register correlation between target_pair and pred_pair.
      If correlation is near zero, the encoder is collapsed (predictions
      don't track targets). If correlation is high, the loss plateau is
      a noise floor of the metric, not encoder underfitting.
    """
    device = resolve_device(args.device)
    t1 = _load_t1(args.t1_encoder, device)
    t2 = _load_t2(args.t2_encoder, device, args)

    rng = np.random.default_rng(args.seed)
    print(f'device={device} n_chunks={args.n_chunks}')

    # Generate a batch of valid chunks.
    rvs = produce_sequence_batch(
        batch_size=args.n_chunks * 2,
        n_inputs=args.n_inputs, max_block_len=10, rng=rng)
    valid_cb = chunk_rvs(rvs, max_chunk_len=16,
                        storage_max_chunk_len=args.max_chunk_len)
    n = min(args.n_chunks, valid_cb.token_ids.shape[0])
    cb = ChunkBatch(
        token_ids=valid_cb.token_ids[:n],
        instr_pad=valid_cb.instr_pad[:n],
        chunk_lens=valid_cb.chunk_lens[:n],
        valid_mask=valid_cb.valid_mask[:n],
        chunk_type=valid_cb.chunk_type[:n],
        reg_delta=valid_cb.reg_delta[:n],
    )
    print(f'Built {n} valid chunks, n_inputs={args.n_inputs}')

    # Encode through T1 + T2.
    t2_in = encode_chunkbatch(cb, t1, device=device)
    with torch.no_grad():
        t2_vecs = t2(t2_in.chunk_emissions, t2_in.chunk_lens)
        norms = t2_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        t2_dir = t2_vecs / norms
        # Apply the trained head (heads read normalized direction during
        # training; same here).
        head_out = t2.reg_effect_head(t2_dir)        # (n, 32)
    head_out = head_out.cpu().float()

    # Compute per-pair targets from reg_delta. (n, n_inputs, 32) -> pairs.
    rd = torch.from_numpy(cb.reg_delta).float()       # (n, n_inputs, 32)
    delta_pair = rd.unsqueeze(1) - rd.unsqueeze(0)    # (n, n, n_inputs, 32)
    target = torch.log1p(
        torch.log1p(delta_pair.abs())).mean(dim=2)    # (n, n, 32)

    pred = (head_out.unsqueeze(1) - head_out.unsqueeze(0)).abs()  # (n, n, 32)

    # Take upper triangle to avoid double counting and self-pairs.
    iu = torch.triu_indices(n, n, offset=1)
    target_pairs = target[iu[0], iu[1]]               # (n_pairs, 32)
    pred_pairs = pred[iu[0], iu[1]]                   # (n_pairs, 32)
    n_pairs = target_pairs.shape[0]
    print(f'\n{n_pairs} pairs × 32 registers = '
          f'{n_pairs * 32:,} pair-register samples')

    # Overall distribution of target values.
    print('\n--- Target distribution (exec_dist_per_reg) ---')
    flat_t = target_pairs.flatten()
    print(f'  mean={flat_t.mean():.4f}  std={flat_t.std():.4f}  '
          f'min={flat_t.min():.4f}  max={flat_t.max():.4f}')
    pcts = [10, 25, 50, 75, 90, 95, 99]
    qs = torch.quantile(flat_t, torch.tensor([p / 100 for p in pcts]))
    for p, q in zip(pcts, qs):
        print(f'  p{p}: {q:.4f}')
    n_zero = (flat_t < 1e-6).float().mean().item()
    print(f'  fraction near-zero (<1e-6): {n_zero:.3f}')

    # Per-register correlation and per-register loss.
    print('\n--- Per-register correlation (pred vs target) ---')
    correlations = []
    per_reg_loss = []
    target_means = []
    target_p99 = []
    for r in range(32):
        t_r = target_pairs[:, r]
        p_r = pred_pairs[:, r]
        # Pearson correlation.
        if t_r.std() < 1e-6:
            corr = float('nan')
        else:
            corr = ((t_r - t_r.mean()) * (p_r - p_r.mean())).mean() / (
                t_r.std() * p_r.std() + 1e-8)
            corr = corr.item()
        loss = (p_r - t_r).square().mean().item()
        correlations.append(corr)
        per_reg_loss.append(loss)
        target_means.append(t_r.mean().item())
        target_p99.append(torch.quantile(t_r, 0.99).item())

    # Group registers: x0 is special (writes are nops); x1..x31 are
    # general purpose. Show summary.
    correlations_t = torch.tensor(correlations)
    finite_mask = torch.isfinite(correlations_t)
    print(f'  registers with finite correlation: {int(finite_mask.sum())}/32')
    print(f'  mean correlation: {correlations_t[finite_mask].mean():.4f}')
    print(f'  median correlation: '
          f'{correlations_t[finite_mask].median():.4f}')
    print(f'  min correlation: {correlations_t[finite_mask].min():.4f}')
    print(f'  max correlation: {correlations_t[finite_mask].max():.4f}')

    # Per-register breakdown for r=0 (x0, special), and a few others.
    print('\n  per-register detail (r, target_mean, target_p99, '
          'pred_mean, corr, mse_loss):')
    print(f'    {"r":>3s} {"t_mean":>8s} {"t_p99":>8s} '
          f'{"p_mean":>8s} {"corr":>8s} {"mse":>8s}')
    pred_per_reg_mean = pred_pairs.mean(dim=0)
    interesting_regs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 31]
    for r in interesting_regs:
        c = correlations[r]
        c_str = f'{c:.4f}' if np.isfinite(c) else 'nan'
        print(f'    {r:>3d} {target_means[r]:>8.4f} {target_p99[r]:>8.4f} '
              f'{pred_per_reg_mean[r].item():>8.4f} '
              f'{c_str:>8s} {per_reg_loss[r]:>8.4f}')

    # Diagnosis hint.
    print('\n--- Diagnosis hint ---')
    median_corr = correlations_t[finite_mask].median().item()
    if median_corr < 0.1:
        print('  Low correlation across registers: the encoder\'s')
        print('  predictions DO NOT track the per-register targets.')
        print('  Diagnosis: encoder is in collapse mode (likely outputs')
        print('  near-constant reg_effect for all chunks). The loss')
        print('  plateau is the encoder failing to fit, not a metric')
        print('  noise floor. Fix: rebalance loss weights, lengthen')
        print('  training, or warm-up curriculum.')
    elif median_corr < 0.5:
        print('  Modest correlation: encoder partially fits but not')
        print('  cleanly. Likely room for improvement via training.')
    else:
        print('  High correlation: encoder IS tracking targets well.')
        print('  Plateau is closer to the noise floor of the metric.')
        print('  Fix would be a richer per-register loss formulation.')

    # Trivial-baseline check: what would per-register MSE be if the
    # encoder simply predicted zero for every pair?
    zero_pred_loss = target_pairs.square().mean().item()
    print(f'\n  Reference: MSE if pred=0 always: {zero_pred_loss:.4f}')
    actual_loss = (pred_pairs - target_pairs).square().mean().item()
    print(f'  Actual MSE on this batch: {actual_loss:.4f}')
    if actual_loss > 0.95 * zero_pred_loss:
        print('  -> Encoder achieves <5% improvement over predict-zero')
        print('     baseline. Effectively in collapse.')


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='probe', required=True)

    def _common(sp):
        sp.add_argument('--t1-encoder', required=True)
        sp.add_argument('--t2-encoder', required=True)
        sp.add_argument('--t1-d-out', type=int, default=64)
        sp.add_argument('--d-model', type=int, default=256)
        sp.add_argument('--n-heads', type=int, default=4)
        sp.add_argument('--n-layers', type=int, default=2)
        sp.add_argument('--d-out', type=int, default=256)
        sp.add_argument('--max-chunk-len', type=int, default=24)
        sp.add_argument('--seed', type=int, default=0)
        sp.add_argument('--device', default='auto')

    for name in ('validity', 'equivalence'):
        sp = sub.add_parser(name)
        _common(sp)
        sp.add_argument('--n-per-class', type=int, default=200)

    sp = sub.add_parser('reg_effect')
    _common(sp)
    sp.add_argument('--n-chunks', type=int, default=500,
                    help='Number of valid chunks to use for the probe. '
                         'Pair count is n_chunks*(n_chunks-1)/2 so 500 '
                         '~ 125K pairs.')
    sp.add_argument('--n-inputs', type=int, default=4,
                    help='Random input states for executing each chunk.')

    args = p.parse_args()
    if args.probe == 'validity':
        run_validity_probe(args)
    elif args.probe == 'equivalence':
        run_equivalence_probe(args)
    else:
        run_reg_effect_probe(args)


if __name__ == '__main__':
    main()
