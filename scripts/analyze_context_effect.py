#!/usr/bin/env python3
"""Analyze how context affects T1 vectors for a fixed instruction.

Takes a trained window_size=2 model, embeds the same target instruction
with many different predecessors, and measures how the vectors cluster.

If context modulates smoothly: vectors cluster by data-flow relevance
(predecessors that write to the target's source registers produce
different vectors; predecessors that don't write to them produce
similar vectors).

If context causes combinatorial explosion: vectors scatter randomly
across S^127 with no structure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from emulator import Instruction, make_ctx, random_regs, run as run_instruction, SparseMemory
from tokenizer import encode_instruction, BOS, EOS, PAD, VOCAB_SIZE
from compressor.model import Compressor


def make_examples(target_instr, predecessors, n_inputs=4, seed=42):
    """Create windowed token sequences and per-instruction deltas.

    Returns:
        token_ids: (N, max_len) int64
        padding_mask: (N, max_len) bool
        deltas: (N, n_inputs, 32) int32
        has_dependency: (N,) bool — True if predecessor writes to
            a source register of the target
    """
    rng = np.random.default_rng(seed)
    ctx = make_ctx()

    target_toks = encode_instruction(target_instr)

    # Determine target's source registers.
    from emulator import R_TYPE, I_TYPE, LOAD_TYPE, STORE_TYPE
    op = target_instr.opcode
    if op in R_TYPE:
        src_regs = {target_instr.args[1], target_instr.args[2]}
    elif op in I_TYPE:
        src_regs = {target_instr.args[1]}
    elif op in LOAD_TYPE:
        src_regs = {target_instr.args[2]}
    elif op in STORE_TYPE:
        src_regs = {target_instr.args[0], target_instr.args[2]}
    else:
        src_regs = set()

    all_tokens = []
    all_deltas = []
    dep_flags = []

    for pred in predecessors:
        pred_toks = encode_instruction(pred)
        toks = [BOS] + pred_toks + target_toks + [EOS]
        all_tokens.append(toks)

        # Determine if predecessor writes to a source register.
        from emulator import B_TYPE
        if pred.opcode in STORE_TYPE or pred.opcode in B_TYPE:
            pred_dest = None
        else:
            pred_dest = pred.args[0]
        dep_flags.append(pred_dest is not None and pred_dest in src_regs
                         and pred_dest != 0)

        # Execute predecessor then target on multiple input states.
        deltas_for_inputs = []
        for s in range(n_inputs):
            regs = random_regs(rng)
            mem = SparseMemory()

            # Run predecessor.
            state, _, mem = run_instruction(
                [pred], regs=regs, pc=0, rng=rng, _ctx=ctx,
                max_steps=1, mem=mem)

            # Snapshot before target.
            before = state.regs.copy()

            # Run target.
            state2, _, _ = run_instruction(
                [target_instr], regs=state.regs, pc=4, rng=rng, _ctx=ctx,
                max_steps=1, mem=mem)
            after = state2.regs

            deltas_for_inputs.append(after - before)

        all_deltas.append(np.stack(deltas_for_inputs, axis=0))  # (n_inputs, 32)

    # Pad tokens.
    max_len = max(len(t) for t in all_tokens)
    N = len(all_tokens)
    token_ids = np.full((N, max_len), PAD, dtype=np.int64)
    padding_mask = np.ones((N, max_len), dtype=np.bool_)
    for j, t in enumerate(all_tokens):
        token_ids[j, :len(t)] = t
        padding_mask[j, :len(t)] = False

    deltas = np.stack(all_deltas, axis=0)  # (N, n_inputs, 32)
    has_dep = np.array(dep_flags)

    return token_ids, padding_mask, deltas, has_dep


def generate_predecessors(target_instr, n=200, seed=0):
    """Generate predecessors: half with data-flow dependency, half without."""
    from emulator import R_TYPE, I_TYPE, LOAD_TYPE, STORE_TYPE
    rng = np.random.default_rng(seed)
    from datagen.seqgen import _data_flow_instruction

    op = target_instr.opcode
    if op in R_TYPE:
        src_regs = {target_instr.args[1], target_instr.args[2]}
    elif op in I_TYPE:
        src_regs = {target_instr.args[1]}
    elif op in LOAD_TYPE:
        src_regs = {target_instr.args[2]}
    elif op in STORE_TYPE:
        src_regs = {target_instr.args[0], target_instr.args[2]}
    else:
        src_regs = set()

    src_regs.discard(0)
    preds = []

    # Half: write to a source register (dependency).
    dep_ops = ['ADD', 'SUB', 'XOR', 'OR', 'AND', 'ADDI', 'ORI', 'ANDI']
    src_list = sorted(src_regs) if src_regs else [1]
    for i in range(n // 2):
        rd = src_list[i % len(src_list)]
        op = str(rng.choice(dep_ops))
        rs1 = int(rng.choice(range(0, 32)))
        if op in ('ADDI', 'ORI', 'ANDI'):
            imm = int(rng.integers(-2048, 2048))
            preds.append(Instruction(op, rd, rs1, imm))
        else:
            rs2 = int(rng.choice(range(0, 32)))
            preds.append(Instruction(op, rd, rs1, rs2))

    # Half: write to a NON-source register (no dependency).
    non_src = [r for r in range(1, 32) if r not in src_regs]
    if not non_src:
        non_src = [31]  # fallback
    for i in range(n - n // 2):
        rd = non_src[i % len(non_src)]
        op = str(rng.choice(dep_ops))
        rs1 = int(rng.choice(range(0, 32)))
        if op in ('ADDI', 'ORI', 'ANDI'):
            imm = int(rng.integers(-2048, 2048))
            preds.append(Instruction(op, rd, rs1, imm))
        else:
            rs2 = int(rng.choice(range(0, 32)))
            preds.append(Instruction(op, rd, rs1, rs2))

    return preds


def analyze(model, target_instr, predecessors, device, n_inputs=8):
    """Embed target with each predecessor and analyze vector distribution."""
    token_ids, padding_mask, deltas, has_dep = make_examples(
        target_instr, predecessors, n_inputs=n_inputs)

    tok = torch.from_numpy(token_ids).to(device)
    pad = torch.from_numpy(padding_mask).to(device)

    model.eval()
    with torch.no_grad():
        vecs = model(tok, pad).cpu().numpy()  # (N, d_out)

    # Pairwise cosine similarity.
    # Vectors are L2-normalized, so dot product = cosine similarity.
    sims = vecs @ vecs.T  # (N, N)

    dep_mask = has_dep
    nodep_mask = ~has_dep

    n_dep = dep_mask.sum()
    n_nodep = nodep_mask.sum()

    print(f'Target: {target_instr.opcode} {target_instr.args}')
    print(f'Predecessors: {len(predecessors)} total '
          f'({n_dep} with dependency, {n_nodep} without)')
    print()

    # Within-group similarity.
    if n_dep >= 2:
        dep_sims = sims[np.ix_(dep_mask, dep_mask)]
        tri = np.triu_indices(n_dep, k=1)
        dep_within = dep_sims[tri].mean()
        print(f'Cosine sim (dep ↔ dep):       {dep_within:.4f}')
    if n_nodep >= 2:
        nodep_sims = sims[np.ix_(nodep_mask, nodep_mask)]
        tri = np.triu_indices(n_nodep, k=1)
        nodep_within = nodep_sims[tri].mean()
        print(f'Cosine sim (nodep ↔ nodep):   {nodep_within:.4f}')
    if n_dep >= 1 and n_nodep >= 1:
        cross_sims = sims[np.ix_(dep_mask, nodep_mask)]
        cross_mean = cross_sims.mean()
        print(f'Cosine sim (dep ↔ nodep):     {cross_mean:.4f}')

    # Overall spread.
    all_tri = np.triu_indices(len(vecs), k=1)
    overall_mean = sims[all_tri].mean()
    overall_std = sims[all_tri].std()
    print(f'Cosine sim (all pairs):       {overall_mean:.4f} ± {overall_std:.4f}')

    # How many effective clusters? Use singular values of centered vectors.
    centered = vecs - vecs.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, compute_uv=False)
    sv_norm = sv / sv.sum()
    n_90 = int((np.cumsum(sv_norm) < 0.90).sum()) + 1
    n_99 = int((np.cumsum(sv_norm) < 0.99).sum()) + 1
    print(f'Singular value dimensions:    {n_90} (90% var), {n_99} (99% var)')

    # Compare exec distance: dep vs nodep
    from compressor.train import exec_distance
    delta_t = torch.from_numpy(deltas)
    ed = exec_distance(delta_t, 'cpu').numpy()
    if n_dep >= 1 and n_nodep >= 1:
        dep_ed = ed[np.ix_(dep_mask, dep_mask)]
        nodep_ed = ed[np.ix_(nodep_mask, nodep_mask)]
        cross_ed = ed[np.ix_(dep_mask, nodep_mask)]
        dep_tri = np.triu_indices(n_dep, k=1)
        nodep_tri = np.triu_indices(n_nodep, k=1)
        print()
        print(f'Exec dist (dep ↔ dep):        {dep_ed[dep_tri].mean():.4f}')
        print(f'Exec dist (nodep ↔ nodep):    {nodep_ed[nodep_tri].mean():.4f}')
        print(f'Exec dist (dep ↔ nodep):      {cross_ed.mean():.4f}')

    print()


def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Analyze context effect on T1 vectors.')
    p.add_argument('--model', required=True, help='Path to model.pt')
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-out', type=int, default=128)
    p.add_argument('--device', default='auto')
    p.add_argument('--n-preds', type=int, default=200)
    args = p.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    model = Compressor(VOCAB_SIZE, args.d_model, args.n_heads,
                       args.n_layers, args.d_out)
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model = model.to(device)

    # Test several target instructions.
    targets = [
        ('R-type ALU',   Instruction('ADD', 5, 3, 7)),
        ('I-type shift', Instruction('SLLI', 5, 3, 1)),
        ('Branch',       Instruction('BEQ', 5, 6, 8)),
        ('SUB-self',     Instruction('SUB', 5, 5, 5)),
        ('Load',         Instruction('LW', 5, 0, 3)),
    ]

    for label, target in targets:
        print(f'{"=" * 60}')
        print(f'{label}')
        print(f'{"=" * 60}')
        preds = generate_predecessors(target, n=args.n_preds)
        analyze(model, target, preds, device)


if __name__ == '__main__':
    main()
