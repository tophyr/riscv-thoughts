#!/usr/bin/env python3
"""Evaluate a trained T0→T1 compressor model.

Usage:
    python scripts/eval_compressor.py runs/20260404_203355
    python scripts/eval_compressor.py runs/20260404_203355 --n-inputs 256
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

from compressor.model import T1Compressor
from compressor.train import tokenize_batch, compute_exec_distance_matrix
from emulator import Instruction
from datagen import generate_sequence


def load_model(run_dir):
    """Load model from a run directory."""
    run_dir = Path(run_dir)

    hparams_path = run_dir / 'hparams.json'
    config_path = run_dir / 'config.json'

    if hparams_path.exists():
        with open(hparams_path) as f:
            hp = json.load(f)
        model = T1Compressor(
            vocab_size=hp.get('vocab_size', 89),
            d_model=hp['d_model'],
            n_heads=hp['n_heads'],
            n_layers=hp['n_layers'],
            d_out=hp['d_out'],
        )
    elif config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        model = T1Compressor(
            vocab_size=cfg['vocab_size'],
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            d_out=cfg['d_out'],
        )
    else:
        raise FileNotFoundError(f'No hparams.json or config.json in {run_dir}')

    model.load_state_dict(
        torch.load(run_dir / 'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    return model


def get_t1(model, instructions):
    ids, mask = tokenize_batch(instructions)
    with torch.no_grad():
        return model(ids, mask)


def t1_dist(model, a, b):
    return torch.cdist(get_t1(model, [a]), get_t1(model, [b])).item()


def eval_equivalence(model):
    """Test execution-equivalent instruction pairs."""
    pairs = [
        (Instruction('ADD', 5, 3, 3), Instruction('SLLI', 5, 3, 1),
         "ADD x5,x3,x3 vs SLLI x5,x3,1"),
        (Instruction('ADDI', 5, 0, 0), Instruction('AND', 5, 0, 0),
         "ADDI x5,x0,0 vs AND x5,x0,x0"),
        (Instruction('ADD', 5, 3, 0), Instruction('ADDI', 5, 3, 0),
         "ADD x5,x3,x0 vs ADDI x5,x3,0"),
        (Instruction('OR', 5, 3, 0), Instruction('ADDI', 5, 3, 0),
         "OR  x5,x3,x0 vs ADDI x5,x3,0"),
        (Instruction('XOR', 5, 3, 0), Instruction('ADDI', 5, 3, 0),
         "XOR x5,x3,x0 vs ADDI x5,x3,0"),
        (Instruction('SUB', 5, 3, 3), Instruction('XOR', 5, 3, 3),
         "SUB x5,x3,x3 vs XOR x5,x3,x3 (both=0)"),
        (Instruction('ADD', 5, 3, 7), Instruction('ADD', 5, 7, 3),
         "ADD x5,x3,x7 vs ADD x5,x7,x3 (commutative)"),
    ]
    print("=" * 60)
    print("Execution equivalence (should be ~0)")
    print("=" * 60)
    for a, b, desc in pairs:
        print(f"  {desc}: {t1_dist(model, a, b):.4f}")


def eval_immediates(model):
    """Test immediate value scaling."""
    print()
    print("=" * 60)
    print("Immediate scaling")
    print("=" * 60)
    base = Instruction('ADDI', 5, 0, 10)
    for imm in [11, 15, 50, 100, 500, 2000]:
        d = t1_dist(model, base, Instruction('ADDI', 5, 0, imm))
        print(f"  ADDI x5,x0,10 vs ADDI x5,x0,{imm:4d}: {d:.4f}")


def eval_registers(model):
    """Test register sensitivity."""
    print()
    print("=" * 60)
    print("Register effects")
    print("=" * 60)
    base = Instruction('ADD', 5, 3, 7)
    tests = [
        (Instruction('ADD', 5, 3, 7), "Identical"),
        (Instruction('ADD', 5, 3, 8), "One src differs"),
        (Instruction('ADD', 5, 1, 2), "Both srcs differ"),
        (Instruction('ADD', 6, 3, 7), "Dest differs"),
        (Instruction('ADD', 10, 1, 2), "All regs differ"),
    ]
    for instr, desc in tests:
        print(f"  {desc:25s} {t1_dist(model, base, instr):.4f}")


def eval_opcodes(model):
    """Test opcode distance matrix."""
    print()
    print("=" * 60)
    print("Opcode groups (same dest & sources: x5 = f(x3, x7))")
    print("=" * 60)
    ops = ['ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU']
    instrs = {op: Instruction(op, 5, 3, 7) for op in ops}
    print("       ", "  ".join(f"{op:5s}" for op in ops))
    for n1 in ops:
        row = []
        for n2 in ops:
            if n1 == n2:
                row.append("  -  ")
            else:
                row.append(f"{t1_dist(model, instrs[n1], instrs[n2]):5.2f}")
        print(f"  {n1:5s}", "  ".join(row))


def eval_correlation(model, n_inputs):
    """Test correlation on held-out random instructions."""
    print()
    print("=" * 60)
    print(f"Correlation on held-out data ({n_inputs} input states)")
    print("=" * 60)
    rng = np.random.default_rng(99)
    instrs = [generate_sequence(1, rng)[0] for _ in range(64)]
    exec_dists = compute_exec_distance_matrix(
        instrs, n_inputs=n_inputs, rng=np.random.default_rng(99))
    t1_vecs = get_t1(model, instrs)
    t1_dists = torch.cdist(t1_vecs, t1_vecs).numpy()
    idx = np.triu_indices(64, k=1)
    r_pearson, _ = pearsonr(exec_dists[idx], t1_dists[idx])
    r_spearman, _ = spearmanr(exec_dists[idx], t1_dists[idx])
    print(f"  Pearson r:  {r_pearson:.4f}")
    print(f"  Spearman r: {r_spearman:.4f}")


def eval_loss_trajectory(run_dir):
    """Print loss trajectory summary."""
    losses_path = Path(run_dir) / 'losses.json'
    if not losses_path.exists():
        return
    with open(losses_path) as f:
        losses = json.load(f)

    print()
    print("=" * 60)
    print("Loss trajectory")
    print("=" * 60)
    print(f"  Steps: {len(losses)}")
    print(f"  Best:  {min(losses):.5f}")
    print(f"  Final: {losses[-1]:.5f}")
    print(f"  Mean last 1000: {np.mean(losses[-1000:]):.5f}")

    # Sample points
    n = len(losses)
    sample_points = [0, n//10, n//4, n//2, 3*n//4, n-1]
    for i in sorted(set(sample_points)):
        print(f"    step {i:7d}: {losses[i]:.5f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained T0→T1 compressor.')
    parser.add_argument('run_dir', type=str, help='Path to run directory')
    parser.add_argument('--n-inputs', type=int, default=256,
                        help='Random input states for correlation test')
    args = parser.parse_args()

    model = load_model(args.run_dir)
    print(f"Loaded model from {args.run_dir}")
    print()

    eval_loss_trajectory(args.run_dir)
    eval_equivalence(model)
    eval_immediates(model)
    eval_registers(model)
    eval_opcodes(model)
    eval_correlation(model, args.n_inputs)


if __name__ == '__main__':
    main()
