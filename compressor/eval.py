"""Evaluation functions for trained T0→T1 compressor models."""

import json
from pathlib import Path

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

from .model import T1Compressor
from .train import tokenize_batch, exec_distance
from emulator import Instruction


def load_model(run_dir):
    """Load a trained model from a run directory."""
    run_dir = Path(run_dir)

    hparams_path = run_dir / 'hparams.json'
    config_path = run_dir / 'config.json'

    if hparams_path.exists():
        with open(hparams_path) as f:
            hp = json.load(f)
    elif config_path.exists():
        with open(config_path) as f:
            hp = json.load(f)
    else:
        raise FileNotFoundError(f'No hparams.json or config.json in {run_dir}')

    model = T1Compressor(
        vocab_size=hp.get('vocab_size', 89),
        d_model=hp['d_model'],
        n_heads=hp['n_heads'],
        n_layers=hp['n_layers'],
        d_out=hp['d_out'],
    )
    model.load_state_dict(
        torch.load(run_dir / 'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    return model


def _device_of(model):
    return next(model.parameters()).device


def get_t1(model, instructions):
    """Get T1 vectors for a list of instructions."""
    device = _device_of(model)
    ids, mask = tokenize_batch(instructions, device=device)
    with torch.no_grad():
        t1, _, _ = model(ids, mask)
        return t1


def t1_dist(model, a, b):
    """T1 distance between two instructions."""
    return torch.cdist(get_t1(model, [a]), get_t1(model, [b])).item()


def eval_equivalence(model):
    """Test execution-equivalent instruction pairs across all types."""
    pairs = [
        # ALU equivalences
        (Instruction('ADD', 5, 3, 3), Instruction('SLLI', 5, 3, 1),
         "ADD x5,x3,x3 vs SLLI x5,x3,1 (double)"),
        (Instruction('ADD', 5, 3, 0), Instruction('ADDI', 5, 3, 0),
         "ADD x5,x3,x0 vs ADDI x5,x3,0 (identity)"),
        (Instruction('SUB', 5, 3, 3), Instruction('XOR', 5, 3, 3),
         "SUB x5,x3,x3 vs XOR x5,x3,x3 (zero)"),
        (Instruction('ADD', 5, 3, 7), Instruction('ADD', 5, 7, 3),
         "ADD x5,x3,x7 vs ADD x5,x7,x3 (commutative)"),
        # Shift equivalences
        (Instruction('SLLI', 5, 3, 0), Instruction('SRLI', 5, 3, 0),
         "SLLI x5,x3,0 vs SRLI x5,x3,0 (shift by 0)"),
        # Branch equivalences
        (Instruction('BEQ', 3, 3, 16), Instruction('BGE', 3, 3, 16),
         "BEQ x3,x3,16 vs BGE x3,x3,16 (always taken)"),
        (Instruction('BLT', 3, 3, 16), Instruction('BNE', 3, 3, 16),
         "BLT x3,x3,16 vs BNE x3,x3,16 (never taken)"),
        # Load width equivalences (same addr, same content if byte < 128)
        (Instruction('LB', 5, 0, 3), Instruction('LBU', 5, 0, 3),
         "LB x5,0(x3) vs LBU x5,0(x3) (same for byte<128)"),
    ]
    print("=" * 60)
    print("Execution equivalence (should be ~0)")
    print("=" * 60)
    for a, b, desc in pairs:
        print(f"  {desc}: {t1_dist(model, a, b):.4f}")


def eval_immediates(model):
    """Test immediate value scaling across instruction types."""
    print()
    print("=" * 60)
    print("Immediate scaling")
    print("=" * 60)

    print("  ADDI (data immediate):")
    base = Instruction('ADDI', 5, 0, 10)
    for imm in [11, 50, 500, 2000]:
        d = t1_dist(model, base, Instruction('ADDI', 5, 0, imm))
        print(f"    vs imm={imm:4d}: {d:.4f}")

    print("  BEQ (branch offset):")
    base = Instruction('BEQ', 1, 2, 8)
    for off in [10, 32, 128, 1024]:
        d = t1_dist(model, base, Instruction('BEQ', 1, 2, off))
        print(f"    vs off={off:4d}: {d:.4f}")

    print("  LUI (upper immediate):")
    base = Instruction('LUI', 5, 0x10)
    for imm in [0x11, 0x20, 0x100, 0x10000]:
        d = t1_dist(model, base, Instruction('LUI', 5, imm))
        print(f"    vs imm=0x{imm:05X}: {d:.4f}")


def eval_registers(model):
    """Test register sensitivity across instruction types."""
    print()
    print("=" * 60)
    print("Register effects")
    print("=" * 60)

    print("  ADD (R-type):")
    base = Instruction('ADD', 5, 3, 7)
    for instr, desc in [
        (Instruction('ADD', 5, 3, 7), "Identical"),
        (Instruction('ADD', 5, 3, 8), "One src differs"),
        (Instruction('ADD', 6, 3, 7), "Dest differs"),
        (Instruction('ADD', 20, 3, 7), "Dest high reg"),
    ]:
        print(f"    {desc:25s} {t1_dist(model, base, instr):.4f}")

    print("  LW (Load):")
    base = Instruction('LW', 5, 0, 3)  # rd=5, imm=0, rs1=3
    for instr, desc in [
        (Instruction('LW', 5, 0, 3), "Identical"),
        (Instruction('LW', 5, 0, 8), "Base reg differs"),
        (Instruction('LW', 6, 0, 3), "Dest differs"),
        (Instruction('LW', 5, 4, 3), "Offset differs"),
    ]:
        print(f"    {desc:25s} {t1_dist(model, base, instr):.4f}")


def eval_opcodes(model):
    """Test opcode distance matrix across all instruction types."""
    print()
    print("=" * 60)
    print("Opcode distances (representative instruction per type)")
    print("=" * 60)

    # One representative instruction per opcode, all using x5/x3/x7 where applicable.
    representatives = {
        'ADD':   Instruction('ADD', 5, 3, 7),
        'SUB':   Instruction('SUB', 5, 3, 7),
        'XOR':   Instruction('XOR', 5, 3, 7),
        'OR':    Instruction('OR', 5, 3, 7),
        'AND':   Instruction('AND', 5, 3, 7),
        'SLL':   Instruction('SLL', 5, 3, 7),
        'SRL':   Instruction('SRL', 5, 3, 7),
        'SRA':   Instruction('SRA', 5, 3, 7),
        'SLT':   Instruction('SLT', 5, 3, 7),
        'SLTU':  Instruction('SLTU', 5, 3, 7),
        'ADDI':  Instruction('ADDI', 5, 3, 42),
        'LW':    Instruction('LW', 5, 0, 3),
        'SW':    Instruction('SW', 7, 0, 3),
        'BEQ':   Instruction('BEQ', 3, 7, 8),
        'JAL':   Instruction('JAL', 5, 100),
        'LUI':   Instruction('LUI', 5, 0x12345),
    }
    ops = list(representatives.keys())

    # Print header
    print("       ", "  ".join(f"{op:5s}" for op in ops))
    for n1 in ops:
        row = []
        for n2 in ops:
            if n1 == n2:
                row.append("  -  ")
            else:
                row.append(f"{t1_dist(model, representatives[n1], representatives[n2]):5.1f}")
        print(f"  {n1:5s}", "  ".join(row))


def eval_type_clustering(model):
    """Test whether instruction types form natural clusters."""
    print()
    print("=" * 60)
    print("Cross-type distances")
    print("=" * 60)
    pairs = [
        (Instruction('ADD', 5, 3, 7), Instruction('ADDI', 5, 3, 42),
         "ADD vs ADDI (both ALU, same dest)"),
        (Instruction('ADD', 5, 3, 7), Instruction('LW', 5, 0, 3),
         "ADD vs LW (both write x5)"),
        (Instruction('ADD', 5, 3, 7), Instruction('SW', 7, 0, 3),
         "ADD vs SW (reg vs mem dest)"),
        (Instruction('ADD', 5, 3, 7), Instruction('BEQ', 3, 7, 8),
         "ADD vs BEQ (ALU vs branch)"),
        (Instruction('ADD', 5, 3, 7), Instruction('JAL', 5, 100),
         "ADD vs JAL (ALU vs jump)"),
        (Instruction('BEQ', 3, 7, 8), Instruction('BLT', 3, 7, 8),
         "BEQ vs BLT (both branches)"),
        (Instruction('LW', 5, 0, 3), Instruction('LB', 5, 0, 3),
         "LW vs LB (both loads)"),
        (Instruction('SW', 7, 0, 3), Instruction('SB', 7, 0, 3),
         "SW vs SB (both stores)"),
        (Instruction('JAL', 5, 100), Instruction('JALR', 5, 3, 0),
         "JAL vs JALR (both jumps)"),
        (Instruction('LUI', 5, 0x100), Instruction('AUIPC', 5, 0x100),
         "LUI vs AUIPC (upper imm)"),
    ]
    for a, b, desc in pairs:
        print(f"  {desc}: {t1_dist(model, a, b):.4f}")


def eval_correlation(model, batches):
    """Test correlation on held-out batches."""
    print()
    print("=" * 60)
    print(f"Correlation on held-out data ({len(batches)} batches)")
    print("=" * 60)
    device = _device_of(model)
    all_pearson = []
    all_spearman = []
    for batch in batches:
        ids = torch.from_numpy(batch.token_ids).to(device)
        mask = torch.from_numpy(batch.padding_mask).to(device)
        with torch.no_grad():
            t1, _, _ = model(ids, mask)
        t1_dists = torch.cdist(t1, t1).cpu().numpy()
        ed = exec_distance(batch.data_vals, batch.pc_vals, device,
                           compiled=False)
        B = t1.shape[0]
        idx = np.triu_indices(B, k=1)
        ed_np = ed.cpu().float().numpy()
        r_p, _ = pearsonr(ed_np[idx], t1_dists[idx])
        r_s, _ = spearmanr(ed_np[idx], t1_dists[idx])
        all_pearson.append(r_p)
        all_spearman.append(r_s)
    print(f"  Pearson r:  {np.mean(all_pearson):.4f}")
    print(f"  Spearman r: {np.mean(all_spearman):.4f}")


def eval_loss_trajectory(run_dir):
    """Print loss trajectory summary."""
    run_dir = Path(run_dir)
    losses_path = run_dir / 'losses.json'
    if not losses_path.exists():
        return
    with open(losses_path) as f:
        losses = json.load(f)

    hparams_path = run_dir / 'hparams.json'
    log_every = 1
    n_steps = len(losses)
    if hparams_path.exists():
        with open(hparams_path) as f:
            hp = json.load(f)
        log_every = hp.get('log_every', 1)
        n_steps = hp.get('steps_completed',
                         hp.get('n_steps', len(losses) * log_every))

    print()
    print("=" * 60)
    print("Loss trajectory")
    print("=" * 60)
    print(f"  Steps: {n_steps} ({len(losses)} logged)")
    print(f"  Best:  {min(losses):.5f}")
    print(f"  Final: {losses[-1]:.5f}")

    n = len(losses)
    sample_points = [0, n//10, n//4, n//2, 3*n//4, n-1]
    for i in sorted(set(sample_points)):
        step = i * log_every
        print(f"    step {step:7d}: {losses[i]:.5f}")


def evaluate(run_dir, batches=None):
    """Run all evaluations on a trained model.

    batches: list of Batch objects for correlation eval. If None,
    correlation eval is skipped.
    """
    model = load_model(run_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Loaded model from {run_dir} (on {device})")
    print()
    eval_loss_trajectory(run_dir)
    eval_equivalence(model)
    eval_immediates(model)
    eval_registers(model)
    eval_opcodes(model)
    eval_type_clustering(model)
    if batches:
        eval_correlation(model, batches)
