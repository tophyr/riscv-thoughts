"""How far the shipped behavioral distance target sits from d*.

For pairs of len-2 chunks from the writers corpus (chain rate 0.10), the
target the trainer builds from a batch (`register_distance_targets` over the
batch's `rf_*` fields) against the reference
distance d* (`datagen.ideal`). Three populations: unrelated natural pairs;
near-misses, a natural chunk against itself with one immediate moved or one
opcode changed; and partner pairs, equivalent by construction.

Reports, per population, the false zeros (target 0, d* > 0), the false
separations (target > 0, d* = 0), the Spearman correlation, and the target's
spread within bins of d*. No model is read.

    python probes/target_vs_ideal.py [n_pairs]
"""
import sys

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

import numpy as np
import torch

from compressor.train import register_distance_targets
from datagen.batch import build_targets, generate_chunks, pack_batch, _make_valid_chunk
from datagen.compare import draw_anchor_states
from datagen.generate import (build_opcode_table, length_cap, load_distribution,
                              sample_chunk_partner_pair)
from datagen.ideal import Behaviour, ideal_distance
from emulator import Instruction

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CONFIG = 'configs/instr_alu_writers_chain010_rewrites050_aim050.json'
_I_IMM = {'ADDI', 'SLTI', 'SLTIU', 'XORI', 'ORI', 'ANDI'}
_R = ['ADD', 'SUB', 'SLL', 'SLT', 'SLTU', 'XOR', 'SRL', 'SRA', 'OR', 'AND']


def shipped_targets(pairs):
    """The trainer's target for each (a, b) pair, both rows packed in one batch."""
    rows = [c for p in pairs for c in p]
    chunks = [_make_valid_chunk(c) for c in rows]
    rng = np.random.default_rng(0)
    aux, ov, oiv, orv = build_targets(
        chunks, lambda k: np.stack([draw_anchor_states(rng, 8) for _ in range(k)]), 2)
    b = pack_batch(chunks, target_B=len(chunks), target_max_tokens=18,
                   target_max_n_instrs=2, out_values=ov, in_role_values=oiv,
                   out_regs_valid=orv, aux_payload=aux)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = {'rf_wo_free': np.float32, 'rf_pos': np.int32, 'rf_wo': np.int32}
    fields = [torch.from_numpy(getattr(b, k).astype(dt.get(k, np.int64))) for k in (
        'rf_pos', 'rf_pos_zero', 'rf_pos_one', 'rf_wo', 'rf_wo_zero', 'rf_wo_one',
        'rf_wo_free', 'n_wo')]
    out = []
    for s in range(0, len(rows), 64):        # pairs stay within a slice
        D = register_distance_targets(*[f[s:s + 64].to(dev) for f in fields]).cpu()
        out += [float(D[k, k + 1]) for k in range(0, min(64, len(rows) - s), 2)]
    return np.array(out)



def near_miss(chunk, rng):
    c = list(chunk)
    k = int(rng.integers(len(c)))
    ins = c[k]
    if ins.opcode in _I_IMM:
        imm = int(ins.args[2]) + int(rng.choice([-1, 1]) * (1 if rng.random() < 0.5 else rng.integers(2, 512)))
        imm = max(-2048, min(2047, imm))
        c[k] = Instruction(ins.opcode, ins.args[0], ins.args[1], imm)
    elif ins.opcode in _R:
        c[k] = Instruction(str(rng.choice([o for o in _R if o != ins.opcode])), *ins.args)
    else:                                   # shift immediates
        c[k] = Instruction(ins.opcode, ins.args[0], ins.args[1], (int(ins.args[2]) + 1) % 32)
    return c


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def report(name, pairs):
    t = shipped_targets(pairs)
    d = np.array([ideal_distance(Behaviour(a), Behaviour(b)) for a, b in pairs])
    fz = (t == 0) & (d > 0)
    fs = (t > 0) & (d == 0)
    print(f'\n{name}: {len(pairs)} pairs')
    print(f'  d* = 0: {np.mean(d == 0):.1%}   target = 0: {np.mean(t == 0):.1%}')
    print(f'  false zeros (target 0, d* > 0): {fz.sum()}'
          + (f'  d* median {np.median(d[fz]):.4f} max {d[fz].max():.4f}' if fz.any() else ''))
    print(f'  false separations (target > 0, d* = 0): {fs.sum()}'
          + (f'  target median {np.median(t[fs]):.4f} max {t[fs].max():.4f}' if fs.any() else ''))
    pos = (d > 0) & (t > 0)
    if pos.sum() > 2:
        print(f'  Spearman(target, d*) over pairs both > 0: {spearman(t[pos], d[pos]):.3f}')
    for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 9)):
        m = (d > lo) & (d <= hi)
        if m.sum():
            q = np.quantile(t[m], [0.05, 0.5, 0.95])
            print(f'  d* in ({lo}, {hi}]: n={m.sum():5d}  target p5/p50/p95 '
                  f'{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}')
    return t, d


def main():
    rng = np.random.default_rng(12345)
    dist = load_distribution(CONFIG)
    gen = generate_chunks(length_cap(2), rng, opcode_table=build_opcode_table(dist),
                          dest_regs=list(range(1, 32)), chain_rate=dist['chain']['rate'])
    natural = [next(gen).instructions for _ in range(2 * N)]
    report('unrelated natural chunks', list(zip(natural[::2], natural[1::2])))
    report('near-misses', [(c, near_miss(c, rng)) for c in natural[:N]])
    report('partner pairs', [sample_chunk_partner_pair(rng, 2) for _ in range(N // 2)])


if __name__ == '__main__':
    main()
