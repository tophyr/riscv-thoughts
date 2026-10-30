#!/usr/bin/env python3
"""Score a tier on the register-function frame: value prediction per written
slot at held-out anchors, the emitted binding, and equivalence collapse over
the classes the structural signature defines.

  vp        — exact 32-bit match of the evaluator's prediction (read from
              the essence through each output's pointer), over every (row,
              output slot, anchor) triple of the corpus, with the failing
              count and the rate by the writing opcode and by the output's
              input count. The corpus must be generated with
              `datagen --heldout-anchors` so no model trained at its states.
              The leak check hands the evaluator ANOTHER chunk's essence with
              this output's pointer and inputs; it must fail if the function
              reaches the evaluator only through the essence.
  binding   — T1: emitted wiring exact per instruction; T2: emitted input and
              output lists exact per chunk (scripts/eval.py's definitions).
  collapse  — essence L2 (unit-normalized) on pairs of one register function:
              rename, reordered independent instructions, dead-code insertion,
              and commutative swaps split by whether a swapped register is
              read elsewhere or is another instruction's result; against an
              aliasing control and unrelated chunks. Classes the frame makes
              structural must read zero on every pair (float noise under 1e-6
              counts as zero: a pair of chunks of different lengths is pooled
              over a different number of picks). Every commutative swap is
              the op-cell's to learn — the encoder is told nothing about which
              operations commute — and reads what training made of it.
  partner groups (T1) — for each group of spellings of one function
              (`datagen.generate.PARTNER_GROUPS`), two spellings on the same
              registers, which must be ~0, against a spelling and the group's
              contrast, a near-miss computing something else.

Usage: frame_scorecard.py <t2_dir|-> <t1_path> <corpus.rvt> [n_batches] [n_pairs]
  `-` as t2_dir scores the T1 alone (the corpus must be `--rule single`).
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.eval import (encode_chunks_t2, t1_binding_accuracy,
                             t2_exact_binding)
from compressor.model import instruction_wiring
from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              _t1_wiring, vp_predict, _value_bits)
from datagen import RVT_FORMAT, Batch, AUX_CE_IGNORE, padding_mask
from datagen.signature import COMMUTATIVE
from datagen.generate import PARTNER_GROUPS, materialize, sample_binding
from emulator import Instruction
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction, encode_instruction, PAD

T2_DIR, T1_PATH, CORPUS = sys.argv[1], sys.argv[2], sys.argv[3]
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 100
N_PAIRS = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
T1_ONLY = T2_DIR == '-'
R_OPS = ['ADD', 'SUB', 'SLL', 'SLT', 'SLTU', 'XOR', 'SRL', 'SRA', 'OR', 'AND']
I_OPS = ['ADDI', 'SLTI', 'SLTIU', 'XORI', 'ORI', 'ANDI']
SH_OPS = ['SLLI', 'SRLI', 'SRAI']

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2 = None if T1_ONLY else load_t2(T2_DIR, t1, dev)[0]
comp = t1 if T1_ONLY else t2


def batches():
    with open(CORPUS, 'rb') as f:
        for i, b in enumerate(RVT_FORMAT.reader(f, Batch)):
            if i >= NB:
                break
            yield b


@torch.no_grad()
def state(batch):
    """The scored tier's emitted (state T, essence) for a batch, or None."""
    if T1_ONLY:
        tok = torch.from_numpy(batch.tokens).to(dev).long()
        pad = torch.from_numpy(padding_mask(batch)).to(dev)
        in0, in1, out = (torch.from_numpy(a).to(dev) for a in _t1_wiring(batch))
        return t1.encode_state(tok, pad, in0, in1, out)
    split = _split_to_per_instruction(batch)
    if split[0].shape[0] == 0:
        return None
    n_chunks, max_ni = batch.instr_lens.shape
    ct1, c0, c1, co, ca = _t2_assemble(t1, split, n_chunks, max_ni, dev)
    return t2.encode_state(ct1, c0, c1, co, ca)


def chunk_instrs(batch, b):
    toks, pos, out = batch.tokens[b].tolist(), 0, []
    for _ in range(int((batch.instr_lens[b] > 0).sum())):
        ins, pos = decode_instruction(toks, pos)
        out.append(ins)
    return out


# ---- vp ---------------------------------------------------------------------
hits = total = leak_hits = 0
by_op, by_nin = {}, {}
perm_rng = torch.Generator().manual_seed(0)
for batch in batches():
    st = state(batch)
    if st is None:
        continue
    T, essence = st
    anchors = torch.from_numpy(batch.anchors).to(dev)
    # A chunk of K instructions writes at most K registers, so output slots
    # past the grid's instruction count are AUX_CE_IGNORE; slicing them off is
    # the same score at a fraction of the evaluator's work.
    n_out = int(batch.instr_lens.shape[1])
    assert (batch.out_slot_regs[:, n_out:] == AUX_CE_IGNORE).all()
    out_slot = torch.from_numpy(batch.out_slot_regs[:, :n_out].astype(np.int64)).to(dev)
    out_inputs = torch.from_numpy(batch.out_inputs[:, :n_out].astype(np.int64)).to(dev)
    out_regs = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
    out_idx = out_slot.clamp(min=0)
    target_bits = _value_bits(out_regs.gather(
        -1, out_idx.unsqueeze(1).expand(-1, anchors.shape[1], -1))).bool()
    valid = (out_slot != AUX_CE_IGNORE) & torch.from_numpy(batch.out_regs_valid).to(dev).unsqueeze(-1)
    va = valid.cpu().numpy()
    # The leak check hands each row ANOTHER row's essence with its own pointer
    # and inputs: if the function reached the head any other way, it would pass.
    other = essence[torch.randperm(essence.shape[0], generator=perm_rng).to(dev)]
    with torch.no_grad():
        ex, lk = (((vp_predict(comp, T, e, anchors, out_slot, out_inputs)[..., :32] > 0)
                   == target_bits).all(-1).cpu().numpy() for e in (essence, other))   # (B,A,O)
    leak_hits += int((lk & va[:, None, :]).sum())
    for b in np.nonzero(batch.out_regs_valid)[0]:
        instrs = chunk_instrs(batch, b)
        writer = {ins.args[0]: ins.opcode for ins in instrs}
        for k in range(out_slot.shape[1]):
            if not va[b, k]:
                continue
            h = int(ex[b, :, k].sum()); n = ex.shape[1]
            hits += h; total += n
            op = writer.get(int(batch.out_slot_regs[b, k]), '?')
            nin = int((batch.out_inputs[b, k] != AUX_CE_IGNORE).sum())
            for d, key in ((by_op, op), (by_nin, nin)):
                f = d.setdefault(key, [0, 0]); f[0] += h; f[1] += n
print(f'\n  vp at held-out anchors: {100 * hits / max(total, 1):.4f}% exact  '
      f'({total - hits} failing of {total} triples)')
for name, d in (('writing opcode', by_op), ('input count', by_nin)):
    rows = sorted(d.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))
    print(f'    by {name}: ' + '  '.join(f'{k}:{100 * h / max(n, 1):.3f}%' for k, (h, n) in rows))
print(f'    leak check, another chunk\'s essence: {100 * leak_hits / max(total, 1):.4f}% exact '
      f'(must be far below the line above)')

# ---- binding / symmetry -----------------------------------------------------
if T1_ONLY:
    r = t1_binding_accuracy(t1, batches(), device=dev, max_batches=NB)
    print(f'  binding: emitted wiring exact on {100 * r["exact"]:.4f}% of {r["n_rows"]} instructions')
    bad = {k: v for k, v in r['by_form'].items() if v[0] < 1.0}
    if bad:
        print('    forms below 100%: ' + ', '.join(f'{k} {100 * a:.1f}% (n={n})' for k, (a, n) in bad.items()))
else:
    r = t2_exact_binding(t1, t2, batches(), device=dev, max_batches=NB)
    print(f'  binding: inputs+outputs exact on {100 * r["exact"]:.4f}% of {r["n_rows"]} chunks '
          f'(inputs {100 * r["exact_in"]:.3f}%, outputs {100 * r["exact_out"]:.3f}%)')

# ---- collapse ---------------------------------------------------------------
rng = np.random.default_rng(0)
L = 1 if T1_ONLY else int(batch.instr_lens.shape[1])


def rand_instr(rd=None):
    kind = rng.integers(3)
    rd = int(rng.integers(1, 32)) if rd is None else rd
    rs1, rs2 = (int(x) for x in rng.integers(1, 32, 2))
    if kind == 0:
        return Instruction(R_OPS[rng.integers(len(R_OPS))], rd, rs1, rs2)
    if kind == 1:
        return Instruction(I_OPS[rng.integers(len(I_OPS))], rd, rs1, int(rng.integers(-2048, 2048)))
    return Instruction(SH_OPS[rng.integers(len(SH_OPS))], rd, rs1, int(rng.integers(0, 32)))


def reads(ins):
    return [r for r in ([ins.args[1], ins.args[2]] if ins.opcode in R_OPS else [ins.args[1]]) if r]


def rename(c):
    perm = np.concatenate([[0], rng.permutation(31) + 1])
    out = []
    for ins in c:
        a = list(ins.args); a[0], a[1] = int(perm[a[0]]), int(perm[a[1]])
        if ins.opcode in R_OPS:
            a[2] = int(perm[a[2]])
        out.append(Instruction(ins.opcode, *a))
    return out


def reorder(c):
    for i in range(len(c) - 1):
        a, b = c[i], c[i + 1]
        if a.args[0] != b.args[0] and a.args[0] not in reads(b) and b.args[0] not in reads(a):
            d = list(c); d[i], d[i + 1] = d[i + 1], d[i]; return d
    return None


def comm_swap(c, shared):
    """A commutative op's operands swapped. `shared` selects swaps where a
    swapped register is read by another instruction or is an earlier
    instruction's result, or where neither holds. Both are the op-cell's to
    learn: with every role ordered the two leaves carry different colours."""
    idx = [i for i, ins in enumerate(c) if ins.opcode in COMMUTATIVE and ins.args[1] != ins.args[2]]
    for i in idx:
        others = [r for j, ins in enumerate(c) if j != i for r in reads(ins)]
        results = [ins.args[0] for ins in c[:i]]          # an operand that is an earlier result
        swapped = c[i].args[1:]
        is_shared = any(r in others or r in results for r in swapped)
        if is_shared == shared:
            d = list(c); a = d[i].args; d[i] = Instruction(d[i].opcode, a[0], a[2], a[1]); return d
    return None


def dead_insert(c):
    for j, ins in enumerate(c):
        rd = ins.args[0]
        if all(rd not in reads(k) for k in c[:j + 1]):
            return [rand_instr(rd=rd)] + list(c)
    return None


def ctrl_alias(c):
    idx = [i for i, ins in enumerate(c) if ins.opcode in R_OPS]
    if not idx:
        return None
    i = int(rng.choice(idx)); a = list(c[i].args)
    a[2] = int(rng.choice([r for r in range(1, 32) if r != a[1]])) if a[1] == a[2] else a[1]
    d = list(c); d[i] = Instruction(d[i].opcode, *a); return d


def pairs(fn):
    """N_PAIRS (base, variant) pairs of `fn`, or fewer if the class is rare,
    or none if the generator never produces one at this length."""
    out, tries = [], 0
    while len(out) < N_PAIRS and tries < 50 * N_PAIRS:
        tries += 1
        base = [rand_instr() for _ in range(L)]
        v = fn(base)
        if v is not None:
            out.append((base, v))
    return out


@torch.no_grad()
def encode(chunks, bs=512):
    out = []
    for i in range(0, len(chunks), bs):
        sub = chunks[i:i + bs]
        if T1_ONLY:
            toks = [encode_instruction(c[0]) for c in sub]
            Lt = max(len(t) for t in toks)
            tok = torch.full((len(sub), Lt), PAD, dtype=torch.long)
            pad = torch.ones(len(sub), Lt, dtype=torch.bool)
            for j, t in enumerate(toks):
                tok[j, :len(t)] = torch.tensor(t); pad[j, :len(t)] = False
            w = torch.tensor([instruction_wiring(c[0]) for c in sub])
            ess = t1.encode_state(tok.to(dev), pad.to(dev), w[:, 0].to(dev), w[:, 1].to(dev), w[:, 2].to(dev))[1]
        else:
            ess = encode_chunks_t2(t1, t2, sub, dev)
        out.append(F.normalize(ess.float(), dim=-1))
    return torch.cat(out)


classes = [('rename', rename, '0 by construction'),
           ('aliasing control', ctrl_alias, 'NOT equivalent — denominator')]
if L > 1:
    # At one instruction a commutative swap is a rename, so the swap classes
    # exist only from two instructions up.
    classes[1:1] = [('reorder independent instrs', reorder, 'MUST be 0'),
                    ('dead-code insertion', dead_insert, 'MUST be 0'),
                    ('commutative swap, fresh operands', lambda c: comm_swap(c, False), 'a rename of two leaves: 0 by construction'),
                    ('commutative swap, shared operand', lambda c: comm_swap(c, True), "the op-cell's")]
print(f'\n  collapse over the signature classes ({N_PAIRS} pairs each, len={L}):')
for name, fn, note in classes:
    ps = pairs(fn)
    if not ps:
        print(f'    {name:34s} no pairs at len={L}')
        continue
    d = (encode([p[0] for p in ps]) - encode([p[1] for p in ps])).norm(dim=-1)
    print(f'    {name:34s} mean={float(d.mean()):.4f}  max={float(d.max()):.2e}  '
          f'zero (<=1e-6): {int((d <= 1e-6).sum())}/{len(ps)}   ({note})')
X = [[rand_instr() for _ in range(L)] for _ in range(N_PAIRS)]
Y = [[rand_instr() for _ in range(L)] for _ in range(N_PAIRS)]
d = (encode(X) - encode(Y)).norm(dim=-1)
print(f'    {"unrelated chunks":34s} mean={float(d.mean()):.4f}')

if L == 1:
    # One register function spelled two ways must be one point; the group's
    # contrast spelling computes something else and must stay apart.
    print(f'\n  partner groups ({N_PAIRS} pairs each): two spellings of one function, '
          f'and a spelling against its contrast')
    for g in PARTNER_GROUPS:
        same, contrast = [], []
        for _ in range(N_PAIRS):
            b = sample_binding(g, rng)
            i, j = rng.choice(len(g.canonical), size=2, replace=False)
            same.append(([materialize(g.canonical[int(i)], b)], [materialize(g.canonical[int(j)], b)]))
            cs = [t for t in g.contrast if t.args[0] == 'rd']
            c = cs[int(rng.integers(len(cs)))]
            contrast.append(([materialize(g.canonical[int(i)], b)], [materialize(c, b)]))
        ds = (encode([p[0] for p in same]) - encode([p[1] for p in same])).norm(dim=-1)
        dc = (encode([p[0] for p in contrast]) - encode([p[1] for p in contrast])).norm(dim=-1)
        print(f'    {g.name:10s} equivalent mean={float(ds.mean()):.4f}  max={float(ds.max()):.2e}  '
              f'contrast mean={float(dc.mean()):.4f}  separation={float(dc.mean() / ds.mean()):.1f}x')
print()
