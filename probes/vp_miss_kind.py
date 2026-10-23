#!/usr/bin/env python3
"""What did the vp head compute when it was wrong?

For every failing (row, output slot, anchor) triple, the head's predicted value
is compared against alternative computations of the slot's writing
instruction, all evaluated in the chunk's canonical frame at that anchor. The
first alternative it equals names the failure:

  operands swapped    — the writer with its two sources exchanged
  composition dropped — the writer reading its sources' values from BEFORE the
                        chunk, i.e. ignoring the other instruction's result
  operand replaced    — one source replaced by another chunk input, x0, or the
                        other instruction's result
  wrong opcode        — another opcode of the same type, same operands
  another slot's value— the target of a different output slot of the row
  one bit flipped     — exactly one bit wrong
  carry run           — off by ±2^k across several bits: a carry or borrow
                        dropped or added
  n bits wrong        — none of the above, binned by Hamming distance

For the bit and carry classes it also records the lowest wrong bit, and whether
that bit's operands make it a carry-sensitive position.

An alternative is checked only where it differs from the true target. The
emulator here proposes candidate values for the head's own output; it never
supplies an answer (METRICS.md, "Do NOT attribute a residual by reconstructing
the instructions and executing them" is about scoring, which this does not do).
The calibration line must read 0 mismatches: it re-executes every chunk in the
canonical frame and compares with the oracle's targets.

Usage: vp_miss_kind.py <t2_dir> <t1_path> [n_batches] [corpus]
"""
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              vp_head_predict, vp_canonical_in_vals)
from datagen import RVT_FORMAT, Batch, AUX_CE_IGNORE
from datagen.compare import (_BOUNDARY_VALUES, _canonical_state, _input_order, _CANON_POSITIONS,
                             _source_regs, _written_regs)
from emulator import R_TYPE
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 300
CORPUS = sys.argv[4] if len(sys.argv) > 4 else \
    f'/tmp/vp_miss_kind_{os.getpid()}.rvt'
M = 0xFFFFFFFF

if not os.path.exists(CORPUS):
    subprocess.run(
        f'python -m datagen --rule "cap=2" --twins 0 --batch-size 128 '
        f'--n-states 16 --heldout-anchors --config configs/instr_alu_writers.json '
        f'--count {NB} > {CORPUS}.part', shell=True, check=True,
        stderr=subprocess.DEVNULL)
    os.replace(f'{CORPUS}.part', CORPUS)


def s32(x):
    x &= M
    return x - (1 << 32) if x >> 31 else x


def alu(op, a, b):
    a, b = a & M, b & M
    sh = b & 31
    return {
        'ADD': a + b, 'SUB': a - b, 'XOR': a ^ b, 'OR': a | b, 'AND': a & b,
        'SLL': a << sh, 'SRL': a >> sh, 'SRA': s32(a) >> sh,
        'SLT': int(s32(a) < s32(b)), 'SLTU': int(a < b),
    }[_BASE.get(op, op)] & M


_BASE = {'ADDI': 'ADD', 'SLTI': 'SLT', 'SLTIU': 'SLTU', 'XORI': 'XOR',
         'ORI': 'OR', 'ANDI': 'AND', 'SLLI': 'SLL', 'SRLI': 'SRL', 'SRAI': 'SRA'}
_R_OPS = ('ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU')


def vclass(v):
    sv = s32(v)
    if sv == 0:
        return 'zero'
    if v in _BOUNDARY_VALUES or (-v & M) in _BOUNDARY_VALUES:
        return 'boundary'
    if v & (v - 1) == 0:
        return 'pow2'
    if abs(sv) < 4096:
        return 'small'
    if abs(sv) < (1 << 20):
        return 'mid'
    return 'large'


def operands(ins):
    """(op name for alu, source regs, immediate or None)."""
    if ins.opcode in R_TYPE:
        return ins.opcode, list(ins.args[1:3]), None
    return ins.opcode, [ins.args[1]], ins.args[2]


def step(ins, regs, srcs=None, op=None, pre=None):
    """Value `ins` writes, reading `regs` (or `pre` for its sources)."""
    o, s, imm = operands(ins)
    s = srcs if srcs is not None else s
    src_regs = pre if pre is not None else regs
    a = src_regs[s[0]] & M if s[0] else 0
    if imm is None:
        b = src_regs[s[1]] & M if s[1] else 0
    else:
        b = imm & M
    return alu(op or o, a, b)


def run(chunk, state, k=None, **kw):
    """Execute the chunk on one register file; `kw` alters instruction k."""
    regs = [int(v) & M for v in state]
    regs[0] = 0
    pre = list(regs)
    for j, ins in enumerate(chunk):
        if kw.get('pre_mode') and j == k:
            v = step(ins, regs, pre=pre)
        else:
            v = step(ins, regs, **({} if j != k else
                                   {x: y for x, y in kw.items() if x != 'pre_mode'}))
        for r in _written_regs(ins):
            regs[r] = v
    return regs


dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)

kinds = defaultdict(Counter)        # bucket -> Counter(kind)
spread = {d: {'rows': Counter(), 'anchors': Counter()} for d in (1, 2)}
row_id = 0
ctx23 = defaultdict(lambda: [0, 0])
vpairs = defaultdict(lambda: [0, 0, 0])
vcls = defaultdict(lambda: [0, 0])
vrel = defaultdict(lambda: [0, 0])
d1pos = defaultdict(lambda: [0, 0])
d1ctx = defaultdict(lambda: [0, 0])   # depth-1 context -> [misses, triples]
cof = Counter()                      # (depth-1 slot misses, depth-2 slot misses) at one anchor
shape2 = defaultdict(lambda: [0, 0])     # depth-2 shape -> [hits, misses]
lowbit = defaultdict(Counter)       # opcode+kind -> Counter(lowest wrong bit // 4)
calib_bad = calib_n = 0
n_fail = 0
with open(CORPUS, 'rb') as f:
    for batch in RVT_FORMAT.reader(f, Batch):
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        nch, mni = batch.instr_lens.shape
        with torch.no_grad():
            ct1, c0, c1, co, cact = _t2_assemble(t1, split, nch, mni, dev,
                                                 route='binding')
            ess = t2.encode(ct1, c0, c1, co, cact)
            anchors = torch.from_numpy(batch.anchors).to(dev)
            in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
            pred = vp_head_predict(t2, ess, vp_canonical_in_vals(anchors, in_slot))
            bits = (pred[..., :32] > 0).long().cpu().numpy()   # (B, A, S, 32)
        pv = (bits << np.arange(32)).sum(-1) & M               # (B, A, S)
        out_slot, out_regs = batch.out_slot_regs, batch.out_regs.astype(np.int64)
        for b in np.nonzero(batch.out_regs_valid)[0]:
            row_id += 1
            toks, p, chunk = batch.tokens[b].tolist(), 0, []
            for _ in range(int((batch.instr_lens[b] > 0).sum())):
                ins, p = decode_instruction(toks, p)
                chunk.append(ins)
            canon = _canonical_state(batch.anchors[b].astype(np.int64),
                                     _input_order(chunk), _CANON_POSITIONS)
            writer = {}
            for k, ins in enumerate(chunk):
                for r in _written_regs(ins):
                    writer[r] = k
            slots = [int(r) for r in out_slot[b] if r != AUX_CE_IGNORE]

            def sdepth(r):
                k = writer[r]
                e = {x for j in range(k) for x in _written_regs(chunk[j])}
                return 2 if any(x in e for x in operands(chunk[k])[1] if x) else 1
            depths = [sdepth(r) for r in slots]
            order = _input_order(chunk)
            for a in range(canon.shape[0]):
                miss = [int(pv[b, a, i]) != (int(out_regs[b, a, r]) & M)
                        for i, r in enumerate(slots)]
                for i, r in enumerate(slots):
                    if depths[i] != 1:
                        continue
                    op1 = chunk[writer[r]].opcode
                    others = [j for j in range(len(slots)) if j != i]
                    ctx = ('only output slot' if not others else
                           'beside a depth-2 slot' if any(depths[j] == 2 for j in others)
                           else 'beside another depth-1 slot')
                    for key in (ctx, f'{op1}: {ctx}'):
                        d1ctx[key][1] += 1; d1ctx[key][0] += miss[i]
                    pos = ','.join(str(order.index(x) + 1) if x in order else 'x0'
                                   for x in operands(chunk[writer[r]])[1])
                    if (op1 in ('SUB', 'ADD') and pos in ('2,3', '1,2', '3,4')
                            and writer[r] == 1):
                        first = chunk[0]
                        fo, fs, _ = operands(first)
                        x = operands(chunk[1])[1][0]          # the slot's rs1
                        y = operands(chunk[1])[1][1]
                        shared = x if pos == '2,3' else None
                        role = ('-' if shared is None else
                                'rs1' if fs[0] == shared else 'rs2')
                        ftype = 'R' if fo in _R_OPS else 'I'
                        for key in (f'{op1} {pos} | first instr {ftype}-type, shared input is its {role}',
                                    f'{op1} {pos} | first instr {fo}'):
                            ctx23[key][1] += 1; ctx23[key][0] += miss[i]
                    sx = operands(chunk[writer[r]])[1]
                    if op1 in ('SUB', 'ADD', 'SLL', 'SRL', 'XOR') and len(sx) == 2 and all(sx):
                        va, vb = int(canon[a][sx[0]]) & M, int(canon[a][sx[1]]) & M
                        dd = s32(va - vb)
                        rel = ('equal' if va == vb else
                               'within 3' if abs(dd) <= 3 else
                               'negated, within 3' if abs(s32(va + vb)) <= 3 else
                               'within 2^12' if abs(dd) < 4096 else
                               'unrelated')
                        if rel == 'unrelated' and op1 in ('SUB', 'ADD'):
                            key = f'{op1}: {vclass(va)} {"-" if op1 == "SUB" else "+"} {vclass(vb)}'
                            vcls[key][1] += 1; vcls[key][0] += miss[i]
                            pr = vpairs[(op1, va, vb)]
                            pr[0] += miss[i]; pr[1] += 1
                            pr[2] = int(pv[b, a, i])
                        vrel[f'{op1}: inputs {rel}'][1] += 1
                        vrel[f'{op1}: inputs {rel}'][0] += miss[i]
                    if op1 in ('SUB', 'ADD') and len(operands(chunk[writer[r]])[1]) == 2:
                        d1pos[f'{op1} reads input positions {pos}'][1] += 1
                        d1pos[f'{op1} reads input positions {pos}'][0] += miss[i]
                    if others and any(depths[j] == 2 for j in others):
                        o = others[0]
                        cof[(miss[i], miss[o])] += 1
                for i, r in enumerate(slots):
                    if depths[i] == 2 and chunk[writer[r]].opcode in ('SUB', 'ADD'):
                        fresh = [x for x in operands(chunk[writer[r]])[1]
                                 if x and x in order and writer.get(x, 9) >= writer[r]]
                        pos = ','.join(str(order.index(x) + 1) for x in fresh) or 'none'
                        key = f'{chunk[writer[r]].opcode} depth 2, fresh input at position {pos}'
                        d1pos[key][1] += 1; d1pos[key][0] += miss[i]
                regs = run(chunk, canon[a])
                for si, r in enumerate(slots):
                    tgt = int(out_regs[b, a, r]) & M
                    calib_n += 1
                    calib_bad += regs[r] != tgt
                    got = int(pv[b, a, si])
                    k = writer[r]
                    earlier = {x for j in range(k) for x in _written_regs(chunk[j])}
                    srcs_k = operands(chunk[k])[1]
                    if any(x in earlier for x in srcs_k if x):
                        prod = chunk[k - 1].opcode
                        pos = '+'.join(str(i) for i, x in enumerate(srcs_k)
                                       if x and x in earlier)
                        other = [x for x in srcs_k if x and x not in earlier]
                        shape2[f'{prod} -> {chunk[k].opcode} [composed operand {pos}'
                               f'{", other a chunk input" if other else ""}]'
                               ][0 if got == tgt else 1] += 1
                    if got == tgt:
                        continue
                    n_fail += 1
                    k = writer[r]
                    ins = chunk[k]
                    op, srcs, imm = operands(ins)
                    earlier = {x for j in range(k) for x in _written_regs(chunk[j])}
                    depth = 2 if any(s in earlier for s in srcs if s) else 1
                    alts = []
                    if imm is None and srcs[0] != srcs[1]:
                        alts.append(('operands swapped',
                                     run(chunk, canon[a], k, srcs=srcs[::-1])[r]))
                    if depth == 2:
                        alts.append(('composition dropped',
                                     run(chunk, canon[a], k, pre_mode=True)[r]))
                    pool = set(_input_order(chunk)) | {0} | earlier
                    for i in range(len(srcs)):
                        for x in pool - {srcs[i]}:
                            ns = list(srcs); ns[i] = x
                            alts.append(('operand replaced',
                                         run(chunk, canon[a], k, srcs=ns)[r]))
                    for o2 in (_R_OPS if imm is None else tuple(_BASE)):
                        if o2 != op:
                            alts.append(('wrong opcode',
                                         run(chunk, canon[a], k, op=o2)[r]))
                    for sj, r2 in enumerate(slots):
                        if r2 != r:
                            alts.append(("another slot's value",
                                         int(out_regs[b, a, r2]) & M))
                    kind = next((nm for nm, v in alts if v != tgt and v == got), None)
                    if kind is None:
                        d = (got - tgt) & M
                        h = bin(got ^ tgt).count('1')
                        if h == 1:
                            kind = 'one bit flipped'
                        elif d & (d - 1) == 0 or (-d & M) & ((-d & M) - 1) == 0:
                            kind = 'carry run'
                        else:
                            kind = ('2 bits wrong' if h <= 2 else
                                    '3-8 bits wrong' if h <= 8 else
                                    '9+ bits wrong')
                    for key in ('ALL', f'depth {depth}', f'{op} depth {depth}'):
                        kinds[key][kind] += 1
                    spread[depth]['rows'][(row_id, si)] += 1
                    spread[depth]['anchors'][a] += 1
                    if kind in ('one bit flipped', 'carry run'):
                        low = ((got ^ tgt) & -(got ^ tgt)).bit_length() - 1
                        lowbit[f'{op} {kind}'][low // 4] += 1

print(f'calibration: {calib_bad} mismatches between re-execution and the '
      f'oracle over {calib_n} triples (must be 0)')
print(f'failing triples: {n_fail}')
order = ['operands swapped', 'composition dropped', 'operand replaced',
         'wrong opcode', "another slot's value", 'one bit flipped', 'carry run',
         '2 bits wrong', '3-8 bits wrong', '9+ bits wrong']
keys = ['ALL', 'depth 1', 'depth 2'] + sorted(
    (k for k in kinds if k[0].isupper() and k != 'ALL'),
    key=lambda k: -sum(kinds[k].values()))[:14]
print(f'{"bucket":<16} {"fails":>6}  ' + '  '.join(f'{o[:14]:>14}' for o in order))
for key in keys:
    c = kinds[key]; n = sum(c.values())
    print(f'{key:<16} {n:>6}  ' +
          '  '.join(f'{100 * c[o] / max(n, 1):13.1f}%' for o in order))

print('\nlowest wrong bit, in groups of four (0 = bits 0-3), for the commonest')
for key in sorted(lowbit, key=lambda k: -sum(lowbit[k].values()))[:8]:
    c = lowbit[key]; n = sum(c.values())
    print(f'{key:<26} {n:>6}  ' + ' '.join(f'{100 * c[g] / n:5.1f}' for g in range(8)))

print('\nhow concentrated the failures are: per failing (row, slot), how many of')
print('its anchors fail; and the share of failures on each held-out anchor')
for d in (1, 2):
    per = Counter(spread[d]['rows'].values())
    n = sum(per.values())
    print(f'depth {d}: {n} failing slots; anchors failing per slot ' +
          ' '.join(f'{k}:{100 * per[k] / n:.0f}%' for k in sorted(per)))
    an = spread[d]['anchors']; tot = sum(an.values())
    print(f'         per anchor ' + ' '.join(f'{100 * an[a] / tot:.0f}' for a in sorted(an)))

print('\ndepth-2 slots by shape (producer -> writer, which writer operand is the')
print('producer\'s result), by share of all depth-2 failures')
tot = sum(m for _, m in shape2.values())
print(f'{"shape":<58} {"triples":>8} {"exact":>8} {"share":>7}')
for k, (h, m) in sorted(shape2.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f'{k:<58} {h + m:>8} {100 * h / (h + m):7.2f}% {100 * m / tot:6.1f}%')

agg = defaultdict(lambda: [0, 0])
for k, (h, m) in shape2.items():
    prod, rest = k.split(' -> ')
    wr = rest.split(' [')[0]
    three = 'other a chunk input' in k
    for key in (f'writer {wr}, {"also reads an input" if three else "reads only the result"}',
                f'{"R" if len(wr) <= 4 and not wr.endswith("I") or wr in ("SLTU",) else "I"}-type writer, '
                f'{"also reads an input" if three else "reads only the result"}'):
        agg[key][0] += h; agg[key][1] += m
print('\ndepth-2 slots aggregated by the writer')
for k, (h, m) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:16]:
    print(f'{k:<46} {h + m:>8} {100 * h / (h + m):7.2f}% {100 * m / tot:6.1f}%')

print('\ndepth-1 slots by what else the chunk outputs: exact, triples')
for k in sorted(d1ctx, key=lambda k: (k.count(':'), k)):
    if k.count(':') and k.split(':')[0] not in ('SUB', 'ADD', 'SLL', 'SRL', 'XOR'):
        continue
    m, n = d1ctx[k]
    print(f'  {k:<40} {100 * (1 - m / n):8.4f}%  of {n}')
n = sum(cof.values())
pd1 = (cof[(True, True)] + cof[(True, False)]) / n
pd2 = (cof[(True, True)] + cof[(False, True)]) / n
print(f'\nrows with a depth-1 and a depth-2 slot, per anchor ({n}):')
print(f'  P(depth-1 wrong) {100 * pd1:.3f}%   P(depth-2 wrong) {100 * pd2:.3f}%')
print(f'  P(both wrong) {100 * cof[(True, True)] / n:.3f}%  vs {100 * pd1 * pd2:.3f}% if independent')
print(f'  P(depth-1 wrong | depth-2 wrong) {100 * cof[(True, True)] / max(1, cof[(True, True)] + cof[(False, True)]):.3f}%'
      f'   | depth-2 right {100 * cof[(True, False)] / max(1, cof[(True, False)] + cof[(False, False)]):.3f}%')

print('\nby the canonical input positions the slot reads: exact, triples')
for k in sorted(d1pos):
    m, n = d1pos[k]
    if n >= 300:
        print(f'  {k:<48} {100 * (1 - m / n):8.3f}%  of {n}')

print('\nsecond-instruction ADD/SUB at depth 1, by the first instruction: exact, triples')
for k in sorted(ctx23):
    m, n = ctx23[k]
    if n >= 200:
        print(f'  {k:<60} {100 * (1 - m / n):8.3f}%  of {n:>6}  fails {m}')

print('\ndepth-1 slots by how the two input VALUES relate at the anchor: exact, triples, share of that opcode\'s depth-1 failures')
for op in ('SUB', 'ADD', 'SLL', 'SRL', 'XOR'):
    tf = sum(m for k, (m, n) in vrel.items() if k.startswith(op + ':'))
    for k in sorted(k for k in vrel if k.startswith(op + ':')):
        m, n = vrel[k]
        print(f'  {k:<32} {100 * (1 - m / n):8.3f}%  of {n:>7}  {100 * m / max(tf, 1):5.1f}% of fails')

print('\nSUB/ADD with unrelated inputs, by value class (a OP b): exact, triples, share of fails')
for op in ('SUB', 'ADD'):
    tf = sum(m for k, (m, n) in vcls.items() if k.startswith(op + ':'))
    for k, (m, n) in sorted(((k, v) for k, v in vcls.items() if k.startswith(op + ':')),
                            key=lambda kv: -kv[1][0])[:12]:
        print(f'  {k:<34} {100 * (1 - m / n):8.3f}%  of {n:>7}  {100 * m / max(tf, 1):5.1f}% of fails')

print('\nthe SUB/ADD input-value pairs carrying the most failures: a, b, true, last prediction')
for (op, va, vb), (m, n, last) in sorted(vpairs.items(), key=lambda kv: -kv[1][0])[:12]:
    t = alu(op, va, vb)
    print(f'  {op} {va:#010x} {vb:#010x} = {t:#010x}  pred {last:#010x} (xor {t ^ last:#010x})'
          f'  wrong {m}/{n}')
