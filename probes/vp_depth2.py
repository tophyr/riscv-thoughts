#!/usr/bin/env python3
"""At dataflow depth 2, which instruction's immediate is the one that fails?

A depth-2 output slot is written by an instruction whose operand is another
instruction's result: `writer(producer(a, b), c)`. The cell that carries the
remaining len=2 vp error is "the chunk has an immediate AND the slot is at depth
2", but a chunk-level axis does not say WHICH of the two instructions carries it,
and the two are different demands on the encoder:

  producer is I-type — the immediate is folded into the intermediate value, and
    has to survive being consumed by a second operation.
  writer is I-type — the immediate is applied TO a computed operand, so the cell
    must combine a literal with its own previous output rather than with a
    seeded leaf.

Splitting on that decides where to look, and no probe so far distinguishes them.

Reported per (producer type -> writer type) cell, and per (producer opcode ->
writer opcode) pair, as exact 32-bit match of the predicted output value —
scored with the model's own trained head, so nothing depends on a fitted
readout. `err mass` is the cell's share of ALL failing triples in the corpus,
including the depth-1 ones not broken out here.

Usage: vp_depth2.py <t2_dir> <t1_path> [n_batches] [config]
"""
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              _value_bits, _value_smooth)
from datagen import (RVT_FORMAT, Batch, AUX_CE_IGNORE, MAX_INPUT_SLOTS,
                     MAX_OUTPUT_SLOTS, N_REGS, make_anchor_states)
from emulator import I_TYPE
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 600
CONFIG = sys.argv[4] if len(sys.argv) > 4 else 'configs/instr_alu_writers.json'
N_ANCHORS, ANCHOR_SEED, NBITS, LEN = 8, 0, 32, 2

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)

CORPUS = f'/tmp/vp_depth2_{CONFIG.split("/")[-1][:-5]}.rvt'
subprocess.run(
    f'python -m datagen --rule "cap={LEN}" --twins 0 --batch-size 128 '
    f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
    f'--config {CONFIG} --count {NB} > {CORPUS}',
    shell=True, check=True, stderr=subprocess.DEVNULL)

anchors = torch.from_numpy(
    np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)

# Opcode classes, so a cell has enough triples to read. A pair of exact opcodes
# has ~50-180 and its ranking is noise.
OPCLASS = {}
for _ops, _c in ((('SLL', 'SRL', 'SRA', 'SLLI', 'SRLI', 'SRAI'), 'shift'),
                 (('XOR', 'OR', 'AND', 'XORI', 'ORI', 'ANDI'), 'logic'),
                 (('ADD', 'SUB', 'ADDI'), 'add'),
                 (('SLT', 'SLTU', 'SLTI', 'SLTIU'), 'cmp')):
    for _o in _ops:
        OPCLASS[_o] = _c

acc = defaultdict(lambda: [0.0, 0.0])

with open(CORPUS, 'rb') as f:
    for batch in RVT_FORMAT.reader(f, Batch):
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        (_, _, chunk_idx, slot_idx, n_per_chunk, w0, w1, wo) = split
        nch, mni = batch.instr_lens.shape
        with torch.no_grad():
            ct1, c0, c1, co, cact = _t2_assemble(
                t1, split, nch, mni, dev, route='binding')
            ess = t2.encode(ct1, c0, c1, co, cact,
                            torch.randn(nch, t2.n_regs, device=dev))
        # Producer/writer roles describe the wiring T2 ACTUALLY RAN ON — T1's
        # emitted binding — not the tokens'. They differ on ~3% of len=2 chunks,
        # and a role read off the tokens would describe a chunk the model was
        # never shown.
        e0, e1, eo = c0.cpu().numpy(), c1.cpu().numpy(), co.cpu().numpy()

        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        vp_mask = torch.from_numpy(batch.out_regs_valid).to(dev)
        valid = (out_slot != AUX_CE_IGNORE) & vp_mask.unsqueeze(-1)

        with torch.no_grad():
            iv = (ANCH_IN.unsqueeze(0)
                  * (in_slot != AUX_CE_IGNORE).to(torch.int64).unsqueeze(1))
            ifeat = torch.cat(
                [_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
                dim=-1).reshape(nch, N_ANCHORS, -1)
            v = F.normalize(ess, dim=-1).unsqueeze(1).expand(-1, N_ANCHORS, -1)
            pred = t2.vp_head(torch.cat([v, ifeat], dim=-1)).reshape(
                nch, N_ANCHORS, MAX_OUTPUT_SLOTS, NBITS + 1)
            tgt = out_regs.gather(-1, out_slot.clamp(min=0).unsqueeze(1)
                                  .expand(-1, N_ANCHORS, -1))
            exact = ((pred[..., :NBITS] > 0).float()
                     == _value_bits(tgt)).all(-1).float().cpu().numpy()

        oreg = out_slot.clamp(min=0).cpu().numpy()
        vnp = valid.cpu().numpy()
        pos = 0
        for b in range(nch):
            n = int(n_per_chunk[b])
            if not n:
                pos += n
                continue
            try:
                toks, p, ins = batch.tokens[b].tolist(), 0, []
                for _ in range(n):
                    i, p = decode_instruction(toks, p)
                    ins.append(i)
            except Exception:
                pos += n
                continue
            if len(ins) != n:
                pos += n
                continue

            # per-slot dataflow depth, and which instruction last wrote it
            depth = np.zeros(N_REGS, dtype=np.int64)
            src = np.full(N_REGS, -1, dtype=np.int64)   # writing instruction
            deep = {}                                   # writer k -> producer j
            for k in range(n):
                i0, i1, o = int(e0[b, k]), int(e1[b, k]), int(eo[b, k])
                # the operand that makes this slot deep is whichever was
                # written earlier in the chunk and sits deepest
                cand = max((i0, i1), key=lambda r: depth[r])
                if depth[cand] > 0:
                    deep[k] = int(src[cand])
                if o != 0:
                    depth[o] = 1 + max(depth[i0], depth[i1])
                    src[o] = k
            per_instr_shared = any(
                ({int(e0[b, a]), int(e1[b, a])} - {0})
                & ({int(e0[b, c]), int(e1[b, c])} - {0})
                for a in range(n) for c in range(a + 1, n))

            for s in np.nonzero(vnp[b])[0]:
                r = int(oreg[b, s])
                if depth[r] != 2:
                    continue
                k = int(src[r])
                j = deep.get(k, -1)
                if j < 0:
                    continue
                wt = 'I' if ins[k].opcode in I_TYPE else 'R'
                pt = 'I' if ins[j].opcode in I_TYPE else 'R'
                e = float(exact[b, :, s].sum())
                keys = [f'producer {pt} -> writer {wt}',
                        f'writer is {wt}-type', f'producer is {pt}-type',
                        f'{ins[j].opcode} -> {ins[k].opcode}']
                # Does the failure track how many bits the literal needs? If it
                # does, the slot is running out of room for it; if a 1-bit
                # immediate fails as often as a 12-bit one, what fails is the
                # structure the immediate sits in, not its width.
                pc, wc = OPCLASS[ins[j].opcode], OPCLASS[ins[k].opcode]
                keys.append(f'{pc} -> {wc}')
                if pt == 'I' and wt == 'I':
                    for nm, i_ in (('producer', ins[j]), ('writer', ins[k])):
                        v = abs(int(i_.args[2]))
                        keys.append(f'I->I, {nm} |imm| bits '
                                    f'{int(v).bit_length():2d}')
                    # Width, measured WITHIN one opcode class, so the 5-bit
                    # shift amount cannot masquerade as a narrow literal.
                    if pc == wc == 'add':
                        v = abs(int(ins[j].args[2]))
                        keys.append(f'add->add, producer |imm| bits '
                                    f'{min(int(v).bit_length(), 11):2d}')
                if not per_instr_shared:
                    keys.append(f'distinct sources: producer {pt} -> writer {wt}')
                for key in keys:
                    a = acc[key]
                    a[0] += e
                    a[1] += N_ANCHORS
            pos += n

        # corpus-wide failing total, so err mass is comparable to vp_budget
        a = acc['#ALL']
        m = valid.unsqueeze(1).expand(-1, N_ANCHORS, -1).cpu().numpy()
        a[0] += float((exact * m).sum())
        a[1] += float(m.sum())

TOT_E, TOT_N = acc['#ALL']
FAIL = TOT_N - TOT_E


def table(title, keys, limit=None):
    ks = sorted(keys, key=lambda k: -(acc[k][1] - acc[k][0]))
    print(f'  {title}')
    print(f'    {"bucket":34s} {"triples":>9s} {"exact":>10s} {"err mass":>9s}')
    for k in ks[:limit]:
        e, n = acc[k]
        if n == 0:
            continue
        print(f'    {k:34s} {int(n):9d} {100 * e / n:9.4f}% '
              f'{100 * (n - e) / max(FAIL, 1):8.1f}%')
    print()


print()
print(f'checkpoint = {T2_DIR}')
print(f'  corpus triples = {int(TOT_N)}   overall exact '
      f'{100 * TOT_E / TOT_N:.4f}%   failing = {int(FAIL)}')
print('  A depth-2 slot is writer(producer(a,b), c). Which one carries the '
      'immediate?')
print()
table('by which instruction is I-type',
      [k for k in acc if k.startswith(('writer is', 'producer is'))])
table('crossed', [k for k in acc if k.startswith('producer ') and '->' in k])
table('crossed, restricted to chunks with no shared source register',
      [k for k in acc if k.startswith('distinct sources:')])
table('by opcode CLASS pair (producer -> writer)',
      [k for k in acc if ' -> ' in k
       and k.split(' -> ')[0] in ('shift', 'logic', 'add', 'cmp')])
table('by exact opcode pair', [k for k in acc if '->' in k
                               and not k.startswith(('producer', 'distinct',
                                                     'I->I', 'add->add'))
                               and ' -> ' not in k],
      limit=10)
for nm in ('producer', 'writer'):
    ks = sorted(k for k in acc if k.startswith(f'I->I, {nm} |imm| bits'))
    print(f'  I->I, by how many bits the {nm}\'s literal needs')
    print(f'    {"bits":>5s} {"triples":>9s} {"exact":>10s}')
    for k in ks:
        e, n = acc[k]
        print(f'    {k[-2:]:>5s} {int(n):9d} {100 * e / n:9.4f}%')
    print()
ks = sorted(k for k in acc if k.startswith('add->add, producer'))
print('  add-class -> add-class only, by the producer literal\'s width — the '
      'same question with opcode held fixed')
print(f'    {"bits":>5s} {"triples":>9s} {"exact":>10s}')
for k in ks:
    e, n = acc[k]
    print(f'    {k[-2:]:>5s} {int(n):9d} {100 * e / n:9.4f}%')
print()
