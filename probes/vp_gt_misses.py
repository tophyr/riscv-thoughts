#!/usr/bin/env python3
"""Where do vp_gt_depth.py's heads still fail, and why? Every miss of one saved
checkpoint, with the writer's operand values recovered by emulation.

Runs the heads of one `<out_prefix>_heads_step<N>.pt` over a held-out corpus
(generated with `datagen --heldout-anchors`) and, for every missed (row, slot,
anchor) triple, records the chunk, its canonical inputs, the target, the
prediction, the wrong bits and their logit margins. The writer's operands come
from executing the chunk on the CPU emulator at those inputs, and the probe
asserts the emulated output equals the target on every triple it emulates, so
a frame or decoding mistake stops it rather than mislabelling operands.

Printed per head: the miss rate by depth, writer opcode, producer opcode,
operand roles and cell, each with its share of all misses; misses per held-out
anchor against the uniform expectation; and, for ADD/SUB writers, the miss rate
by the longest carry chain in the word and the carry ripple arriving at wrong
bits against correct bits of the same words.

Usage: vp_gt_misses.py <corpus> <heads.pt> <spec> [out_json]
       e.g. vp_gt_misses.py /tmp/h64_chain100.rvt \
                logs/gt10m_resln_heads_step10000000.pt 6:he:resln,12:he:resln
corpus: generated on demand; the datagen skill has the command for the
composed-dense one (configs/instr_alu_writers_chain100.json).
spec: the heads' depth[:init[:arch]] list exactly as vp_gt_depth.py trained
them (the checkpoint's step JSON records it). out_json, if given, receives
every miss record.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

CORPUS, HEADS, SPEC = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# vp_gt_depth parses its argv at import; hand it the spec so ARCH/INIT match.
sys.argv = ['vp_gt_depth.py', CORPUS, '-', '-', '1', SPEC]
import vp_gt_depth as P                                      # noqa: E402
from emulator import run, R_TYPE                             # noqa: E402

u32 = lambda v: int(v) & 0xFFFFFFFF


def carries(a, b, sub):
    """Per bit, how far the arriving carry rippled (0 = none), and the longest
    run of consecutive carry-outs in the word, for a+b or a-b."""
    a, b, c = u32(a), u32(b), 0
    if sub:
        b, c = (~b) & 0xFFFFFFFF, 1
    ripple, run_len, best = [], 0, 0
    for i in range(32):
        ripple.append(run_len if c else 0)
        x, y = (a >> i) & 1, (b >> i) & 1
        c = (x & y) | (c & (x ^ y))
        run_len = run_len + 1 if c else 0
        best = max(best, run_len)
    return ripple, best


def keys(cell):
    comp = '->' in cell
    d = 'd2' if comp else 'd1'
    return {'depth': 'depth 2' if comp else 'depth 1',
            'writer': f"{cell.split('-> ')[-1].split(' [')[0]} ({d})",
            'producer': cell.split(' ->')[0] if comp else None,
            'roles': f"{cell.split('[')[1].rstrip(']')} ({d})",
            'cell': cell}


def main():
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True    # as vp_gt_depth trains and scores
    torch.backends.cudnn.allow_tf32 = True
    d_in = P.K * P.D_INSTR + P.N_IN * 33
    nets = {}
    for d, sd in torch.load(HEADS, map_location='cpu').items():
        net = (P.ResLN if P.ARCH[d] == 'resln' else P.mlp)(d_in, d, P.INIT[d])
        net.load_state_dict(sd)
        nets[d] = net.to(dev).eval()
    tot = {k: Counter() for k in ('depth', 'writer', 'producer', 'roles', 'cell')}
    misses = {d: [] for d in nets}
    chain = {d: {hm: Counter() for hm in ('hit', 'miss')} for d in nets}
    n_emul = 0
    with torch.no_grad(), open(CORPUS, 'rb') as fh:
        for bi, batch in enumerate(P.RVT_FORMAT.reader(fh, P.Batch)):
            f, iv, t, m, labels = P.chunk_features(batch)
            x = P.head_in(torch.from_numpy(f).to(dev), torch.from_numpy(iv).to(dev))
            bits = P._value_bits(torch.from_numpy(t).to(dev))
            logit = {d: n(x).view(*x.shape[:2], P.N_OUT, 32) for d, n in nets.items()}
            ok = {d: ((lg > 0).float() == bits).all(-1).cpu().numpy()
                  for d, lg in logit.items()}
            A = x.shape[1]
            for b, s in zip(*np.nonzero(m)):
                depth, op, _, cell = labels[b][s]
                for k, v in keys(cell).items():
                    if v is not None:
                        tot[k][v] += A
                addsub = op in ('ADD', 'SUB')
                if not addsub and all(ok[d][b, :, s].all() for d in nets):
                    continue
                toks, p, chunk = batch.tokens[b].tolist(), 0, []
                for _ in range(int((batch.instr_lens[b] > 0).sum())):
                    ins, p = P.decode_instruction(toks, p); chunk.append(ins)
                order = P._input_order(chunk)
                reg = [int(r) for r in batch.out_slot_regs[b] if r != P.AUX_CE_IGNORE][s]
                k = max(i for i, ins in enumerate(chunk) if reg in P._written_regs(ins))
                w = chunk[k]
                for a in range(A):
                    regs = np.zeros(32, dtype=np.int64)
                    for j, r in enumerate(order[:P.N_IN]):
                        regs[r] = iv[b, a, j]
                    regs = (regs & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
                    pre = run(chunk[:k], regs)[0].regs if k else regs
                    assert u32(run(chunk, regs)[0].regs[reg]) == u32(t[b, a, s]), \
                        f'batch {bi} row {b}: emulation disagrees with the target'
                    n_emul += 1
                    ops = ([int(pre[r]) for r in w.args[1:3]] if w.opcode in R_TYPE
                           else [int(pre[w.args[1]]), w.args[2]])
                    rip, longest = carries(*ops, op == 'SUB') if addsub else (None, None)
                    for d in nets:
                        hit = bool(ok[d][b, a, s])
                        if addsub:
                            chain[d]['hit' if hit else 'miss'][(depth, op, longest)] += 1
                        if hit:
                            continue
                        lg = logit[d][b, a, s].float().cpu().numpy()
                        pred = sum(1 << i for i in range(32) if lg[i] > 0)
                        wrong = [i for i in range(32) if ((pred ^ u32(t[b, a, s])) >> i) & 1]
                        misses[d].append(dict(
                            batch=bi, row=int(b), slot=int(s), anchor=a, cell=cell,
                            chunk=[repr(i) for i in chunk], operands=[u32(v) for v in ops],
                            inputs=[u32(v) for v in iv[b, a]], target=u32(t[b, a, s]),
                            pred=pred, wrong_bits=wrong,
                            wrong_logit_abs=[round(abs(float(lg[i])), 3) for i in wrong],
                            ripple=rip))
    print(f'emulated {n_emul} (slot, anchor) cases, every one equal to its target')
    if OUT:
        json.dump({str(d): v for d, v in misses.items()}, open(OUT, 'w'))
    report(misses, tot, chain)


def report(misses, tot, chain):
    n_all = sum(tot['depth'].values())
    for d, v in misses.items():
        print(f'\n######## {d} layers: {len(v)} misses / {n_all} triples = '
              f'{100 * len(v) / n_all:.4f}%')
        mis = {k: Counter() for k in tot}
        for e in v:
            for k, val in keys(e['cell']).items():
                if val is not None:
                    mis[k][val] += 1
        for k, top in (('depth', 2), ('writer', 8), ('producer', 8), ('roles', 8), ('cell', 12)):
            print(f'  -- by {k}')
            for val, n in mis[k].most_common(top):
                print(f'     {val:<32} {n:6d} / {tot[k][val]:9d} = '
                      f'{100 * n / tot[k][val]:7.3f}%   share {100 * n / len(v):5.1f}%')
        per = Counter(e['anchor'] for e in v)
        top6 = per.most_common(6)
        print(f'  -- by held-out anchor: uniform {len(v) / 64:.1f} each; {64 - len(per)} '
              f'with none; top six {top6} hold '
              f'{100 * sum(n for _, n in top6) / max(len(v), 1):.1f}%')
        mx = [max(e['wrong_logit_abs']) for e in v]
        print(f'  -- single-bit misses {sum(len(e["wrong_bits"]) == 1 for e in v)}; largest '
              f'wrong-bit |logit| <0.5: {sum(x < .5 for x in mx)}, '
              f'0.5-2: {sum(.5 <= x < 2 for x in mx)}, >=2: {sum(x >= 2 for x in mx)}')
        bins = ((0, 4), (5, 8), (9, 16), (17, 24), (25, 32))
        for depth in (1, 2):
            for op in ('ADD', 'SUB'):
                row = []
                for lo, hi in bins:
                    mm = sum(c for (dd, o, cc), c in chain[d]['miss'].items()
                             if dd == depth and o == op and lo <= cc <= hi)
                    hh = sum(c for (dd, o, cc), c in chain[d]['hit'].items()
                             if dd == depth and o == op and lo <= cc <= hi)
                    row.append(f'{lo}-{hi}: {mm}/{mm + hh}')
                print(f'  -- depth {depth} {op}, misses by longest carry chain: ' + '  '.join(row))
        w, r = Counter(), Counter()
        for e in v:
            if e['ripple'] is not None:
                for i, x in enumerate(e['ripple']):
                    (w if i in e['wrong_bits'] else r)[min(x, 16) // 8] += 1
        if w:
            tw, tr = sum(w.values()), sum(r.values())
            print('  -- carry ripple arriving at ADD/SUB wrong bits vs correct bits: ' + '  '.join(
                f'{lab}: {100 * w[i] / tw:.1f}% vs {100 * r[i] / tr:.1f}%'
                for i, lab in ((0, '0-7'), (1, '8-15'), (2, '16+'))))


if __name__ == '__main__':
    main()
