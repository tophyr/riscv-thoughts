#!/usr/bin/env python3
"""Is the essence describing the same chunk the vp target describes?

Every capacity- and architecture-side explanation for the len=2 vp residual has
been refuted, which is what a MISMATCH looks like: if the encoder is fed one
chunk and graded against another, no amount of width or depth helps. There are
exactly two places the two descriptions can drift apart, and this probe measures
both against the model's own trained head.

  wiring — T2's operand and destination slots come from the FROZEN T1's binding
    predictions, never from the tokens (`_t1_predicted_wiring`). When T1
    mispredicts, the essence describes a different chunk than the one that was
    executed to build the target, and the row is unsatisfiable. The reference
    is the ORACLE's binding, not the tokens': both are behavioral, so a read
    T1 omits because the output does not depend on it is agreement rather than
    error.

  frame — a chunk input's identity in the encoder is its canonical READ
    POSITION, seeded on first read and counted causally over registers read
    before they are written (`seed_mode='rank'`). The target's canonical frame
    is assigned over the chunk's LIVE syntactic inputs (`live_input_regs`), and
    the head is fed the values at those positions. A dead input is read by the
    encoder and skipped by the oracle, so every later input's position shifts
    and the essence names a position the head fills with another input's value.

Both are invisible to a probe that splits on capacity, and both would land
hardest where the encoder has the most to say per register — which is why the
cells crossed here are the three that carry the len=2 error (`vp_budget.py`):
sources shared across the two instructions, an immediate in the chunk, and a
slot whose operand is the other instruction's result.

The mismatch is reported by KIND, and vp is scored within each kind — a
comparison that stays inside the mispredicted population and so cannot be
explained by mispredicted chunks simply being harder:

  dest       the destination register differs.
  wrong-src  a source register neither dropped nor added — a different register.
  add-src    a source T1 emits that the oracle does not list.
  drop-src   a source the oracle lists and T1 omits.
  swap       the same two sources in the other operand position. `in_slot_regs`
             is in first-read order and the `in_slot` loss trains exactly that
             order, so this is a genuine ranking error in `in_score`.

Usage: vp_frame.py <t2_dir> <t1_path> [chunk_len] [n_batches] [config] [route]
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
from datagen.compare import (to_ssa, live_nodes, live_input_regs,
                             precompute_chunk)
from emulator import I_TYPE
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 600
CONFIG = sys.argv[5] if len(sys.argv) > 5 else 'configs/instr_alu_writers.json'
# 'binding' is how the model trains and how the tier recursion is defined: T2
# reads T1's predicted wiring. 'tokens' overrides it with the true wiring, which
# is off-distribution for a binding-trained model but isolates what the
# prediction costs.
ROUTE = sys.argv[6] if len(sys.argv) > 6 else 'binding'
N_ANCHORS, ANCHOR_SEED, NBITS = 8, 0, 32

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)

CORPUS = f'/tmp/vp_frame_len{LEN}_{CONFIG.split("/")[-1][:-5]}.rvt'
subprocess.run(
    f'python -m datagen --rule "cap={LEN}" --twins 0 --batch-size 128 '
    f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
    f'--config {CONFIG} --count {NB} > {CORPUS}',
    shell=True, check=True, stderr=subprocess.DEVNULL)

anchors_np = np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))
anchors = torch.from_numpy(anchors_np).to(dev)
behav_cache = {}
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)


def encoder_ranks(i0, i1, o, n):
    """The canonical read position the encoder seeds each register with.

    Replays `Compressor._stamp_reads` under read_time_mode='rank': operand 0
    then operand 1 of each instruction, a register numbered on its first read
    and only if it has not already been written here. Returns reg -> position.
    """
    rank, written, nxt = {}, set(), 0
    for k in range(n):
        for r in (int(i0[k]), int(i1[k])):
            if r != 0 and r not in rank and r not in written:
                nxt += 1
                rank[r] = nxt
        if int(o[k]) != 0:
            written.add(int(o[k]))
    return rank


def oracle_wiring(instr, cache):
    """The wiring the ORACLE describes for one instruction: its behavioral
    inputs in first-read order, and the register it modifies.

    This is the reference T1 is trying to emit, and comparing against the
    TOKENS instead is wrong in two ways that both read as T1 errors when they
    are T1 agreeing with the oracle. The tokens say `ADD x3,x5,x5` reads two
    registers; the oracle lists one, because that is how many distinct values
    the output depends on. The tokens say `SLTIU x10,x22,-k` reads x22; the
    oracle lists none, because a sign-extended negative immediate makes the
    comparison constant. T1's binding heads are behavioral and correctly emit
    the oracle's answer in both cases.
    """
    key = (instr.opcode, tuple(instr.args))
    if key not in cache:
        try:
            pre = precompute_chunk([instr], anchors_np)
            ins_regs = [int(x) for x in pre.in_slot_regs if x != AUX_CE_IGNORE]
            out = pre.out_slot_regs[0]
            cache[key] = (tuple(ins_regs[:2]) + (0, 0))[:2], (
                int(out) if out != AUX_CE_IGNORE else 0)
        except Exception:
            cache[key] = ((0, 0), 0)
    return cache[key]


def wiring_kind(p0, p1, po, instrs, cache):
    """How the wiring T2 ran on differs from the wiring the ORACLE describes.
    Returns the most consequential kind over the chunk's instructions, or 'ok'.

    `swap` is the one that survives this reference: `in_slot_regs` is in
    first-read order and `_t1_predicted_wiring` takes a top-2 over `in_score`,
    whose loss trains exactly that order, so a reversed pair is a genuine
    misprediction rather than a convention mismatch.
    """
    order = ('dest', 'wrong-src', 'add-src', 'drop-src', 'swap')
    kinds, dropped = set(), []
    for k, instr in enumerate(instrs):
        (o0, o1), oout = oracle_wiring(instr, cache)
        pred, true = (int(p0[k]), int(p1[k])), (o0, o1)
        if int(po[k]) != oout:
            kinds.add('dest')
        if pred == true:
            continue
        ps, ts = set(pred) - {0}, set(true) - {0}
        if ps == ts:
            kinds.add('swap')
        elif ps < ts:
            kinds.add('drop-src')
            dropped += [(k, r) for r in ts - ps]
        elif ps > ts:
            kinds.add('add-src')
        else:
            kinds.add('wrong-src')
    for kind in order:
        if kind in kinds:
            return kind, dropped
    return 'ok', dropped


def slot_depth(i0, i1, o, n):
    d = np.zeros(N_REGS, dtype=np.int64)
    for k in range(n):
        nd = 1 + max(d[int(i0[k])], d[int(i1[k])])
        if int(o[k]) != 0:
            d[int(o[k])] = nd
    return d


# bucket -> [exact hits, triples]
acc = defaultdict(lambda: [0.0, 0.0])
# bucket -> histogram of how many bits a failing triple gets wrong (9 = 9+)
bits = defaultdict(lambda: defaultdict(float))
examples = []
# how often each mismatch occurs, per chunk
chunks = defaultdict(float)

with open(CORPUS, 'rb') as f:
    for batch in RVT_FORMAT.reader(f, Batch):
        split = _split_to_per_instruction(batch)
        if split[0].shape[0] == 0:
            continue
        (_, _, chunk_idx, slot_idx, n_per_chunk, w0, w1, wo) = split
        nch, mni = batch.instr_lens.shape
        with torch.no_grad():
            # Classification always uses T1's PREDICTED wiring, whatever the
            # encode runs on, so the same rows are labelled the same way under
            # both routes and the two runs are comparable row for row.
            _, c0, c1, co, _ = _t2_assemble(
                t1, split, nch, mni, dev, route='binding')
            ct1, e0, e1, eo, cact = _t2_assemble(
                t1, split, nch, mni, dev, route=ROUTE)
            ess = t2.encode(ct1, e0, e1, eo, cact,
                            torch.randn(nch, t2.n_regs, device=dev))

        # The wiring T2 actually ran on (T1's predictions) beside the wiring the
        # tokens specify and the target was executed from.
        p0, p1, po = (c0.cpu().numpy(), c1.cpu().numpy(), co.cpu().numpy())
        t0 = np.zeros((nch, mni), dtype=np.int64)
        t1w = np.zeros((nch, mni), dtype=np.int64)
        to = np.zeros((nch, mni), dtype=np.int64)
        t0[chunk_idx, slot_idx] = w0
        t1w[chunk_idx, slot_idx] = w1
        to[chunk_idx, slot_idx] = wo

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
            hit = (pred[..., :NBITS] > 0).float() == _value_bits(tgt)
            exact = hit.all(-1).float().cpu().numpy()
            # How BADLY a failing triple misses. One or two wrong bits is a
            # different defect from a value that is wholly wrong.
            wrong_bits = (NBITS - hit.float().sum(-1)).cpu().numpy()
            # the head's prediction as a signed 32-bit value, to read beside
            # the target
            pw = torch.tensor([1 << i for i in range(NBITS)], device=dev)
            pv = ((pred[..., :NBITS] > 0).long() * pw).sum(-1)
            pred_val = pv - (pv >> 31 << 32)

        isl = batch.in_slot_regs.astype(np.int64)
        oreg = out_slot.clamp(min=0).cpu().numpy()
        vnp = valid.cpu().numpy()

        for b in range(nch):
            n = int(n_per_chunk[b])
            if not n:
                continue
            try:
                toks, p, ins = batch.tokens[b].tolist(), 0, []
                for _ in range(n):
                    i, p = decode_instruction(toks, p)
                    ins.append(i)
            except Exception:
                continue
            if len(ins) != n:
                continue

            # Compared against the ORACLE's per-instruction binding, not the
            # tokens': every mismatch below is T1 disagreeing with the target,
            # never a difference of convention.
            kind, dropped = wiring_kind(p0[b], p1[b], po[b], ins, behav_cache)
            wiring_ok = kind == 'ok'
            # The encoder's frame comes from the wiring it ran on; the target's
            # comes from the oracle. Comparing them is the whole question.
            ranks = encoder_ranks(p0[b], p1[b], po[b], n)
            try:
                g = to_ssa(ins)
                live = live_input_regs(g, live_nodes(g))
            except Exception:
                continue
            frame_ok = all(ranks.get(r) == i + 1 for i, r in enumerate(live))
            # The head is fed anchor[k+1] for input slot k, so the BEHAVIORAL
            # input list must also be the live list in order for slot k to name
            # canonical position k+1 (the documented degenerate-operand caveat).
            # Whether the behavioral input list IS the live syntactic one. When
            # it is, the two canonical frames coincide and the chunk's vp target
            # is byte-identical under either — the only rows on which a
            # behavioral-frame model and a syntactic-frame model can be compared
            # without comparing two different tasks.
            beh = [int(x) for x in isl[b] if x != AUX_CE_IGNORE]
            slot_ok = beh == live

            dv = slot_depth(t0[b], t1w[b], to[b], n)
            per_instr = [{int(t0[b, k]), int(t1w[b, k])} - {0} for k in range(n)]
            cross = any(per_instr[x] & per_instr[y]
                        for x in range(n) for y in range(x + 1, n))
            self_rep = any(int(t0[b, k]) == int(t1w[b, k]) != 0
                           for k in range(n))
            how = ('distinct' if not (cross or self_rep) else
                   'cross-instr' if cross and not self_rep else
                   'same-instr' if self_rep else 'both')
            anyi = 'imm' if any(i.opcode in I_TYPE for i in ins) else 'no-imm'

            chunks['chunks'] += 1
            chunks['wiring mismatch'] += (not wiring_ok)
            chunks['frame mismatch'] += (not frame_ok)
            chunks['in_slot mismatch'] += (not slot_ok)
            chunks[f'{how}|{anyi}: chunks'] += 1
            chunks[f'{how}|{anyi}: wiring mismatch'] += (not wiring_ok)
            chunks[f'{how}|{anyi}: frame mismatch'] += (not frame_ok)

            tag = ('clean' if wiring_ok and frame_ok else
                   'BAD wiring' if not wiring_ok and frame_ok else
                   'BAD frame' if wiring_ok else 'BAD both')
            # Which instruction wrote each register, so a failing slot can be
            # attributed to the instruction whose read was dropped or to the
            # other one.
            writer = {int(to[b, k]): k for k in range(n) if int(to[b, k]) != 0}
            droppers = {k for k, _ in dropped}

            for s in np.nonzero(vnp[b])[0]:
                d = int(dv[oreg[b, s]])
                e = float(exact[b, :, s].sum())
                cell = f'{how}|{anyi}|depth {d}'
                keys = ['ALL', tag, cell, f'{cell}  {tag}',
                        f'wiring {kind}', f'{cell}  wiring {kind}',
                        f'frames coincide: {slot_ok}',
                        f'{cell}  coincide={slot_ok}']
                if kind.startswith('drop-src'):
                    w = writer.get(int(oreg[b, s]))
                    keys.append('drop: slot written by the DROPPING instruction'
                                if w in droppers else
                                'drop: slot written by the OTHER instruction')
                    keys.append(f'drop x frame_ok={frame_ok}')
                for key in keys:
                    a = acc[key]
                    a[0] += e
                    a[1] += N_ANCHORS
                if (kind.startswith('drop-src') and frame_ok
                        and exact[b, :, s].min() == 0 and len(examples) < 25):
                    a_ = int(np.argmin(exact[b, :, s]))
                    examples.append(
                        (' ; '.join(str(i) for i in ins),
                         [(int(p0[b, k]), int(p1[b, k]), int(po[b, k]))
                          for k in range(n)],
                         [(int(t0[b, k]), int(t1w[b, k]), int(to[b, k]))
                          for k in range(n)],
                         dropped, int(oreg[b, s]), a_,
                         int(out_regs[b, a_, oreg[b, s]].item()),
                         int(pred_val[b, a_, s].item()),
                         [int(x) for x in isl[b] if x != AUX_CE_IGNORE]))
                miss = wrong_bits[b, :, s]
                for nb_ in miss[miss > 0]:
                    bits[('drop-src' if kind.startswith('drop-src')
                          else kind if kind != 'ok' else
                          f'ok, depth {d}')][min(int(nb_), 9)] += 1

TOT_E, TOT_N = acc['ALL']
FAIL = TOT_N - TOT_E


def table(title, keys):
    print(f'  {title}')
    print(f'    {"bucket":38s} {"triples":>9s} {"exact":>10s} {"err mass":>9s}')
    for k in keys:
        e, n = acc[k]
        if n == 0:
            continue
        print(f'    {k:38s} {int(n):9d} {100 * e / n:9.4f}% '
              f'{100 * (n - e) / max(FAIL, 1):8.1f}%')
    print()


print()
print(f'checkpoint = {T2_DIR}')
print(f'  len={LEN}  triples = {int(TOT_N)}  '
      f'exact {100 * TOT_E / TOT_N:.4f}%  failing = {int(FAIL)}')
print()
nc = chunks['chunks']
print('  how often the encoder and the target describe different chunks')
print(f'    chunks                                {int(nc):9d}')
for k in ('wiring mismatch', 'frame mismatch', 'in_slot mismatch'):
    print(f'    {k:38s} {int(chunks[k]):9d} {100 * chunks[k] / nc:8.3f}%')
print()
print('  ... by cell (sources | immediate)')
print(f'    {"cell":38s} {"chunks":>9s} {"wiring":>9s} {"frame":>9s}')
for k in sorted(x for x in chunks if x.endswith(': chunks')):
    cell = k[:-len(': chunks')]
    n = chunks[k]
    print(f'    {cell:38s} {int(n):9d} '
          f'{100 * chunks[cell + ": wiring mismatch"] / n:8.3f}% '
          f'{100 * chunks[cell + ": frame mismatch"] / n:8.3f}%')
print()
table('vp by whether the descriptions agree',
      ['clean', 'BAD wiring', 'BAD frame', 'BAD both'])
table('the three axes crossed (as vp_budget reports them)',
      sorted((k for k in acc if k.count('|') == 2 and '  ' not in k),
             key=lambda k: -(acc[k][1] - acc[k][0])))
table('the same cells split by whether the descriptions agree',
      sorted((k for k in acc if '  ' in k and 'wiring' not in k),
             key=lambda k: -(acc[k][1] - acc[k][0]))[:12])
table('vp by KIND of wiring mismatch — the comparison that stays inside the '
      'mispredicted population',
      sorted((k for k in acc if k.startswith('wiring ')),
             key=lambda k: -(acc[k][1] - acc[k][0])))
table('the worst cell (shared source, immediate, depth 1) by kind',
      sorted((k for k in acc
              if k.startswith('cross-instr|imm|depth 1  wiring')),
             key=lambda k: -(acc[k][1] - acc[k][0])))
table('the other worst cell (composed operand, immediate) by kind',
      sorted((k for k in acc
              if k.startswith('distinct|imm|depth 2  wiring')),
             key=lambda k: -(acc[k][1] - acc[k][0])))
table('where a dropped read costs its output slot',
      sorted(k for k in acc if k.startswith('drop')))
table('rows whose target is IDENTICAL under the behavioral and syntactic '
      'canonical frames — the only fair cross-frame comparison',
      sorted(k for k in acc if k.startswith('frames coincide')))
table('... by cell',
      sorted((k for k in acc if 'coincide=True' in k),
             key=lambda k: -(acc[k][1] - acc[k][0]))[:8])
print('  failing triples where T1 dropped a read and the frame still lines up')
for (txt, pw_, tw, dr, oslot, a_, tgt_, pv_, isl_) in examples[:20]:
    print(f'    {txt}')
    print(f'      T1 wiring {pw_}   tokens {tw}   dropped {dr}')
    print(f'      in_slots {isl_}   out slot x{oslot} @anchor {a_}: '
          f'target {tgt_}  predicted {pv_}  delta {pv_ - tgt_}')
print()
print('  how many of the 32 bits a FAILING triple gets wrong (share of that '
      'bucket\'s failures)')
print(f'    {"bucket":26s} ' + ' '.join(f'{i if i < 9 else "9+":>5}'
                                        for i in range(1, 10)))
for k in sorted(bits):
    tot = sum(bits[k].values())
    print(f'    {k:26s} ' + ' '.join(
        f'{100 * bits[k][i] / tot:4.0f}%' for i in range(1, 10)))
print()
