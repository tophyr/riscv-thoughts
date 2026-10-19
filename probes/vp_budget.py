#!/usr/bin/env python3
"""Where is the vp error? The whole budget, one model, one corpus, one pass.

Scored with the model's OWN trained head, so nothing here depends on a fitted
readout's capacity — the failure mode that made several earlier probes disagree
with themselves. The unit is the (row, output slot, anchor) triple that
`t2_value_predict_loss` supervises.

Every table reports the same three things, and the third is the one to read:

  exact    — exact 32-bit match of the predicted output value. THE criterion.
  per-bit  — mean over the 32 bits. Reported only because the training log and
             older records use it; it flatters a broken subset badly (a bucket
             at 96% per-bit can be at 74% exact).
  err mass — the bucket's share of ALL failing triples in the corpus. A bucket
             can have a terrible rate and carry almost none of the error, or an
             excellent rate and carry most of it, purely by how common it is.
             Both facts matter and neither implies the other.

Splits, coarse to fine: dataflow depth; operand type; the two crossed; the
opcode that wrote the slot; whether the chunk's instructions share a source
register; and the per-bit error curve by bit position.

Usage: vp_budget.py <t2_dir> <t1_path> [chunk_len] [n_batches] [config] [corpus]
                    [n_anchors]

Scored at the fixed held-out anchor set (`datagen --heldout-anchors`), which no
model trains on; `n_anchors` (default 16) is how many of its states each row is
executed at. Each row is scored at the anchors its batch carries, so a corpus
generated without `--heldout-anchors` scores at whatever it was drawn with.

Without `corpus` each call draws a fresh corpus. To compare models per bucket,
pass the same `corpus` path to every call: the first generates it, the rest
read it.
"""
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              _value_bits, vp_head_predict,
                              vp_canonical_in_vals)
from datagen import (RVT_FORMAT, Batch, AUX_CE_IGNORE, MAX_OUTPUT_SLOTS,
                     N_REGS)
from emulator import I_TYPE
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import decode_instruction

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
# The corpus decides which population the percentages describe, and that has
# already been mistaken for a property of the model once: a subset measured on
# an R-only corpus reads a point better than the same subset on the mixed
# corpus the model trains on.
CONFIG = sys.argv[5] if len(sys.argv) > 5 else 'configs/instr_alu_writers.json'
NBITS = 32
N_ANCHORS = int(sys.argv[7]) if len(sys.argv) > 7 else 16

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)

CORPUS = sys.argv[6] if len(sys.argv) > 6 else \
    f'/tmp/vp_budget_len{LEN}_{CONFIG.split("/")[-1][:-5]}_{os.getpid()}.rvt'
if not os.path.exists(CORPUS):
    # Renamed into place only once complete, so a reused corpus is never partial.
    subprocess.run(
        f'python -m datagen --rule "cap={LEN}" --twins 0 --batch-size 128 '
        f'--n-states {N_ANCHORS} --heldout-anchors '
        f'--config {CONFIG} --count {NB} > {CORPUS}.part',
        shell=True, check=True, stderr=subprocess.DEVNULL)
    os.replace(f'{CORPUS}.part', CORPUS)



def slot_depth(in0, in1, out, n):
    d = np.zeros(N_REGS, dtype=np.int64)
    for k in range(n):
        i0, i1, o = int(in0[k]), int(in1[k]), int(out[k])
        nd = 1 + max(d[i0], d[i1])
        if o != 0:
            d[o] = nd
    return d


# per-bucket [exact hits, bit hits, triples]
acc = defaultdict(lambda: [0.0, 0.0, 0.0])
bit_err = torch.zeros(NBITS, device=dev)
bit_n = 0.0

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
            ess = t2.encode(ct1, c0, c1, co, cact)
        # Cells describe the wiring T2 ACTUALLY RAN ON — T1's emitted binding —
        # not the tokens'. The two differ on ~3% of len=2 chunks, and a cell
        # labelled from the tokens would be describing a chunk the model was
        # never shown.
        e0, e1, eo = c0.cpu().numpy(), c1.cpu().numpy(), co.cpu().numpy()

        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        out_slot = torch.from_numpy(batch.out_slot_regs.astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(batch.out_regs.astype(np.int64)).to(dev)
        vp_mask = torch.from_numpy(batch.out_regs_valid).to(dev)
        valid = (out_slot != AUX_CE_IGNORE) & vp_mask.unsqueeze(-1)

        anchors = torch.from_numpy(batch.anchors).to(dev)
        with torch.no_grad():
            iv = vp_canonical_in_vals(anchors, in_slot)
            pred = vp_head_predict(t2, ess, iv)
            tgt = out_regs.gather(-1, out_slot.clamp(min=0).unsqueeze(1)
                                  .expand(-1, anchors.shape[-2], -1))
            hit = ((pred[..., :NBITS] > 0).float()
                   == _value_bits(tgt))               # (B, A, MAX_OUT, 32)
            exact = hit.all(-1).float()               # (B, A, MAX_OUT)
            perbit = hit.float().mean(-1)

        # per-slot categories
        oreg = out_slot.clamp(min=0).cpu().numpy()
        vnp = valid.cpu().numpy()
        cats = {}          # (b, s) -> tuple of bucket labels
        for b in range(nch):
            n = int(n_per_chunk[b])
            if n:
                dv = slot_depth(e0[b], e1[b], eo[b], n)
                try:
                    toks, p, ins = batch.tokens[b].tolist(), 0, []
                    for _ in range(n):
                        i, p = decode_instruction(toks, p)
                        ins.append(i)
                except Exception:
                    ins = []
                if len(ins) == n:
                    # the opcode that last wrote each register
                    last = {int(eo[b, k]): ins[k].opcode
                            for k in range(n) if int(eo[b, k]) != 0}
                    # Two different things get called "a shared source", and
                    # they are different questions for the encoder: one
                    # instruction reading the same register twice (ADD x3,x5,x5)
                    # collapses two operands of ONE op, while two instructions
                    # reading a common register ties two ops to one canonical
                    # input position.
                    per_instr = [{int(e0[b, k]), int(e1[b, k])} - {0}
                                 for k in range(n)]
                    self_rep = any(int(e0[b, k]) == int(e1[b, k]) != 0
                                   for k in range(n))
                    cross_rep = any(per_instr[a] & per_instr[b]
                                    for a in range(n) for b in range(a + 1, n))
                    srcs = [x for st in per_instr for x in st]
                    shared = self_rep or cross_rep
                    how = ('src: distinct' if not shared else
                           'src: cross-instr' if cross_rep and not self_rep else
                           'src: same-instr' if self_rep and not cross_rep else
                           'src: both')
                    # Whether an IMMEDIATE is present anywhere in the chunk,
                    # which is not the same as the slot's own instruction being
                    # I-type: the interaction between a shared source and an
                    # immediate elsewhere in the chunk is invisible to a probe
                    # that splits on either alone.
                    anyi = 'chunk has an immediate' if any(
                        i.opcode in I_TYPE for i in ins) else 'chunk is all-R'
                    for s in np.nonzero(vnp[b])[0]:
                        op = last.get(int(oreg[b, s]))
                        if op:
                            cats[(b, int(s))] = (
                                int(dv[oreg[b, s]]), op,
                                'I' if op in I_TYPE else 'R', shared,
                                how, anyi)

        ex, pb = exact.cpu().numpy(), perbit.cpu().numpy()
        for (b, s), (d, op, typ, shared, how, anyi) in cats.items():
            e, p_, nn = float(ex[b, :, s].sum()), float(pb[b, :, s].sum()), anchors.shape[-2]
            for key in (f'depth {d}', f'{typ}-type', f'{typ}-type depth {d}',
                        f'op {op}', f'shared source: {shared}',
                        f'depth {d}, shared {shared}',
                        f'{typ}-type, shared {shared}', how,
                        f'{how} ({typ})', f'{how} | {anyi}',
                        f'{how} | {anyi} | depth {d}',
                        f'depth {d}: op {op}'):
                a = acc[key]
                a[0] += e; a[1] += p_; a[2] += nn
            a = acc['ALL']
            a[0] += e; a[1] += p_; a[2] += nn
        m = valid.unsqueeze(1).expand(-1, anchors.shape[-2], -1)
        bit_err += ((1 - hit.float()) * m.unsqueeze(-1).float()).sum((0, 1, 2))
        bit_n += float(m.sum())

TOT_E, TOT_B, TOT_N = acc['ALL']
FAIL = TOT_N - TOT_E


def table(title, keys):
    print(f'  {title}')
    print(f'    {"bucket":26s} {"triples":>9s} {"exact":>9s} {"per-bit":>9s} '
          f'{"err mass":>9s}')
    for k in keys:
        e, p, n = acc[k]
        if n == 0:
            continue
        print(f'    {k:26s} {int(n):9d} {100 * e / n:8.4f}% '
              f'{100 * p / n:8.4f}% {100 * (n - e) / FAIL:8.1f}%')
    print()


print()
print(f'checkpoint = {T2_DIR}')
print(f'  len={LEN}   supervised (row, slot, anchor) triples = {int(TOT_N)}')
print(f'  OVERALL   exact {100 * TOT_E / TOT_N:.4f}%   '
      f'per-bit {100 * TOT_B / TOT_N:.4f}%   '
      f'failing triples = {int(FAIL)}')
print()
def only(prefix):
    """Top-level buckets with `prefix`, excluding the crossed ones."""
    return sorted(k for k in acc
                  if k.startswith(prefix) and ',' not in k and ':' not in k)


table('by dataflow depth', only('depth '))
table('by operand type', only('') and sorted(
    k for k in acc if k.endswith('-type')))
table('depth x type', sorted(k for k in acc if 'type depth' in k))
table('by the opcode that wrote the slot',
      sorted((k for k in acc if k.startswith('op ')),
             key=lambda k: -(acc[k][2] - acc[k][0])))
table('do the chunk\'s instructions share a source register',
      sorted(k for k in acc if k.startswith('shared')))
table('type x shared source', sorted(k for k in acc if '-type, shared' in k))
table('HOW the sources repeat',
      sorted(k for k in acc if k.startswith('src: ') and '(' not in k))
table('how the sources repeat x type',
      sorted(k for k in acc if k.startswith('src: ') and '(' in k))
table('how the sources repeat x whether the CHUNK carries an immediate',
      sorted(k for k in acc if k.count('|') == 1))
table('the three axes crossed — sources x immediate x depth',
      sorted((k for k in acc if k.count('|') == 2
              and k.startswith(('src: distinct', 'src: cross-instr'))),
             key=lambda k: -(acc[k][2] - acc[k][0])))
table('depth x shared source — are these the same error or two',
      sorted(k for k in acc if ', shared' in k))
table('within depth 2, by opcode',
      sorted((k for k in acc if k.startswith('depth 2: op')),
             key=lambda k: -(acc[k][2] - acc[k][0]))[:8])
print('  per-bit error rate by bit position (0 = LSB; an immediate occupies '
      '0-11)')
r = (100 * bit_err / max(bit_n, 1)).tolist()
for lo in range(0, NBITS, 8):
    print('    bits %2d-%2d  ' % (lo, lo + 7)
          + '  '.join(f'{r[i]:6.3f}%' for i in range(lo, lo + 8)))
print()
