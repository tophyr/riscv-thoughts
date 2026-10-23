#!/usr/bin/env python3
"""Can a vp-head-shaped readout EVALUATE a two-instruction chunk at all, given
a perfect description of it? How deep does it need to be?

Heads of the vp head's shape (plain ReLU MLPs, 1024 wide) at several depths are
trained side by side on identical batches. Instead of an essence each is handed
ground-truth chunk features — per instruction its opcode, its 12-bit immediate,
what feeds each operand (x0, canonical input k, the other instruction's result,
or a read the output ignores) and which output slot it writes — plus the
canonical input values exactly as the vp head receives them, and predicts every
output slot's value. No encoder is involved, so the result is a property of
the readout alone: if no depth reaches exact, the vp head's shape cannot
evaluate these functions however good the essence is.

Training rows come from the live stream (anchors drawn per row, as in
training); the score is at a held-out corpus, next to the trained T2's own vp
head on the same slots.

Usage: vp_gt_depth.py <heldout_corpus> <t2_dir> <t1_path> [steps] [depths]
                       [eval_every] [out_prefix]
       e.g. vp_gt_depth.py h64.rvt runs/<t2> runs/<t1>/encoder.pt 10000000 \
                6:default,12:he 1000000 logs/gt10m
depths: depth[:init[:arch]] per head, init 'he' (default) or 'default', arch
'plain' (default, the vp head's Linear+ReLU stack) or 'resln' (pre-activation
residual blocks with LayerNorm, the same Linear count). With
out_prefix, each evaluation writes <out_prefix>_step<N>.json — per slot cell
(writer opcode, its operands, and for a composed slot the producer's opcode)
the held-out hits per head and the training stream's count of that cell — and
the heads are saved to <out_prefix>_heads_step<N>.pt.
"""
import math
import multiprocessing as mp
import subprocess
import sys
import time
import json
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import (_split_to_per_instruction, _t2_assemble,
                              _value_bits, _value_feat, vp_canonical_in_vals,
                              vp_head_predict)
from datagen import RVT_FORMAT, Batch, AUX_CE_IGNORE
from datagen.compare import _input_order, _written_regs
from emulator import R_TYPE
from tokenizer import decode_instruction

HELDOUT, T2_DIR, T1_PATH = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1000000
# depth[:init[:arch]] per head; init is 'he' (default) or 'default'
# (nn.Linear's own); arch is 'plain' (default) or 'resln'
SPEC = [(int(f[0]), f[1] if len(f) > 1 else 'he', f[2] if len(f) > 2 else 'plain')
        for f in (x.split(':') for x in
                  (sys.argv[5] if len(sys.argv) > 5 else '4,6,8,12').split(','))]
DEPTHS = [d for d, _, _ in SPEC]
assert len(set(DEPTHS)) == len(DEPTHS), 'heads are keyed by depth'
INIT = {d: i for d, i, _ in SPEC}
ARCH = {d: a for d, _, a in SPEC}
EVAL_EVERY = int(sys.argv[6]) if len(sys.argv) > 6 else 250000
OUT = sys.argv[7] if len(sys.argv) > 7 else None
N_IN, N_OUT, K = 4, 2, 2
OPS = ('ADD SUB XOR OR AND SLL SRL SRA SLT SLTU '
       'ADDI SLTI SLTIU XORI ORI ANDI SLLI SRLI SRAI').split()
N_ROLE = N_IN + 3                  # x0, inputs 1..4, other's result, ignored
_ROLE_NAMES = ['x0'] + [f'in{i}' for i in range(1, N_IN + 1)] + ['result', 'ignored']
D_INSTR = len(OPS) + 12 + 2 * N_ROLE + (N_OUT + 1)


def chunk_features(batch):
    """Per row: ground-truth features, canonical input values, targets, slot
    mask, and per-slot labels (depth, writer opcode, composed + fresh)."""
    B = batch.tokens.shape[0]
    A = batch.anchors.shape[1]
    feat = np.zeros((B, K * D_INSTR), dtype=np.float32)
    labels = [[None] * N_OUT for _ in range(B)]
    mask = np.zeros((B, N_OUT), dtype=bool)
    tgt = np.zeros((B, A, N_OUT), dtype=np.int64)
    for b in np.nonzero(batch.out_regs_valid)[0]:
        n = int((batch.instr_lens[b] > 0).sum())
        toks, p, chunk = batch.tokens[b].tolist(), 0, []
        for _ in range(n):
            ins, p = decode_instruction(toks, p); chunk.append(ins)
        order = _input_order(chunk)
        outs = [int(r) for r in batch.out_slot_regs[b] if r != AUX_CE_IGNORE]
        last = {}
        for k, ins in enumerate(chunk):
            for r in _written_regs(ins):
                last[r] = k
        written = set()
        for k, ins in enumerate(chunk):
            o = k * D_INSTR
            feat[b, o + OPS.index(ins.opcode)] = 1
            if ins.opcode in R_TYPE:
                srcs, imm = list(ins.args[1:3]), 0
            else:
                srcs, imm = [ins.args[1], 0], ins.args[2]
            for i in range(12):
                feat[b, o + len(OPS) + i] = (imm >> i) & 1
            composed, roles = False, []
            for j, x in enumerate(srcs):
                role = (0 if x == 0 else N_IN + 1 if x in written
                        else 1 + order.index(x) if x in order else N_IN + 2)
                composed |= role == N_IN + 1
                roles.append(_ROLE_NAMES[role])
                feat[b, o + len(OPS) + 12 + j * N_ROLE + role] = 1
            ws = [r for r in _written_regs(ins) if last.get(r) == k and r in outs]
            si = outs.index(ws[0]) + 1 if ws else 0
            feat[b, o + len(OPS) + 12 + 2 * N_ROLE + si] = 1
            if si:
                fresh = any(x and x not in written and x in order for x in srcs)
                kind = ('R-type writer, composed + fresh input'
                        if composed and fresh and ins.opcode in R_TYPE else
                        'other depth-2' if composed else 'depth 1')
                ops = roles if ins.opcode in R_TYPE else roles[:1] + ['imm']
                cell = (f'{chunk[0].opcode} -> ' if composed else '') + \
                    f'{ins.opcode} [{",".join(ops)}]'
                labels[b][si - 1] = (2 if composed else 1, ins.opcode, kind, cell)
                mask[b, si - 1] = True
                tgt[b, :, si - 1] = batch.out_regs[b, :, ws[0]]
            written.update(_written_regs(ins))
    n_in = (batch.in_slot_regs != AUX_CE_IGNORE).sum(1)
    iv = batch.anchors[:, :, 1:1 + N_IN].astype(np.int64) * (
        np.arange(N_IN)[None, None, :] < n_in[:, None, None])
    return feat, iv, tgt, mask, labels


def worker(q, seed, n_gen):
    proc = subprocess.Popen(
        f'python -m datagen --rule cap=2 --twins 2 --batch-size 128 --n-states 8 '
        f'--config configs/instr_alu_writers_chain010.json --seed {seed} '
        f'--gen-count {n_gen}', shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, cwd='/home/tophy/Projects/riscv-thoughts')
    for batch in RVT_FORMAT.reader(proc.stdout, Batch):
        f, iv, t, m, labels = chunk_features(batch)
        cells = [labels[b][s][3] for b, s in zip(*np.nonzero(m))]
        q.put((f, iv, t, m, cells))


class ResLN(nn.Module):
    """Pre-activation residual MLP: h = W0·x, then h += W·relu(LN(h)) per
    block, then out = W·relu(LN(h)). A block whose ReLUs all die passes h
    through unchanged, and LN stops activation scale compounding with depth."""

    def __init__(self, d_in, depth, init):
        super().__init__()
        self.inp = nn.Linear(d_in, 1024)
        self.norms = nn.ModuleList(nn.LayerNorm(1024) for _ in range(depth))
        self.lins = nn.ModuleList(nn.Linear(1024, 1024) for _ in range(depth - 1))
        self.out = nn.Linear(1024, N_OUT * 32)
        if init == 'he':
            for lin in (self.inp, *self.lins):
                nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
                nn.init.zeros_(lin.bias)

    def forward(self, x):
        h = self.inp(x)
        for norm, lin in zip(self.norms, self.lins):
            h = h + lin(F.relu(norm(h)))
        return self.out(F.relu(self.norms[-1](h)))


def mlp(d_in, depth, init):
    """With init 'he', hidden layers are He-initialised: nn.Linear's default
    weight variance is 1/(3·fan_in), a sixth of what keeps a ReLU stack's signal
    from shrinking, so at depth 12 it leaves the output independent of the
    input."""
    layers, d = [], d_in
    for _ in range(depth):
        lin = nn.Linear(d, 1024)
        if init == 'he':
            nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
            nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU()]; d = 1024
    return nn.Sequential(*layers, nn.Linear(d, N_OUT * 32))


def head_in(f, iv):
    A = iv.shape[1]
    return torch.cat([f.unsqueeze(1).expand(-1, A, -1),
                      _value_feat(iv).reshape(*iv.shape[:2], -1)], -1)


def evaluate(nets, dev, reference):
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with torch.no_grad(), open(HELDOUT, 'rb') as fh:
        for batch in RVT_FORMAT.reader(fh, Batch):
            f, iv, t, m, labels = chunk_features(batch)
            x = head_in(torch.from_numpy(f).to(dev), torch.from_numpy(iv).to(dev))
            bits = _value_bits(torch.from_numpy(t).to(dev))      # (B, A, S, 32)
            res = {}
            for d, net in nets.items():
                pred = net(x).view(*x.shape[:2], N_OUT, 32) > 0
                res[f'{d} layers'] = (pred.float() == bits).all(-1)
            if reference is not None:
                res['trained T2 head'] = reference(batch, t)
            for b, s in zip(*np.nonzero(m)):
                depth, op, kind, cell = labels[b][s]
                for name, ok in res.items():
                    c = cells[cell][name]
                    c[0] += int(ok[b, :, s].sum()); c[1] += ok.shape[1]
                keys = list(dict.fromkeys(['all', f'depth {depth}', kind]))
                if depth == 2 and op in ('SUB', 'ADD', 'XOR', 'SLL', 'ADDI'):
                    keys.append(f'depth 2, writer {op}')
                for key in keys:
                    for name, ok in res.items():
                        a = acc[key][name]
                        a[0] += int(ok[b, :, s].sum()); a[1] += ok.shape[1]
    return acc, cells


def make_reference(dev):
    """The trained T2's own vp head on the same slots, through T1's wiring."""
    from scripts._common import load_frozen_encoder, load_t2
    t1 = load_frozen_encoder(T1_PATH, dev)
    t2, _ = load_t2(T2_DIR, t1, dev)

    def ref(batch, t):
        split = _split_to_per_instruction(batch)
        nch, mni = batch.instr_lens.shape
        ct1, c0, c1, co, cact = _t2_assemble(t1, split, nch, mni, dev,
                                             route='binding')
        ess = t2.encode(ct1, c0, c1, co, cact)
        anchors = torch.from_numpy(batch.anchors).to(dev)
        in_slot = torch.from_numpy(batch.in_slot_regs.astype(np.int64)).to(dev)
        pred = vp_head_predict(t2, ess, vp_canonical_in_vals(anchors, in_slot))
        bits = _value_bits(torch.from_numpy(t).to(dev))
        return ((pred[..., :N_OUT, :32] > 0).float() == bits).all(-1)
    return ref


def report(acc, step):
    names = [f'{d} layers' for d in DEPTHS] + (
        ['trained T2 head'] if 'trained T2 head' in acc['all'] else [])
    print(f'\n== held-out, step {step} ==', flush=True)
    print(f'{"slots":<42} {"triples":>8} ' + ' '.join(f'{n:>15}' for n in names))
    order = ['all', 'depth 1', 'depth 2', 'R-type writer, composed + fresh input',
             'other depth-2'] + sorted(k for k in acc if k.startswith('depth 2, writer'))
    for k in order:
        if k in acc:
            print(f'{k:<42} {acc[k][names[0]][1]:>8} ' + ' '.join(
                f'{100 * acc[k][n][0] / acc[k][n][1]:14.3f}%' for n in names), flush=True)


def main():
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True   # as the trainers run
    torch.backends.cudnn.allow_tf32 = True
    q = mp.Queue(maxsize=256)
    procs = [mp.Process(target=worker, args=(q, 100 + 7 * i, 2), daemon=True)
             for i in range(12)]
    for p_ in procs:
        p_.start()
    d_in = K * D_INSTR + N_IN * 33
    nets = {d: (ResLN if ARCH[d] == 'resln' else mlp)(d_in, d, INIT[d]).to(dev)
            for d in DEPTHS}
    seen = Counter()               # training-stream count of each slot cell
    opts = {d: torch.optim.Adam(n.parameters(), lr=3e-4) for d, n in nets.items()}
    t0 = time.time()
    for step in range(1, STEPS + 1):
        f, iv, t, m, cell_list = q.get()
        seen.update(cell_list)
        x = head_in(torch.from_numpy(f).to(dev), torch.from_numpy(iv).to(dev))
        bits = _value_bits(torch.from_numpy(t).to(dev))
        mk = torch.from_numpy(m).to(dev)[:, None, :, None].float()
        lr = 3e-4 * min(1, step / 2000) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        for d, net in nets.items():
            out = net(x).view(*x.shape[:2], N_OUT, 32)
            loss = (F.binary_cross_entropy_with_logits(out, bits, reduction='none')
                    * mk).sum() / (mk.sum() * 32 * x.shape[1] / mk.shape[1]).clamp(min=1)
            for g in opts[d].param_groups:
                g['lr'] = lr
            opts[d].zero_grad(set_to_none=True)
            loss.backward()
            opts[d].step()
        if step % 10000 == 0:
            print(f'step {step}  {step / (time.time() - t0):.0f} steps/s', flush=True)
        if step % EVAL_EVERY == 0 or step == STEPS:
            final = step == STEPS
            for n in nets.values():
                n.eval()
            acc, cells = evaluate(nets, dev, make_reference(dev) if final else None)
            report(acc, step)
            if OUT:
                with open(f'{OUT}_step{step}.json', 'w') as fh:
                    json.dump({'heads': {f'{d} layers': f'{INIT[d]}:{ARCH[d]}'
                                         for d in DEPTHS},
                               'train_count': dict(seen),
                               'heldout': {c: dict(v) for c, v in cells.items()}}, fh)
                torch.save({d: n.state_dict() for d, n in nets.items()},
                           f'{OUT}_heads_step{step}.pt')
            for n in nets.values():
                n.train()
    for p_ in procs:
        p_.terminate()


if __name__ == '__main__':
    mp.set_start_method('fork')
    main()
