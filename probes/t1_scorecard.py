#!/usr/bin/env python3
"""Is this T1 fit to be the tier below T2? Two properties, one frozen model.

`scripts/eval.py` covers T1's equivariance and binding accuracy but stops
there: its GVN rows call `gvn_collapse`, which requires a T2. So the two
properties T2 actually consumes from T1 had no T1-only scorer.

  A. FAITHFULNESS — exact 32-bit match of T1's own vp head on single
     instructions, split by operand type. T1 is the tier whose essence T2 is
     fed, so a value T1 cannot describe is one T2 can never recover. Reported
     as exact match rather than as the training log's per-bit figure, which
     flatters a broken subset badly.

  B. EQUIVALENCE GEOMETRY — the distance between two instructions computing
     the SAME function, against the distance between two computing different
     ones. Every pair here has an identical binding (one live input, the same
     source register, the same destination), so the whole difference is in the
     essence and no rename is involved.

     `separation` = different / equivalent, larger is better. The classes are
     ordered by how much the architecture has to LEARN to see the equivalence:
     a commutative operand swap is free when the pool is a multiset over
     register slots, while recognising that `ADD x,y,y` and `SLLI x,y,1` agree
     requires knowing two different operations compute one function.

     The commutative row is the one to watch: a slot's leaf identity is its
     canonical read position, so read-order is present in the state and
     `a+b` versus `b+a` is representable. Whether that invariance holds
     structurally (the ranks move between registers, leaving the pool's
     multiset unchanged) or has to be trained per opcode is unsettled.

The encode is deterministic, so a distance between two sets reflects only the
instructions, never a resampling artifact.

Usage: t1_scorecard.py <t1_path> [n_pairs] [n_batches] [config]
"""
import contextlib
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.train import (_split_to_per_instruction, _value_bits,
                              _value_smooth)
from compressor.model import instruction_wiring
from datagen import (RVT_FORMAT, Batch, AUX_CE_IGNORE, MAX_INPUT_SLOTS,
                     MAX_OUTPUT_SLOTS, N_REGS, make_anchor_states)
from emulator import Instruction, I_TYPE
from emulator import run as emu_run
from scripts._common import resolve_device, load_frozen_encoder
from tokenizer import decode_instruction, encode_instruction, PAD

MAX_INSTR_TOKENS = 9      # tokenizer.MAX_INSTR_TOKENS is not exported; see
                          # compressor.train._MAX_INSTR_TOKENS

T1_PATH = sys.argv[1]
N_PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 400
CONFIG = sys.argv[4] if len(sys.argv) > 4 else 'configs/instr_alu_writers.json'
N_ANCHORS, ANCHOR_SEED, NBITS = 8, 0, 32

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)

anchors = torch.from_numpy(
    np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)


def encode(rows, bs=1024):
    """[Instruction, ...] -> (essence, unit-normalized essence). The wiring is
    the instruction's own, which is what T1 is given (registers stripped to
    wiring)."""
    out = []
    for i in range(0, len(rows), bs):
        sub = rows[i:i + bs]
        toks, pads, w0, w1, wo = [], [], [], [], []
        for ins in sub:
            t = encode_instruction(ins)
            assert len(t) <= MAX_INSTR_TOKENS, (ins, len(t))
            toks.append(t + [PAD] * (MAX_INSTR_TOKENS - len(t)))
            pads.append([False] * len(t) + [True] * (MAX_INSTR_TOKENS - len(t)))
            a, b, o = instruction_wiring(ins)
            w0.append(a); w1.append(b); wo.append(o)
        with torch.no_grad():
            ess = t1.encode(
                torch.tensor(toks, dtype=torch.long, device=dev),
                torch.tensor(pads, dtype=torch.bool, device=dev),
                torch.tensor(w0, dtype=torch.long, device=dev),
                torch.tensor(w1, dtype=torch.long, device=dev),
                torch.tensor(wo, dtype=torch.long, device=dev))
        out.append(F.normalize(ess.float(), dim=-1))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# A. Faithfulness: T1's own vp head, exact 32-bit match, on a real corpus.
# ---------------------------------------------------------------------------

CORPUS = f'/tmp/t1_scorecard_{CONFIG.split("/")[-1][:-5]}_{os.getpid()}.rvt'
# n_batches=0 runs part B alone. Part A generates a corpus with a local
# all-cores datagen, which perturbs any training run sharing the box; part B
# builds its instructions in-process and needs neither.
if NB:
    subprocess.run(
        f'python -m datagen --rule single --twins 0 --batch-size 128 '
        f'--n-states {N_ANCHORS} --anchor-seed {ANCHOR_SEED} '
        f'--config {CONFIG} --count {NB} > {CORPUS}',
        shell=True, check=True, stderr=subprocess.DEVNULL)

acc = defaultdict(lambda: [0.0, 0.0, 0.0])      # [exact, per-bit, triples]

with (open(CORPUS, 'rb') if NB else contextlib.nullcontext(())) as f:
    for batch in (RVT_FORMAT.reader(f, Batch) if NB else ()):
        split = _split_to_per_instruction(batch)
        (itok, ipad, chunk_idx, _, n_per_chunk, w0, w1, wo) = split
        if itok.shape[0] == 0:
            continue
        nch = batch.instr_lens.shape[0]
        # --rule single: one instruction per row, so instruction i IS row i.
        assert itok.shape[0] == nch and (n_per_chunk == 1).all(), \
            'part A needs a single-instruction corpus (--rule single)'
        with torch.no_grad():
            ess = t1.encode(
                torch.from_numpy(itok.astype(np.int64)).to(dev),
                torch.from_numpy(ipad).to(dev),
                torch.from_numpy(w0).to(dev), torch.from_numpy(w1).to(dev),
                torch.from_numpy(wo).to(dev))

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
            pred = t1.vp_head(torch.cat([v, ifeat], dim=-1)).reshape(
                nch, N_ANCHORS, MAX_OUTPUT_SLOTS, NBITS + 1)
            tgt = out_regs.gather(-1, out_slot.clamp(min=0).unsqueeze(1)
                                  .expand(-1, N_ANCHORS, -1))
            hit = ((pred[..., :NBITS] > 0).float() == _value_bits(tgt))
            exact = hit.all(-1).float()                 # (B, A, MAX_OUT)
            perbit = hit.float().mean(-1)

        vnp = valid.cpu().numpy()
        ex = exact.cpu().numpy(); pb = perbit.cpu().numpy()
        for b in range(nch):
            try:
                ins, _ = decode_instruction(batch.tokens[b].tolist(), 0)
            except Exception:
                continue
            kind = 'I-type' if ins.opcode in I_TYPE else 'R-type'
            for s in range(MAX_OUTPUT_SLOTS):
                if not vnp[b, s]:
                    continue
                for key in ('all', kind, f'op:{ins.opcode}'):
                    a = acc[key]
                    a[0] += ex[b, :, s].sum(); a[1] += pb[b, :, s].sum()
                    a[2] += N_ANCHORS


def show(key, label):
    hit, bit, n = acc[key]
    if not n:
        return
    print(f'  {label:<12} exact {100 * hit / n:>9.4f}%   '
          f'per-bit {100 * bit / n:>8.4f}%   ({int(n)} triples)')


if NB:
    print(f'== A. vp faithfulness — T1\'s own head, {CONFIG.split("/")[-1]} ==')
    show('all', 'all')
    show('R-type', 'R-type')
    show('I-type', 'I-type')
    print('  by opcode:')
    for k in sorted(k for k in acc if k.startswith('op:')):
        show(k, '  ' + k[3:])


# ---------------------------------------------------------------------------
# B. Equivalence geometry: same function vs different function, same binding.
# ---------------------------------------------------------------------------

# Each entry builds (a, b) from one (rd, rs, rt) draw. `equal` is what the
# emulator is asserted to agree with, so a family that stops being equivalent
# fails loudly instead of quietly scoring well.
FAMILIES = [
    ('commutative swap', True,
     lambda o, d, s, t, r: (Instruction(o, d, s, t), Instruction(o, d, t, s)),
     ('ADD', 'AND', 'OR', 'XOR')),
    ('identity spelling', True,
     lambda o, d, s, t, r: (Instruction('ADDI', d, s, 0),
                            Instruction(o, d, s, 0)),
     ('ORI', 'XORI', 'SLLI')),
    ('double vs shift', True,
     lambda o, d, s, t, r: (Instruction('ADD', d, s, s),
                            Instruction('SLLI', d, s, 1)),
     ('SLLI',)),
    ('different shift', False,
     lambda o, d, s, t, r: (Instruction('SLLI', d, s, 1),
                            Instruction('SLLI', d, s, 2)),
     ('SLLI',)),
    ('different opcode', False,
     lambda o, d, s, t, r: (Instruction('ADD', d, s, t),
                            Instruction('SUB', d, s, t)),
     ('SUB',)),
]

print()
print('== B. equivalence geometry — identical binding, distance in the essence ==')
rng = np.random.default_rng(0)
means = {}
for name, want_equal, make, opcodes in FAMILIES:
    A, Bp = [], []
    for _ in range(N_PAIRS):
        op = opcodes[int(rng.integers(len(opcodes)))]
        d = int(rng.integers(1, 32))
        s = int(rng.integers(1, 32))
        t = int(rng.integers(1, 32))
        if name == 'commutative swap' and s == t:
            continue            # a swap of one register with itself is not a pair
        a, b = make(op, d, s, t, rng)
        A.append(a); Bp.append(b)
    # Hand-built instructions: check the tokens round-trip and that the pair
    # really is (or is not) behaviorally identical, rather than trusting the
    # construction.
    for k in range(0, len(A), max(1, len(A) // 40)):
        for ins in (A[k], Bp[k]):
            toks = encode_instruction(ins)
            got, _ = decode_instruction(list(toks), 0)
            assert str(got) == str(ins), (got, ins)
        st = np.zeros(32, dtype=np.int32)
        st[1:] = rng.integers(-(2 ** 20), 2 ** 20, 31).astype(np.int32)
        ra = emu_run([A[k]], st.copy())[0].regs
        rb = emu_run([Bp[k]], st.copy())[0].regs
        assert np.array_equal(ra, rb) == want_equal, (A[k], Bp[k], want_equal)

    d = (encode(A) - encode(Bp)).norm(dim=-1)
    means[name] = float(d.mean())
    print(f'  {name:<18} {"equivalent" if want_equal else "DIFFERENT ":<11} '
          f'mean {means[name]:.4f}  median {float(d.median()):.4f}  '
          f'(n={len(A)})')

denom = means.get('different opcode')
if denom:
    print()
    print('  separation against `different opcode` (larger is better):')
    for name, want_equal, _, _ in FAMILIES:
        if want_equal and name in means:
            print(f'    {name:<18} {denom / means[name]:>6.1f}x')
