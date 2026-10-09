#!/usr/bin/env python3
"""Does T1 dropping an irrelevant read cost the OTHER instruction its value?

On the corpus, chunks where T1's emitted wiring omits a source register the
tokens read score far worse than chunks where it does not, and the failure lands
on the slot the other instruction wrote. That comparison selects on the chunk
containing a degenerate instruction, so it cannot separate "the drop costs
information" from "these chunks are harder".

This builds the comparison instead of finding it. The SECOND instruction is held
fixed — same opcode, same source register, same immediate, same destination —
and only the FIRST instruction varies:

  drop      `SLTIU rdA, rs, -k`. A 12-bit immediate sign-extends to a huge
            UNSIGNED value, so the comparison is true for every rs and the
            output is the constant 1. T1's live_in head is behavioral and
            correctly reports rs as not read, so the emitted wiring omits it.
  keep-slt  `SLTI rdA, rs, +k`. The signed comparison against a small immediate
            is decided by rs's sign bit, so this one register-and-immediate
            compare does depend on rs and the read is emitted. The nearest
            possible control: same shape of instruction, opposite outcome.
  keep-add  `ADDI rdA, rs, +k`. A second no-drop control on an unrelated
            opcode, so the result cannot be an artifact of the compare family.
  drop-far  `SLTIU rdA, rsB, -k` on a register the second instruction does not
            read. The drop still happens, but now it removes a register that is
            nothing to the scored slot — separating "a read was dropped" from
            "the scored slot's own source was dropped elsewhere".

The second instruction is at dataflow depth 1 in every group, and its
destination is a register the first instruction does not touch. Scoring is
restricted to that slot, so any difference between the groups is caused by the
first instruction alone.

Usage: vp_drop_control.py <t2_dir> <t1_path> [n_chunks]
"""
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/tophy/Projects/riscv-thoughts')

from compressor.model import instruction_wiring
from compressor.train import (_t2_assemble, _t1_predicted_wiring, _value_bits,
                              _value_smooth)
from datagen import AUX_CE_IGNORE, MAX_INPUT_SLOTS, MAX_OUTPUT_SLOTS, N_REGS
from datagen.compare import make_anchor_states, precompute_chunk
from emulator import Instruction
from scripts._common import resolve_device, load_frozen_encoder, load_t2
from tokenizer import encode_instruction, PAD

MAX_INSTR_TOKENS = 9
N_ANCHORS, ANCHOR_SEED, NBITS = 8, 0, 32

T2_DIR, T1_PATH = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

dev = resolve_device('auto')
t1 = load_frozen_encoder(T1_PATH, dev)
t2, _ = load_t2(T2_DIR, t1, dev)

anchors_np = np.asarray(make_anchor_states(N_ANCHORS, ANCHOR_SEED))
anchors = torch.from_numpy(anchors_np).to(dev)
canon_pos = torch.arange(1, MAX_INPUT_SLOTS + 1, device=dev).clamp(max=N_REGS - 1)
ANCH_IN = anchors[:, canon_pos].to(torch.int64)

# The second instruction's opcode pool: I-type ALU ops whose output depends on
# both the register and the immediate, so the slot being scored is a real
# 32-bit value rather than a constant or a flag.
OP2 = ['ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI']
SHIFTS = {'SLLI', 'SRLI'}


def build(chunks):
    """[[Instruction, ...], ...] -> the split tuple `_t2_assemble` consumes."""
    toks, pads, ci, si, w0, w1, wo = [], [], [], [], [], [], []
    for c, instrs in enumerate(chunks):
        for s, ins in enumerate(instrs):
            t = encode_instruction(ins)
            toks.append(t + [PAD] * (MAX_INSTR_TOKENS - len(t)))
            pads.append([False] * len(t) + [True] * (MAX_INSTR_TOKENS - len(t)))
            a, b, o = instruction_wiring(ins)
            ci.append(c); si.append(s); w0.append(a); w1.append(b); wo.append(o)
    return (np.array(toks, dtype=np.int64), np.array(pads, dtype=bool),
            np.array(ci), np.array(si),
            np.array([len(x) for x in chunks]),
            np.array(w0), np.array(w1), np.array(wo))


def score(chunks, target_reg, bs=256):
    """Exact-32-bit vp rate on the slot `target_reg[i]`, plus how often T1's
    emitted wiring omits a source the tokens read.

    Targets come from the same oracle the trainer supervises against
    (`precompute_chunk`), and the prediction is the model's own trained head —
    nothing here is fitted."""
    hits = drops = trip = 0.0
    for i in range(0, len(chunks), bs):
        sub, tgt_regs = chunks[i:i + bs], target_reg[i:i + bs]
        split = build(sub)
        n = len(sub)
        mni = max(len(x) for x in sub)
        with torch.no_grad():
            ct1, c0, c1, co, cact = _t2_assemble(
                t1, split, n, mni, dev, route='binding')
            ess = t2.encode(ct1, c0, c1, co, cact)

        pre = [precompute_chunk(c, anchors_np) for c in sub]
        in_slot = torch.from_numpy(
            np.stack([p.in_slot_regs for p in pre]).astype(np.int64)).to(dev)
        out_regs = torch.from_numpy(
            np.stack([p.out_regs for p in pre]).astype(np.int64)).to(dev)
        # The scored slot is the second instruction's destination, looked up by
        # register rather than by slot index (the oracle orders slots by SSA
        # write order, which is not the instruction order in general).
        sel = torch.tensor([int(r) for r in tgt_regs], device=dev)
        tgt = out_regs.gather(
            -1, sel.view(-1, 1, 1).expand(-1, N_ANCHORS, 1)).squeeze(-1)

        with torch.no_grad():
            iv = (ANCH_IN.unsqueeze(0)
                  * (in_slot != AUX_CE_IGNORE).to(torch.int64).unsqueeze(1))
            ifeat = torch.cat(
                [_value_bits(iv), _value_smooth(iv.float()).unsqueeze(-1)],
                dim=-1).reshape(n, N_ANCHORS, -1)
            v = F.normalize(ess, dim=-1).unsqueeze(1).expand(-1, N_ANCHORS, -1)
            pred = t2.vp_head(torch.cat([v, ifeat], dim=-1)).reshape(
                n, N_ANCHORS, MAX_OUTPUT_SLOTS, NBITS + 1)
        # find the slot index whose register is the one being scored
        oslot = torch.from_numpy(
            np.stack([p.out_slot_regs for p in pre]).astype(np.int64)).to(dev)
        idx = (oslot == sel.unsqueeze(-1)).float().argmax(-1)
        p = pred[torch.arange(n, device=dev), :, :, :][
            torch.arange(n, device=dev).unsqueeze(1),
            torch.arange(N_ANCHORS, device=dev).unsqueeze(0),
            idx.unsqueeze(1)]                                # (n, A, NBITS+1)
        hits += float(((p[..., :NBITS] > 0).float()
                       == _value_bits(tgt)).all(-1).float().sum())
        trip += n * N_ANCHORS

        # Did T1 actually drop the read? Compare its emitted wiring against the
        # tokens' — the property this probe is built to vary.
        (_, _, chunk_idx, slot_idx, _, w0, w1, _) = split
        pw0 = c0.cpu().numpy()[chunk_idx, slot_idx]
        pw1 = c1.cpu().numpy()[chunk_idx, slot_idx]
        for j in range(len(w0)):
            ps = {int(pw0[j]), int(pw1[j])} - {0}
            ts = {int(w0[j]), int(w1[j])} - {0}
            drops += ps < ts
    return 100 * hits / trip, drops, trip


rng = np.random.default_rng(0)
groups = defaultdict(lambda: ([], []))
for _ in range(N):
    rs, rd_a, rd_2, rs_b = rng.choice(np.arange(1, 32), size=4, replace=False)
    op2 = OP2[int(rng.integers(len(OP2)))]
    imm2 = int(rng.integers(1, 32) if op2 in SHIFTS
               else rng.integers(-2048, 2048))
    k = int(rng.integers(1, 2048))
    second = Instruction(op2, int(rd_2), int(rs), imm2)
    for name, first in (
            ('drop      SLTIU rdA,rs,-k', Instruction('SLTIU', int(rd_a),
                                                      int(rs), -k)),
            ('keep-slt  SLTI  rdA,rs,+k', Instruction('SLTI', int(rd_a),
                                                      int(rs), k)),
            ('keep-add  ADDI  rdA,rs,+k', Instruction('ADDI', int(rd_a),
                                                      int(rs), k)),
            ('drop-far  SLTIU rdA,rsB,-k', Instruction('SLTIU', int(rd_a),
                                                       int(rs_b), -k))):
        groups[name][0].append([first, second])
        groups[name][1].append(int(rd_2))

print()
print(f'checkpoint = {T2_DIR}')
print(f'  the second instruction is identical across the three groups; only '
      f'the first varies   (n={N})')
print()
print(f'    {"first instruction":28s} {"triples":>9s} {"exact":>10s} '
      f'{"reads dropped":>14s}')
for name in sorted(groups):
    chunks, tgt = groups[name]
    pct, drops, trip = score(chunks, tgt)
    print(f'    {name:28s} {int(trip):9d} {pct:9.4f}% '
          f'{100 * drops / (2 * len(chunks)):13.2f}%')
print()
