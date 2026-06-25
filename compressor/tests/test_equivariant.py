"""Equivariance is the load-bearing property of the new T1 encoder: a
register rename must permute the emitted state EXACTLY and leave the
invariant essence unchanged. These pin that by construction (no training),
plus the head shapes the binding losses depend on.
"""
import numpy as np
import torch

from emulator import Instruction, R_TYPE, I_TYPE, B_TYPE
from tokenizer import encode_instruction, PAD, VOCAB_SIZE
from compressor.model import T1Compressor, RegisterStateMachine, instruction_wiring


def _tokenize(instrs):
    enc = [encode_instruction(i) for i in instrs]
    L = max(len(e) for e in enc)
    tok = np.full((len(enc), L), PAD, dtype=np.int64)
    pad = np.ones((len(enc), L), dtype=bool)
    for i, e in enumerate(enc):
        tok[i, :len(e)] = e
        pad[i, :len(e)] = False
    w = np.array([instruction_wiring(i) for i in instrs], dtype=np.int64)
    return (torch.from_numpy(tok), torch.from_numpy(pad),
            torch.from_numpy(w[:, 0]), torch.from_numpy(w[:, 1]),
            torch.from_numpy(w[:, 2]))


def _relabel(instr, perm):
    op, a = instr.opcode, list(instr.args)
    if op in R_TYPE:
        a = [perm[a[0]], perm[a[1]], perm[a[2]]]
    elif op in (I_TYPE | {'JALR'}):
        a = [perm[a[0]], perm[a[1]], a[2]]
    elif op in B_TYPE:
        a = [perm[a[0]], perm[a[1]], a[2]]
    elif op in ('LUI', 'AUIPC', 'JAL'):
        a = [perm[a[0]], a[1]]
    return Instruction(op, *a)


def _model():
    torch.manual_seed(0)
    return T1Compressor(vocab_size=VOCAB_SIZE, d_model=64, n_heads=4,
                        n_layers=2, max_window=16, d_out=48, d_event=16).eval()


def test_rename_permutes_state_and_fixes_essence():
    """enc(pi.C) = pi.enc(C) on the state, essence invariant — exactly."""
    m = _model()
    R = m.n_regs
    instrs = [Instruction('ADD', 5, 3, 4), Instruction('ADDI', 7, 2, 100),
              Instruction('SUB', 9, 1, 8), Instruction('XOR', 2, 5, 6)]
    B = len(instrs)
    tok, pad, in0, in1, out = _tokenize(instrs)
    tags = torch.randn(B, R); tags[:, 0] = 0.0

    g = torch.Generator().manual_seed(3)
    perm = torch.arange(R)
    perm[1:] = torch.randperm(R - 1, generator=g) + 1     # fix slot 0 (x0)

    instrs_p = [_relabel(i, perm.tolist()) for i in instrs]
    tok_p, pad_p, in0_p, in1_p, out_p = _tokenize(instrs_p)
    tags_p = torch.empty_like(tags); tags_p[:, perm] = tags

    with torch.no_grad():
        T, ess = m.encode_state(tok, pad, in0, in1, out, tags)
        T_p, ess_p = m.encode_state(tok_p, pad_p, in0_p, in1_p, out_p, tags_p)

    T_exp = torch.empty_like(T); T_exp[:, perm] = T
    assert (T_p - T_exp).abs().max().item() < 1e-4
    assert (ess_p - ess).abs().max().item() < 1e-4


def test_predicted_wiring_is_equivariant():
    """T2 routes off T1's PREDICTED binding. The derived wiring must
    permute EXACTLY under a register rename (argmax/topk of equivariant
    per-slot readouts permute with π; counts are rename-invariant) — this is
    what keeps the principled T2 exactly equivariant."""
    from compressor.train import _t1_predicted_wiring
    m = _model()
    R = m.n_regs
    instrs = [Instruction('ADD', 5, 3, 4), Instruction('ADDI', 7, 2, 100),
              Instruction('SUB', 9, 1, 8), Instruction('XOR', 2, 5, 6)]
    B = len(instrs)
    tok, pad, in0, in1, out = _tokenize(instrs)
    tags = torch.randn(B, R); tags[:, 0] = 0.0

    g = torch.Generator().manual_seed(5)
    perm = torch.arange(R)
    perm[1:] = torch.randperm(R - 1, generator=g) + 1
    instrs_p = [_relabel(i, perm.tolist()) for i in instrs]
    tok_p, pad_p, in0_p, in1_p, out_p = _tokenize(instrs_p)
    tags_p = torch.empty_like(tags); tags_p[:, perm] = tags

    with torch.no_grad():
        T, _ = m.encode_state(tok, pad, in0, in1, out, tags)
        T_p, _ = m.encode_state(tok_p, pad_p, in0_p, in1_p, out_p, tags_p)
    w0, w1, wo = _t1_predicted_wiring(m, T)
    w0p, w1p, wop = _t1_predicted_wiring(m, T_p)

    pl = perm.tolist()
    relab = lambda x: pl[x] if x != 0 else 0   # x0 (absent) is fixed
    assert [relab(x) for x in w0.tolist()] == w0p.tolist()
    assert [relab(x) for x in w1.tolist()] == w1p.tolist()
    assert [relab(x) for x in wo.tolist()] == wop.tolist()


def test_value_preserved_on_read():
    """Core invariant for T2 dataflow: reading a slot never changes its
    value channels."""
    torch.manual_seed(0)
    core = RegisterStateMachine(d_value=8, d_event=4, d_content=6, n_regs=32)
    B = 3
    content = torch.randn(B, 2, 6)
    in0 = torch.tensor([[3, 5]]).expand(B, 2).clone()    # step1 reads r5
    in1 = torch.tensor([[4, 0]]).expand(B, 2).clone()
    out = torch.tensor([[5, 9]]).expand(B, 2).clone()    # step0 writes r5
    active = torch.ones(B, 2)
    tags = torch.randn(B, 32); tags[:, 0] = 0.0
    with torch.no_grad():
        T0, _ = core(content[:, :1], in0[:, :1], in1[:, :1], out[:, :1],
                     active[:, :1], tags)
        T1, _ = core(content, in0, in1, out, active, tags)
    assert (T0[:, 5, :8] - T1[:, 5, :8]).abs().max().item() < 1e-6


def test_head_shapes():
    m = _model()
    tok, pad, in0, in1, out = _tokenize([Instruction('ADD', 5, 3, 4)])
    tags = torch.randn(1, m.n_regs); tags[:, 0] = 0.0
    T, ess = m.encode_state(tok, pad, in0, in1, out, tags)
    assert ess.shape == (1, m.d_out)
    core = m.compressor.core
    assert core.apply_slot_head(m.live_in_head, T).shape == (1, m.n_regs)
    assert core.apply_slot_head(m.in_score_head, T).shape == (1, m.n_regs)
    assert m.pc_writes_head(ess).shape == (1, 1)
