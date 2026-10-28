"""Equivariance is the load-bearing property of the new T1 encoder: a
register rename must permute the emitted state EXACTLY and leave the
invariant essence unchanged. These pin that by construction (no training),
plus the head shapes the binding losses depend on.
"""
import pytest
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

    g = torch.Generator().manual_seed(3)
    perm = torch.arange(R)
    perm[1:] = torch.randperm(R - 1, generator=g) + 1     # fix slot 0 (x0)

    instrs_p = [_relabel(i, perm.tolist()) for i in instrs]
    tok_p, pad_p, in0_p, in1_p, out_p = _tokenize(instrs_p)

    with torch.no_grad():
        T, ess = m.encode_state(tok, pad, in0, in1, out)
        T_p, ess_p = m.encode_state(tok_p, pad_p, in0_p, in1_p, out_p)

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

    g = torch.Generator().manual_seed(5)
    perm = torch.arange(R)
    perm[1:] = torch.randperm(R - 1, generator=g) + 1
    instrs_p = [_relabel(i, perm.tolist()) for i in instrs]
    tok_p, pad_p, in0_p, in1_p, out_p = _tokenize(instrs_p)

    with torch.no_grad():
        T, _ = m.encode_state(tok, pad, in0, in1, out)
        T_p, _ = m.encode_state(tok_p, pad_p, in0_p, in1_p, out_p)
    w0, w1, wo = _t1_predicted_wiring(m, T)
    w0p, w1p, wop = _t1_predicted_wiring(m, T_p)

    pl = perm.tolist()
    relab = lambda x: pl[x] if x != 0 else 0   # x0 (absent) is fixed
    assert [relab(x) for x in w0.tolist()] == w0p.tolist()
    assert [relab(x) for x in w1.tolist()] == w1p.tolist()
    assert [relab(x) for x in wo.tolist()] == wop.tolist()


def test_value_preserved_on_read(monkeypatch):
    """Core invariant for T2 dataflow: reading a slot never changes its
    value channels. A leaf's colour depends on the whole chunk's graph, so the
    one-step prefix is compared with the two-step run at a fixed colouring
    (zero refinement rounds); the write at step 0 must survive step 1's read."""
    import compressor.model as M
    monkeypatch.setattr(M, 'WL_ROUNDS', 0)
    torch.manual_seed(0)
    core = RegisterStateMachine(d_value=8, d_event=4, d_content=6, n_regs=32)
    B = 3
    content = torch.randn(B, 2, 6)
    in0 = torch.tensor([[3, 5]]).expand(B, 2).clone()    # step1 reads r5
    in1 = torch.tensor([[4, 0]]).expand(B, 2).clone()
    out = torch.tensor([[5, 9]]).expand(B, 2).clone()    # step0 writes r5
    active = torch.ones(B, 2)
    with torch.no_grad():
        T0, _ = core(content[:, :1], in0[:, :1], in1[:, :1], out[:, :1],
                     active[:, :1])
        T1, _ = core(content, in0, in1, out, active)
    assert (T0[:, 5, :8] - T1[:, 5, :8]).abs().max().item() < 1e-6


def test_pool_projects_touched_slots_like_all_slots():
    """The pool projects only the slots a chunk wrote and fills the rest with
    the projection's bias. That must equal the dense projection of the state
    with every unwritten slot zeroed — essence and every parameter gradient —
    including chunks with padding steps, repeated registers and x0 operands."""
    torch.manual_seed(0)
    core = RegisterStateMachine(d_value=16, d_event=4, d_content=6, n_regs=32,
                                op_cell_depth=2, n_pool_heads=4)
    B, K = 6, 3
    content = torch.randn(B, K, 6)
    in0 = torch.randint(0, 32, (B, K))
    in1 = torch.randint(0, 32, (B, K))
    out = torch.randint(0, 32, (B, K))
    in1[0] = in0[0]                        # one register read twice
    in0[1, 0] = 0                          # an x0 operand
    active = torch.ones(B, K)
    active[2, 2] = 0                       # a padding step

    def dense(pool_src, written, max_touched):
        src = pool_src * written.unsqueeze(-1)
        return core.pool_k(src), core.pool_v(src)

    def run(project):
        core.zero_grad()
        orig = core._project_touched
        core._project_touched = project
        try:
            T, ess = core(content, in0, in1, out, active)
        finally:
            core._project_touched = orig
        ess.pow(2).sum().backward()
        return ess.detach(), {n: p.grad.clone() for n, p in core.named_parameters()
                              if p.grad is not None}

    ess_t, g_t = run(core._project_touched)
    ess_d, g_d = run(dense)
    assert torch.allclose(ess_t, ess_d, atol=1e-6)
    assert g_t.keys() == g_d.keys()
    for n in g_d:
        assert torch.allclose(g_t[n], g_d[n], atol=1e-5), n


def test_head_on_touched_slots_matches_the_dense_head():
    """apply_head_touched evaluates a per-slot head only on slots a chunk can
    have touched and gives the rest head(0). On a real core state it must equal
    the dense head — output and every gradient — and too small a bound must
    not."""
    from compressor.model import apply_head_touched
    torch.manual_seed(0)
    core = RegisterStateMachine(d_value=16, d_event=4, d_content=6, n_regs=32)
    head = torch.nn.Sequential(torch.nn.Linear(16 + 4 + core.d_colour, 8), torch.nn.ReLU(),
                               torch.nn.Linear(8, 1))
    B, K = 5, 3
    content = torch.randn(B, K, 6)
    in0, in1, out = (torch.randint(0, 32, (B, K)) for _ in range(3))
    active = torch.ones(B, K)
    active[1, 2] = 0

    def run(f):
        core.zero_grad(); head.zero_grad()
        T, _ = core(content, in0, in1, out, active)
        y = f(T)
        (y * torch.arange(32.0).view(1, -1, 1)).sum().backward()
        params = list(core.parameters()) + list(head.parameters())
        return y.detach(), [p.grad.clone() for p in params if p.grad is not None]

    y_d, g_d = run(head)
    y_t, g_t = run(lambda T: apply_head_touched(head, T, 3 * K))
    assert torch.allclose(y_t, y_d, atol=1e-6)
    assert len(g_t) == len(g_d)
    for a, b in zip(g_t, g_d):
        assert torch.allclose(a, b, atol=1e-5)
    y_short, _ = run(lambda T: apply_head_touched(head, T, 1))
    assert not torch.allclose(y_short, y_d, atol=1e-6)


def test_head_shapes():
    m = _model()
    tok, pad, in0, in1, out = _tokenize([Instruction('ADD', 5, 3, 4)])
    T, ess = m.encode_state(tok, pad, in0, in1, out)
    assert ess.shape == (1, m.d_out)
    core = m.compressor.core
    assert core.apply_slot_head(m.live_in_head, T).shape == (1, m.n_regs)
    assert core.apply_slot_head(m.in_score_head, T).shape == (1, m.n_regs)
    assert m.pc_writes_head(ess).shape == (1, 1)


@pytest.mark.parametrize('chunk, shape', [
    ([('ADD', 5, 6, 7), ('SUB', 8, 9, 10)], 'plain'),
    ([('ADDI', 5, 6, 1), ('ADD', 8, 5, 9)], 'composed'),
    ([('ADD', 5, 6, 7), ('SUB', 8, 6, 9)], 'shared input'),
    ([('ADD', 5, 6, 7), ('ADDI', 5, 8, 1)], 'dead write'),
    ([('ADD', 5, 5, 0), ('SUB', 8, 6, 9)], 'self-write + x0 operand'),
    ([('XOR', 5, 6, 6), ('ADDI', 5, 5, 1)], 'aliased + composed + ignored read + self-write'),
])
def test_chunk_shape(chunk, shape):
    from compressor.eval import _chunk_shape
    assert _chunk_shape([Instruction(*i) for i in chunk]) == shape
