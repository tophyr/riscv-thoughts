"""CPU vs GPU emulator parity.

For a battery of random (instruction, register file, pc) across every
ALU / shift / compare / branch / jump / upper-immediate opcode, run ONE
CPU step and ONE GPU batch_execute lane and assert that the destination
register value and the final PC match.

The two emulators are independent implementations (sequential numpy vs
batched torch.where), so agreement is a real cross-check. The GPU
returns ``data_val = final_regs[rd]`` for register-writing ops and 0 for
branches, plus ``final_pc``.

DOCUMENTED DIVERGENCE (scoped out, NOT a bug): loads and stores use
different memory models — the CPU uses SparseMemory backed by per-pair
RNG fill, the GPU uses a deterministic address hash. They are not
expected to agree on loaded values, so LOAD/STORE opcodes are excluded
from this parity battery. See test_load_store_divergence below, which
asserts the divergence is real (so this exclusion stays honest).
"""

import numpy as np
import pytest
import torch

from emulator import Instruction, run, make_ctx
from emulator import batch_execute, instructions_to_tensors

SEED = 12345
N_PER_OP = 50
DEVICE = 'cpu'

R_OPS = ['ADD', 'SUB', 'XOR', 'OR', 'AND', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU']
I_OPS = ['ADDI', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI', 'SLTI', 'SLTIU']
B_OPS = ['BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU']
U_OPS = ['LUI', 'AUIPC']
J_OPS = ['JAL', 'JALR']

PARITY_OPS = R_OPS + I_OPS + B_OPS + U_OPS + J_OPS


def _random_instruction(op, rng):
    """Build a random Instruction of the given opcode."""
    rd = int(rng.integers(1, 32))
    rs1 = int(rng.integers(0, 32))
    rs2 = int(rng.integers(0, 32))
    if op in R_OPS:
        return Instruction(op, rd, rs1, rs2)
    if op in I_OPS:
        if op in ('SLLI', 'SRLI', 'SRAI'):
            imm = int(rng.integers(0, 32))      # shift amount
        else:
            imm = int(rng.integers(-2048, 2048))   # 12-bit signed
        return Instruction(op, rd, rs1, imm)
    if op in B_OPS:
        imm = int(rng.integers(-1024, 1024)) * 2   # 13-bit signed, even
        return Instruction(op, rs1, rs2, imm)
    if op in U_OPS:
        imm = int(rng.integers(0, 2**20))          # 20-bit
        return Instruction(op, rd, imm)
    if op == 'JAL':
        imm = int(rng.integers(-1024, 1024)) * 2
        return Instruction(op, rd, imm)
    if op == 'JALR':
        imm = int(rng.integers(-2048, 2048))
        return Instruction(op, rd, rs1, imm)
    raise ValueError(op)


def _gpu_step(instr, regs, pc):
    """Run one GPU lane; return (data_val_int32, final_pc_uint32)."""
    op, rd, rs1, rs2, imm = instructions_to_tensors([instr], DEVICE)
    gregs = torch.from_numpy(regs.reshape(1, 32).copy())
    gpc = torch.tensor([pc], dtype=torch.int32)
    data_val, final_pc = batch_execute(op, rd, rs1, rs2, imm, gregs, gpc)
    # data_val is int64; reinterpret low 32 bits as signed int32 so it
    # compares against the CPU's int32 register value.
    dv = int(np.int32(np.uint32(int(data_val[0]) & 0xFFFFFFFF)))
    return dv, int(final_pc[0]) & 0xFFFFFFFF


@pytest.mark.parametrize('op', PARITY_OPS)
def test_cpu_gpu_parity(op):
    rng = np.random.default_rng(SEED + hash(op) % 1000)
    ctx = make_ctx()
    for _ in range(N_PER_OP):
        regs = rng.integers(-2**31, 2**31, size=32, dtype=np.int32)
        regs[0] = 0
        pc = int(rng.integers(0, 1024)) * 4
        instr = _random_instruction(op, rng)
        rd = instr.args[0] if op not in B_OPS else None

        state, cpu_pc, _ = run([instr], regs=regs.copy(), pc=pc,
                               max_steps=1, _ctx=ctx)
        gpu_rd, gpu_pc = _gpu_step(instr, regs, pc)

        assert int(np.uint32(cpu_pc)) == gpu_pc, (
            f'{op}: PC mismatch cpu={cpu_pc} gpu={gpu_pc} instr={instr!r}')
        if op not in B_OPS:
            assert int(state.regs[rd]) == gpu_rd, (
                f'{op}: rd value mismatch cpu={int(state.regs[rd])} '
                f'gpu={gpu_rd} instr={instr!r}')


def test_x0_as_rd_parity():
    """Writing to x0 is a no-op on both emulators."""
    instr = Instruction('ADDI', 0, 0, 5)
    state, _, _ = run([instr], max_steps=1)
    gpu_rd, _ = _gpu_step(instr, np.zeros(32, dtype=np.int32), 0)
    assert state.regs[0] == 0
    assert gpu_rd == 0


def test_load_store_divergence_is_real():
    """Documented divergence: CPU (SparseMemory+RNG) and GPU (address
    hash) use different memory models, so loaded values are NOT expected
    to match. This pins that they actually differ, keeping the parity
    battery's LOAD/STORE exclusion honest. If a future change unified
    the memory models, this test would flag it for review.
    """
    regs = np.zeros(32, dtype=np.int32)
    regs[3] = 1000      # base address far from 0
    instr = Instruction('LW', 5, 0, 3)   # TinyFive order (rd, imm, rs1)
    state, _, _ = run([instr], regs=regs.copy(),
                      rng=np.random.default_rng(1), max_steps=1)
    gpu_rd, _ = _gpu_step(instr, regs, 0)
    assert int(state.regs[5]) != gpu_rd
