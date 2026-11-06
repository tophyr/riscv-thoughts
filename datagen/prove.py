"""Exact equivalence of two ALU chunks, by bit-vector proof.

Two chunks are equivalent when their register-file maps agree on every
register file (RISCV_DESIGN, "Collapse", "The behaviour the distance
measures"). Sampled states cannot establish it — two comparisons can disagree
on one input in 2³² — so a pair whose equivalence becomes a training label is
proved: both chunks are executed symbolically over 32-bit registers (x0 = 0)
and z3 is asked for a register file on which any register differs.
"""
import z3

from emulator import R_TYPE, I_TYPE

_SHIFT = {'SLL', 'SRL', 'SRA', 'SLLI', 'SRLI', 'SRAI'}
_BASE = {'ADDI': 'ADD', 'SLTI': 'SLT', 'SLTIU': 'SLTU', 'XORI': 'XOR',
         'ORI': 'OR', 'ANDI': 'AND', 'SLLI': 'SLL', 'SRLI': 'SRL', 'SRAI': 'SRA'}


def _op(op, a, b):
    one, zero = z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)
    if op in ('SLL', 'SRL', 'SRA'):
        b = b & 31
    return {
        'ADD': lambda: a + b, 'SUB': lambda: a - b, 'XOR': lambda: a ^ b,
        'OR': lambda: a | b, 'AND': lambda: a & b, 'SLL': lambda: a << b,
        'SRL': lambda: z3.LShR(a, b), 'SRA': lambda: a >> b,
        'SLT': lambda: z3.If(a < b, one, zero),
        'SLTU': lambda: z3.If(z3.ULT(a, b), one, zero),
    }[op]()


def run(chunk, regs):
    """The symbolic register file after `chunk`, from `regs` (list of 32)."""
    regs = list(regs)
    for ins in chunk:
        op, args = ins.opcode, ins.args
        if op in R_TYPE:
            rd, a, b = args
            v = _op(op, regs[a], regs[b])
        elif op in I_TYPE and op in _BASE:
            rd, a, k = args
            imm = z3.BitVecVal(k & 0x1F if op in _SHIFT else k & 0xFFFFFFFF, 32)
            v = _op(_BASE[op], regs[a], imm)
        else:
            raise NotImplementedError(op)
        if rd != 0:
            regs[rd] = v
    return regs


def equivalent(a, b, timeout_ms=2000):
    """True iff chunks a and b leave every register equal for every register
    file; None if the solver gives up."""
    regs = [z3.BitVecVal(0, 32)] + [z3.BitVec(f'x{i}', 32) for i in range(1, 32)]
    ra, rb = run(a, regs), run(b, regs)
    s = z3.Solver()
    s.set('timeout', timeout_ms)
    s.add(z3.Or([ra[i] != rb[i] for i in range(1, 32)]))
    res = s.check()
    if res == z3.unsat:
        return True
    if res == z3.sat:
        return False
    return None
