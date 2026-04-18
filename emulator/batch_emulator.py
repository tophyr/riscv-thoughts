"""Batched RV32I emulator using PyTorch tensor operations.

Executes B instructions in parallel on GPU. No Python loops over
instructions — everything is torch.where cascades on (B,) tensors.

Covers all ALU, branch, jump, and upper-immediate instructions.
LOAD/STORE are not yet supported (return data_val=0).

Usage:
    op, rd, rs1, rs2, imm = instructions_to_tensors(instrs, device)
    regs = random_regs_gpu(B, device=device)
    pc = torch.randint(0, 1024, (B,), device=device) * 4
    data_val, final_pc = batch_execute(op, rd, rs1, rs2, imm, regs, pc)
"""

import torch
import numpy as np

from tokenizer.tokenizer import _OP_TO_TOKEN, _TOKEN_TO_ID
from emulator.emulator import Instruction

# ---------------------------------------------------------------------------
# Token ID constants
# ---------------------------------------------------------------------------

# Opcodes
_ADD = _OP_TO_TOKEN['ADD']
_SUB = _OP_TO_TOKEN['SUB']
_AND = _OP_TO_TOKEN['AND']
_OR  = _OP_TO_TOKEN['OR']
_XOR = _OP_TO_TOKEN['XOR']
_SLL = _OP_TO_TOKEN['SLL']
_SRL = _OP_TO_TOKEN['SRL']
_SRA = _OP_TO_TOKEN['SRA']
_SLT = _OP_TO_TOKEN['SLT']
_SLTU = _OP_TO_TOKEN['SLTU']

_ADDI  = _OP_TO_TOKEN['ADDI']
_ANDI  = _OP_TO_TOKEN['ANDI']
_ORI   = _OP_TO_TOKEN['ORI']
_XORI  = _OP_TO_TOKEN['XORI']
_SLTI  = _OP_TO_TOKEN['SLTI']
_SLTIU = _OP_TO_TOKEN['SLTIU']
_SLLI  = _OP_TO_TOKEN['SLLI']
_SRLI  = _OP_TO_TOKEN['SRLI']
_SRAI  = _OP_TO_TOKEN['SRAI']

_LUI   = _OP_TO_TOKEN['LUI']
_AUIPC = _OP_TO_TOKEN['AUIPC']
_JAL   = _OP_TO_TOKEN['JAL']
_JALR  = _OP_TO_TOKEN['JALR']

_BEQ  = _OP_TO_TOKEN['BEQ']
_BNE  = _OP_TO_TOKEN['BNE']
_BLT  = _OP_TO_TOKEN['BLT']
_BGE  = _OP_TO_TOKEN['BGE']
_BLTU = _OP_TO_TOKEN['BLTU']
_BGEU = _OP_TO_TOKEN['BGEU']

_LB  = _OP_TO_TOKEN['LB']
_LH  = _OP_TO_TOKEN['LH']
_LW  = _OP_TO_TOKEN['LW']
_LBU = _OP_TO_TOKEN['LBU']
_LHU = _OP_TO_TOKEN['LHU']
_SB  = _OP_TO_TOKEN['SB']
_SH  = _OP_TO_TOKEN['SH']
_SW  = _OP_TO_TOKEN['SW']

_LOAD_OPS = {_LB, _LH, _LW, _LBU, _LHU}
_STORE_OPS = {_SB, _SH, _SW}
_BRANCH_OPS = {_BEQ, _BNE, _BLT, _BGE, _BLTU, _BGEU}

# Reverse mapping: opcode string → token ID
_OP_STR_TO_ID = {op: tok for op, tok in _OP_TO_TOKEN.items()}

# Token ranges for GPU parsing.
_REG_BASE = _TOKEN_TO_ID['X0']     # 41
_HEX_BASE = _TOKEN_TO_ID['H0']    # 73
_NEG_TOK = _TOKEN_TO_ID['<NEG>']  # 3
_MIN_OP = min(_OP_TO_TOKEN.values())  # 4 (ADD)
_MAX_OP = max(_OP_TO_TOKEN.values())  # 40 (XORI)

_R_OPS = {_ADD, _SUB, _AND, _OR, _XOR, _SLL, _SRL, _SRA, _SLT, _SLTU}
_I_OPS = {_ADDI, _ANDI, _ORI, _XORI, _SLTI, _SLTIU, _SLLI, _SRLI, _SRAI}


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def batch_execute(op, rd, rs1, rs2, imm, regs, pc):
    """Execute B instructions in parallel.

    Args:
        op:   (B,) int32 — opcode token IDs
        rd:   (B,) int32 — destination register index (0-31)
        rs1:  (B,) int32 — source register 1 index (0-31)
        rs2:  (B,) int32 — source register 2 index (0-31)
        imm:  (B,) int32 — immediate value (sign-extended)
        regs: (B, 32) int32 — register file
        pc:   (B,) int32 — program counter

    Returns:
        (data_val, final_pc) as (B,) int64 tensors.
        data_val = final_regs[rd] for ALU/upper/jump, 0 for branch/load/store.
    """
    B = op.shape[0]
    device = op.device
    arange_B = torch.arange(B, device=device)

    src1 = regs[arange_B, rs1]
    src2 = regs[arange_B, rs2]

    # For unsigned operations, promote to int64 and mask.
    u1 = src1.to(torch.int64) & 0xFFFFFFFF
    u2 = src2.to(torch.int64) & 0xFFFFFFFF
    uimm = imm.to(torch.int64) & 0xFFFFFFFF
    shamt_r = (src2 & 31).to(torch.int32)
    shamt_i = (imm & 31).to(torch.int32)

    result = torch.zeros(B, dtype=torch.int32, device=device)

    # R-type ALU
    result = torch.where(op == _ADD, src1 + src2, result)
    result = torch.where(op == _SUB, src1 - src2, result)
    result = torch.where(op == _AND, src1 & src2, result)
    result = torch.where(op == _OR,  src1 | src2, result)
    result = torch.where(op == _XOR, src1 ^ src2, result)
    result = torch.where(op == _SLL, src1 << shamt_r, result)
    result = torch.where(op == _SRL, (u1 >> shamt_r).to(torch.int32), result)
    result = torch.where(op == _SRA, src1 >> shamt_r, result)
    result = torch.where(op == _SLT, (src1 < src2).to(torch.int32), result)
    result = torch.where(op == _SLTU, (u1 < u2).to(torch.int32), result)

    # I-type ALU
    result = torch.where(op == _ADDI, src1 + imm, result)
    result = torch.where(op == _ANDI, src1 & imm, result)
    result = torch.where(op == _ORI,  src1 | imm, result)
    result = torch.where(op == _XORI, src1 ^ imm, result)
    result = torch.where(op == _SLTI, (src1 < imm).to(torch.int32), result)
    result = torch.where(op == _SLTIU, (u1 < uimm).to(torch.int32), result)
    result = torch.where(op == _SLLI, src1 << shamt_i, result)
    result = torch.where(op == _SRLI, (u1 >> shamt_i).to(torch.int32), result)
    result = torch.where(op == _SRAI, src1 >> shamt_i, result)

    # Upper immediate
    result = torch.where(op == _LUI,   (imm << 12).to(torch.int32), result)
    result = torch.where(op == _AUIPC, (pc + (imm << 12)).to(torch.int32), result)

    # Jumps (link value = pc + 4)
    pc4 = (pc + 4).to(torch.int32)
    result = torch.where(op == _JAL,  pc4, result)
    result = torch.where(op == _JALR, pc4, result)

    # --- LOADs: hash-based deterministic memory ---
    # Each byte at address a = ((a * 2654435761) >> 16) & 0xFF.
    # Deterministic: original and decoded instructions reading the
    # same address get the same value. No memory allocation needed.
    load_addr = ((src1 + imm).to(torch.int64)) & 0xFFFFFFFF
    _HASH_C = 2654435761
    b0 = ((load_addr * _HASH_C) >> 16) & 0xFF
    b1 = (((load_addr + 1) * _HASH_C) >> 16) & 0xFF
    b2 = (((load_addr + 2) * _HASH_C) >> 16) & 0xFF
    b3 = (((load_addr + 3) * _HASH_C) >> 16) & 0xFF

    word = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(torch.int32)
    half = (b0 | (b1 << 8)).to(torch.int32)
    byte = b0.to(torch.int32)

    # LW: full word
    result = torch.where(op == _LW, word, result)
    # LH: sign-extend 16 bits
    result = torch.where(op == _LH,
                         torch.where(half >= 0x8000, half - 0x10000, half),
                         result)
    # LHU: zero-extend 16 bits
    result = torch.where(op == _LHU, half, result)
    # LB: sign-extend 8 bits
    result = torch.where(op == _LB,
                         torch.where(byte >= 0x80, byte - 0x100, byte),
                         result)
    # LBU: zero-extend 8 bits
    result = torch.where(op == _LBU, byte, result)

    # --- STOREs: data_val = stored value truncated to width ---
    store_val = src2.to(torch.int32)
    result = torch.where(op == _SB, store_val & 0xFF, result)
    result = torch.where(op == _SH, store_val & 0xFFFF, result)
    result = torch.where(op == _SW, store_val, result)

    # x0 is hardwired to 0 — only for register-writing instructions.
    has_rd = torch.ones(B, dtype=torch.bool, device=device)
    for tok in _BRANCH_OPS | _STORE_OPS:
        has_rd = has_rd & (op != tok)
    result = torch.where(has_rd & (rd == 0),
                         torch.zeros_like(result), result)

    # --- Final PC ---
    final_pc = (pc + 4).to(torch.int64)

    # Branches
    taken = torch.zeros(B, dtype=torch.bool, device=device)
    taken = taken | ((op == _BEQ)  & (src1 == src2))
    taken = taken | ((op == _BNE)  & (src1 != src2))
    taken = taken | ((op == _BLT)  & (src1 < src2))
    taken = taken | ((op == _BGE)  & (src1 >= src2))
    taken = taken | ((op == _BLTU) & (u1 < u2))
    taken = taken | ((op == _BGEU) & (u1 >= u2))
    final_pc = torch.where(taken, (pc + imm).to(torch.int64), final_pc)

    # JAL: pc += imm
    final_pc = torch.where(op == _JAL, (pc + imm).to(torch.int64), final_pc)

    # JALR: pc = (rs1 + imm) & ~1
    final_pc = torch.where(op == _JALR,
                           ((src1 + imm).to(torch.int64) & ~1),
                           final_pc)

    # Wrap PC to unsigned 32-bit.
    final_pc = final_pc & 0xFFFFFFFF

    # --- data_val ---
    # Branches return 0; everything else returns result.
    is_branch = torch.zeros(B, dtype=torch.bool, device=device)
    for tok in _BRANCH_OPS:
        is_branch = is_branch | (op == tok)
    data_val = torch.where(is_branch,
                           torch.zeros(B, dtype=torch.int64, device=device),
                           result.to(torch.int64))

    return data_val, final_pc


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def instructions_to_tensors(instrs, device='cpu'):
    """Convert a list of Instruction objects to batched tensors.

    Returns (op, rd, rs1, rs2, imm) as (B,) int32 tensors.
    Handles all instruction types by mapping args to the standard
    5-field representation.
    """
    from tokenizer.tokenizer import R_TYPE, I_TYPE, LOAD_TYPE, STORE_TYPE, B_TYPE

    B = len(instrs)
    op_arr  = np.zeros(B, dtype=np.int32)
    rd_arr  = np.zeros(B, dtype=np.int32)
    rs1_arr = np.zeros(B, dtype=np.int32)
    rs2_arr = np.zeros(B, dtype=np.int32)
    imm_arr = np.zeros(B, dtype=np.int32)

    for i, instr in enumerate(instrs):
        op_arr[i] = _OP_STR_TO_ID[instr.opcode]
        a = instr.args

        if instr.opcode in R_TYPE:
            rd_arr[i], rs1_arr[i], rs2_arr[i] = a[0], a[1], a[2]
        elif instr.opcode in I_TYPE:
            rd_arr[i], rs1_arr[i], imm_arr[i] = a[0], a[1], a[2]
        elif instr.opcode in LOAD_TYPE:
            # TinyFive arg order: (rd, imm, rs1)
            rd_arr[i], imm_arr[i], rs1_arr[i] = a[0], a[1], a[2]
        elif instr.opcode in STORE_TYPE:
            # TinyFive arg order: (rs2, imm, rs1)
            rs2_arr[i], imm_arr[i], rs1_arr[i] = a[0], a[1], a[2]
        elif instr.opcode in B_TYPE:
            rs1_arr[i], rs2_arr[i], imm_arr[i] = a[0], a[1], a[2]
        elif instr.opcode in ('LUI', 'AUIPC'):
            rd_arr[i], imm_arr[i] = a[0], a[1]
        elif instr.opcode == 'JAL':
            rd_arr[i], imm_arr[i] = a[0], a[1]
        elif instr.opcode == 'JALR':
            rd_arr[i], rs1_arr[i], imm_arr[i] = a[0], a[1], a[2]

    # Clamp register indices to [0, 31] for safety (decoded
    # instructions from REINFORCE may have unexpected values).
    rd_arr  = np.clip(rd_arr, 0, 31)
    rs1_arr = np.clip(rs1_arr, 0, 31)
    rs2_arr = np.clip(rs2_arr, 0, 31)

    return (torch.from_numpy(op_arr).to(device),
            torch.from_numpy(rd_arr).to(device),
            torch.from_numpy(rs1_arr).to(device),
            torch.from_numpy(rs2_arr).to(device),
            torch.from_numpy(imm_arr).to(device))


def random_regs_gpu(B, device='cpu', rng=None):
    """Generate B random register files on GPU.

    Returns (B, 32) int32 tensor. x0 is always 0.
    """
    regs = torch.randint(
        -2**31, 2**31, (B, 32), dtype=torch.int32, device=device)
    regs[:, 0] = 0
    return regs


# ---------------------------------------------------------------------------
# GPU token parser
# ---------------------------------------------------------------------------

def _decode_imm_gpu(token_ids, start_pos, n_digits, device):
    """Decode hex-encoded immediates from token tensor.

    start_pos: scalar int, position where NEG or first hex digit is.
    Returns (B,) int32 immediate values.
    """
    B = token_ids.shape[0]
    T = token_ids.shape[1]
    arange_B = torch.arange(B, device=device)

    start = torch.full((B,), start_pos, dtype=torch.long, device=device)
    neg = (token_ids[arange_B, start.clamp(max=T - 1)] == _NEG_TOK)
    digit_start = start + neg.long()

    indices = digit_start.unsqueeze(1) + torch.arange(
        n_digits, device=device, dtype=torch.long)
    indices = indices.clamp(0, T - 1)

    digit_toks = torch.gather(token_ids, 1, indices)
    digits = (digit_toks - _HEX_BASE).clamp(0, 15).to(torch.int32)

    value = torch.zeros(B, dtype=torch.int32, device=device)
    for d in range(n_digits):
        value = value | (digits[:, d] << (4 * (n_digits - 1 - d)))

    return torch.where(neg, -value, value)


def batch_parse_tokens(token_ids, lengths, device):
    """Parse (B, T) token IDs into instruction fields on GPU.

    Handles all RV32I instruction types. Invalid opcode tokens
    produce valid=False; other invalid tokens are clamped.

    Args:
        token_ids: (B, T) int64 tensor of token IDs.
        lengths:   (B,) int tensor of valid token counts.

    Returns:
        (op, rd, rs1, rs2, imm, valid) — all (B,) tensors.
        op is the raw opcode token ID. valid is a bool mask.
    """
    B = token_ids.shape[0]
    token_ids = token_ids.to(device)

    op = token_ids[:, 0].to(torch.int32)
    valid = (op >= _MIN_OP) & (op <= _MAX_OP)

    rd  = torch.zeros(B, dtype=torch.int32, device=device)
    rs1 = torch.zeros(B, dtype=torch.int32, device=device)
    rs2 = torch.zeros(B, dtype=torch.int32, device=device)
    imm = torch.zeros(B, dtype=torch.int32, device=device)

    # --- Type masks ---
    is_r = torch.zeros(B, dtype=torch.bool, device=device)
    for tok in _R_OPS:
        is_r = is_r | (op == tok)

    is_i = torch.zeros(B, dtype=torch.bool, device=device)
    for tok in _I_OPS:
        is_i = is_i | (op == tok)

    is_load = torch.zeros(B, dtype=torch.bool, device=device)
    for tok in _LOAD_OPS:
        is_load = is_load | (op == tok)

    is_store = torch.zeros(B, dtype=torch.bool, device=device)
    for tok in _STORE_OPS:
        is_store = is_store | (op == tok)

    is_branch = torch.zeros(B, dtype=torch.bool, device=device)
    for tok in _BRANCH_OPS:
        is_branch = is_branch | (op == tok)

    is_lui_auipc = (op == _LUI) | (op == _AUIPC)
    is_jal = (op == _JAL)
    is_jalr = (op == _JALR)

    # --- Register fields ---
    # Position 1: rd for most, rs1 for branch, rs2 for store.
    tok1 = (token_ids[:, 1] - _REG_BASE).clamp(0, 31).to(torch.int32)
    rd  = torch.where(is_r | is_i | is_load | is_jalr | is_lui_auipc | is_jal,
                      tok1, rd)
    rs1 = torch.where(is_branch, tok1, rs1)
    rs2 = torch.where(is_store, tok1, rs2)

    # Position 2: rs1 for R/I/LOAD/STORE/JALR/BRANCH(rs2), rs2 for R.
    tok2 = (token_ids[:, 2] - _REG_BASE).clamp(0, 31).to(torch.int32)
    rs1 = torch.where(is_r | is_i | is_load | is_store | is_jalr, tok2, rs1)
    rs2 = torch.where(is_branch, tok2, rs2)

    # Position 3 for R-type: rs2.
    tok3 = (token_ids[:, 3] - _REG_BASE).clamp(0, 31).to(torch.int32)
    rs2 = torch.where(is_r, tok3, rs2)

    # --- Immediates ---
    # 3-digit (I-type, LOAD, STORE, JALR) starting at position 3.
    imm_3 = _decode_imm_gpu(token_ids, 3, 3, device)
    imm = torch.where(is_i | is_load | is_store | is_jalr, imm_3, imm)

    # 4-digit (BRANCH) starting at position 3.
    imm_4 = _decode_imm_gpu(token_ids, 3, 4, device)
    imm = torch.where(is_branch, imm_4, imm)

    # 5-digit (LUI, AUIPC) starting at position 2.
    imm_5 = _decode_imm_gpu(token_ids, 2, 5, device)
    imm = torch.where(is_lui_auipc, imm_5, imm)

    # 6-digit (JAL) starting at position 2.
    imm_6 = _decode_imm_gpu(token_ids, 2, 6, device)
    imm = torch.where(is_jal, imm_6, imm)

    return op, rd, rs1, rs2, imm, valid
