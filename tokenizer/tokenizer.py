"""Structural tokenizer for RV32I instruction sequences.

Converts between Instruction objects and integer token sequences.
Uses hex-digit tokenization for immediates: each hex digit is a separate
token, enabling compositional understanding of numerical values.

Token layout per instruction type (all start with opcode token):
  R-type:  [OP] [RD] [RS1] [RS2]
  I-type:  [OP] [RD] [RS1] [NEG?] [H] [H] [H]
  Load:    [OP] [RD] [RS1] [NEG?] [H] [H] [H]
  Store:   [OP] [RS2] [RS1] [NEG?] [H] [H] [H]
  Branch:  [OP] [RS1] [RS2] [NEG?] [H] [H] [H] [H]
  U-type:  [OP] [RD] [H] [H] [H] [H] [H]
  JAL:     [OP] [RD] [NEG?] [H] [H] [H] [H] [H] [H]
  JALR:    [OP] [RD] [RS1] [NEG?] [H] [H] [H]

Note: Load/Store token order is [dest/src] [base_reg] [offset], which
differs from TinyFive's internal argument order.
"""

from emulator import (
    Instruction, R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE, ALL_OPCODES,
)

# --- Vocabulary definition ---

# Special tokens
PAD = 0
BOS = 1
EOS = 2
NEG = 3

_SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<NEG>']

# Opcode tokens (sorted for deterministic ordering)
_OPCODES = sorted(ALL_OPCODES)
_OPCODE_OFFSET = len(_SPECIAL_TOKENS)

# Register tokens: X0 .. X31
_REG_OFFSET = _OPCODE_OFFSET + len(_OPCODES)

# Hex digit tokens: H0 .. HF
_HEX_OFFSET = _REG_OFFSET + 32

VOCAB_SIZE = _HEX_OFFSET + 16

# Build string representations for the full vocabulary.
VOCAB = (
    _SPECIAL_TOKENS
    + _OPCODES
    + [f'X{i}' for i in range(32)]
    + [f'H{d:X}' for d in range(16)]
)
assert len(VOCAB) == VOCAB_SIZE

# Reverse lookup: string -> token id
_TOKEN_TO_ID = {name: i for i, name in enumerate(VOCAB)}

# Opcode string -> token id
_OP_TO_TOKEN = {op: _OPCODE_OFFSET + i for i, op in enumerate(_OPCODES)}
_TOKEN_TO_OP = {v: k for k, v in _OP_TO_TOKEN.items()}


def _reg_token(reg: int) -> int:
    assert 0 <= reg < 32
    return _REG_OFFSET + reg


def _token_to_reg(token: int) -> int:
    return token - _REG_OFFSET


def _hex_token(digit: int) -> int:
    assert 0 <= digit < 16
    return _HEX_OFFSET + digit


def _token_to_hex(token: int) -> int:
    return token - _HEX_OFFSET


def _encode_imm(value: int, n_digits: int) -> list[int]:
    """Encode a signed immediate as optional NEG + n_digits hex tokens."""
    tokens = []
    if value < 0:
        tokens.append(NEG)
        value = -value
    for i in range(n_digits - 1, -1, -1):
        digit = (value >> (4 * i)) & 0xF
        tokens.append(_hex_token(digit))
    return tokens


def _decode_imm(tokens: list[int], pos: int, n_digits: int) -> tuple[int, int]:
    """Decode hex digit tokens back to a signed immediate.

    Returns (value, new_pos).
    """
    negative = False
    if tokens[pos] == NEG:
        negative = True
        pos += 1
    value = 0
    for _ in range(n_digits):
        value = (value << 4) | _token_to_hex(tokens[pos])
        pos += 1
    if negative:
        value = -value
    return value, pos


# Immediate width (hex digits) per instruction category.
_I_IMM_DIGITS = 3   # 12-bit signed
_S_IMM_DIGITS = 3   # 12-bit signed
_B_IMM_DIGITS = 4   # 13-bit signed (multiples of 2)
_U_IMM_DIGITS = 5   # 20-bit
_J_IMM_DIGITS = 6   # 21-bit signed (multiples of 2)

# Worst case: JAL = op + rd + NEG + 6 imm digits.
MAX_INSTR_TOKENS = 1 + 1 + 1 + _J_IMM_DIGITS


def encode_instruction(instr: Instruction) -> list[int]:
    """Convert a single Instruction to a list of token IDs."""
    op = instr.opcode
    args = instr.args
    tokens = [_OP_TO_TOKEN[op]]

    if op in R_TYPE:
        # args: (rd, rs1, rs2)
        tokens += [_reg_token(args[0]), _reg_token(args[1]), _reg_token(args[2])]

    elif op in I_TYPE:
        # args: (rd, rs1, imm)
        tokens += [_reg_token(args[0]), _reg_token(args[1])]
        tokens += _encode_imm(args[2], _I_IMM_DIGITS)

    elif op in LOAD_TYPE:
        # TinyFive args: (rd, imm, rs1) -> tokens: [RD] [RS1] [IMM]
        tokens += [_reg_token(args[0]), _reg_token(args[2])]
        tokens += _encode_imm(args[1], _I_IMM_DIGITS)

    elif op in STORE_TYPE:
        # TinyFive args: (rs2, imm, rs1) -> tokens: [RS2] [RS1] [IMM]
        tokens += [_reg_token(args[0]), _reg_token(args[2])]
        tokens += _encode_imm(args[1], _S_IMM_DIGITS)

    elif op in B_TYPE:
        # args: (rs1, rs2, imm)
        tokens += [_reg_token(args[0]), _reg_token(args[1])]
        tokens += _encode_imm(args[2], _B_IMM_DIGITS)

    elif op in ('LUI', 'AUIPC'):
        # args: (rd, imm)
        tokens += [_reg_token(args[0])]
        tokens += _encode_imm(args[1], _U_IMM_DIGITS)

    elif op == 'JAL':
        # args: (rd, imm)
        tokens += [_reg_token(args[0])]
        tokens += _encode_imm(args[1], _J_IMM_DIGITS)

    elif op == 'JALR':
        # args: (rd, rs1, imm)
        tokens += [_reg_token(args[0]), _reg_token(args[1])]
        tokens += _encode_imm(args[2], _I_IMM_DIGITS)

    return tokens


def decode_instruction(tokens: list[int], pos: int = 0) -> tuple[Instruction, int]:
    """Decode token IDs starting at pos back into an Instruction.

    Returns (instruction, new_pos).
    """
    op = _TOKEN_TO_OP[tokens[pos]]
    pos += 1

    if op in R_TYPE:
        rd = _token_to_reg(tokens[pos]); pos += 1
        rs1 = _token_to_reg(tokens[pos]); pos += 1
        rs2 = _token_to_reg(tokens[pos]); pos += 1
        return Instruction(op, rd, rs1, rs2), pos

    elif op in I_TYPE:
        rd = _token_to_reg(tokens[pos]); pos += 1
        rs1 = _token_to_reg(tokens[pos]); pos += 1
        imm, pos = _decode_imm(tokens, pos, _I_IMM_DIGITS)
        return Instruction(op, rd, rs1, imm), pos

    elif op in LOAD_TYPE:
        rd = _token_to_reg(tokens[pos]); pos += 1
        rs1 = _token_to_reg(tokens[pos]); pos += 1
        imm, pos = _decode_imm(tokens, pos, _I_IMM_DIGITS)
        # TinyFive order: (rd, imm, rs1)
        return Instruction(op, rd, imm, rs1), pos

    elif op in STORE_TYPE:
        rs2 = _token_to_reg(tokens[pos]); pos += 1
        rs1 = _token_to_reg(tokens[pos]); pos += 1
        imm, pos = _decode_imm(tokens, pos, _S_IMM_DIGITS)
        # TinyFive order: (rs2, imm, rs1)
        return Instruction(op, rs2, imm, rs1), pos

    elif op in B_TYPE:
        rs1 = _token_to_reg(tokens[pos]); pos += 1
        rs2 = _token_to_reg(tokens[pos]); pos += 1
        imm, pos = _decode_imm(tokens, pos, _B_IMM_DIGITS)
        return Instruction(op, rs1, rs2, imm), pos

    elif op in ('LUI', 'AUIPC'):
        rd = _token_to_reg(tokens[pos]); pos += 1
        imm, pos = _decode_imm(tokens, pos, _U_IMM_DIGITS)
        return Instruction(op, rd, imm), pos

    elif op == 'JAL':
        rd = _token_to_reg(tokens[pos]); pos += 1
        imm, pos = _decode_imm(tokens, pos, _J_IMM_DIGITS)
        return Instruction(op, rd, imm), pos

    elif op == 'JALR':
        rd = _token_to_reg(tokens[pos]); pos += 1
        rs1 = _token_to_reg(tokens[pos]); pos += 1
        imm, pos = _decode_imm(tokens, pos, _I_IMM_DIGITS)
        return Instruction(op, rd, rs1, imm), pos

    raise ValueError(f'unhandled opcode: {op}')


def encode_sequence(instructions: list[Instruction], add_bos_eos: bool = True) -> list[int]:
    """Encode a full instruction sequence to token IDs."""
    tokens = []
    if add_bos_eos:
        tokens.append(BOS)
    for instr in instructions:
        tokens.extend(encode_instruction(instr))
    if add_bos_eos:
        tokens.append(EOS)
    return tokens


def decode_sequence(tokens: list[int]) -> list[Instruction]:
    """Decode a token ID sequence back to Instructions.

    Stops at EOS or end of list. Skips BOS/EOS/PAD tokens.
    """
    instructions = []
    pos = 0
    while pos < len(tokens):
        if tokens[pos] in (PAD, BOS):
            pos += 1
            continue
        if tokens[pos] == EOS:
            break
        instr, pos = decode_instruction(tokens, pos)
        instructions.append(instr)
    return instructions


def tokens_to_str(tokens: list[int]) -> str:
    """Convert token IDs to a human-readable string."""
    return ' '.join(VOCAB[t] for t in tokens)
