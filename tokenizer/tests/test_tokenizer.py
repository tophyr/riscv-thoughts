"""Unit tests for the RV32I structural tokenizer."""

import pytest
from emulator import Instruction, run


def execute(instructions, initial_regs=None):
    """Test helper: run instructions, return only register state."""
    state, _, _ = run(instructions, regs=initial_regs)
    return state
from tokenizer import (
    PAD, BOS, EOS, NEG, VOCAB, VOCAB_SIZE,
    encode_instruction, decode_instruction,
    encode_sequence, decode_sequence,
    tokens_to_str,
)
# MAX_INSTR_TOKENS is not re-exported by the package __init__.
from tokenizer.tokenizer import MAX_INSTR_TOKENS
from emulator import ALL_OPCODES, R_TYPE, I_TYPE, B_TYPE, LOAD_TYPE, STORE_TYPE


class TestVocabulary:
    def test_vocab_size(self):
        assert len(VOCAB) == VOCAB_SIZE

    def test_no_duplicate_tokens(self):
        assert len(set(VOCAB)) == len(VOCAB)

    def test_special_tokens_at_start(self):
        assert VOCAB[PAD] == '<PAD>'
        assert VOCAB[BOS] == '<BOS>'
        assert VOCAB[EOS] == '<EOS>'
        assert VOCAB[NEG] == '<NEG>'

    def test_all_opcodes_present(self):
        from emulator import ALL_OPCODES
        for op in ALL_OPCODES:
            assert op in VOCAB

    def test_all_registers_present(self):
        for i in range(32):
            assert f'X{i}' in VOCAB

    def test_all_hex_digits_present(self):
        for d in range(16):
            assert f'H{d:X}' in VOCAB


class TestEncodeDecodeRType:
    @pytest.mark.parametrize('op', ['ADD', 'SUB', 'XOR', 'OR', 'AND',
                                     'SLL', 'SRL', 'SRA', 'SLT', 'SLTU'])
    def test_round_trip(self, op):
        instr = Instruction(op, 5, 3, 7)
        tokens = encode_instruction(instr)
        decoded, pos = decode_instruction(tokens)
        assert decoded.opcode == op
        assert decoded.args == (5, 3, 7)
        assert pos == len(tokens)

    def test_token_count(self):
        tokens = encode_instruction(Instruction('ADD', 5, 3, 7))
        assert len(tokens) == 4  # [OP] [RD] [RS1] [RS2]


class TestEncodeDecodeIType:
    @pytest.mark.parametrize('op', ['ADDI', 'XORI', 'ORI', 'ANDI',
                                     'SLTI', 'SLTIU'])
    def test_round_trip_positive(self, op):
        instr = Instruction(op, 5, 3, 42)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.opcode == op
        assert decoded.args == (5, 3, 42)

    @pytest.mark.parametrize('op', ['ADDI', 'XORI', 'ORI', 'ANDI'])
    def test_round_trip_negative(self, op):
        instr = Instruction(op, 5, 3, -1)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.opcode == op
        assert decoded.args == (5, 3, -1)

    def test_positive_no_neg_token(self):
        tokens = encode_instruction(Instruction('ADDI', 5, 3, 42))
        assert NEG not in tokens

    def test_negative_has_neg_token(self):
        tokens = encode_instruction(Instruction('ADDI', 5, 3, -1))
        assert NEG in tokens

    def test_zero_immediate(self):
        instr = Instruction('ADDI', 5, 3, 0)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.args == (5, 3, 0)

    def test_token_count_positive(self):
        tokens = encode_instruction(Instruction('ADDI', 5, 3, 42))
        assert len(tokens) == 6  # [OP] [RD] [RS1] [H] [H] [H]

    def test_token_count_negative(self):
        tokens = encode_instruction(Instruction('ADDI', 5, 3, -1))
        assert len(tokens) == 7  # [OP] [RD] [RS1] [NEG] [H] [H] [H]

    def test_hex_encoding(self):
        # 42 = 0x02A
        tokens = encode_instruction(Instruction('ADDI', 5, 0, 42))
        s = tokens_to_str(tokens)
        assert 'H0 H2 HA' in s

    @pytest.mark.parametrize('op', ['SLLI', 'SRLI', 'SRAI'])
    def test_shift_round_trip(self, op):
        instr = Instruction(op, 5, 3, 4)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.opcode == op
        assert decoded.args == (5, 3, 4)

    def test_max_positive_12bit(self):
        instr = Instruction('ADDI', 1, 0, 2047)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.args == (1, 0, 2047)

    def test_max_negative_12bit(self):
        instr = Instruction('ADDI', 1, 0, -2048)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.args == (1, 0, -2048)


class TestEncodeDecodeLoad:
    @pytest.mark.parametrize('op', ['LB', 'LBU', 'LH', 'LHU', 'LW'])
    def test_round_trip(self, op):
        # TinyFive args: (rd, imm, rs1)
        instr = Instruction(op, 5, 16, 3)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.opcode == op
        assert decoded.args == (5, 16, 3)

    def test_token_order_reordered(self):
        # TinyFive: (rd=5, imm=16, rs1=3) -> tokens: [OP] [RD=X5] [RS1=X3] [IMM]
        tokens = encode_instruction(Instruction('LW', 5, 16, 3))
        s = tokens_to_str(tokens)
        assert s.startswith('LW X5 X3')

    def test_negative_offset(self):
        instr = Instruction('LW', 5, -4, 3)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.args == (5, -4, 3)


class TestEncodeDecodeStore:
    @pytest.mark.parametrize('op', ['SB', 'SH', 'SW'])
    def test_round_trip(self, op):
        # TinyFive args: (rs2, imm, rs1)
        instr = Instruction(op, 5, 8, 3)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.opcode == op
        assert decoded.args == (5, 8, 3)

    def test_token_order_reordered(self):
        # TinyFive: (rs2=5, imm=8, rs1=3) -> tokens: [OP] [RS2=X5] [RS1=X3] [IMM]
        tokens = encode_instruction(Instruction('SW', 5, 8, 3))
        s = tokens_to_str(tokens)
        assert s.startswith('SW X5 X3')


class TestEncodeDecodeBranch:
    @pytest.mark.parametrize('op', ['BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU'])
    def test_round_trip(self, op):
        instr = Instruction(op, 1, 2, 16)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.opcode == op
        assert decoded.args == (1, 2, 16)

    def test_negative_offset(self):
        instr = Instruction('BEQ', 1, 2, -8)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.args == (1, 2, -8)

    def test_branch_uses_4_hex_digits(self):
        tokens = encode_instruction(Instruction('BEQ', 1, 2, 16))
        # [OP] [RS1] [RS2] [H] [H] [H] [H] = 7 tokens
        assert len(tokens) == 7


class TestEncodeDecodeUType:
    @pytest.mark.parametrize('op', ['LUI', 'AUIPC'])
    def test_round_trip(self, op):
        instr = Instruction(op, 5, 0x12345)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.opcode == op
        assert decoded.args == (5, 0x12345)

    def test_uses_5_hex_digits(self):
        tokens = encode_instruction(Instruction('LUI', 5, 0x12345))
        # [OP] [RD] [H] [H] [H] [H] [H] = 7 tokens
        assert len(tokens) == 7


class TestEncodeDecodeJump:
    def test_jal_round_trip(self):
        instr = Instruction('JAL', 1, 8)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.opcode == 'JAL'
        assert decoded.args == (1, 8)

    def test_jal_negative(self):
        instr = Instruction('JAL', 1, -16)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.args == (1, -16)

    def test_jal_uses_6_hex_digits(self):
        tokens = encode_instruction(Instruction('JAL', 1, 8))
        # [OP] [RD] [H] [H] [H] [H] [H] [H] = 8 tokens
        assert len(tokens) == 8

    def test_jalr_round_trip(self):
        instr = Instruction('JALR', 1, 5, 0)
        tokens = encode_instruction(instr)
        decoded, _ = decode_instruction(tokens)
        assert decoded.opcode == 'JALR'
        assert decoded.args == (1, 5, 0)


class TestSequenceEncoding:
    def test_bos_eos(self):
        prog = [Instruction('ADDI', 1, 0, 10)]
        tokens = encode_sequence(prog, add_bos_eos=True)
        assert tokens[0] == BOS
        assert tokens[-1] == EOS

    def test_no_bos_eos(self):
        prog = [Instruction('ADDI', 1, 0, 10)]
        tokens = encode_sequence(prog, add_bos_eos=False)
        assert tokens[0] != BOS
        assert tokens[-1] != EOS

    def test_multi_instruction_round_trip(self):
        prog = [
            Instruction('ADDI', 1, 0, 10),
            Instruction('ADDI', 2, 0, 20),
            Instruction('ADD', 3, 1, 2),
        ]
        tokens = encode_sequence(prog)
        decoded = decode_sequence(tokens)
        assert len(decoded) == 3
        for orig, dec in zip(prog, decoded):
            assert orig.opcode == dec.opcode
            assert orig.args == dec.args

    def test_mixed_types_round_trip(self):
        prog = [
            Instruction('LUI', 1, 0x10),
            Instruction('ADDI', 1, 1, 0x20),
            Instruction('SW', 2, 0, 1),
            Instruction('BEQ', 1, 2, 8),
            Instruction('ADD', 3, 1, 2),
        ]
        tokens = encode_sequence(prog)
        decoded = decode_sequence(tokens)
        assert len(decoded) == 5
        for orig, dec in zip(prog, decoded):
            assert orig.opcode == dec.opcode
            assert orig.args == dec.args


class TestSequenceDecoding:
    def test_skips_pad(self):
        prog = [Instruction('ADDI', 1, 0, 10)]
        tokens = [PAD, PAD] + encode_sequence(prog)
        decoded = decode_sequence(tokens)
        assert len(decoded) == 1
        assert decoded[0].opcode == 'ADDI'

    def test_stops_at_eos(self):
        prog = [Instruction('ADDI', 1, 0, 10)]
        extra = encode_instruction(Instruction('ADDI', 2, 0, 20))
        tokens = [BOS] + encode_instruction(prog[0]) + [EOS] + extra
        decoded = decode_sequence(tokens)
        assert len(decoded) == 1


class TestExecutionAfterRoundTrip:
    """Verify that encode -> decode produces execution-equivalent programs."""

    def test_arithmetic_chain(self):
        prog = [
            Instruction('ADDI', 1, 0, 10),
            Instruction('ADDI', 2, 0, 20),
            Instruction('ADD', 3, 1, 2),
            Instruction('SUB', 4, 3, 1),
        ]
        tokens = encode_sequence(prog)
        decoded = decode_sequence(tokens)
        assert execute(prog) == execute(decoded)

    def test_with_branches(self):
        prog = [
            Instruction('ADDI', 1, 0, 5),
            Instruction('ADDI', 2, 0, 5),
            Instruction('BEQ', 1, 2, 8),
            Instruction('ADDI', 3, 0, 99),
            Instruction('ADDI', 4, 0, 42),
        ]
        tokens = encode_sequence(prog)
        decoded = decode_sequence(tokens)
        assert execute(prog) == execute(decoded)


class TestTokensToStr:
    def test_basic(self):
        tokens = encode_instruction(Instruction('ADD', 5, 3, 7))
        assert tokens_to_str(tokens) == 'ADD X5 X3 X7'

    def test_with_neg(self):
        tokens = encode_instruction(Instruction('ADDI', 5, 0, -1))
        s = tokens_to_str(tokens)
        assert '<NEG>' in s

    def test_full_sequence(self):
        prog = [Instruction('ADDI', 1, 0, 10)]
        tokens = encode_sequence(prog)
        s = tokens_to_str(tokens)
        assert s.startswith('<BOS>')
        assert s.endswith('<EOS>')


class TestMaxInstrTokens:
    """MAX_INSTR_TOKENS must bound the longest single-instruction encoding
    the encoder can actually produce — the worst case is JAL with a
    negative immediate (op + rd + NEG + 6 hex digits)."""

    def _max_neg_instr(self, op):
        """A negative-immediate (worst-case length) instruction per type."""
        if op in R_TYPE:
            return Instruction(op, 5, 3, 7)        # no immediate
        if op in I_TYPE:
            return Instruction(op, 5, 3, -1)
        if op in LOAD_TYPE:
            return Instruction(op, 5, -1, 3)        # (rd, imm, rs1)
        if op in STORE_TYPE:
            return Instruction(op, 5, -1, 3)        # (rs2, imm, rs1)
        if op in B_TYPE:
            return Instruction(op, 1, 2, -8)
        if op in ('LUI', 'AUIPC'):
            return Instruction(op, 5, 1)            # U-imm unsigned, fixed 7
        if op == 'JAL':
            return Instruction(op, 1, -16)
        if op == 'JALR':
            return Instruction(op, 1, 5, -1)
        raise ValueError(op)

    def test_no_encoding_exceeds_max(self):
        for op in ALL_OPCODES:
            n = len(encode_instruction(self._max_neg_instr(op)))
            assert n <= MAX_INSTR_TOKENS, f'{op} encodes to {n} > {MAX_INSTR_TOKENS}'

    def test_jal_negative_hits_max(self):
        # The worst case is realized, so the bound is tight, not loose.
        tokens = encode_instruction(Instruction('JAL', 1, -16))
        assert len(tokens) == MAX_INSTR_TOKENS


class TestNegativeImmediateRoundTrip:
    def test_jalr_negative_offset(self):
        instr = Instruction('JALR', 1, 5, -4)
        tokens = encode_instruction(instr)
        assert NEG in tokens
        decoded, pos = decode_instruction(tokens)
        assert decoded.opcode == 'JALR'
        assert decoded.args == (1, 5, -4)
        assert pos == len(tokens)

    def test_store_negative_offset(self):
        # TinyFive store arg order: (rs2, imm, rs1).
        instr = Instruction('SW', 5, -8, 3)
        tokens = encode_instruction(instr)
        assert NEG in tokens
        decoded, _ = decode_instruction(tokens)
        assert decoded.opcode == 'SW'
        assert decoded.args == (5, -8, 3)


class TestImmediateOverflow:
    """Immediates wider than the encoded hex-digit budget are silently
    truncated (the encoder masks each digit, so high bits are dropped).
    JALR uses 3 hex digits = 12 bits, so 0x1000 truncates to 0."""

    def test_jalr_imm_truncates_above_12_bits(self):
        instr = Instruction('JALR', 1, 5, 0x1000)   # needs a 4th hex digit
        decoded, _ = decode_instruction(encode_instruction(instr))
        assert decoded.args[2] == 0                  # 0x1000 & 0xFFF == 0

    def test_jalr_imm_keeps_low_12_bits(self):
        instr = Instruction('JALR', 1, 5, 0x1ABC)
        decoded, _ = decode_instruction(encode_instruction(instr))
        assert decoded.args[2] == 0xABC              # low 12 bits retained
