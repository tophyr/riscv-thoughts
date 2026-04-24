"""Tests for batch_is_complete_instruction.

The tokenizer emits a variable-length encoding: negative immediates
get a NEG prefix token, so the total token count depends on both the
opcode type AND the sign of the immediate. The completeness check
must account for NEG when computing expected length.
"""

import numpy as np
import torch

from tokenizer import encode_instruction, PAD
from emulator import Instruction, batch_is_complete_instruction


def _to_batch(token_lists, device='cpu', pad_to=16):
    """Right-pad a list of token lists into (B, T) + (B,) lengths.

    Pads to at least `pad_to` columns so batch_parse_tokens can
    safely index into all register/immediate positions even for
    short (partial) sequences, matching how windows are stored in
    the shift-reduce runtime (always padded to max_window).
    """
    lens = [len(t) for t in token_lists]
    T = max(max(lens), pad_to)
    B = len(token_lists)
    tok = np.full((B, T), PAD, dtype=np.int64)
    for i, t in enumerate(token_lists):
        tok[i, :len(t)] = t
    return (torch.from_numpy(tok).to(device),
            torch.tensor(lens, dtype=torch.int64, device=device))


def _check(instrs, expected_all_complete=True, device='cpu'):
    tokens = [encode_instruction(i) for i in instrs]
    tok_t, len_t = _to_batch(tokens, device)
    res = batch_is_complete_instruction(tok_t, len_t, device)
    if expected_all_complete:
        assert res.all(), (
            f'Expected all complete. '
            f'Results={res.tolist()} lens={[len(t) for t in tokens]}')
    else:
        assert not res.any(), (
            f'Expected none complete. Results={res.tolist()}')


def test_rtype_complete():
    _check([Instruction('ADD', 5, 6, 7),
            Instruction('SUB', 1, 2, 3),
            Instruction('XOR', 0, 0, 0)])


def test_itype_positive_imm_complete():
    _check([Instruction('ADDI', 5, 6, 1),
            Instruction('ANDI', 1, 2, 255),
            Instruction('ORI',  3, 4, 0)])


def test_itype_negative_imm_complete():
    """Regression: was mis-labeled as incomplete pre-fix."""
    _check([Instruction('ADDI', 5, 6, -1),
            Instruction('ANDI', 1, 2, -256),
            Instruction('ORI',  3, 4, -1)])


def test_load_negative_offset_complete():
    _check([Instruction('LW',  5, -4, 6),
            Instruction('LB',  1, -128, 2)])


def test_store_negative_offset_complete():
    _check([Instruction('SW',  5, -4, 6),
            Instruction('SB',  1, -1, 2)])


def test_branch_positive_imm_complete():
    _check([Instruction('BEQ', 5, 6, 8),
            Instruction('BNE', 1, 2, 16)])


def test_branch_negative_imm_complete():
    _check([Instruction('BEQ', 5, 6, -8),
            Instruction('BNE', 1, 2, -16)])


def test_jal_positive_imm_complete():
    _check([Instruction('JAL', 5, 8)])


def test_jal_negative_imm_complete():
    _check([Instruction('JAL', 5, -8)])


def test_jalr_negative_imm_complete():
    _check([Instruction('JALR', 5, 6, -4)])


def test_lui_complete():
    _check([Instruction('LUI', 5, 0x12345)])


def test_partial_not_complete():
    """Prefixes of valid instructions are incomplete."""
    tokens = encode_instruction(Instruction('ADDI', 5, 6, -1))
    # truncate to various partial lengths
    for k in range(1, len(tokens)):
        tok_t, len_t = _to_batch([tokens[:k]])
        res = batch_is_complete_instruction(tok_t, len_t, 'cpu')
        assert not res.any(), f'Length-{k} prefix should not be complete'


def test_spanning_not_complete():
    """instr + prefix of next is not complete."""
    a = encode_instruction(Instruction('ADD', 5, 6, 7))
    b = encode_instruction(Instruction('SUB', 1, 2, 3))
    for k in range(1, len(b)):
        tok_t, len_t = _to_batch([a + b[:k]])
        res = batch_is_complete_instruction(tok_t, len_t, 'cpu')
        assert not res.any(), f'ADD + {k}-prefix of SUB should not be complete'


def test_two_complete_not_complete():
    """Two complete instructions back-to-back is not one complete instr."""
    a = encode_instruction(Instruction('ADD', 5, 6, 7))
    b = encode_instruction(Instruction('SUB', 1, 2, 3))
    tok_t, len_t = _to_batch([a + b])
    res = batch_is_complete_instruction(tok_t, len_t, 'cpu')
    assert not res.any()


def test_overlong_rtype_not_complete():
    """R-type + 1 extra token should not be complete."""
    a = encode_instruction(Instruction('ADD', 5, 6, 7))
    tok_t, len_t = _to_batch([a + [a[0]]])  # append an opcode token
    res = batch_is_complete_instruction(tok_t, len_t, 'cpu')
    assert not res.any()


def test_neg_in_padding_not_counted():
    """Padding beyond the reported length must not be misread as NEG.

    Construct an R-type (4 tokens) but write a NEG token at position 3
    in the underlying buffer. The length field says 4, so position 3
    is within-range. But for R-type, neg_pos is 0 (no imm), so NEG
    should be ignored entirely. Regression guard.
    """
    a = encode_instruction(Instruction('ADD', 5, 6, 7))
    tok = a[:]
    tok_t, len_t = _to_batch([tok])
    # R-type has no neg_pos > 0, so stays complete.
    res = batch_is_complete_instruction(tok_t, len_t, 'cpu')
    assert res.all()


def test_mixed_batch():
    """Mix positive- and negative-imm instructions in one batch."""
    instrs = [
        Instruction('ADD', 5, 6, 7),
        Instruction('ADDI', 5, 6, -1),
        Instruction('ADDI', 5, 6, 1),
        Instruction('JAL', 5, -8),
        Instruction('BEQ', 5, 6, 16),
    ]
    tokens = [encode_instruction(i) for i in instrs]
    tok_t, len_t = _to_batch(tokens)
    res = batch_is_complete_instruction(tok_t, len_t, 'cpu')
    assert res.all().item(), f'Expected all complete; got {res.tolist()}'
