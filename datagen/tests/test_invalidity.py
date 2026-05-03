"""Tests for invalid-window generators.

Each type of generator should:
- Produce a token list of plausible length.
- Produce a window that batch_is_complete_instruction labels as
  NOT a single complete instruction (modulo the rare bogus case
  that happens to parse).
"""

import numpy as np
import pytest
import torch

from datagen.invalidity import (
    DEFAULT_TYPE_WEIGHTS, build_type_table,
    gen_partial, gen_spanning, gen_multi, gen_bogus, generate_invalid,
)
from datagen.generate import DEFAULT_DISTRIBUTION, _build_opcode_table
from emulator import batch_is_complete_instruction


@pytest.fixture
def opcode_table():
    return _build_opcode_table(DEFAULT_DISTRIBUTION)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _is_complete(tokens, max_len=32):
    """Wrap batch_is_complete_instruction for a single token list."""
    T = max(len(tokens), max_len)
    tok = np.zeros((1, T), dtype=np.int64)
    tok[0, :len(tokens)] = tokens
    tok_t = torch.from_numpy(tok)
    len_t = torch.tensor([len(tokens)], dtype=torch.int64)
    return batch_is_complete_instruction(tok_t, len_t, 'cpu')[0].item()


def test_partial_is_not_complete(rng, opcode_table):
    for _ in range(50):
        toks = gen_partial(rng, opcode_table, max_window=32)
        assert not _is_complete(toks), (
            f'partial window labeled complete: {toks}')


def test_spanning_is_not_complete(rng, opcode_table):
    for _ in range(50):
        toks = gen_spanning(rng, opcode_table, max_window=32)
        assert not _is_complete(toks), (
            f'spanning window labeled complete: {toks}')


def test_multi_is_not_complete(rng, opcode_table):
    for _ in range(50):
        toks = gen_multi(rng, opcode_table, max_window=32)
        assert not _is_complete(toks), (
            f'multi window labeled complete: {toks}')


def test_bogus_rarely_valid(rng):
    """Bogus may occasionally parse as valid; verify rarity."""
    n = 500
    valid = 0
    for _ in range(n):
        toks = gen_bogus(rng, max_window=32)
        if _is_complete(toks):
            valid += 1
    # Empirically ~3% on prior runs; assert < 10% to catch regressions.
    assert valid / n < 0.10, f'{valid}/{n} bogus windows parsed as valid'


def test_partial_length_bounds(rng, opcode_table):
    for _ in range(50):
        toks = gen_partial(rng, opcode_table, max_window=32)
        assert 1 <= len(toks) <= 32


def test_spanning_length_bounds(rng, opcode_table):
    for _ in range(50):
        toks = gen_spanning(rng, opcode_table, max_window=32)
        assert 1 <= len(toks) <= 32


def test_multi_length_bounds(rng, opcode_table):
    for _ in range(50):
        toks = gen_multi(rng, opcode_table, max_window=32)
        # Must be at least 2 instrs × min-len (R-type = 4).
        assert 8 <= len(toks) <= 32


def test_bogus_no_special_tokens(rng):
    for _ in range(50):
        toks = gen_bogus(rng, max_window=32)
        # Special tokens are PAD=0, BOS=1, EOS=2, NEG=3.
        for t in toks:
            assert t >= 4, f'bogus generated special token {t}'


def test_generate_invalid_respects_type_table(rng, opcode_table):
    """Type sampler should return only types in the table."""
    type_table = build_type_table({'partial': 1.0})  # only 'partial'
    names = set()
    for _ in range(20):
        _, name = generate_invalid(rng, opcode_table, 32, type_table)
        names.add(name)
    assert names == {'partial'}


def test_build_type_table_rejects_empty():
    with pytest.raises(ValueError):
        build_type_table({})
    with pytest.raises(ValueError):
        build_type_table({'partial': 0, 'bogus': 0})


def test_default_weights_cover_all_types():
    names = {name for _, name in build_type_table(DEFAULT_TYPE_WEIGHTS)}
    assert names == {'partial', 'spanning', 'multi', 'bogus'}
