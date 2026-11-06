"""Tests for datagen.ideal — the reference behavioral distance d*.

d* is 0 exactly on register-file equivalents, depends on behaviour only, and
moves a little for a small change."""

import numpy as np

from emulator import Instruction as I, run as emu_run
from datagen.compare import draw_anchor_states
from datagen.generate import random_perm, relabel, sample_chunk_partner_pair
from datagen.ideal import ideal_distance

A = [I('ADDI', 4, 3, 42)]


def test_a_rename_is_at_zero():
    rng = np.random.default_rng(0)
    for chunk in (A, [I('ADD', 5, 6, 7), I('SUB', 8, 5, 6)], [I('ADDI', 5, 5, 3), I('XOR', 6, 5, 9)]):
        for _ in range(5):
            assert ideal_distance(chunk, relabel(chunk, random_perm(rng))) == 0.0


def test_the_minimising_rename_reproduces_the_emulator():
    """B relabelled by the rename d* chooses, run on the emulator, gives the
    register file d* compared with A's."""
    a, b = [I('ADD', 5, 6, 7)], [I('ADD', 9, 11, 10)]
    d, perm = ideal_distance(a, b, return_rename=True)
    assert d == 0.0
    b2 = relabel(b, perm)
    for s in draw_anchor_states(np.random.default_rng(1), 50):
        assert np.array_equal(emu_run(a, s.copy())[0].regs, emu_run(b2, s.copy())[0].regs)


def test_a_write_that_changes_nothing_is_no_output_and_a_copy_is_one():
    assert ideal_distance(A, A + [I('ADDI', 7, 7, 0)]) == 0.0
    assert ideal_distance(A, A + [I('XORI', 7, 7, 0)]) == 0.0
    assert ideal_distance(A, A + [I('ADDI', 8, 7, 0)]) > 0.5


def test_an_in_place_step_is_a_small_step():
    """x7 += 1 sits where an immediate changed by one sits, far inside a copy."""
    step = ideal_distance(A, A + [I('ADDI', 7, 7, 1)])
    imm = ideal_distance(A, [I('ADDI', 4, 3, 43)])
    assert abs(step - imm) < 1e-9
    assert step < ideal_distance(A, A + [I('ADDI', 8, 7, 1)]) / 3


def test_spellings_of_one_function_are_one_point_to_a_third():
    zero_a, zero_b, one = [I('ADDI', 5, 0, 0)], [I('SUB', 5, 6, 6)], [I('ADDI', 5, 0, 1)]
    assert ideal_distance(zero_a, zero_b) == 0.0
    assert ideal_distance(zero_a, one) == ideal_distance(zero_b, one)


def test_a_rare_change_is_small_and_not_zero():
    base = [I('SLTIU', 6, 5, 1)]
    d = ideal_distance(base + [I('ADD', 7, 7, 6)], base)
    assert 0.0 < d < 0.05


def test_every_partner_rewrite_is_at_zero():
    rng = np.random.default_rng(3)
    for _ in range(200):
        x, y = sample_chunk_partner_pair(rng, 2)
        assert ideal_distance(x, y) == 0.0, (x, y)
