"""Tests for datagen.aim — anchor states aimed at a chunk's boundaries.

Every check executes on the CPU emulator, independently of the aiming code's
own evaluator."""

import numpy as np

from emulator import Instruction, run as emu_run
from datagen.aim import aim_anchor_states
from datagen.batch import collect_into_batches, generate_chunks
from datagen.compare import draw_anchor_states, precompute_chunk
from datagen.generate import build_opcode_table, length_cap, relabel, random_perm

_M = (1 << 32) - 1


def _executed(chunk, states):
    """The register files the chunk is executed from, and after it, as
    precompute_chunk places the draw."""
    pre = precompute_chunk(chunk, states)
    return pre.anchors.astype(np.int64) & _M, pre.out_regs.astype(np.int64) & _M


def test_a_comparison_against_an_immediate_sees_both_sides_of_it():
    rng = np.random.default_rng(0)
    for k in (-2048, -1, 0, 1, 600, 2047):
        for op in ('SLTI', 'SLTIU'):
            chunk = [Instruction(op, 5, 9, k)]
            draw = draw_anchor_states(rng, 8)
            regs, out = _executed(chunk, aim_anchor_states(chunk, draw, rng, 0.25))
            if op == 'SLTIU' and k == 0:
                continue                              # constant: no boundary
            seen = set(regs[:, 9].tolist())
            assert {(k - 1) & _M, k & _M} <= seen, (op, k)
            assert set(out[:, 5].tolist()) == {0, 1}


def test_a_comparison_is_also_straddled_at_a_distance():
    """The second pair puts the input at k - d and k - 1 + d for one d, so
    the output still differs between them."""
    rng = np.random.default_rng(7)
    ds = []
    for _ in range(200):
        k = int(rng.integers(-2048, 2048))
        chunk = [Instruction('SLTI', 5, 9, k)]
        draw = draw_anchor_states(rng, 8)
        out = aim_anchor_states(chunk, draw, rng, 0.5)
        regs, res = _executed(chunk, out)
        x = [v - (1 << 32) if v >= 1 << 31 else v for v in regs[:, 9].tolist()]
        lo = [k - v for v in x if 2 <= k - v <= 2048]
        assert any(k - 1 + d in x for d in lo), (k, x)
        ds.append(max(lo))
        assert set(res[:, 5].tolist()) == {0, 1}
    assert max(ds) > 512 and min(ds) < 8      # the scales spread


def test_a_boundary_is_walked_back_through_the_chain_that_feeds_it():
    """`ADDI t,x,5; ADDI t,t,-20; SLTI r,t,600` flips at x = 615."""
    rng = np.random.default_rng(1)
    chunk = [Instruction('ADDI', 6, 9, 5), Instruction('ADDI', 6, 6, -20),
             Instruction('SLTI', 5, 6, 600)]
    for _ in range(20):
        regs, out = _executed(chunk, aim_anchor_states(
            chunk, draw_anchor_states(rng, 8), rng, 0.25))
        assert {614, 615} <= set(regs[:, 9].tolist())
        assert set(out[:, 5].tolist()) == {0, 1}


def test_a_comparison_of_two_registers_sees_them_equal_and_one_apart():
    rng = np.random.default_rng(2)
    chunk = [Instruction('ADD', 6, 7, 8), Instruction('SLT', 5, 6, 9)]
    for _ in range(20):
        regs, out = _executed(chunk, aim_anchor_states(
            chunk, draw_anchor_states(rng, 8), rng, 0.25))
        a = (regs[:, 7] + regs[:, 8]) & _M
        assert {_M, 0} <= set(((a - regs[:, 9]) & _M).tolist())


def test_an_add_is_aimed_across_a_long_carry():
    rng = np.random.default_rng(3)
    chunk = [Instruction('ADDI', 5, 9, -7)]
    for _ in range(20):
        _, out = _executed(chunk, aim_anchor_states(
            chunk, draw_anchor_states(rng, 8), rng, 0.25))
        vals = set(out[:, 5].tolist())
        assert any(((1 << h) - 1) & _M in vals and (1 << h) & _M in vals
                   for h in range(4, 33))


def test_a_rename_aims_the_same_draw():
    """Targets come from the chunk's structure, so a register-relabelled
    chunk at the same draw and rng state gets the identical draw back."""
    rng = np.random.default_rng(4)
    alu = build_opcode_table({'R_ALU': 0.55, 'I_ALU': 0.45})
    chunks = generate_chunks(length_cap(4), rng, opcode_table=alu)
    for _ in range(200):
        chunk = next(chunks).instructions
        renamed = relabel(chunk, random_perm(rng))
        draw = draw_anchor_states(rng, 8)
        a = aim_anchor_states(chunk, draw, np.random.default_rng(9), 0.5)
        b = aim_anchor_states(renamed, draw, np.random.default_rng(9), 0.5)
        assert np.array_equal(a, b)


def test_aiming_leaves_unaimed_states_and_the_x0_column_alone():
    rng = np.random.default_rng(5)
    chunk = [Instruction('SLTI', 5, 9, 600)]
    draw = draw_anchor_states(rng, 8)
    out = aim_anchor_states(chunk, draw, rng, 0.25)
    changed = (out != draw).any(1)
    assert changed.sum() <= 2 and (out[:, 0] == 0).all()
    assert out is not draw and np.array_equal(draw, draw_anchor_states(
        np.random.default_rng(5), 8))


def test_an_aimed_batch_ships_the_anchors_its_targets_were_executed_at():
    """INVARIANT 1 with aiming on: every row's out_regs re-executes from its
    shipped anchors."""
    from tokenizer import decode_instruction
    rng = np.random.default_rng(6)
    alu = build_opcode_table({'R_ALU': 0.55, 'I_ALU': 0.45})
    gen = collect_into_batches(
        generate_chunks(length_cap(3), rng, opcode_table=alu), batch_size=16,
        draw_anchors=lambda k: draw_anchor_states(rng, k * 8).reshape(k, 8, -1),
        aim=lambda instrs, states: aim_anchor_states(instrs, states, rng, 0.5),
        max_chunk_len=3)
    b = next(gen)
    for i in np.nonzero(b.out_regs_valid)[0]:
        toks, pos, chunk = b.tokens[i, :b.token_lens[i]].tolist(), 0, []
        while pos < len(toks):
            ins, pos = decode_instruction(toks, pos)
            chunk.append(ins)
        for a in range(b.anchors.shape[1]):
            regs = emu_run(chunk, b.anchors[i, a].astype(np.int32).copy())[0].regs
            assert np.array_equal(regs, b.out_regs[i, a])
