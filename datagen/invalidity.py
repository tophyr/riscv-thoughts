"""Invalid-window generators for encoder validity training.

Four types of windows that are deliberately NOT a single complete
RV32I instruction. The encoder learns to map these near the origin
of T1 space, leaving valid single-instruction windows near the
unit sphere. See EXPERIMENT_LOG Phase 9 for motivation.

Types:
- partial:  encode(A)[j:k] where (j,k) != (0, len(A)). A contiguous
            slice of one instruction that excludes the full
            instruction itself. Includes prefixes, suffixes, and
            interior slices.
- spanning: encode(A)[j:] + encode(B)[:k] where (j,k) != (0, len(B)).
            A slice crossing an instruction boundary with at least
            one end mis-aligned. Includes full-A + partial-B,
            partial-A + full-B, and partial-A + partial-B.
- multi:    encode(A) + encode(B) [+ encode(C)]. Two or three
            complete instructions concatenated. Each "is" a valid
            instruction, but the window as a whole is not a single
            emittable thought.
- bogus:    random tokens of random length drawn from the non-
            special vocabulary. May, very rarely, happen to form a
            valid instruction; that label noise is tolerated.
"""

import numpy as np

from tokenizer import encode_instruction, VOCAB_SIZE, PAD, BOS, EOS, NEG
from .generate import random_instruction


# Default weight distribution over invalid types. Callers can
# override per-config via the 'invalidity.types' sub-dict.
DEFAULT_TYPE_WEIGHTS = {
    'partial': 0.4,
    'spanning': 0.3,
    'multi': 0.2,
    'bogus': 0.1,
}

VALID_TYPES = frozenset(DEFAULT_TYPE_WEIGHTS.keys())

# First non-special token id — anything below this is PAD/BOS/EOS/NEG.
_FIRST_NON_SPECIAL = 4
assert {PAD, BOS, EOS, NEG} == {0, 1, 2, 3}, \
    'bogus generator assumes the 4 special tokens are ids 0..3'


def build_type_table(types):
    """Normalize a {type_name: weight} dict into a [(weight, name)] list.

    Missing or zero-weight entries are dropped. Weights are
    normalized to sum to 1.0 across the surviving entries.
    """
    positive = {k: v for k, v in types.items() if v > 0}
    if not positive:
        raise ValueError('invalidity.types must have at least one positive weight')
    total = sum(positive.values())
    return [(v / total, k) for k, v in positive.items()]


def _sample_type(rng, type_table):
    roll = rng.random()
    c = 0.0
    for w, name in type_table:
        c += w
        if roll < c:
            return name
    return type_table[-1][1]


def gen_partial(rng, opcode_table, max_window):
    """encode(A)[j:k] where (j, k) != (0, len(A)).

    slice_len is drawn from [1, len(A)-1]; start is drawn so the
    slice lands within encode(A). Excludes the degenerate full-
    instruction case.
    """
    while True:
        instr = random_instruction(rng, opcode_table=opcode_table)
        toks = encode_instruction(instr)
        L = len(toks)
        if L < 2:
            # Can't take a non-full slice of a 1-token instruction.
            continue
        slice_len = int(rng.integers(1, L))   # [1, L-1]
        start = int(rng.integers(0, L - slice_len + 1))
        out = toks[start:start + slice_len]
        if len(out) <= max_window:
            return out


def gen_spanning(rng, opcode_table, max_window):
    """encode(A)[j:] + encode(B)[:k] where (j=0, k=len(B)) is excluded.

    Covers partial-A+full-B, full-A+partial-B, and
    partial-A+partial-B. The full-A+full-B case is 'multi', not
    spanning, and is retried if sampled.
    """
    for _ in range(20):
        instr_a = random_instruction(rng, opcode_table=opcode_table)
        instr_b = random_instruction(rng, opcode_table=opcode_table)
        tA = encode_instruction(instr_a)
        tB = encode_instruction(instr_b)
        j = int(rng.integers(0, len(tA)))       # [0, len(A)-1]
        k = int(rng.integers(1, len(tB) + 1))   # [1, len(B)]
        if j == 0 and k == len(tB):
            continue  # that's 'multi'
        out = tA[j:] + tB[:k]
        if 0 < len(out) <= max_window:
            return out
    # Deterministic fallback: pure partial-start, no next instr.
    instr = random_instruction(rng, opcode_table=opcode_table)
    tokens = encode_instruction(instr)
    return tokens[1:] if len(tokens) > 1 else tokens


def gen_multi(rng, opcode_table, max_window):
    """Two or three complete instructions concatenated.

    n=2 with probability 0.7, else n=3. Retries on oversize; falls
    back to truncation of the 2-instr case.
    """
    for _ in range(20):
        n = 2 if rng.random() < 0.7 else 3
        toks = []
        for _ in range(n):
            instr = random_instruction(rng, opcode_table=opcode_table)
            toks.extend(encode_instruction(instr))
        if len(toks) <= max_window:
            return toks
    # Fallback: always-fits truncation of a 2-instr case.
    instr_a = random_instruction(rng, opcode_table=opcode_table)
    instr_b = random_instruction(rng, opcode_table=opcode_table)
    toks = encode_instruction(instr_a) + encode_instruction(instr_b)
    return toks[:max_window]


def gen_bogus(rng, max_window, min_len=2):
    """Random tokens from the non-special vocabulary.

    May, very rarely, happen to form a valid RV32I instruction.
    That label noise is tolerated — per the default type weights
    it contributes well under 0.1% of total examples.
    """
    length = int(rng.integers(min_len, max_window + 1))
    return rng.integers(_FIRST_NON_SPECIAL, VOCAB_SIZE, size=length).tolist()


def generate_invalid(rng, opcode_table, max_window, type_table):
    """Sample one invalid window. Returns (token_list, type_name)."""
    type_name = _sample_type(rng, type_table)
    if type_name == 'partial':
        return gen_partial(rng, opcode_table, max_window), type_name
    if type_name == 'spanning':
        return gen_spanning(rng, opcode_table, max_window), type_name
    if type_name == 'multi':
        return gen_multi(rng, opcode_table, max_window), type_name
    if type_name == 'bogus':
        return gen_bogus(rng, max_window), type_name
    raise ValueError(f'Unknown invalidity type: {type_name!r}')
