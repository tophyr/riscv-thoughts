"""Known-bits analysis of a chunk's outputs, and the behavioral distance it
bounds.

For every register a chunk writes, `output_known_bits` gives the bits that are
forced to 0 or to 1 whatever the chunk's inputs hold: `SLT` can only set bit 0,
`SRLI x,4` clears the top four bits, `ANDI t,a,0xFF; SLLI r,t,4` can only set
bits 4-11. It is the compiler analysis of that name, run one transfer function
per instruction in program order, with every external input unknown and x0
known to be 0. It is conservative: a bit it calls unknown may in fact never
vary, so the value sets it implies may be larger than the true ones, never
smaller.

`output_distance` measures how far apart two outputs are, as a fraction of the
largest distance those two outputs could ever be apart (`max_distance`, from
their known bits), on a doubly-logarithmic scale.

Scope: R- and I-type ALU instructions, as `datagen.signature`.
"""
from dataclasses import dataclass
import math

from emulator import R_TYPE, I_TYPE

MASK = 0xFFFFFFFF
_HALF = 1 << 31
_FULL = 1 << 32


@dataclass(frozen=True)
class KnownBits:
    """`zero`: bits known to be 0; `one`: bits known to be 1 (32-bit masks,
    disjoint)."""
    zero: int
    one: int

    @staticmethod
    def unknown():
        return KnownBits(0, 0)

    @staticmethod
    def constant(v):
        v &= MASK
        return KnownBits(~v & MASK, v)

    def is_constant(self):
        return (self.zero | self.one) == MASK

    def umin(self):
        return self.one

    def umax(self):
        return ~self.zero & MASK

    def smin(self):
        """Smallest signed value: the sign bit set unless it is known 0."""
        v = self.one | (0 if self.zero & _HALF else _HALF)
        return v - _FULL if v & _HALF else v

    def smax(self):
        v = (~self.zero & MASK) & ~(0 if self.one & _HALF else _HALF) & MASK
        return v - _FULL if v & _HALF else v


def _sext12(imm):
    return imm & MASK


def _and(a, b):
    return KnownBits(a.zero | b.zero, a.one & b.one)


def _or(a, b):
    return KnownBits(a.zero & b.zero, a.one | b.one)


def _xor(a, b):
    known = (a.zero | a.one) & (b.zero | b.one)
    v = (a.one ^ b.one) & known
    return KnownBits(~v & known & MASK, v)


def _add(a, b, carry_in=0):
    """Known bits of a + b + carry_in (mod 2^32), propagating which carries
    are known (the rule LLVM's KnownBits::computeForAddSub implements)."""
    sum_max = (a.umax() + b.umax() + carry_in) & MASK      # every unknown bit 1
    sum_min = (a.umin() + b.umin() + carry_in) & MASK      # every unknown bit 0
    carry_known_zero = ~(sum_max ^ a.zero ^ b.zero) & MASK
    carry_known_one = (sum_min ^ a.one ^ b.one) & MASK
    known = ((a.zero | a.one) & (b.zero | b.one)
             & (carry_known_zero | carry_known_one))
    return KnownBits(~sum_max & known & MASK, sum_min & known)


def _not(a):
    return KnownBits(a.one, a.zero)


def _sub(a, b):
    return _add(a, _not(b), carry_in=1)


def _shift_const(op, a, k):
    k &= 31
    if op == 'SLL':
        return KnownBits(((a.zero << k) | ((1 << k) - 1)) & MASK, (a.one << k) & MASK)
    if op == 'SRL':
        hi = (MASK << (32 - k)) & MASK if k else 0
        return KnownBits((a.zero >> k) | hi, a.one >> k)
    # SRA: the sign bit is copied into the top k bits.
    hi = (MASK << (32 - k)) & MASK if k else 0
    zero, one = a.zero >> k, a.one >> k
    if a.zero & _HALF:
        zero |= hi
    elif a.one & _HALF:
        one |= hi
    return KnownBits(zero, one)


def _shift_var(op, a, b):
    """A shift by a register: intersect the known bits over every shift
    amount b's low five known bits allow."""
    out = None
    for k in range(32):
        if (k & b.zero & 31) or (~k & b.one & 31):
            continue
        r = _shift_const(op, a, k)
        out = r if out is None else KnownBits(out.zero & r.zero, out.one & r.one)
    return out


def _compare(a, b, signed):
    """Bit 0 is the comparison; bits 1-31 are 0. Bit 0 is known when the
    operands' value ranges decide it."""
    lo_a, hi_a = (a.smin(), a.smax()) if signed else (a.umin(), a.umax())
    lo_b, hi_b = (b.smin(), b.smax()) if signed else (b.umin(), b.umax())
    if hi_a < lo_b:
        return KnownBits.constant(1)
    if lo_a >= hi_b:
        return KnownBits.constant(0)
    return KnownBits(MASK & ~1, 0)


# The operation an I-type opcode applies to its register and immediate.
_I_BASE = {'ADDI': 'ADD', 'XORI': 'XOR', 'ORI': 'OR', 'ANDI': 'AND',
           'SLLI': 'SLL', 'SRLI': 'SRL', 'SRAI': 'SRA', 'SLTI': 'SLT',
           'SLTIU': 'SLTU'}


def transfer(op, a, b):
    """Known bits of `op`'s result from its operands' known bits. For an
    I-type op, `b` is the immediate's KnownBits.constant."""
    base = _I_BASE.get(op, op)
    if base == 'ADD':
        return _add(a, b)
    if base == 'SUB':
        return _sub(a, b)
    if base == 'AND':
        return _and(a, b)
    if base == 'OR':
        return _or(a, b)
    if base == 'XOR':
        return _xor(a, b)
    if base in ('SLL', 'SRL', 'SRA'):
        if b.is_constant():
            return _shift_const(base, a, b.one)
        return _shift_var(base, a, b)
    if base == 'SLT':
        return _compare(a, b, signed=True)
    if base == 'SLTU':
        return _compare(a, b, signed=False)
    raise NotImplementedError(f'known bits: {op} is out of scope')


def output_known_bits(chunk):
    """{register: KnownBits} for every register the chunk writes, its final
    value's known bits, with every register the chunk reads from outside
    unknown and x0 known to be 0."""
    state = {0: KnownBits.constant(0)}
    out = {}
    for ins in chunk:
        op, args = ins.opcode, ins.args
        if op in R_TYPE:
            rd, a, b = args[0], state.get(args[1], KnownBits.unknown()), state.get(args[2], KnownBits.unknown())
        elif op in I_TYPE:
            rd, a, b = args[0], state.get(args[1], KnownBits.unknown()), KnownBits.constant(_sext12(args[2]))
        else:
            raise NotImplementedError(f'known bits: {op} is out of scope')
        r = transfer(op, a, b)
        if rd:
            state[rd] = r
            out[rd] = r
    return out


def _max_wrap_distance(lo, hi):
    """Largest wrap-around distance |t| mod 2^32, folded to [0, 2^31], over
    the integers t in [lo, hi]."""
    if hi - lo >= _FULL:
        return _HALF
    # A t congruent to 2^31 inside the interval reaches the maximum.
    first = lo + ((_HALF - lo) % _FULL)
    if first <= hi:
        return _HALF
    wrap = lambda t: min(t % _FULL, _FULL - t % _FULL)
    return max(wrap(lo), wrap(hi))


def max_distance(ka, kb):
    """The largest wrap-around distance between any value `ka` allows and
    any value `kb` allows. Each of the unsigned and the signed value ranges
    bounds it from above; the tighter bound is kept. Two outputs that can
    only be 0 or 1 are at most 1 apart."""
    u = _max_wrap_distance(ka.umin() - kb.umax(), ka.umax() - kb.umin())
    s = _max_wrap_distance(ka.smin() - kb.smax(), ka.smax() - kb.smin())
    return min(u, s)


def wrap_distance(a, b):
    """How far apart two 32-bit register values are, the shorter way round
    mod 2^32, computed exactly on integers.

    RISK: a register carries no signedness; signed or unsigned is how the
    instruction that reads it interprets its bits. The wrap-around reading
    matches ADD/SUB/ADDI (x and x+1 are 1 apart across the wrap) and needs no
    per-instruction choice, but it puts INT_MAX next to INT_MIN, which a signed
    comparison downstream would treat as the farthest apart two values can be.
    """
    d = (int(a) - int(b)) % _FULL
    return min(d, _FULL - d)


def output_distance(a, b, max_dist):
    """`a` and `b` apart as a fraction of `max_dist`, doubly logarithmic:
    log1p(log1p(d)) / log1p(log1p(max_dist)). 0 when the values are equal,
    1 at the largest distance the two outputs could ever be apart.

    ALTERNATIVE: a single log1p(d) / log1p(max_dist) is linear in orders of
    magnitude, so 1e6 vs 1e9 would count as much as 1e3 vs 1e6, and a
    difference of 1 maps to 0.03 instead of 0.17, close to an equivalent
    pair's 0. The nested log saturates the top and spreads the bottom.
    """
    d = wrap_distance(a, b)
    if d == 0:
        return 0.0
    return math.log1p(math.log1p(d)) / math.log1p(math.log1p(max_dist))
