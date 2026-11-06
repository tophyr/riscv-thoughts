"""Known bits — the bits a value column never changes — and the behavioral
distance they bound.

A column's known bits (`observed_bits`) are read from its values across a
large fixed state family, so they are a function of the chunk's behaviour,
never of how it is spelled: `SLT` can only set bit 0, `SRLI x,4` clears the top
four bits. `max_distance` is the largest wrap-around difference two columns'
known bits allow, and `column_distance` is the mean over states of how far
apart two columns' values are as a fraction of it, on a doubly-logarithmic
scale (RISCV_DESIGN, "Collapse"). The trainer computes the same distance on
the GPU (`compressor.train._column_distance`).
"""
from dataclasses import dataclass

import numpy as np

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
    def constant(v):
        v &= MASK
        return KnownBits(~v & MASK, v)

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


def observed_bits(col):
    """(zero, one) masks of the bits constant across a value column."""
    u = col.astype(np.int64) & MASK
    return (int(np.bitwise_and.reduce(~u & MASK)), int(np.bitwise_and.reduce(u)))


def column_distance(a, a_bits, b, b_bits):
    """Mean over states of the doubly-logarithmic wrap-around distance between
    two value columns, log1p(log1p(d)) / log1p(log1p(max_dist)), max_dist the
    largest difference their constant bits (`observed_bits`) allow: 0 for equal
    columns, 1 at the largest distance they could be apart. The difference is
    taken the shorter way round mod 2^32, which reads x and x+1 as 1 apart
    across the wrap and INT_MAX next to INT_MIN."""
    m = max_distance(KnownBits(*a_bits), KnownBits(*b_bits))
    if m == 0:
        return 0.0
    d = (a.astype(np.int64) - b.astype(np.int64)) & MASK
    d = np.minimum(d, _FULL - d).astype(np.float64)
    return float(np.mean(np.log1p(np.log1p(d)) / np.log1p(np.log1p(float(m)))))
