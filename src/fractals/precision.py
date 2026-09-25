"""Arbitrary-precision numbers, and the bridge from them to float64.

Deep zoom needs two different kinds of number:

* **Positions** (the view centre, the reference orbit) need as many digits as
  the zoom is deep: at a pixel spacing of 1e-300 the centre must be known to
  better than 1e-300. They are stored as :class:`decimal.Decimal` — exact
  decimal input, unlimited exponent range, standard library — and iterated as
  fixed-point Python integers, which is the fastest arbitrary-precision
  arithmetic CPython offers without extra dependencies.

* **Differences** (pixel offsets from the centre, perturbations of an orbit)
  need only ~16 significant digits, but can be far smaller than the smallest
  float64 (≈ 1e-308). They are carried as a float64 mantissa and a separate
  base-2 exponent, :class:`FloatExp`.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext

#: Extra binary digits carried beyond what the pixel spacing strictly needs.
GUARD_BITS = 64

#: log2(10), for converting a decimal exponent into a binary one.
_LOG2_10 = math.log2(10)


def to_decimal(value) -> Decimal:
    """Convert ``str``/``int``/``float``/``Decimal`` to an exact ``Decimal``.

    Strings are the way to enter deep coordinates: ``"-0.7436438870371587"``
    is kept digit for digit, whereas a float literal is already rounded.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, str):
        return Decimal(value.strip())
    if isinstance(value, numbers.Integral):
        return Decimal(int(value))
    if isinstance(value, numbers.Real):
        return Decimal(repr(float(value)))
    raise TypeError(f"cannot convert {type(value).__name__} to Decimal")


def bits_for_spacing(spacing: Decimal) -> int:
    """Binary precision needed to resolve a pixel ``spacing`` (plus guard bits)."""
    if spacing <= 0:
        raise ValueError("pixel spacing must be positive")
    return max(GUARD_BITS, int(-float(spacing.adjusted()) * _LOG2_10) + GUARD_BITS)


def decimal_context(bits: int) -> Context:
    """A decimal context with at least ``bits`` binary digits of precision."""
    ctx = Context(prec=int(bits / _LOG2_10) + 10)
    ctx.Emin, ctx.Emax = -10**9, 10**9
    return ctx


def to_fixed(value: Decimal, bits: int) -> int:
    """``round(value · 2**bits)`` — ``value`` as a fixed-point integer."""
    with localcontext(decimal_context(bits + max(0, value.adjusted()) * 4 + 16)):
        return int((value * (1 << bits)).to_integral_value())


def fixed_to_float(x: int, bits: int) -> float:
    """``x / 2**bits`` rounded to float64, keeping full *relative* precision."""
    shift = x.bit_length() - 64 if x >= 0 else (-x).bit_length() - 64
    if shift > 0:
        return math.ldexp(float(x >> shift), shift - bits)
    return math.ldexp(float(x), -bits)


@dataclass(frozen=True)
class FloatExp:
    """``mantissa · 2**exponent``: a float64 with an unbounded exponent.

    ``0.5 <= |mantissa| < 1`` for non-zero values, as returned by
    :func:`math.frexp`.
    """

    mantissa: float
    exponent: int

    @classmethod
    def from_decimal(cls, value: Decimal) -> "FloatExp":
        if value == 0:
            return cls(0.0, 0)
        # Pull out a power of two in exact arithmetic, then frexp what is left.
        e2 = int(math.floor(value.adjusted() * _LOG2_10))
        with localcontext(decimal_context(128 + abs(e2))):
            scaled = value * (Decimal(2) ** -e2)
        m, e = math.frexp(float(scaled))
        return cls(m, e2 + e)

    def __float__(self) -> float:
        return math.ldexp(self.mantissa, self.exponent)

    def log10(self) -> float:
        return math.log10(abs(self.mantissa)) + self.exponent * math.log10(2)

    def __str__(self) -> str:
        """Scientific notation at any exponent, e.g. ``3.2e-1000``."""
        if self.mantissa == 0:
            return "0"
        lg = self.log10()
        ex = math.floor(lg)
        sign = "-" if self.mantissa < 0 else ""
        return f"{sign}{10 ** (lg - ex):.3g}e{ex}"

    def __repr__(self) -> str:
        return f"FloatExp({self})"


def format_decimal(value: Decimal, digits: int) -> str:
    """``value`` rounded to ``digits`` significant digits, as a plain string."""
    with localcontext(Context(prec=max(digits, 1), Emin=-10**9, Emax=10**9)):
        return str(+value)
