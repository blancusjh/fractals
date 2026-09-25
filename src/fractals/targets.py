"""Exact zoom targets: points you can zoom into forever.

A zoom only stays interesting if the centre is *on* the boundary, to every
digit the zoom uses. Two kinds of Mandelbrot point have closed definitions
and so can be computed to any precision by Newton's method:

* **Misiurewicz points** ``M(k, p)``: ``c`` whose critical orbit becomes
  periodic after ``k`` steps, ``f^{k+p}(0) = f^k(0)``. Around them the set is
  asymptotically self-similar — spirals and branches repeat at every scale.
* **Nuclei** of period ``p``: ``f^p(0) = 0``, the centres of hyperbolic
  components — including the minibrots.

Both are solved in fixed-point integers at the requested precision.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

from .precision import decimal_context, fixed_to_float, to_decimal, to_fixed


def _cmul(a, b, c, d, bits):
    return (a * c - b * d) >> bits, (a * d + b * c) >> bits


def _cdiv(a, b, c, d, bits):
    den = (c * c + d * d) >> bits
    if den == 0:
        raise ZeroDivisionError("Newton derivative vanished")
    return ((a * c + b * d) // den), ((b * c - a * d) // den)


def _orbit_with_derivative(cx, cy, n, bits):
    """``f^n(0)`` and ``d/dc f^n(0)`` for ``z² + c``, each step recorded."""
    one = 1 << bits
    x = y = dx = dy = 0
    out = [(0, 0, 0, 0)]
    for _ in range(n):
        # dz ← 2·z·dz + 1, then z ← z² + c
        tx, ty = _cmul(x, y, dx, dy, bits)
        dx, dy = 2 * tx + one, 2 * ty
        x, y = ((x * x - y * y) >> bits) + cx, ((x * y) >> (bits - 1)) + cy
        out.append((x, y, dx, dy))
    return out


def _newton(seed_re, seed_im, residual, digits: int, max_steps: int):
    bits = int(digits * 3.33) + 64
    cx, cy = to_fixed(to_decimal(seed_re), bits), to_fixed(to_decimal(seed_im), bits)
    tol = 1 << 32  # ~2**-(bits-32): converged once the step is below this
    for _ in range(max_steps):
        gx, gy, dgx, dgy = residual(cx, cy, bits)
        sx, sy = _cdiv(gx, gy, dgx, dgy, bits)
        cx, cy = cx - sx, cy - sy
        if abs(sx) < tol and abs(sy) < tol:
            break
    else:
        raise RuntimeError("Newton did not converge; try a closer seed")
    with localcontext(decimal_context(bits)):
        scale = Decimal(2) ** bits
        re, im = Decimal(cx) / scale, Decimal(cy) / scale
    with localcontext(decimal_context(int(digits * 3.33))):
        return +re, +im


def misiurewicz(seed_re, seed_im, preperiod: int, period: int, digits: int = 100,
                max_steps: int = 200) -> tuple[Decimal, Decimal]:
    """The Misiurewicz point ``M(preperiod, period)`` nearest ``seed``.

    Solves ``f^{k+p}(0) − f^k(0) = 0`` by Newton. The root found may have a
    *smaller* (pre)period than asked — check with :func:`preperiod_period`.
    """
    k, p = preperiod, period

    def residual(cx, cy, bits):
        orb = _orbit_with_derivative(cx, cy, k + p, bits)
        x1, y1, dx1, dy1 = orb[k + p]
        x0, y0, dx0, dy0 = orb[k]
        return x1 - x0, y1 - y0, dx1 - dx0, dy1 - dy0

    return _newton(seed_re, seed_im, residual, digits, max_steps)


def nucleus(seed_re, seed_im, period: int, digits: int = 100,
            max_steps: int = 200) -> tuple[Decimal, Decimal]:
    """The centre of the period-``period`` component nearest ``seed``."""

    def residual(cx, cy, bits):
        return _orbit_with_derivative(cx, cy, period, bits)[-1]

    return _newton(seed_re, seed_im, residual, digits, max_steps)


def ball_period(re, im, radius, max_period: int = 100000) -> int | None:
    """Period of the lowest-period nucleus near ``c`` (the ball method).

    Iterates ``z`` and ``dz/dc`` at ``c``; the first ``n`` at which the disc of
    ``radius`` around ``c`` is mapped over 0 (``|zₙ| < |dzₙ/dc|·radius``) is
    the period of a hyperbolic component — a minibrot or a bulb — close by.
    """
    radius = to_decimal(radius)
    bits = max(64, int(-radius.adjusted() * 3.33) + 64)
    cx, cy = to_fixed(to_decimal(re), bits), to_fixed(to_decimal(im), bits)
    rr = to_fixed(radius, bits)
    one = 1 << bits
    x = y = dx = dy = 0
    for n in range(1, max_period + 1):
        tx, ty = _cmul(x, y, dx, dy, bits)
        dx, dy = 2 * tx + one, 2 * ty
        x, y = ((x * x - y * y) >> bits) + cx, ((x * y) >> (bits - 1)) + cy
        z2 = x * x + y * y
        if z2 > (4 * one * one) << 16:
            return None                              # escaped: nothing nearby
        if z2 << (2 * bits) < (dx * dx + dy * dy) * rr * rr:
            return n
    return None


def minibrot_size(re, im, period: int, digits: int = 40) -> complex:
    """Complex size of the minibrot with nucleus ``c``: near it the set is
    approximately ``c + size·M`` — ``|size|`` its scale, ``arg(size)`` its
    rotation (the atom-domain formula)."""
    bits = int(digits * 3.33) + 64
    cx, cy = to_fixed(to_decimal(re), bits), to_fixed(to_decimal(im), bits)
    x = y = 0
    lam = 1 + 0j
    b = 1 + 0j
    for _ in range(1, period):
        x, y = ((x * x - y * y) >> bits) + cx, ((x * y) >> (bits - 1)) + cy
        lam *= 2 * complex(fixed_to_float(x, bits), fixed_to_float(y, bits))
        b += 1 / lam
    return 1 / (b * lam * lam)


def minibrot_near(re, im, radius, digits: int | None = None,
                  max_period: int = 100000) -> tuple[Decimal, Decimal, int, complex] | None:
    """The minibrot found by :func:`ball_period` near ``c``, exactly.

    Returns ``(re, im, period, size)`` — the nucleus to ``digits`` digits
    (default: enough for zooming to ~1e-6 of its size) — or ``None``.
    """
    p = ball_period(re, im, radius, max_period)
    if p is None:
        return None
    guess = int(-to_decimal(radius).adjusted()) + 20
    nre, nim = nucleus(re, im, p, digits=digits or 2 * guess)
    size = minibrot_size(nre, nim, p)
    if digits is None:
        digits = int(-math.log10(abs(size))) + 30
        nre, nim = nucleus(nre, nim, p, digits=digits)
    return nre, nim, p, size


def preperiod_period(re, im, max_preperiod: int = 200, max_period: int = 64,
                     digits: int = 40) -> tuple[int, int] | None:
    """``(k, p)`` of a Misiurewicz point or nucleus (``k = 0``), else ``None``."""
    bits = int(digits * 3.33) + 64
    cx, cy = to_fixed(to_decimal(re), bits), to_fixed(to_decimal(im), bits)
    orb = _orbit_with_derivative(cx, cy, max_preperiod + max_period + 1, bits)
    zs = [(fixed_to_float(x, bits), fixed_to_float(y, bits)) for x, y, _, _ in orb]
    eps = 1e-12
    for k in range(max_preperiod + 1):
        for p in range(1, max_period + 1):
            if all(abs(zs[k + p + m][0] - zs[k + m][0]) < eps
                   and abs(zs[k + p + m][1] - zs[k + m][1]) < eps for m in range(2)):
                return k, p
    return None


#: Named places worth zooming into — exact definitions, not rounded digits.
LANDMARKS = {
    # A preperiod-24 Misiurewicz point in the seahorse valley's neighbour:
    # spirals within spirals, at every depth.
    "spiral": ("misiurewicz", "-0.77568377", "0.13646737", 24, 1),
    # The tip of the antenna, c = −2 (M(2, 1)).
    "antenna": ("misiurewicz", "-2", "0", 2, 1),
    # c = i, where the dendrite branches in three (M(2, 2)).
    "dendrite": ("misiurewicz", "0", "1", 2, 2),
}


def landmark(name: str, digits: int = 100) -> tuple[Decimal, Decimal]:
    """Coordinates of a :data:`LANDMARKS` entry, to ``digits`` digits."""
    kind, re, im, *args = LANDMARKS[name]
    if kind == "misiurewicz":
        return misiurewicz(re, im, *args, digits=digits)
    return nucleus(re, im, *args, digits=digits)
