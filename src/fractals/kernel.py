"""Perturbation rendering: one arbitrary-precision orbit, float64 for the rest.

Write each pixel's orbit as a reference orbit plus a small difference,
``zₙ = Zₙ + δₙ``. The reference ``Zₙ`` is iterated once, exactly, at the view
centre (:mod:`fractals.render`); every pixel then iterates only its ``δₙ``,
which obeys a recurrence obtained by subtracting the two maps — for
``z² + c`` it is ``δₙ₊₁ = 2Zₙδₙ + δₙ² + δc``. ``δ`` only has to be accurate
*relative to itself*, so float64 is enough at any depth.

Two refinements make that robust:

* **Rebasing** (Zhuoran, 2021). When ``|zₙ| < |δₙ|`` the pixel has come
  closer to the origin than to the reference, and ``δ`` is about to lose its
  relative precision. The pixel then restarts from the beginning of the
  reference: ``δ ← zₙ − Z₀``, ``n ← 0``. The same happens when the
  reference orbit runs out. This removes perturbation "glitches" without
  needing a second reference.

* **Extended range.** Past a zoom of ~1e300 a pixel offset underflows float64.
  While ``|δ|`` is below ``2**SCALED_LIMIT`` it is kept as ``w · 2**s`` —
  float64 mantissa, integer exponent — renormalised whenever ``w`` leaves
  ``2**±64``; once it grows into float64 range it is unscaled. Only zooms
  deeper than ~1e270 ever use this mode, and only while ``δ`` is that small.

The kernel returns the continuous (smooth) iteration count per pixel, and
``-1`` for pixels that did not escape.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .families import KIND_BURNING_SHIP, KIND_TRICORN

#: |δ| below ~2**SCALED_LIMIT is carried as mantissa × 2**exponent.
SCALED_LIMIT = -900
_RENORM_HI = 2.0**64
_RENORM_LO = 2.0**-64


@njit(cache=True, inline="always")
def _diffabs(c, d):
    """``|c + d| − |c|`` without cancellation."""
    if c >= 0.0:
        if c + d >= 0.0:
            return d
        return -(2.0 * c + d)
    if c + d > 0.0:
        return 2.0 * c + d
    return -d


@njit(cache=True, inline="always")
def _step(kind, zx, zy, dx, dy, cx, cy, f):
    """One perturbation step, ``δ ← f(Z + δ) − f(Z) + δc``, in units of ``2**s``.

    ``f = 2**s`` multiplies the terms quadratic in ``δ`` (``f = 1`` when
    unscaled); ``(cx, cy)`` is ``δc`` in the same units as ``δ``.
    """
    lin_x = 2.0 * (zx * dx - zy * dy)
    sq_x = f * (dx * dx - dy * dy)
    if kind == KIND_BURNING_SHIP:
        # Im: 2|xy| — difference of absolute values, in the scaled units.
        c = zx * zy
        d = zx * dy + dx * zy + f * dx * dy
        cs = c / f if f != 0.0 else math.inf  # c in units of 2**s
        if abs(cs) > 1e300:                   # |c| ≫ |δ|: the sign of c decides
            if c > 0.0:
                ny = 2.0 * d
            elif c < 0.0:
                ny = -2.0 * d
            else:
                ny = 2.0 * abs(d)
        else:
            ny = 2.0 * _diffabs(cs, d)
        return lin_x + sq_x + cx, ny + cy
    lin_y = 2.0 * (zx * dy + zy * dx)
    sq_y = f * 2.0 * dx * dy
    if kind == KIND_TRICORN:
        return lin_x + sq_x + cx, -(lin_y + sq_y) + cy
    return lin_x + sq_x + cx, lin_y + sq_y + cy


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def perturbation_kernel(
    ref_x, ref_y,            # reference orbit Z₀..Z_N (float64)
    kind, dynamical_plane,   # map, and whether the pixel is z₀ (Julia) or c
    spacing_m, spacing_e,    # pixel spacing = spacing_m · 2**spacing_e
    width, height, imag_down,
    max_iter, bailout2,
    sa_n0=0, sa_e=0, sa_coef=np.zeros(6), sa_r=1.0,
):
    out = np.empty((height, width), dtype=np.float64)
    n_ref = ref_x.shape[0] - 1
    z0x = ref_x[0]
    z0y = ref_y[0]
    log2 = math.log(2.0)
    for idx in prange(width * height):
        j = idx // width
        i = idx - j * width
        # Pixel offset from the centre, as a mantissa at exponent spacing_e.
        ox = (i + 0.5 - 0.5 * width) * spacing_m
        oy = (0.5 * height - j - 0.5) * spacing_m
        if imag_down:
            oy = -oy
        if dynamical_plane:
            wx, wy, pcx, pcy = ox, oy, 0.0, 0.0
        else:
            wx, wy, pcx, pcy = 0.0, 0.0, ox, oy
        # δ = w · 2**s  (scaled);  unscaled δc for later.
        s = spacing_e
        dcx = math.ldexp(pcx, spacing_e)
        dcy = math.ldexp(pcy, spacing_e)
        scaled = True
        rescale = True
        f = scx = scy = 0.0
        n = 0
        it = 0
        if sa_n0 > 0:
            # Series approximation: δ at iteration n₀ is a cubic in u = δc / r
            # (or δz₀ / r), shared by every pixel — skip straight to it.
            ux = ox / sa_r
            uy = oy / sa_r
            u2x = ux * ux - uy * uy
            u2y = 2.0 * ux * uy
            u3x = u2x * ux - u2y * uy
            u3y = u2x * uy + u2y * ux
            ar, ai, br, bi, cr, ci = sa_coef[0], sa_coef[1], sa_coef[2], sa_coef[3], sa_coef[4], sa_coef[5]
            wx = ar * ux - ai * uy + br * u2x - bi * u2y + cr * u3x - ci * u3y
            wy = ar * uy + ai * ux + br * u2y + bi * u2x + cr * u3y + ci * u3x
            s = sa_e
            m = max(abs(wx), abs(wy))
            if m != 0.0:
                e = math.frexp(m)[1]
                wx = math.ldexp(wx, -e)
                wy = math.ldexp(wy, -e)
                s += e
            if s > SCALED_LIMIT:
                wx = math.ldexp(wx, s)
                wy = math.ldexp(wy, s)
                scaled = False
            n = sa_n0
            it = sa_n0
        mu = -1.0
        while it < max_iter:
            if scaled:
                if rescale:  # the exponent moved: refresh the scale factors
                    f = math.ldexp(1.0, s)
                    if not dynamical_plane:
                        g = math.ldexp(1.0, spacing_e - s)
                        scx = pcx * g
                        scy = pcy * g
                    rescale = False
                wx, wy = _step(kind, ref_x[n], ref_y[n], wx, wy, scx, scy, f)
                n += 1
                it += 1
                # Keep the mantissa within 2**±64; leave scaled mode when possible.
                m = max(abs(wx), abs(wy))
                if m != 0.0 and (m > _RENORM_HI or m < _RENORM_LO):
                    e = math.frexp(m)[1]
                    wx = math.ldexp(wx, -e)
                    wy = math.ldexp(wy, -e)
                    s += e
                    rescale = True
                if s > SCALED_LIMIT or n == n_ref:
                    wx = math.ldexp(wx, s)
                    wy = math.ldexp(wy, s)
                    scaled = False
                else:
                    # |δ| ≲ 2**SCALED_LIMIT: the pixel is where the reference is.
                    zx = ref_x[n]
                    zy = ref_y[n]
                    r2 = zx * zx + zy * zy
                    if r2 > bailout2:
                        mu = it + 1.0 - math.log(0.5 * math.log(r2)) / log2
                        break
                    continue
            else:
                wx, wy = _step(kind, ref_x[n], ref_y[n], wx, wy, dcx, dcy, 1.0)
                n += 1
                it += 1
            zx = ref_x[n] + wx
            zy = ref_y[n] + wy
            r2 = zx * zx + zy * zy
            if r2 > bailout2:
                mu = it + 1.0 - math.log(0.5 * math.log(r2)) / log2
                break
            if r2 < wx * wx + wy * wy or n == n_ref:
                wx = zx - z0x
                wy = zy - z0y
                n = 0
        out[j, i] = mu
    return out
