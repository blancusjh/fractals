"""From a fractal and a viewport to escape data and pixels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import color as _color
from .families import KIND_QUADRATIC, Fractal, get
from .kernel import perturbation_kernel
from .precision import FloatExp, fixed_to_float, to_fixed
from .view import Viewport

#: Escape radius. Large, so the smooth iteration count is smooth.
BAILOUT = 256.0


@dataclass(frozen=True)
class ReferenceOrbit:
    """The exact orbit of the view centre, rounded to float64 for the kernel."""

    x: np.ndarray
    y: np.ndarray
    bits: int

    @property
    def escaped_at(self) -> int | None:
        r2 = self.x[-1] ** 2 + self.y[-1] ** 2
        return len(self.x) - 1 if r2 > BAILOUT**2 else None


def reference_orbit(fractal: Fractal, view: Viewport, max_iter: int) -> ReferenceOrbit:
    """Iterate the centre of ``view`` in fixed point, at the precision it needs."""
    bits = view.bits
    px, py = to_fixed(view.re, bits), to_fixed(view.im, bits)
    x, y, cx, cy = fractal.start(px, py, bits)
    bail = BAILOUT**2
    xs = [fixed_to_float(x, bits)]
    ys = [fixed_to_float(y, bits)]
    step = fractal.step_fixed
    for _ in range(max_iter):
        x, y = step(x, y, cx, cy, bits)
        fx, fy = fixed_to_float(x, bits), fixed_to_float(y, bits)
        xs.append(fx)
        ys.append(fy)
        if fx * fx + fy * fy > bail:
            break
    return ReferenceOrbit(np.array(xs), np.array(ys), bits)


@dataclass(frozen=True)
class SeriesStart:
    """Where every pixel may start: ``δ_{n₀} ≈ a·u + b·u² + c·u³``.

    ``u`` is the pixel offset divided by ``r`` (so ``|u| ≤ 1`` on the image),
    in the kernel's mantissa units; the coefficients are mantissas at
    exponent ``e`` (actual value = mantissa · 2**e).
    """

    n0: int
    coef: np.ndarray   # (a.re, a.im, b.re, b.im, c.re, c.im)
    e: int
    r: float


#: Stop the series once its first neglected (4th-order) term exceeds this
#: fraction of the linear one.
SERIES_TOLERANCE = 2.0**-42


def series_start(fractal: Fractal, ref: ReferenceOrbit, view: Viewport,
                 tolerance: float = SERIES_TOLERANCE) -> SeriesStart | None:
    """Series approximation of the shared start of every pixel's orbit.

    For ``z² + c`` the perturbation is a power series in the pixel offset,
    ``δₙ = Aₙ·δc + Bₙ·δc² + Cₙ·δc³ + …`` with coefficients that depend only on
    the reference: ``Aₙ₊₁ = 2ZₙAₙ + 1``, ``Bₙ₊₁ = 2ZₙBₙ + Aₙ²``,
    ``Cₙ₊₁ = 2ZₙCₙ + 2AₙBₙ`` (and ``Dₙ₊₁ = 2ZₙDₙ + 2AₙCₙ + Bₙ²``, the error
    estimate). While the cubic is accurate for the farthest pixel, iterating
    is unnecessary — deep zooms skip most of their iterations this way.
    Only the holomorphic map ``z² + c`` (Mandelbrot, Julia) has such a series.
    """
    if fractal.kind != KIND_QUADRATIC:
        return None
    sp = FloatExp.from_decimal(view.spacing)
    r = sp.mantissa * math.hypot(view.width / 2, view.height / 2)
    julia = fractal.dynamical_plane
    e = sp.exponent
    a = complex(r) if julia else 0j
    b = c = d = 0j
    best = None
    zs = ref.x + 1j * ref.y
    for n in range(len(zs) - 2):
        two_z = 2 * complex(zs[n])
        k = math.ldexp(1.0, e)                 # 2**e, may underflow to 0
        a, b, c, d = (
            two_z * a + (0 if julia else r * math.ldexp(1.0, sp.exponent - e)),
            two_z * b + a * a * k,
            two_z * c + 2 * a * b * k,
            two_z * d + (2 * a * c + b * b) * k,
        )
        m = abs(a)
        if m > 2.0**32 or 0 < m < 2.0**-32:
            shift = math.frexp(m)[1]
            scale = math.ldexp(1.0, -shift)
            a, b, c, d = a * scale, b * scale, c * scale, d * scale
            e += shift
        if not (abs(d) <= tolerance * abs(a)):   # also catches inf/nan
            break
        best = (n + 1, a, b, c, e)
    if best is None or best[0] < 2:
        return None
    n0, a, b, c, e = best
    coef = np.array([a.real, a.imag, b.real, b.imag, c.real, c.imag])
    return SeriesStart(n0, coef, e, r)


def escape_time(fractal: Fractal | str, view: Viewport, max_iter: int = 1000,
                series: bool = True, device: str = "cpu") -> np.ndarray:
    """Smooth iteration count per pixel, shape ``(height, width)``; ``-1`` inside.

    Works at any zoom: one reference orbit in arbitrary precision, then
    float64 perturbation (with extended exponent range) for every pixel.
    ``series`` lets the pixels skip their shared first iterations by series
    approximation (Mandelbrot and Julia). ``device="gpu"`` runs the same
    algorithm as a GLSL shader (:mod:`fractals.gpu`, needs OpenGL via VisPy).
    """
    if device == "gpu":
        from .gpu import default_renderer
        return default_renderer().escape_time(fractal, view, max_iter, series)
    if device != "cpu":
        raise ValueError(f"device must be 'cpu' or 'gpu', not {device!r}")
    fractal = get(fractal)
    ref = reference_orbit(fractal, view, max_iter)
    sp = FloatExp.from_decimal(view.spacing)
    sa = series_start(fractal, ref, view) if series else None
    extra = () if sa is None else (sa.n0, sa.e, sa.coef, sa.r)
    return perturbation_kernel(
        ref.x, ref.y,
        fractal.kind, fractal.dynamical_plane,
        sp.mantissa, sp.exponent,
        view.width, view.height, view.imag_down,
        int(max_iter), BAILOUT**2,
        *extra,
    )


def auto_iterations(view: Viewport, base: int = 400, per_decade: int = 120) -> int:
    """A reasonable iteration budget for a zoom depth (it grows with depth)."""
    decades = max(0.0, view.magnification.log10())
    return int(base + per_decade * decades)


def render(
    fractal: Fractal | str,
    view: Viewport,
    max_iter: int | None = None,
    palette: str = "fire",
    supersample: int = 1,
    device: str = "cpu",
    **color_kw,
) -> np.ndarray:
    """An RGB ``uint8`` image, shape ``(height, width, 3)``.

    ``supersample=k`` renders ``k×k`` samples per pixel and averages their
    colours — anti-aliasing for the filaments near the boundary.
    """
    if max_iter is None:
        max_iter = auto_iterations(view)
    k = int(supersample)
    fine = view.resized(view.width * k, view.height * k) if k > 1 else view
    rgb = _color.colorize(escape_time(fractal, fine, max_iter, device=device), palette, **color_kw)
    if k > 1:
        rgb = rgb.reshape(view.height, k, view.width, k, 3).mean(axis=(1, 3))
        rgb = np.round(rgb).astype(np.uint8)
    return rgb
