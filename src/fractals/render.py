"""From a fractal and a viewport to escape data and pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as _color
from .families import Fractal, get
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


def escape_time(fractal: Fractal | str, view: Viewport, max_iter: int = 1000) -> np.ndarray:
    """Smooth iteration count per pixel, shape ``(height, width)``; ``-1`` inside.

    Works at any zoom: one reference orbit in arbitrary precision, then
    float64 perturbation (with extended exponent range) for every pixel.
    """
    fractal = get(fractal)
    ref = reference_orbit(fractal, view, max_iter)
    sp = FloatExp.from_decimal(view.spacing)
    return perturbation_kernel(
        ref.x, ref.y,
        fractal.kind, fractal.dynamical_plane,
        sp.mantissa, sp.exponent,
        view.width, view.height, view.imag_down,
        int(max_iter), BAILOUT**2,
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
    rgb = _color.colorize(escape_time(fractal, fine, max_iter), palette, **color_kw)
    if k > 1:
        rgb = rgb.reshape(view.height, k, view.width, k, 3).mean(axis=(1, 3))
        rgb = np.round(rgb).astype(np.uint8)
    return rgb
