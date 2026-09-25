"""Colouring: smooth iteration counts to RGB."""

from __future__ import annotations

import numpy as np

#: Cyclic palettes as control points (the last point wraps to the first).
PALETTES: dict[str, np.ndarray] = {
    # The classic "Ultra Fractal" gradient: deep blue, white, amber, black.
    "classic": np.array([
        (0, 7, 100), (32, 107, 203), (237, 255, 255), (255, 170, 0), (0, 2, 0),
    ], dtype=float),
    "fire": np.array([
        (10, 2, 20), (120, 20, 40), (230, 90, 20), (255, 210, 90),
        (255, 250, 230), (70, 130, 200), (20, 30, 80),
    ], dtype=float),
    "ocean": np.array([
        (2, 10, 30), (10, 70, 120), (60, 170, 190), (230, 250, 240),
        (240, 180, 90), (110, 40, 60),
    ], dtype=float),
    "gray": np.array([(10, 10, 10), (245, 245, 245)], dtype=float),
}

# The cosine colormap of the original GLSL viewer (examples/original/gloo.py):
# 0.5 + 0.5·cos(3 + 2π·(t, 1.5t, 2t)), sampled over its full period t ∈ [0, 2).
_t = np.linspace(0.0, 2.0, 96, endpoint=False)[:, None]
PALETTES["glsl"] = 255 * (0.5 + 0.5 * np.cos(3.0 + 2 * np.pi * _t * np.array([1.0, 1.5, 2.0])))
del _t


def palette_lookup(t: np.ndarray, palette: str | np.ndarray) -> np.ndarray:
    """Sample a cyclic palette at positions ``t`` (period 1). Returns floats."""
    stops = PALETTES[palette] if isinstance(palette, str) else np.asarray(palette, float)
    k = len(stops)
    u = np.mod(t, 1.0) * k
    i0 = np.floor(u).astype(int) % k
    i1 = (i0 + 1) % k
    w = (u - np.floor(u))[..., None]
    w = w * w * (3 - 2 * w)          # smoothstep between stops
    return stops[i0] * (1 - w) + stops[i1] * w


def colorize(
    mu: np.ndarray,
    palette: str | np.ndarray = "fire",
    period: float | None = None,
    offset: float = 0.0,
    interior=(0, 0, 0),
    banded: bool = True,
) -> np.ndarray:
    """RGB ``uint8`` image from smooth iteration counts (``-1`` = inside).

    With ``period=None`` the palette follows ``log(1 + μ − μ_min)``, the
    iterations above the image's own minimum — good for a single image at any
    depth (deep views have μ in the tens of thousands, all within a narrow
    band, so plain ``log(μ)`` would be one flat colour). With a number, one palette
    cycle spans ``period`` iterations — stable from frame to frame, which is
    what a zoom animation wants.

    ``banded=True`` (the default) colours by the whole number of iterations,
    so each escape band is one flat colour — the classic escape-time look.
    ``banded=False`` uses the continuous count for smooth gradients.
    """
    escaped = mu >= 0
    m = np.where(escaped, mu, 0.0)
    if banded:
        m = np.floor(m)
    if period is None:
        base = m[escaped].min() if escaped.any() else 0.0
        t = 1.4 * np.log1p(np.maximum(m - base, 0.0)) + offset
    else:
        t = m / period + offset
    rgb = palette_lookup(t, palette)
    rgb[~escaped] = interior
    return np.clip(rgb, 0, 255).astype(np.uint8)
