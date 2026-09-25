"""Perturbation against direct iteration — shallow in float64, deep in big ints."""

import math

import numpy as np
import pytest

import fractals as fr
from fractals.precision import fixed_to_float, to_fixed
from fractals.render import BAILOUT

FAMILIES = [fr.Mandelbrot(), fr.Julia(), fr.Tricorn(), fr.BurningShip()]


def direct_float(fractal, view, max_iter):
    """Plain complex128 iteration of every pixel (valid only for shallow views)."""
    h, w = view.height, view.width
    s = float(view.spacing)
    i = (np.arange(w) + 0.5 - w / 2) * s
    j = (h / 2 - np.arange(h) - 0.5) * s
    if view.imag_down:
        j = -j
    p = float(view.re) + i[None, :] + 1j * (float(view.im) + j[:, None])
    if fractal.dynamical_plane:
        z, c = p, complex(float(fr.to_decimal(fractal.c[0])), float(fr.to_decimal(fractal.c[1])))
    else:
        z, c = np.zeros_like(p), p
    mu = np.full(p.shape, -1.0)
    alive = np.ones(p.shape, bool)
    for n in range(1, max_iter + 1):
        if fractal.kind == fr.families.KIND_TRICORN:
            z = np.conj(z) ** 2 + c
        elif fractal.kind == fr.families.KIND_BURNING_SHIP:
            z = (np.abs(z.real) + 1j * np.abs(z.imag)) ** 2 + c
        else:
            z = z * z + c
        r2 = np.abs(z) ** 2
        esc = alive & (r2 > BAILOUT**2)
        mu[esc] = n + 1 - np.log2(0.5 * np.log(r2[esc]))
        alive &= ~esc
        z[~alive] = 0
    return mu


def direct_exact(fractal, view, i, j, max_iter):
    """One pixel, iterated exactly in fixed point at the view's precision."""
    bits = view.bits + 32
    re, im = view.point(i, j)
    x, y, cx, cy = fractal.start(to_fixed(re, bits), to_fixed(im, bits), bits)
    for n in range(1, max_iter + 1):
        x, y = fractal.step_fixed(x, y, cx, cy, bits)
        fx, fy = fixed_to_float(x, bits), fixed_to_float(y, bits)
        r2 = fx * fx + fy * fy
        if r2 > BAILOUT**2:
            return n + 1 - math.log2(0.5 * math.log(r2))
    return -1.0


@pytest.mark.parametrize("fractal", FAMILIES, ids=str)
def test_shallow_matches_float64(fractal):
    view = fr.Viewport.home(fractal, 96, 72)
    mu = fr.escape_time(fractal, view, 300)
    ref = direct_float(fractal, view, 300)
    # A few pixels sit on chaotic orbits, where *any* two finite precisions —
    # float64 direct, float64 perturbed, even 100-bit exact — part ways after
    # enough iterations (the Burning Ship has the most of them). The rest agree.
    same_class = (mu < 0) == (ref < 0)
    assert same_class.mean() > 0.99
    both = (mu >= 0) & (ref >= 0)
    close = np.abs(mu[both] - ref[both]) < 1e-6
    assert close.mean() > 0.98


def julia_beta(c_re, c_im, digits=150):
    """The repelling fixed point z = z² + c — a point of the Julia set, exactly."""
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = digits + 10
        cr, ci = Decimal(c_re), Decimal(c_im)
        x, y = Decimal("1.5"), Decimal(0)
        for _ in range(60):   # Newton on g(z) = z² − z + c
            gr, gi = x * x - y * y - x + cr, 2 * x * y - y + ci
            dr, di = 2 * x - 1, 2 * y
            den = dr * dr + di * di
            x, y = x - (gr * dr + gi * di) / den, y - (gi * dr - gr * di) / den
        return +x, +y


# Points on each boundary, exact to every digit the zoom uses.
DEEP = {
    "mandelbrot": fr.targets.landmark("spiral", digits=150),   # Misiurewicz point
    "julia": julia_beta(*fr.Julia().c),                          # repelling fixed point
    "tricorn": ("-2", "0"),                                      # preperiodic tip
    "burning_ship": ("-2", "0"),                                 # preperiodic tip
}


@pytest.mark.parametrize("fractal", FAMILIES, ids=str)
@pytest.mark.parametrize("depth", [1e20, 1e60])
def test_deep_matches_exact(fractal, depth):
    re, im = DEEP[fractal.name]
    # Offset a little so the reference is not symmetric with the image.
    view = fr.Viewport(re, im, fr.to_decimal(fractal.default_radius), 24, 24,
                       fractal.imag_down).zoom(depth).pan(3.3, -2.7)
    max_iter = fr.auto_iterations(view)
    mu = fr.escape_time(fractal, view, max_iter)
    escaped = mu >= 0
    assert escaped.mean() > 0.2, "too few escaping pixels to test anything"
    assert np.ptp(mu[escaped]) > 1, "no structure at this depth"
    rng = np.random.default_rng(0)
    pix = [(int(a), int(b)) for a, b in rng.integers(0, 24, size=(12, 2))]
    checked = 0
    for i, j in pix:
        exact = direct_exact(fractal, view, i, j, max_iter)
        # Near chaotic orbits (the Burning Ship's mast is the fully chaotic
        # map x² − 2) the answer depends on digits no finite precision holds.
        # Compare only where it does not: moving by 1e-9 pixel changes nothing.
        nudged = direct_exact(fractal, view, i + 1e-9, j, max_iter)
        if abs(exact - nudged) > 1e-6:
            continue
        checked += 1
        got = mu[j, i]
        assert (got < 0) == (exact < 0), (i, j, got, exact)
        if exact >= 0:
            assert got == pytest.approx(exact, abs=1e-4), (i, j)
    assert checked >= 6


def test_beyond_float64_exponent_range():
    """1e-400 is not a float64; the extended-range path must still be exact."""
    re, im = fr.targets.landmark("spiral", digits=450)
    view = fr.Viewport(re, im, "1e-400", 16, 16)
    assert view.bits > 1300
    max_iter = fr.auto_iterations(view)
    mu = fr.escape_time("mandelbrot", view, max_iter)
    assert (mu >= 0).mean() > 0.2
    for i, j in [(0, 0), (15, 3), (7, 8), (4, 13), (11, 11)]:
        exact = direct_exact(fr.Mandelbrot(), view, i, j, max_iter)
        assert mu[j, i] == pytest.approx(exact, abs=1e-4)
    # And the image is not flat: structure survives at this depth.
    assert np.ptp(mu[mu >= 0]) > 1
