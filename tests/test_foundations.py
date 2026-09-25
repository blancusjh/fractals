"""Precision, viewport navigation and zoom targets."""

from decimal import Decimal

import pytest

import fractals as fr
from fractals.precision import FloatExp, fixed_to_float, to_decimal, to_fixed
from fractals.targets import landmark, misiurewicz, nucleus, preperiod_period


# --- precision ---------------------------------------------------------------

def test_fixed_point_round_trip_keeps_relative_precision():
    for v in ["3.8e-21", "-1.25", "1e-300", "-0.7756837680090537974694"]:
        d = Decimal(v)
        assert fixed_to_float(to_fixed(d, 2000), 2000) == pytest.approx(float(d), rel=1e-15)


def test_floatexp_beyond_float64():
    fe = FloatExp.from_decimal(Decimal("3e-1000"))
    assert 0.5 <= fe.mantissa < 1
    assert fe.log10() == pytest.approx(-1000 + 0.4771212547, abs=1e-9)
    assert float(FloatExp.from_decimal(Decimal("0.75"))) == 0.75


def test_to_decimal_is_exact_for_strings():
    s = "-0.74364388703715870475219150611477352910780986578023927616"
    assert str(to_decimal(s)) == s
    with pytest.raises(TypeError):
        to_decimal(object())


# --- viewport ----------------------------------------------------------------

@pytest.mark.parametrize("imag_down", [False, True])
def test_zoom_keeps_the_point_under_the_cursor(imag_down):
    v = fr.Viewport("-0.75", "0.1", "1e-40", 300, 200, imag_down)
    about = (41.5, 170.25)
    w = v.zoom(1e30, about=about)
    before, after = v.point(*about), w.point(*about)
    tol = w.spacing * Decimal("1e-6")
    assert abs(before[0] - after[0]) < tol and abs(before[1] - after[1]) < tol
    assert w.bits > v.bits + 90


@pytest.mark.parametrize("imag_down", [False, True])
def test_pan_moves_content_with_the_mouse(imag_down):
    v = fr.Viewport("0.3", "-0.2", "1e-30", 100, 100, imag_down)
    w = v.pan(10, -4)
    # What was under pixel (20, 30) is now under (30, 26).
    a, b = v.point(20, 30), w.point(30, 26)
    assert abs(a[0] - b[0]) < v.spacing * Decimal("1e-9")
    assert abs(a[1] - b[1]) < v.spacing * Decimal("1e-9")


def test_viewport_home_and_catalog():
    assert set(fr.CATALOG) == {"mandelbrot", "julia", "tricorn", "burning_ship"}
    for name in fr.CATALOG:
        v = fr.Viewport.home(fr.get(name), 64, 48)
        assert v.imag_down == fr.get(name).imag_down
    with pytest.raises(KeyError):
        fr.get("sierpinski")


# --- targets -----------------------------------------------------------------

def test_misiurewicz_landmark_is_preperiodic():
    re, im = landmark("spiral", digits=200)
    assert preperiod_period(re, im) == (24, 1)
    # 200 digits really are 200 digits: recomputing at 250 agrees to ~1e-195.
    re2, im2 = landmark("spiral", digits=250)
    assert abs(re - re2) < Decimal("1e-195") and abs(im - im2) < Decimal("1e-195")


def test_nucleus_of_period_3():
    re, im = nucleus("-1.75", "0", 3, digits=60)          # the airship minibrot
    assert preperiod_period(re, im) == (0, 3)
    assert abs(float(re) + 1.7548776662466927) < 1e-15


def test_misiurewicz_known_value():
    re, im = misiurewicz("0.1", "0.9", 2, 2, digits=50)   # c = i
    assert abs(re) < Decimal("1e-45") and abs(im - 1) < Decimal("1e-45")


def test_floatexp_prints_at_any_exponent():
    assert str(FloatExp.from_decimal(Decimal("3.2e-1000"))) == "3.2e-1000"
    assert str(FloatExp.from_decimal(Decimal("-5e40"))) == "-5e40"


# --- viewer navigation (no GUI needed) ----------------------------------------

def test_navigator_preview_follows_zoom_and_pan():
    from fractals.viewer import Navigator

    nav = Navigator.start("mandelbrot", 200, 100)
    nav.shown = nav.view
    assert nav.preview_transform() == (1.0, 0.0, 0.0)
    nav.wheel((150.0, 50.0), 4)            # zoom 1.25**4 about a point right of centre
    scale, dx, dy = nav.preview_transform()
    assert scale == pytest.approx(1.25**4)
    # The pixel under the cursor did not move: centre + scale·(p − centre) + d = p.
    assert 100 + scale * 50 + dx == pytest.approx(150.0)
    assert dy == pytest.approx(0.0, abs=1e-9)
    nav.shown = nav.view
    nav.drag(7, -3)
    assert nav.preview_transform() == pytest.approx((1.0, 7.0, -3.0))


def test_navigator_keys():
    from fractals.viewer import Navigator

    nav = Navigator.start("mandelbrot", 32, 24)
    n0 = nav.max_iter
    nav.more_iterations(2)
    assert nav.max_iter == 2 * n0
    nav.select("burning_ship")
    assert nav.fractal.name == "burning_ship" and nav.view.imag_down
    p = nav.palette
    nav.next_palette()
    assert nav.palette != p
    assert nav.render().shape == (24, 32, 3)
    assert "burning_ship" in nav.describe()
