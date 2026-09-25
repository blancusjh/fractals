"""Render the README's garden: twelve places in four fractals.

    python examples/garden.py            # docs/garden/*.png
"""

from pathlib import Path

from PIL import Image

import fractals as fr
from fractals.targets import landmark, minibrot_near

OUT = Path(__file__).resolve().parents[1] / "docs" / "garden"

M, T, B = fr.Mandelbrot(), fr.Tricorn(), fr.BurningShip()
SEAHORSE = ("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139")


def places():
    spiral = landmark("spiral", digits=60)
    nre, nim, _, size = minibrot_near(*SEAHORSE, "1e-10")
    # (file, fractal, centre, radius, iterations, palette, iterations per colour cycle)
    return [
        ("seahorse_valley", M, ("-0.7453", "0.1127"), "0.0012", 3000, "fire", 24),
        ("elephant_valley", M, ("0.27514", "0.00685"), "0.0008", 3000, "ocean", 24),
        ("triple_spiral", M, ("-0.08840", "0.65467"), "0.0004", 4000, "classic", 24),
        ("endless_spiral", M, spiral, "2e-8", 3000, "fire", 32),
        ("star", M, ("-0.1011", "0.9563"), "0.0006", 4000, "fire", 24),
        ("tendrils", M, ("-1.25066", "0.02012"), "0.00025", 3000, "fire", 24),
        ("minibrot", M, (nre, nim), str(2.2 * abs(size)), 200000, "classic", 998 * 4),
        ("rabbit", fr.Julia(c=("-0.123", "0.745")), ("0", "0"), "1.15", 1500, "classic", 16),
        ("julia_spirals", fr.Julia(c=("-0.75", "0.11")), ("0", "0"), "1.1", 1500, "ocean", 16),
        ("tricorn", T, ("-0.3", "0"), "1.3", 1500, "fire", 16),
        ("armada", B, ("-1.7621", "-0.0290"), "0.052", 1500, "fire", 16),
        ("lone_ship", B, ("-1.861", "-0.0017"), "0.012", 2000, "classic", 16),
    ]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fractal, (re, im), radius, n, palette, period in places():
        view = fr.Viewport(re, im, radius, 400, 300, fractal.imag_down)
        img = fr.render(fractal, view, n, palette, supersample=2, period=period)
        Image.fromarray(img).quantize(256, dither=Image.Dither.NONE).save(OUT / f"{name}.png", optimize=True)
        print(name)
