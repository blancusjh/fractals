"""Render the README's still images.

* ``docs/gallery.png`` — the four fractals of the catalogue.
* ``docs/depths.png``  — one Misiurewicz point at magnifications from 1 to
  1e600, far past float64's range (≈1e308) and its precision (≈1e16).
* ``docs/precision.png`` — the same views in plain float32 / float64 (what a
  GLSL shader or a NumPy loop computes) and in arbitrary precision.

    python examples/gallery.py
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fractals as fr

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

PANELS = [
    # (fractal, centre, radius, palette, caption)
    (fr.Mandelbrot(), ("-0.6", "0"), "1.25", "classic", "Mandelbrot   z^2 + c"),
    (fr.Julia(c=("-0.75", "0.11")), ("0", "0"), "1.1", "ocean", "Julia   z^2 - 0.75 + 0.11i"),
    (fr.Tricorn(), ("-0.3", "0"), "1.3", "fire", "Tricorn   conj(z)^2 + c"),
    (fr.BurningShip(), ("-1.7621", "-0.0290"), "0.052", "fire", "Burning Ship   (|x| + i|y|)^2 + c"),
]

DEPTHS = ["1", "1e3", "1e8", "1e20", "1e60", "1e150", "1e320", "1e600"]


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def caption(img, text, size=15):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = font(size)
    x, y = 10, im.height - 10
    box = d.textbbox((x, y), text, font=f, anchor="ls")
    d.rectangle((box[0] - 5, box[1] - 4, box[2] + 5, box[3] + 4), fill=(0, 0, 0))
    d.text((x, y), text, font=f, fill=(255, 255, 255), anchor="ls")
    return np.asarray(im)


def gallery(w=480, h=360):
    tiles = []
    for fractal, (re, im), radius, palette, text in PANELS:
        view = fr.Viewport(re, im, radius, w, h, fractal.imag_down)
        tiles.append(caption(fr.render(fractal, view, max_iter=1500, palette=palette,
                                       supersample=3), text))
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    Image.fromarray(grid).save(DOCS / "gallery.png", optimize=True)


def depths(size=240):
    re, im = fr.targets.landmark("spiral", digits=650)
    tiles = []
    for d in DEPTHS:
        view = fr.Viewport(re, im, 2, size, size).zoom(d)
        t = time.time()
        img = fr.render("mandelbrot", view, palette="fire", period=64)
        print(f"zoom {d:>6}: {view.bits:5d} bits, {time.time() - t:5.1f} s")
        tiles.append(caption(img, f"zoom {d}", size=14))
    grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    Image.fromarray(grid).save(DOCS / "depths.png", optimize=True)


def direct(view, max_iter, dtype):
    """Plain z² + c with every coordinate in ``dtype`` — no perturbation."""
    real = np.float32 if dtype == np.complex64 else np.float64
    s = real(float(view.spacing))
    i = (np.arange(view.width, dtype=real) + real(0.5) - real(view.width / 2)) * s
    j = (real(view.height / 2) - np.arange(view.height, dtype=real) - real(0.5)) * s
    c = (real(float(view.re)) + i[None, :]) + 1j * (real(float(view.im)) + j[:, None])
    c = c.astype(dtype)
    z = np.zeros_like(c)
    mu = np.full(c.shape, -1.0)
    alive = np.ones(c.shape, bool)
    for n in range(1, max_iter + 1):
        z[alive] = z[alive] ** 2 + c[alive]
        r2 = np.abs(z.astype(np.complex128)) ** 2
        esc = alive & (r2 > 256.0**2)
        mu[esc] = n + 1 - np.log2(0.5 * np.log(r2[esc]))
        alive &= ~esc
    return mu


def precision(size=300):
    re, im = fr.targets.landmark("spiral", digits=60)
    rows = []
    for depth, dtype, name in [("3e6", np.complex64, "float32"),
                               ("1e15", np.complex128, "float64")]:
        view = fr.Viewport(re, im, 2, size, size).zoom(depth)
        n = fr.auto_iterations(view)
        plain = fr.colorize(direct(view, n, dtype), "fire", period=64)
        exact = fr.colorize(fr.escape_time("mandelbrot", view, n), "fire", period=64)
        rows.append(np.hstack([caption(plain, f"zoom {depth}: {name}", 14),
                               caption(exact, f"zoom {depth}: fractals ({view.bits} bits)", 14)]))
    Image.fromarray(np.vstack(rows)).save(DOCS / "precision.png", optimize=True)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    gallery()
    depths()
    precision()
