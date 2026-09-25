"""Render the README's infinite-zoom GIF: a dive that loops forever.

The dive starts on the whole Mandelbrot set, goes down Seahorse Valley past
spirals, double spirals and embedded Julia sets, and ends on a period-998
minibrot at 10¹⁵× — a copy of the whole set. Everything is exact: the target
is the minibrot's nucleus, found by the ball method and Newton's method, and
its complex size gives the copy's scale and rotation.

Because the minibrot *is* the set again, the last frame can be the first one:
the camera drifts and rotates so the minibrot ends up framed exactly like the
opening view, and the final frames cross-fade onto that opening view (the
two differ only in colour, since escape counts near the minibrot are ~998×
larger). The GIF then loops seamlessly — zooming in forever.

    python examples/deep_zoom_gif.py            # docs/infinite_zoom.gif
    python examples/deep_zoom_gif.py --gpu      # render with the GLSL kernel
"""

import argparse
import cmath
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

import fractals as fr
from fractals.targets import minibrot_near

ROOT = Path(__file__).resolve().parents[1]

# The classic Seahorse Valley coordinates (33 digits); the minibrot found
# near them is then computed exactly.
SEAHORSE = ("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139")
HOME = complex(-0.6, 0.0)     # the opening view: whole set, centred here
HOME_RADIUS = 1.5


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def escape_square(centre, radius, size, max_iter, device):
    """Escape counts on a square √2 larger than the frame, so any rotation of
    it still covers the ``size``² frame."""
    big = int(math.ceil(size * math.sqrt(2))) | 1
    view = fr.Viewport(*centre, fr.to_decimal(radius * big / size), big, big)
    n = max_iter(view) if callable(max_iter) else max_iter
    return fr.escape_time("mandelbrot", view, n, device=device).astype(np.float32), n


def colour(mu, angle, size, base, levels=12):
    """Colour escape counts in flat bands, then rotate the image and crop.

    The palette position follows the (log) escape count above the frame's
    base, stepped into ``levels`` flat bands per palette cycle — contour
    bands rather than a smooth gradient, at any depth.
    """
    escaped = mu >= 0
    t = 0.42 * np.log1p(np.maximum(np.where(escaped, mu, 0) - base, 0.0))
    t = np.floor(t * levels) / levels
    rgb = fr.color.palette_lookup(t, "fire")
    rgb[~escaped] = 0
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    img = img.rotate(math.degrees(angle), resample=Image.NEAREST)
    off = (img.width - size) // 2
    return np.asarray(img.crop((off, off, off + size, off + size)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=int, default=270)
    ap.add_argument("--fade", type=int, default=32, help="cross-fade frames at the end")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--fps", type=float, default=25)
    ap.add_argument("--colors", type=int, default=64)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "infinite_zoom.gif"))
    args = ap.parse_args()
    device = "gpu" if args.gpu else "cpu"

    nre, nim, period, size = minibrot_near(*SEAHORSE, "1e-10")
    N = complex(float(nre), float(nim))
    unit = size / abs(size)
    phi = cmath.phase(size)
    print(f"target: period-{period} minibrot, size {abs(size):.3e}, "
          f"rotated {math.degrees(phi):.1f}°")

    K = args.frames
    r0, r1 = HOME_RADIUS, HOME_RADIUS * abs(size)
    t0 = time.time()
    deep, mapped = [], []
    for k in range(K):
        t = k / (K - 1)
        R = r0 * (r1 / r0) ** t
        # Offset of the view centre from the nucleus: starts at the home view,
        # ends at the minibrot's image of the home view (size · HOME).
        D = (R / r0) * ((HOME - N) * (1 - t) + unit * HOME * t)
        centre = (nre + fr.to_decimal(D.real), nim + fr.to_decimal(D.imag))
        # Near the minibrot one step of the set's own dynamics costs `period`
        # iterations, so its surroundings need a far larger budget.
        if R < 300 * abs(size):
            budget = 200_000
        elif R < 1e-9:
            budget = lambda v: max(fr.auto_iterations(v), 30 * period)  # noqa: E731
        else:
            budget = fr.auto_iterations
        mu, n_iter = escape_square(centre, R, args.size, budget, device)
        deep.append(mu)
        if k >= K - args.fade:
            # The same frame on the whole set, through c' = (c − N) / size.
            Cp = D / size
            mu2, _ = escape_square((fr.to_decimal(Cp.real), fr.to_decimal(Cp.imag)),
                                   R / abs(size), args.size, fr.auto_iterations, device)
            mapped.append(mu2)
        if k % 20 == 0 or k == K - 1:
            print(f"frame {k:3d}/{K}  radius {R:.2e}  {n_iter} it  {time.time() - t0:6.0f} s",
                  flush=True)

    # Colour base: each frame's minimum escape count, smoothed over time so the
    # palette glides instead of flickering.
    def bases(mus):
        b = np.array([m[m >= 0].min() if (m >= 0).any() else 0.0 for m in mus])
        lb = np.log1p(b)
        sm = np.convolve(np.pad(lb, 6, mode="edge"), np.ones(13) / 13, mode="valid")
        return np.expm1(sm)

    bd = bases(deep)
    home_base = bd[0]            # the fade ends on frame 0, coloured as frame 0
    frames = []
    for k in range(K):
        t = k / (K - 1)
        img = colour(deep[k], -t * phi, args.size, bd[k]).astype(float)
        j = k - (K - args.fade)
        if j >= 0:
            a = smoothstep((j + 1) / args.fade)
            img2 = colour(mapped[j], -t * phi + phi, args.size, home_base).astype(float)
            img = (1 - a) * img + a * img2
        frames.append(np.round(img).astype(np.uint8))
    # The last frame equals the first: drop it so the loop does not stutter.
    frames = frames[:-1]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fr.save_gif(frames, args.out, fps=args.fps, colors=args.colors)
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
