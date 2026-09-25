"""Command line: ``python -m fractals {view,render,zoom} ...``."""

from __future__ import annotations

import argparse
import sys
import time

from . import targets
from .color import PALETTES
from .families import CATALOG, Julia, get
from .precision import to_decimal
from .render import auto_iterations, render
from .view import Viewport
from .zoom import ZoomPath, save_gif, zoom_frames


def _fractal(args):
    if args.fractal == "julia" and args.c:
        return Julia(c=tuple(args.c))
    return get(args.fractal)


def _view(args, fractal):
    view = Viewport.home(fractal, args.width, args.height)
    if args.re is not None:
        view = view.centered_on(args.re, view.im)
    if args.im is not None:
        view = view.centered_on(view.re, args.im)
    if args.radius is not None:
        view = Viewport(view.re, view.im, args.radius, view.width, view.height,
                        view.imag_down)
    return view


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fractals", description="Escape-time fractals at any depth.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("fractal", nargs="?", default="mandelbrot", choices=sorted(CATALOG))
    common.add_argument("--re", help="centre, real part (a string: keep every digit)")
    common.add_argument("--im", help="centre, imaginary part")
    common.add_argument("--radius", help="half-height of the view, e.g. 1e-120")
    common.add_argument("--c", nargs=2, metavar=("RE", "IM"), help="Julia parameter")
    common.add_argument("--width", type=int, default=800)
    common.add_argument("--height", type=int, default=800)
    common.add_argument("--palette", default="fire", choices=sorted(PALETTES))

    sub.add_parser("view", parents=[common], help="interactive window (needs vispy)")

    r = sub.add_parser("render", parents=[common], help="one image to PNG")
    r.add_argument("-o", "--out", default="fractal.png")
    r.add_argument("--iterations", type=int)
    r.add_argument("--supersample", type=int, default=1)

    z = sub.add_parser("zoom", parents=[common], help="zoom animation to GIF")
    z.add_argument("--target", choices=sorted(targets.LANDMARKS),
                   help="exact Mandelbrot landmark (overrides --re/--im)")
    z.add_argument("--depth", default="1e30", help="final magnification")
    z.add_argument("--frames", type=int, default=240)
    z.add_argument("--fps", type=float, default=25)
    z.add_argument("--colors", type=int, default=128)
    z.add_argument("-o", "--out", default="zoom.gif")

    args = ap.parse_args(argv)
    fractal = _fractal(args)

    if args.cmd == "view":
        from .viewer import run
        run(fractal, args.re, args.im, args.radius, (args.width, args.height), args.palette)
        return 0

    if args.cmd == "render":
        from PIL import Image
        view = _view(args, fractal)
        n = args.iterations or auto_iterations(view)
        t = time.time()
        img = render(fractal, view, n, args.palette, supersample=args.supersample)
        Image.fromarray(img).save(args.out)
        print(f"{args.out}: {view.width}×{view.height}, {n} iterations, "
              f"{view.bits} bits, {time.time() - t:.1f} s")
        return 0

    # zoom
    end_radius = 2 / to_decimal(args.depth)
    if args.target:
        digits = int(-end_radius.log10()) + 40
        re, im = targets.landmark(args.target, digits=digits)
    else:
        home = Viewport.home(fractal)
        re = args.re or home.re
        im = args.im or home.im
    path = ZoomPath(fractal, re, im, end_radius, to_decimal(args.radius or 2),
                    args.frames, args.width, args.height)

    def progress(k, view, mu):
        print(f"\rframe {k + 1}/{args.frames}  zoom {view.magnification}", end="", flush=True)

    save_gif(zoom_frames(path, palette=args.palette, progress=progress), args.out,
             fps=args.fps, colors=args.colors)
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
