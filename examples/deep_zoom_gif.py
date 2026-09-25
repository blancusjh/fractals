"""Render the README's deep-zoom GIF.

The target is a Misiurewicz point, computed by Newton's method to as many
digits as the zoom needs, so it sits exactly on the boundary and new detail
keeps coming at every depth. Each frame is rendered by perturbation around an
arbitrary-precision reference orbit; the label shows the magnification and the
binary precision the centre needs at that depth.

    python examples/deep_zoom_gif.py                      # docs/deep_zoom.gif
    python examples/deep_zoom_gif.py --depth 1e100 --frames 600 --size 480
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fractals as fr

ROOT = Path(__file__).resolve().parents[1]


def label(img: np.ndarray, view: fr.Viewport) -> np.ndarray:
    """Magnification and working precision, bottom-left."""
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(11, img.shape[0] // 22))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    mag = view.magnification.log10()
    e = math.floor(mag)
    text = f"zoom {10 ** (mag - e):.1f}e{e}   {view.bits} bits"
    x, y = 8, img.shape[0] - 8
    box = draw.textbbox((x, y), text, font=font, anchor="ls")
    draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255), anchor="ls")
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", default="spiral", choices=sorted(fr.targets.LANDMARKS))
    ap.add_argument("--depth", default="1e27", help="final magnification")
    ap.add_argument("--frames", type=int, default=270)
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--fps", type=float, default=25)
    ap.add_argument("--colors", type=int, default=64, help="GIF colours per frame")
    ap.add_argument("--palette", default="fire", choices=sorted(fr.PALETTES))
    ap.add_argument("--out", default=str(ROOT / "docs" / "deep_zoom.gif"))
    args = ap.parse_args()

    end_radius = 2 / fr.to_decimal(args.depth)
    digits = int(-end_radius.log10()) + 40
    re, im = fr.targets.landmark(args.target, digits=digits)
    path = fr.ZoomPath("mandelbrot", re, im, end_radius, start_radius=2,
                       frames=args.frames, width=args.size, height=args.size)

    t0 = time.time()
    frames = []
    for k, (view, img) in enumerate(zip(path.views(),
                                        fr.zoom_frames(path, palette=args.palette, period=64))):
        frames.append(label(img, view))
        if k % 20 == 0:
            print(f"frame {k:4d}/{args.frames}  zoom {view.magnification}  "
                  f"{view.bits} bits  {time.time() - t0:6.1f} s")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fr.save_gif(frames, args.out, fps=args.fps, colors=args.colors)
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
