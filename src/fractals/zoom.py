"""Zoom animations: a path into the plane, its frames, and a GIF of them."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterator

import numpy as np

from .color import colorize
from .families import Fractal, get
from .precision import to_decimal
from .render import auto_iterations, escape_time
from .view import Viewport


@dataclass(frozen=True)
class ZoomPath:
    """An exponential zoom from ``start_radius`` to ``end_radius`` on a target.

    The magnification grows by the same factor every frame, which is what
    makes a zoom look like steady flight. ``target_re``/``target_im`` should be
    given as strings with at least as many digits as the final zoom needs.
    """

    fractal: Fractal
    target_re: Decimal
    target_im: Decimal
    end_radius: Decimal
    start_radius: Decimal = Decimal(2)
    frames: int = 120
    width: int = 320
    height: int = 320

    def __post_init__(self):
        object.__setattr__(self, "fractal", get(self.fractal))
        for name in ("target_re", "target_im", "end_radius", "start_radius"):
            object.__setattr__(self, name, to_decimal(getattr(self, name)))

    @property
    def factor_per_frame(self) -> float:
        decades = float(self.start_radius.log10() - self.end_radius.log10())
        return 10 ** (decades / max(self.frames - 1, 1))

    def views(self) -> Iterator[Viewport]:
        view = Viewport(self.target_re, self.target_im, self.start_radius,
                        self.width, self.height, self.fractal.imag_down)
        k = self.factor_per_frame
        for _ in range(self.frames):
            yield view
            view = view.zoom(k)


def zoom_frames(
    path: ZoomPath,
    max_iter=None,
    palette: str = "fire",
    period: float = 48.0,
    progress=None,
) -> Iterator[np.ndarray]:
    """RGB frames along ``path``.

    ``max_iter`` may be an int, a callable ``view → int``, or ``None`` for
    :func:`~fractals.render.auto_iterations`. Colour cycles every ``period``
    iterations, so bands keep their colour from one frame to the next.
    """
    for k, view in enumerate(path.views()):
        if max_iter is None:
            n = auto_iterations(view)
        elif callable(max_iter):
            n = int(max_iter(view))
        else:
            n = int(max_iter)
        mu = escape_time(path.fractal, view, n)
        if progress is not None:
            progress(k, view, mu)
        yield colorize(mu, palette, period=period)


def save_gif(frames, filename, fps: float = 20.0, colors: int = 256,
             loop: bool = True, pingpong: bool = False):
    """Write RGB frames to an animated GIF (needs Pillow).

    ``colors`` per frame (≤ 256): fewer colours make much smaller files, since
    deep-zoom frames are mostly fine filaments that compress poorly.
    """
    from PIL import Image

    frames = [np.asarray(f) for f in frames]
    if pingpong:
        frames = frames + frames[-2:0:-1]
    images = [Image.fromarray(f).quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                                          dither=Image.Dither.NONE) for f in frames]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000 / fps)),
        loop=0 if loop else 1,
        optimize=True,
        disposal=1,
    )
    return filename


__all__ = ["ZoomPath", "zoom_frames", "save_gif"]
