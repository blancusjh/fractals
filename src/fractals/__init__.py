"""Escape-time fractals at any zoom depth.

The library is ordered by what each thing *is*:

========================  ====================================================
``fractals.precision``    arbitrary-precision numbers and their float64 bridge
``fractals.families``     the catalogue — what each fractal iterates
``fractals.view``         the camera — a window onto the plane, at any depth
``fractals.kernel``       perturbation iteration (Numba), glitch-free
``fractals.render``       fractal + view → escape data → image
``fractals.color``        escape data → RGB
``fractals.targets``      exact zoom targets (Misiurewicz points, nuclei)
``fractals.zoom``         zoom paths, frame sequences and GIFs
``fractals.viewer``       interactive deep-zoom window (VisPy, optional)
========================  ====================================================

Typical use::

    import fractals as fr

    re, im = fr.targets.landmark("spiral", digits=80)   # exact to 80 digits
    view = fr.Viewport(re, im, radius="1e-60", width=800, height=600)
    img = fr.render("mandelbrot", view)                  # (600, 800, 3) uint8
"""

from . import targets
from .color import PALETTES, colorize
from .families import CATALOG, BurningShip, Fractal, Julia, Mandelbrot, Tricorn, get
from .precision import FloatExp, to_decimal
from .render import auto_iterations, escape_time, reference_orbit, render
from .view import Viewport
from .zoom import ZoomPath, save_gif, zoom_frames

__all__ = [
    "CATALOG", "PALETTES", "BurningShip", "FloatExp", "Fractal", "Julia",
    "Mandelbrot", "Tricorn", "Viewport", "ZoomPath", "auto_iterations",
    "colorize", "escape_time", "get", "reference_orbit", "render", "save_gif",
    "targets", "to_decimal", "zoom_frames",
]

__version__ = "0.2.0"
