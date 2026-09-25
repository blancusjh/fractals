"""Where the camera is: a window onto the complex plane, at any depth."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, localcontext

from .families import Fractal, default_view_args
from .precision import (
    FloatExp,
    bits_for_spacing,
    decimal_context,
    format_decimal,
    to_decimal,
)


@dataclass(frozen=True)
class Viewport:
    """A ``width × height`` pixel window centred on ``center``.

    ``radius`` is the half-extent of the *shorter* side, in complex-plane
    units. Every coordinate is a :class:`~decimal.Decimal`, so the window
    can be as small as you like; the arithmetic precision follows the zoom.

    Pixel ``(i, j)`` is column ``i`` from the left and row ``j`` from the top;
    its centre sits at ``center + ((i + ½ − W/2)·s, ±(H/2 − j − ½)·s)`` with
    ``s`` the pixel :attr:`spacing` and the sign set by ``imag_down``.
    """

    re: Decimal
    im: Decimal
    radius: Decimal
    width: int = 640
    height: int = 480
    imag_down: bool = False

    def __post_init__(self):
        object.__setattr__(self, "re", to_decimal(self.re))
        object.__setattr__(self, "im", to_decimal(self.im))
        object.__setattr__(self, "radius", to_decimal(self.radius))
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.width < 1 or self.height < 1:
            raise ValueError("width and height must be positive")

    # --- construction --------------------------------------------------------

    @classmethod
    def home(cls, fractal: Fractal, width: int = 640, height: int = 480) -> "Viewport":
        """The fractal's natural first view."""
        re, im, radius = default_view_args(fractal)
        return cls(re, im, radius, width, height, fractal.imag_down)

    # --- derived quantities --------------------------------------------------

    @property
    def spacing(self) -> Decimal:
        """Distance between neighbouring pixel centres."""
        with localcontext(decimal_context(bits_for_spacing(self.radius) + 32)):
            return 2 * self.radius / min(self.width, self.height)

    @property
    def bits(self) -> int:
        """Binary precision the centre and the reference orbit need."""
        return bits_for_spacing(self.spacing)

    @property
    def magnification(self) -> FloatExp:
        """How much deeper than a radius-2 view this is (``2 / radius``)."""
        with localcontext(decimal_context(128)):
            return FloatExp.from_decimal(2 / self.radius)

    def offset(self, i: float, j: float) -> tuple[Decimal, Decimal]:
        """Complex-plane offset of (sub)pixel ``(i, j)`` from the centre."""
        s = self.spacing
        with localcontext(decimal_context(self.bits)):
            dx = (to_decimal(i) + Decimal("0.5") - Decimal(self.width) / 2) * s
            dy = (Decimal(self.height) / 2 - to_decimal(j) - Decimal("0.5")) * s
        return dx, (-dy if self.imag_down else dy)

    def point(self, i: float, j: float) -> tuple[Decimal, Decimal]:
        """Complex-plane coordinates of (sub)pixel ``(i, j)``."""
        dx, dy = self.offset(i, j)
        with localcontext(decimal_context(self.bits)):
            return self.re + dx, self.im + dy

    # --- navigation ----------------------------------------------------------

    def zoom(self, factor: float | str | Decimal, about: tuple[float, float] | None = None) -> "Viewport":
        """Magnify by ``factor`` (> 1 zooms in), keeping pixel ``about`` fixed.

        With ``about=None`` the centre stays put.
        """
        factor = to_decimal(factor)
        if not factor.is_finite() or factor <= 0:
            raise ValueError("zoom factor must be positive and finite — beyond "
                             "float range, pass a string such as '1e400'")
        with localcontext(decimal_context(self.bits + 64)):
            radius = self.radius / factor
        new = replace(self, radius=radius)
        if about is None:
            return new
        # The point under `about` must be the same before and after.
        pre = self.point(*about)
        dx, dy = new.offset(*about)
        with localcontext(decimal_context(new.bits)):
            return replace(new, re=pre[0] - dx, im=pre[1] - dy)

    def pan(self, di: float, dj: float) -> "Viewport":
        """Move the content by ``(di, dj)`` pixels (as when dragged by the mouse)."""
        s = self.spacing
        with localcontext(decimal_context(self.bits)):
            dx = -to_decimal(di) * s
            dy = to_decimal(dj) * s
            if self.imag_down:
                dy = -dy
            return replace(self, re=self.re + dx, im=self.im + dy)

    def centered_on(self, re, im) -> "Viewport":
        return replace(self, re=to_decimal(re), im=to_decimal(im))

    def resized(self, width: int, height: int) -> "Viewport":
        return replace(self, width=int(width), height=int(height))

    # --- presentation --------------------------------------------------------

    def describe(self) -> str:
        digits = max(6, int(self.bits * 0.30103) - 16)
        return (
            f"re = {format_decimal(self.re, digits)}\n"
            f"im = {format_decimal(self.im, digits)}\n"
            f"radius = {format_decimal(self.radius, 6)}  "
            f"(zoom {self.magnification}, {self.bits} bits)"
        )
