"""The catalogue: what each fractal iterates.

A fractal here is a quadratic escape-time map ``z → f(z) + c`` on the complex
plane. Each family states its map three times, once per arithmetic:

* ``step_fixed`` — exact, on fixed-point integers, for the reference orbit;
* a numbered ``kind`` — the same map as a perturbation ``δ → f(Z+δ) − f(Z)``,
  implemented in :mod:`fractals.kernel`;
* ``start`` — which of ``z₀`` and ``c`` the pixel sets (parameter plane for
  Mandelbrot-like sets, dynamical plane for Julia sets).

Adding a family means adding those three things; nothing else in the library
knows which fractal it is drawing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .precision import to_decimal, to_fixed

# Kernel identifiers — the perturbation formula for each map lives in kernel.py.
KIND_QUADRATIC = 0   # z² + c
KIND_TRICORN = 1     # conj(z)² + c
KIND_BURNING_SHIP = 2  # (|Re z| + i|Im z|)² + c


@dataclass(frozen=True)
class Fractal:
    """Base class: an escape-time map and its natural first view."""

    name: str = field(init=False, default="fractal")
    kind: int = field(init=False, default=KIND_QUADRATIC)
    #: True when the pixel is the starting point ``z₀`` (Julia sets);
    #: False when it is the parameter ``c`` (Mandelbrot-like sets).
    dynamical_plane: bool = field(init=False, default=False)
    #: Draw with the imaginary axis pointing down (the Burning Ship convention).
    imag_down: bool = field(init=False, default=False)
    default_center: tuple[str, str] = field(init=False, default=("0", "0"))
    default_radius: str = field(init=False, default="2")

    def start(self, px: int, py: int, bits: int) -> tuple[int, int, int, int]:
        """``(z₀x, z₀y, cx, cy)`` in fixed point, for the pixel ``(px, py)``."""
        return 0, 0, px, py

    def step_fixed(self, x: int, y: int, cx: int, cy: int, bits: int) -> tuple[int, int]:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Mandelbrot(Fractal):
    """``z → z² + c``, ``z₀ = 0``, pixel = ``c``."""

    name: str = field(init=False, default="mandelbrot")
    default_center: tuple[str, str] = field(init=False, default=("-0.6", "0"))
    default_radius: str = field(init=False, default="1.5")

    def step_fixed(self, x, y, cx, cy, bits):
        return ((x * x - y * y) >> bits) + cx, ((x * y) >> (bits - 1)) + cy


@dataclass(frozen=True)
class Julia(Fractal):
    """``z → z² + k`` for a fixed ``k``, pixel = ``z₀``."""

    c: tuple[str, str] = ("-0.75", "0.11")
    name: str = field(init=False, default="julia")
    dynamical_plane: bool = field(init=False, default=True)
    default_center: tuple[str, str] = field(init=False, default=("0", "0"))
    default_radius: str = field(init=False, default="1.2")

    def __post_init__(self):
        object.__setattr__(self, "c", (str(self.c[0]), str(self.c[1])))

    def start(self, px, py, bits):
        return px, py, to_fixed(to_decimal(self.c[0]), bits), to_fixed(to_decimal(self.c[1]), bits)

    def step_fixed(self, x, y, cx, cy, bits):
        return ((x * x - y * y) >> bits) + cx, ((x * y) >> (bits - 1)) + cy


@dataclass(frozen=True)
class Tricorn(Fractal):
    """``z → conj(z)² + c`` — the Mandelbar set, with three-fold symmetry."""

    name: str = field(init=False, default="tricorn")
    kind: int = field(init=False, default=KIND_TRICORN)
    default_center: tuple[str, str] = field(init=False, default=("-0.3", "0"))
    default_radius: str = field(init=False, default="1.7")

    def step_fixed(self, x, y, cx, cy, bits):
        return ((x * x - y * y) >> bits) + cx, -((x * y) >> (bits - 1)) + cy


@dataclass(frozen=True)
class BurningShip(Fractal):
    """``z → (|Re z| + i|Im z|)² + c`` — the Burning Ship."""

    name: str = field(init=False, default="burning_ship")
    kind: int = field(init=False, default=KIND_BURNING_SHIP)
    imag_down: bool = field(init=False, default=True)
    default_center: tuple[str, str] = field(init=False, default=("-0.45", "-0.5"))
    default_radius: str = field(init=False, default="1.7")

    def step_fixed(self, x, y, cx, cy, bits):
        return ((x * x - y * y) >> bits) + cx, (abs(x * y) >> (bits - 1)) + cy


#: Every family, by name.
CATALOG: dict[str, type[Fractal]] = {
    "mandelbrot": Mandelbrot,
    "julia": Julia,
    "tricorn": Tricorn,
    "burning_ship": BurningShip,
}


def get(name: str | Fractal) -> Fractal:
    """A fractal instance from its catalogue name (or pass one through)."""
    if isinstance(name, Fractal):
        return name
    try:
        return CATALOG[name.lower().replace("-", "_").replace(" ", "_")]()
    except KeyError:
        raise KeyError(f"unknown fractal {name!r}; choose from {sorted(CATALOG)}") from None


def default_view_args(fractal: Fractal) -> tuple[Decimal, Decimal, Decimal]:
    re, im = fractal.default_center
    return to_decimal(re), to_decimal(im), to_decimal(fractal.default_radius)
