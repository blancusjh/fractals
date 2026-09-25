"""Interactive deep-zoom viewer (VisPy).

Controls
--------
mouse wheel      zoom toward the pointer
left drag        pan
1 2 3 4          Mandelbrot, Julia, Tricorn, Burning Ship
+ / -            more / fewer iterations (×1.5)
P                next palette
S                save a PNG of the current view (full resolution)
C                print the current coordinates (paste them back with --re/--im/--radius)
R                back to the fractal's home view
Esc              quit

The window shows the last finished render; while you zoom or drag, that image
is scaled and moved as a preview, and a new render starts once you pause.
Every render is exact at any depth: coordinates are ``Decimal`` and the pixels
come from perturbation around an arbitrary-precision reference orbit.

The navigation logic lives in :class:`Navigator`, which has no GUI dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np

from .color import PALETTES
from .families import CATALOG, Fractal, get
from .precision import decimal_context
from .render import auto_iterations, render
from .view import Viewport


@dataclass
class Navigator:
    """The viewer's state and what each input does to it — no GUI involved."""

    fractal: Fractal
    view: Viewport
    iter_scale: float = 1.0
    palette: str = "fire"
    #: The view the currently displayed image was rendered for.
    shown: Viewport | None = None

    @classmethod
    def start(cls, fractal="mandelbrot", width=800, height=800, **kw) -> "Navigator":
        f = get(fractal)
        return cls(f, Viewport.home(f, width, height), **kw)

    # --- inputs --------------------------------------------------------------

    def wheel(self, pos: tuple[float, float], clicks: float, step: float = 1.25):
        """Zoom about window position ``pos`` (pixels from the top-left corner)."""
        # Pixel i spans [i, i+1); its centre, i + ½, is the Viewport's pixel i.
        self.view = self.view.zoom(step ** clicks, about=(pos[0] - 0.5, pos[1] - 0.5))

    def drag(self, di: float, dj: float):
        self.view = self.view.pan(di, dj)

    def resize(self, width: int, height: int):
        self.view = self.view.resized(max(1, width), max(1, height))

    def select(self, name: str):
        self.fractal = get(name)
        self.home()

    def home(self):
        self.view = Viewport.home(self.fractal, self.view.width, self.view.height)

    def more_iterations(self, factor: float = 1.5):
        self.iter_scale *= factor

    def next_palette(self):
        names = list(PALETTES)
        self.palette = names[(names.index(self.palette) + 1) % len(names)]

    # --- outputs -------------------------------------------------------------

    @property
    def max_iter(self) -> int:
        return max(16, int(auto_iterations(self.view) * self.iter_scale))

    def render(self) -> np.ndarray:
        img = render(self.fractal, self.view, self.max_iter, self.palette)
        self.shown = self.view
        return img

    def preview_transform(self) -> tuple[float, float, float]:
        """``(scale, dx, dy)`` placing the shown image in the current view.

        In the current view's pixels: the shown image is magnified by ``scale``
        about the window centre, then moved by ``(dx, dy)`` (x right, y down).
        """
        if self.shown is None:
            return 1.0, 0.0, 0.0
        old, new = self.shown, self.view
        with localcontext(decimal_context(max(old.bits, new.bits) + 16)):
            scale = old.radius / new.radius
            sx = (old.re - new.re) / new.spacing
            sy = (old.im - new.im) / new.spacing
        dy = float(sy) if new.imag_down else -float(sy)
        return float(scale), float(sx), dy

    def describe(self) -> str:
        head = f"{self.fractal.name}"
        if self.fractal.dynamical_plane:
            head += f"  c = {self.fractal.c[0]} {self.fractal.c[1]}i"
        return f"{head}\n{self.view.describe()}\nmax_iter = {self.max_iter}"


VERTEX = """
attribute vec2 a_position;
attribute vec2 a_texcoord;
uniform vec2 u_scale;
uniform vec2 u_offset;
varying vec2 v_texcoord;
void main() {
    gl_Position = vec4(a_position * u_scale + u_offset, 0.0, 1.0);
    v_texcoord = a_texcoord;
}
"""

FRAGMENT = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;
void main() {
    gl_FragColor = texture2D(u_texture, v_texcoord);
}
"""


def build(fractal="mandelbrot", re=None, im=None, radius=None, size=(800, 800),
          palette="fire", idle_delay=0.15):
    """Create the viewer window; returns ``(canvas, navigator)`` without blocking."""
    from vispy import app, gloo

    canvas = app.Canvas(keys="interactive", size=size, title="fractals")
    nav = Navigator.start(fractal, *canvas.physical_size, palette=palette)
    if re is not None and im is not None:
        nav.view = nav.view.centered_on(re, im)
    if radius is not None:
        nav.view = Viewport(nav.view.re, nav.view.im, Decimal(str(radius)),
                            nav.view.width, nav.view.height, nav.view.imag_down)

    program = gloo.Program(VERTEX, FRAGMENT)
    program["a_position"] = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], np.float32)
    program["a_texcoord"] = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], np.float32)
    texture = gloo.Texture2D(np.zeros((1, 1, 3), np.uint8), interpolation="linear")
    program["u_texture"] = texture
    state = {"drag": None, "dirty": True}
    keys = {str(i + 1): name for i, name in enumerate(CATALOG)}

    def to_pixels(pos):
        ratio = np.asarray(canvas.physical_size, float) / np.asarray(canvas.size, float)
        return float(pos[0] * ratio[0]), float(pos[1] * ratio[1])

    def rerender():
        t = time.perf_counter()
        img = nav.render()
        texture.set_data(img)
        state["dirty"] = False
        dt = time.perf_counter() - t
        canvas.title = (f"fractals — {nav.fractal.name}  zoom {nav.view.magnification}  "
                        f"{nav.view.bits} bits  {nav.max_iter} it  {dt:.2f} s")
        canvas.update()

    def changed():
        state["dirty"] = True
        timer.stop()
        timer.start()
        canvas.update()

    timer = app.Timer(idle_delay, connect=lambda ev: rerender(), iterations=1, start=False)

    @canvas.connect
    def on_draw(event):
        gloo.clear("black")
        scale, dx, dy = nav.preview_transform()
        w, h = nav.view.width, nav.view.height
        program["u_scale"] = (scale, scale)
        program["u_offset"] = (2 * dx / w, -2 * dy / h)
        program.draw("triangle_strip")

    @canvas.connect
    def on_resize(event):
        gloo.set_viewport(0, 0, *event.physical_size)
        nav.resize(*event.physical_size)
        changed()

    @canvas.connect
    def on_mouse_wheel(event):
        nav.wheel(to_pixels(event.pos), event.delta[1])
        changed()

    @canvas.connect
    def on_mouse_press(event):
        if event.button == 1:
            state["drag"] = to_pixels(event.pos)

    @canvas.connect
    def on_mouse_release(event):
        state["drag"] = None

    @canvas.connect
    def on_mouse_move(event):
        if state["drag"] is not None and event.is_dragging:
            p = to_pixels(event.pos)
            nav.drag(p[0] - state["drag"][0], p[1] - state["drag"][1])
            state["drag"] = p
            changed()

    @canvas.connect
    def on_key_press(event):
        k = event.text.lower() if event.text else ""
        if k in keys:
            nav.select(keys[k])
        elif k in ("+", "="):
            nav.more_iterations(1.5)
        elif k == "-":
            nav.more_iterations(1 / 1.5)
        elif k == "p":
            nav.next_palette()
        elif k == "r":
            nav.home()
        elif k == "c":
            print(nav.describe(), flush=True)
            return
        elif k == "s":
            from PIL import Image
            name = f"{nav.fractal.name}_{int(time.time())}.png"
            Image.fromarray(nav.render()).save(name)
            print(f"saved {name}", flush=True)
            return
        else:
            return
        changed()

    canvas.show()
    rerender()
    return canvas, nav


def run(*args, **kwargs):
    """Open the viewer window (see :func:`build`). Blocks until it is closed."""
    from vispy import app

    canvas, nav = build(*args, **kwargs)
    app.run()
    return nav


if __name__ == "__main__":
    run()
