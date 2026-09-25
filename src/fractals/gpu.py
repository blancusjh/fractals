"""GPU rendering: the perturbation kernel as a GLSL fragment shader (VisPy).

The same algorithm as :mod:`fractals.kernel` — arbitrary-precision reference
orbit on the CPU, perturbation per pixel on the GPU, rebasing, extended
exponent range, series-approximation start — so it zooms without limit too.
It runs on any OpenGL 2.1 GPU through VisPy (no CUDA needed).

The GPU works in float32: ``δ`` carries 24 bits instead of 53, so images are
visually identical but individual escape counts can differ slightly from the
float64 CPU kernel. Extended range starts at ``2**GPU_SCALED_LIMIT`` rather
than ``2**-900``, since float32 underflows at ~1e-38.

The result is read back as float escape counts, so colouring, supersampling
and GIF output are shared with the CPU path. (For a plain float32 shader that
colours on the GPU, see the original viewer, ``examples/original/gloo.py``.)
"""

from __future__ import annotations

import numpy as np

from .families import Fractal, get
from .precision import FloatExp
from .render import BAILOUT, reference_orbit, series_start
from .view import Viewport

#: |δ| below ~2**GPU_SCALED_LIMIT is carried as float32 mantissa × 2**exponent.
GPU_SCALED_LIMIT = -60

#: Reference-orbit texture width (the orbit is wrapped into rows).
REF_TEXTURE_WIDTH = 4096

#: Pixel-iterations per draw call; long draws are split into strips so the
#: GPU driver's watchdog does not kill them.
WORK_PER_DRAW = 3e8

VERTEX = """
attribute vec2 a_position;
void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
"""

FRAGMENT = """
uniform sampler2D u_ref;
uniform vec2 u_ref_shape;
uniform float u_nref;
uniform vec2 u_z0;
uniform int u_kind;
uniform float u_julia;
uniform float u_imag_down;
uniform vec2 u_size;
uniform float u_sp_m;
uniform float u_sp_e;
uniform float u_max_iter;
uniform float u_bail2;
uniform float u_limit;
uniform float u_sa_n0;
uniform float u_sa_e;
uniform float u_sa_r;
uniform vec2 u_sa_a;
uniform vec2 u_sa_b;
uniform vec2 u_sa_c;

vec2 ref(float n) {
    float row = floor(n / u_ref_shape.x);
    float col = n - row * u_ref_shape.x;
    return texture2D(u_ref, vec2((col + 0.5) / u_ref_shape.x,
                                 (row + 0.5) / u_ref_shape.y)).xy;
}

vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

float diffabs(float c, float d) {
    if (c >= 0.0) { return (c + d >= 0.0) ? d : -(2.0 * c + d); }
    return (c + d > 0.0) ? 2.0 * c + d : -d;
}


vec2 pstep(vec2 Z, vec2 d, vec2 dc, float f) {
    float lin_x = 2.0 * (Z.x * d.x - Z.y * d.y);
    float sq_x = f * (d.x * d.x - d.y * d.y);
    if (u_kind == 2) {
        float c = Z.x * Z.y;
        float dd = Z.x * d.y + d.x * Z.y + f * d.x * d.y;
        float ny;
        if (f == 0.0 || abs(c) > 1e30 * f) {
            ny = (c > 0.0) ? 2.0 * dd : ((c < 0.0) ? -2.0 * dd : 2.0 * abs(dd));
        } else {
            ny = 2.0 * diffabs(c / f, dd);
        }
        return vec2(lin_x + sq_x + dc.x, ny + dc.y);
    }
    float lin_y = 2.0 * (Z.x * d.y + Z.y * d.x);
    float sq_y = f * 2.0 * d.x * d.y;
    if (u_kind == 1) { return vec2(lin_x + sq_x + dc.x, -(lin_y + sq_y) + dc.y); }
    return vec2(lin_x + sq_x + dc.x, lin_y + sq_y + dc.y);
}

void main() {
    float j = u_size.y - gl_FragCoord.y - 0.5;
    float i = gl_FragCoord.x - 0.5;
    vec2 o = vec2((i + 0.5 - 0.5 * u_size.x) * u_sp_m,
                  (0.5 * u_size.y - j - 0.5) * u_sp_m);
    if (u_imag_down > 0.5) { o.y = -o.y; }
    bool julia = u_julia > 0.5;
    vec2 pc = julia ? vec2(0.0) : o;
    vec2 w = julia ? o : vec2(0.0);
    float s = u_sp_e;
    float n = 0.0;
    float it = 0.0;
    if (u_sa_n0 > 0.0) {
        vec2 u = o / u_sa_r;
        vec2 u2 = cmul(u, u);
        w = cmul(u_sa_a, u) + cmul(u_sa_b, u2) + cmul(u_sa_c, cmul(u2, u));
        s = u_sa_e;
        n = u_sa_n0;
        it = u_sa_n0;
    }
    float m = max(abs(w.x), abs(w.y));
    if (m > 0.0) { float e = floor(log2(m)) + 1.0; w *= exp2(-e); s += e; }
    vec2 dc = pc * exp2(u_sp_e);
    bool scaled = true;
    if (s > u_limit) { w *= exp2(s); scaled = false; }
    float mu = -1.0;
    for (int k = 0; k < 100000000; ++k) {
        if (it >= u_max_iter) { break; }
        vec2 Z = ref(n);
        if (scaled) {
            vec2 sc = julia ? vec2(0.0) : pc * exp2(u_sp_e - s);
            w = pstep(Z, w, sc, exp2(s));
            n += 1.0;
            it += 1.0;
            m = max(abs(w.x), abs(w.y));
            if (m > 1048576.0 || (m > 0.0 && m < 1.0 / 1048576.0)) {
                float e = floor(log2(m)) + 1.0;
                w *= exp2(-e);
                s += e;
            }
            if (s > u_limit || n >= u_nref) {
                w *= exp2(s);
                scaled = false;
            } else {
                vec2 Zn = ref(n);
                float q = dot(Zn, Zn);
                if (q > u_bail2) { mu = it + 1.0 - log2(0.5 * log(q)); break; }
                continue;
            }
        } else {
            w = pstep(Z, w, dc, 1.0);
            n += 1.0;
            it += 1.0;
        }
        vec2 z = ref(n) + w;
        float r2 = dot(z, z);
        if (r2 > u_bail2) { mu = it + 1.0 - log2(0.5 * log(r2)); break; }
        if (r2 < dot(w, w) || n >= u_nref) { w = z - u_z0; n = 0.0; }
    }
    gl_FragColor = vec4(mu, 0.0, 0.0, 1.0);
}
"""


class GpuRenderer:
    """Escape-time rendering on the GPU. Needs an OpenGL context (VisPy).

    Pass the ``canvas`` of an existing window to share its context; with none,
    a hidden canvas is created. Use it from the thread that owns the context.
    """

    def __init__(self, canvas=None):
        from vispy import app, gloo

        self._gloo = gloo
        self.canvas = canvas if canvas is not None else app.Canvas(show=False, size=(16, 16))
        self.canvas.set_current()
        self.program = gloo.Program(VERTEX, FRAGMENT)
        self.program["a_position"] = np.array(
            [[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)
        self._fbo = None
        self._shape = None

    def _target(self, width, height):
        gloo = self._gloo
        if self._shape != (height, width):
            tex = gloo.Texture2D(shape=(height, width, 4), format="rgba",
                                 internalformat="rgba32f", interpolation="nearest")
            self._fbo = gloo.FrameBuffer(color=tex)
            self._shape = (height, width)
        return self._fbo

    def escape_time(self, fractal: Fractal | str, view: Viewport, max_iter: int = 1000,
                    series: bool = True) -> np.ndarray:
        """Same contract as :func:`fractals.render.escape_time`."""
        gloo = self._gloo
        self.canvas.set_current()
        fractal = get(fractal)
        ref = reference_orbit(fractal, view, max_iter)
        sa = series_start(fractal, ref, view) if series else None
        sp = FloatExp.from_decimal(view.spacing)

        n = len(ref.x)
        cols = min(n, REF_TEXTURE_WIDTH)
        rows = -(-n // cols)
        data = np.zeros((rows * cols, 4), np.float32)
        data[:n, 0], data[:n, 1] = ref.x, ref.y
        p = self.program
        p["u_ref"] = gloo.Texture2D(data.reshape(rows, cols, 4), internalformat="rgba32f",
                                    interpolation="nearest")
        p["u_ref_shape"] = (cols, rows)
        p["u_nref"] = n - 1
        p["u_z0"] = (ref.x[0], ref.y[0])
        p["u_kind"] = int(fractal.kind)
        p["u_julia"] = float(fractal.dynamical_plane)
        p["u_imag_down"] = float(view.imag_down)
        p["u_size"] = (view.width, view.height)
        p["u_sp_m"] = sp.mantissa
        p["u_sp_e"] = float(sp.exponent)
        p["u_max_iter"] = float(max_iter)
        p["u_bail2"] = BAILOUT**2
        p["u_limit"] = float(GPU_SCALED_LIMIT)
        if sa is None:
            p["u_sa_n0"] = 0.0
        else:
            a, b, c = sa.coef[0:2], sa.coef[2:4], sa.coef[4:6]
            p["u_sa_n0"], p["u_sa_e"], p["u_sa_r"] = float(sa.n0), float(sa.e), sa.r
            p["u_sa_a"], p["u_sa_b"], p["u_sa_c"] = tuple(a), tuple(b), tuple(c)

        w, h = view.width, view.height
        fbo = self._target(w, h)
        rows_per_draw = max(1, int(WORK_PER_DRAW / max(1, w * max_iter)))
        with fbo:
            gloo.set_viewport(0, 0, w, h)
            gloo.set_state(scissor_test=True, blend=False, depth_test=False)
            for y in range(0, h, rows_per_draw):
                gloo.set_scissor(0, y, w, min(rows_per_draw, h - y))
                p.draw("triangle_strip")
                gloo.finish()
            gloo.set_state(scissor_test=False)
            out = gloo.read_pixels((0, 0, w, h), alpha=True, out_type=np.float32)
        return np.ascontiguousarray(out[..., 0], dtype=np.float64)


_default: GpuRenderer | None = None


def default_renderer() -> GpuRenderer:
    """A shared renderer on a hidden canvas (created on first use)."""
    global _default
    if _default is None:
        _default = GpuRenderer()
    return _default
