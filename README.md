# Fractals

Escape-time fractals at **any zoom depth**. The view centre and a single
reference orbit are computed in arbitrary precision; every pixel is then
iterated in float64 as a *perturbation* of that orbit, which is exact enough
at 1e30, 1e300 or 1e600 as it is at 1.

![An endless zoom into the Mandelbrot set: down Seahorse Valley past spirals and embedded Julia sets to a minibrot at 10^15×, which becomes the whole set again and the zoom starts over.](docs/infinite_zoom.gif)

*An endless zoom. The dive runs down Seahorse Valley, past spirals, double
spirals, a period-78 minibrot and the 998-fold embedded Julia sets, to a
**period-998 minibrot** at 10¹⁵×. That minibrot is an exact copy of the whole set,
so the last frame *is* the first one and the GIF loops without a seam. Nothing
is hand-tuned: the minibrot's nucleus is found by the ball method and Newton's
method, and its complex size gives the copy's scale and rotation, which the
camera undoes on the way down. Rendered by
[`examples/deep_zoom_gif.py`](examples/deep_zoom_gif.py) in ~3 minutes on 4
CPU cores.*

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[viewer,gif]"      # or: pip install -r requirements.txt && pip install -e .

python -m fractals view                        # interactive deep-zoom window
python -m fractals view --gpu                  # the same, rendered by the GPU
```

Viewer controls:

| Input | Action |
| --- | --- |
| mouse wheel | zoom toward the pointer (no depth limit) |
| left drag | pan |
| `1` `2` `3` `4` | Mandelbrot, Julia, Tricorn, Burning Ship |
| `+` / `-` | more / fewer iterations |
| `P` | next palette |
| `S` | save a PNG of the current view |
| `C` | print the exact coordinates of the view |
| `R` | back to the home view |
| `Esc` | quit |

The window never freezes. While you zoom or drag, the last image is scaled and
moved as a preview; once you pause, a quarter-resolution pass appears almost
immediately, followed by the full-resolution one. CPU renders run in a
background thread (the Numba kernel releases the GIL); GPU renders run between
frames on the window's own OpenGL context. The first CPU render includes
Numba's compilation time (cached after that).

### From the command line

```bash
# One image, 10^60× deep: coordinates are strings, so every digit is kept
python -m fractals render mandelbrot --radius 1e-60 --supersample 2 -o deep.png \
    --re -0.77568376800905379746948350393474104572824650461580039479495200 \
    --im  0.13646736829469012473327440961784876519777211159246769640780020

# A zoom GIF onto an exact landmark (computed to the digits the depth needs)
python -m fractals zoom --target spiral --depth 1e100 --frames 300 -o zoom.gif

# Any command takes --gpu to render with the GLSL kernel instead of the CPU
python -m fractals render tricorn --gpu --supersample 2 -o tricorn.png

# The other families
python -m fractals render burning_ship --re -1.7621 --im -0.029 --radius 0.05 -o ship.png
python -m fractals render julia --c -0.123 0.745 -o rabbit.png
```

### From Python

```python
import fractals as fr

re, im = fr.targets.landmark("spiral", digits=120)       # exact, 120 digits
view = fr.Viewport(re, im, radius="1e-100", width=800, height=600)

img = fr.render("mandelbrot", view, supersample=2)        # (600, 800, 3) uint8
img = fr.render("mandelbrot", view, device="gpu")         # same, on the GPU
mu  = fr.escape_time("mandelbrot", view, max_iter=20000)  # smooth iteration counts

view = view.zoom("1e50", about=(200, 150))                # keep pixel (200,150) fixed
view = view.pan(-40, 12)                                  # drag by pixels
print(view.describe())                                    # coordinates, zoom, bits
```

---

## The catalogue

![The four fractals of the catalogue: the Mandelbrot set, a Julia set, the Tricorn and the Burning Ship's armada.](docs/gallery.png)

| Name | Map | Pixel is | Notes |
| --- | --- | --- | --- |
| `mandelbrot` | `z → z² + c` | `c` | the reference; exact zoom targets in `fractals.targets` |
| `julia` | `z → z² + k` | `z₀` | `Julia(c=("-0.75", "0.11"))`, any `k` |
| `tricorn` | `z → conj(z)² + c` | `c` | the Mandelbar set; anti-holomorphic, three-fold symmetric |
| `burning_ship` | `z → (\|Re z\| + i\|Im z\|)² + c` | `c` | drawn with the imaginary axis pointing down, by convention |

All four zoom without limit: each one has its exact map (for the reference
orbit) and its perturbation formula (for the pixels). For the Burning Ship,
that means differencing absolute values without cancellation.

---

## How infinite zoom works

A float64 has 53 significant bits. Once neighbouring pixels differ by less than
that, around 10¹⁴× for a whole-set view, they collapse into the same number and
the image turns to blocks. A float32 GLSL shader gets there by about 10⁶×:

![The same two views in plain float32/float64 and in arbitrary precision: the plain renders dissolve into blocks, the perturbation render stays sharp.](docs/precision.png)

The library avoids the problem instead of carrying more digits per pixel:

1. **Positions are exact.** A `Viewport` keeps its centre and radius as
   `decimal.Decimal`, and every zoom or pan is done at the precision that
   depth needs (`view.bits`: about 3.3 bits per decade, plus guard bits).
2. **One orbit, in arbitrary precision.** The centre's orbit `Zₙ` is iterated
   once in fixed-point Python integers ([`render.py`](src/fractals/render.py)).
3. **Every pixel as a difference.** Each pixel iterates only `δₙ = zₙ − Zₙ`,
   e.g. `δₙ₊₁ = 2Zₙδₙ + δₙ² + δc` for `z² + c`. `δ` only needs to be accurate
   *relative to itself*, so float64 suffices at any depth. The Numba kernel is
   [`kernel.py`](src/fractals/kernel.py).
4. **Glitch-free by rebasing.** When a pixel comes closer to 0 than to the
   reference (`|zₙ| < |δₙ|`), or the reference orbit ends, the pixel restarts
   from the beginning of the reference with `δ ← zₙ`. That is Zhuoran's
   method, and it avoids the need for secondary references.
5. **Past float64's exponent range.** Beyond ~1e300 a pixel offset underflows
   float64 itself, so while `δ` is that small it is carried as a mantissa plus
   an integer exponent and renormalised as it grows.
6. **Skipping the shared iterations.** At depth, every pixel spends thousands
   of iterations doing nearly the same thing. *Series approximation* expands
   `δₙ` as a cubic in the pixel offset, with coefficients that depend only on
   the reference, and jumps every pixel straight to the last iteration where
   the cubic is still exact to 2⁻⁴² (checked with the 4th-order term). At
   10³²⁰× that skips 18,962 of ~19,200 iterations: a 240×240 frame takes
   0.15 s instead of 6.3 s. (Mandelbrot and Julia; the Tricorn and Burning Ship
   are not holomorphic and have no such series.)
7. **Or on the GPU.** [`gpu.py`](src/fractals/gpu.py) is the same algorithm
   (reference texture, perturbation, rebasing, extended range, series start)
   as a GLSL fragment shader, through VisPy on any OpenGL 2.1 GPU. It computes
   in float32, so escape counts can differ slightly from the float64 CPU
   kernel; the images are visually identical, and the tests check them against
   each other.

The tests check steps 2–5 against **direct iteration in exact big-integer
arithmetic**, pixel by pixel, for all four fractals at 10²⁰× and 10⁶⁰×, and
for the Mandelbrot set at 10⁴⁰⁰×. Here is one point, far past the float64
exponent limit:

![One Misiurewicz point at magnifications 1, 1e3, 1e8, 1e20, 1e60, 1e150, 1e320 and 1e600: the same spiral structure persists at every depth.](docs/depths.png)

*The same spiral at 10⁰ … 10⁶⁰⁰×. Around a Misiurewicz point the set is
asymptotically self-similar, so the zoom never runs out of detail. The last
panel needs 2063 bits for its centre.*

### Exact zoom targets

A deep zoom is only as good as its target: a centre that is off the boundary by
1e-20 looks fine down to 1e20× and then turns into a flat colour.
[`fractals.targets`](src/fractals/targets.py) solves for points with a
closed definition, by Newton's method in fixed point, to any number of digits:

```python
from fractals.targets import misiurewicz, nucleus, preperiod_period, landmark, minibrot_near

re, im = misiurewicz("-0.77568377", "0.13646737", preperiod=24, period=1, digits=500)
preperiod_period(re, im)                  # (24, 1)
re, im = nucleus("-1.75", "0", period=3)  # centre of the period-3 minibrot
re, im = landmark("spiral", digits=1000)  # named places: spiral, antenna, dendrite

# The minibrot nearest a point: nucleus (exact), period, and complex size
re, im, period, size = minibrot_near("-0.743643887037158704752191506114774",
                                     "0.131825904205311970493132056385139", "1e-10")
# → period 998, |size| 6.3e-16: near it the set is  c + size·M  (scale and rotation)
```

---

## Library layout

The package is ordered by what each thing *is*, from numbers up to pixels:

| Module | Holds | Key names |
| --- | --- | --- |
| [`precision.py`](src/fractals/precision.py) | arbitrary-precision numbers and the bridge to float64 | `to_decimal`, `to_fixed`, `fixed_to_float`, `FloatExp` |
| [`families.py`](src/fractals/families.py) | the catalogue: what each fractal iterates | `Mandelbrot`, `Julia`, `Tricorn`, `BurningShip`, `CATALOG`, `get` |
| [`view.py`](src/fractals/view.py) | the camera: a window onto the plane, at any depth | `Viewport` (`.zoom`, `.pan`, `.point`, `.bits`) |
| [`kernel.py`](src/fractals/kernel.py) | perturbation iteration (Numba, parallel, GIL-free) | `perturbation_kernel` |
| [`gpu.py`](src/fractals/gpu.py) | the same kernel as a GLSL shader | `GpuRenderer` |
| [`render.py`](src/fractals/render.py) | fractal + view → escape data → image | `reference_orbit`, `series_start`, `escape_time`, `render` |
| [`color.py`](src/fractals/color.py) | escape data → RGB | `colorize`, `PALETTES` |
| [`targets.py`](src/fractals/targets.py) | exact zoom targets | `misiurewicz`, `nucleus`, `minibrot_near`, `landmark` |
| [`zoom.py`](src/fractals/zoom.py) | zoom paths, frame sequences, GIFs | `ZoomPath`, `zoom_frames`, `save_gif` |
| [`viewer.py`](src/fractals/viewer.py) | interactive window (VisPy), non-blocking; logic in a GUI-free `Navigator` | `run`, `build`, `Navigator` |
| [`__main__.py`](src/fractals/__main__.py) | the command line | `view`, `render`, `zoom` |

To add a fractal, subclass `Fractal` in `families.py` with its exact map
(`step_fixed`), and if it is not one of the three existing kernels, add its
perturbation step to `kernel._step`. Nothing else knows which fractal it is
drawing.

```text
.
├── src/fractals/             the library (above)
├── examples/
│   ├── deep_zoom_gif.py      the endless zoom at the top of this page
│   ├── gallery.py            gallery.png, depths.png, precision.png
│   └── original/             the original renderers, kept as they were
│       ├── gloo.py           GLSL viewer (float32), unchanged
│       ├── numba_mandelbrot.py
│       └── numpy_mandelbrot.py
├── tests/                    perturbation vs exact big-int iteration, GPU vs CPU, views, targets
├── docs/                     figures
├── pyproject.toml
└── requirements.txt
```

## The original renderers

The three original renderers live, unchanged in substance, in
[`examples/original/`](examples/original). The GLSL viewer is byte-for-byte the
original: the fastest way to fly around the set interactively, until float32
runs out at about 10⁶×. Its cosine colormap is also available to the library
as the `glsl` palette (`palette="glsl"`, or `P` in the viewer).

```bash
python examples/original/gloo.py              # GLSL, float32: instant, until ~1e6×
python examples/original/numba_mandelbrot.py  # Numba, float64: until ~1e14×
python examples/original/numpy_mandelbrot.py  # NumPy, float64, static image
```

(They were renamed from `mandelbrot/numpy.py` and `mandelbrot/numba.py`: run as
scripts, those names shadowed the `numpy` and `numba` packages they import.)

## Tests

```bash
pip install -e ".[dev]"
python -m pytest            # 28 tests (+6 GPU tests, skipped without an OpenGL context)
xvfb-run python -m pytest   # headless machines: GPU tests on Mesa's software OpenGL
```

The render tests compare the perturbation kernel with independent direct
iteration: float64 for shallow views, and exact fixed-point integers pixel by
pixel for deep ones (10²⁰×, 10⁶⁰× and 10⁴⁰⁰×). The series approximation and
the GPU kernel are each checked against the plain CPU kernel. Pixels whose orbit is
chaotic (the answer changes if the pixel moves by 10⁻⁹ of its width) are
excluded, since no finite precision agrees there.

## Limitations

- **Minibrots of very high period are expensive.** Near a period-p minibrot one
  step of the set's own dynamics costs p iterations, and interior pixels run to
  the iteration limit. The endless zoom stops at a period-998 minibrot for
  that reason; the period-8007 one further down the same path needs ~1.6 M
  iterations per pixel to render crisply. Periodicity checking would help with
  interior pixels, but it needs derivative tracking to be reliable at depth,
  where exterior orbits shadow cycles for thousands of iterations.
- **Series approximation covers `z² + c` only** (Mandelbrot, Julia). The Tricorn
  and Burning Ship still iterate every step (bilinear approximation would
  cover them).
- **The GPU kernel is float32.** It keeps the zoom depth unlimited, but a
  pixel's escape count may differ from the float64 CPU result by a fraction of
  an iteration, more on chaotic filaments.
- **Chaotic regions.** Where orbits are chaotic (parts of the Burning Ship,
  the Mandelbrot antenna), individual pixels are not reproducible by *any*
  finite precision. The picture is right statistically, not pixel by pixel.
