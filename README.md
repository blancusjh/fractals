# Fractals

Interactive implementations of the Mandelbrot set, intended for comparing three
rendering approaches: vectorised NumPy, parallel Numba, and an interactive GLSL
viewer powered by VisPy Gloo.

<img width="1824" height="1880" alt="image" src="https://github.com/user-attachments/assets/cd2cd3db-bbfb-4d29-a432-e159f65bb9c5" />



## Quick start

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python mandelbrot/gloo.py
```

## Interactive GPU viewer

The Gloo viewer is the primary application:

```bash
python mandelbrot/gloo.py
```

Controls:

- Mouse wheel: zoom toward the pointer.
- Left mouse drag: pan the complex plane.
- `Esc`: close the window (provided by VisPy's interactive key handling).

The current renderer uses single-precision GPU coordinates. It is responsive for
normal exploration, but extremely deep zooms will eventually lose numerical
precision. Arbitrary-precision and perturbation rendering are planned future
work.

## Alternative implementations

The `mandelbrot/` directory contains independent renderers for comparison:

```bash
# Vectorised NumPy implementation
python mandelbrot/numpy.py

# Numba-accelerated implementation
python mandelbrot/numba.py
```

The Numba example redraws after mouse interactions. Its first render includes
Numba's just-in-time compilation cost.

## Project layout

```text
.
├── mandelbrot/
│   ├── gloo.py               # Interactive GPU implementation
│   ├── numba.py              # Parallel CPU implementation
│   └── numpy.py              # Vectorised CPU implementation
├── requirements.txt          # Runtime dependencies
└── README.md
```

## Limitations

The most strong limitation is that arbitrary precision arithmetic implementation was not implemented,
so the fractal is limited by the float representation used, for infinite zoom it is required solve this gap.
