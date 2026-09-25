"""The GLSL kernel against the CPU kernel (skipped without an OpenGL context)."""

import numpy as np
import pytest

import fractals as fr


@pytest.fixture(scope="module")
def gpu():
    try:
        from fractals.gpu import GpuRenderer
        return GpuRenderer()
    except Exception as exc:  # no VisPy backend / no display
        pytest.skip(f"no OpenGL context: {exc}")


CASES = [(f, None) for f in ["mandelbrot", "julia", "tricorn", "burning_ship"]] + [
    ("mandelbrot", "1e20"), ("mandelbrot", "1e320")]


@pytest.mark.parametrize("name,depth", CASES)
def test_gpu_matches_cpu(gpu, name, depth):
    f = fr.get(name)
    if depth is None:
        view = fr.Viewport.home(f, 96, 72)
    else:
        re, im = fr.targets.landmark("spiral", digits=400)
        view = fr.Viewport(re, im, 2, 96, 72).zoom(depth)
    n = fr.auto_iterations(view)
    cpu = fr.escape_time(f, view, n)
    gpu_mu = gpu.escape_time(f, view, n)
    assert ((cpu < 0) == (gpu_mu < 0)).mean() > 0.99
    both = (cpu >= 0) & (gpu_mu >= 0)
    # float32 on the GPU: most pixels agree closely, chaotic filaments do not.
    assert (np.abs(cpu - gpu_mu)[both] < 0.05).mean() > 0.9
