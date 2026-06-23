import numpy as np
from vispy import app, scene

# Image size
width, height = 800, 800

# Bounds in the complex plane
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5

# Max iterations
max_iter = 200

# Create a grid of complex numbers
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y

Z = np.zeros_like(C)
escape = np.zeros(C.shape, dtype=int)

# Mandelbrot iterations
for i in range(max_iter):
    mask = np.abs(Z) <= 2
    Z[mask] = Z[mask]**2 + C[mask]
    escape[mask & (np.abs(Z) > 2)] = i

# Normalize to 0-255
normalized = (escape / escape.max() * 255).astype(np.uint8)

# Create RGB image (grayscale)
image = np.dstack([normalized] * 3)

# --- Display with VisPy ---
canvas = scene.SceneCanvas(keys='interactive', size=(width, height), show=True)
view = canvas.central_widget.add_view()
img = scene.visuals.Image(image, parent=view.scene, cmap='inferno')
view.camera = scene.PanZoomCamera(aspect=1)
view.camera.set_range()
app.run()
