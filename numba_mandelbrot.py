import numpy as np
from vispy import app, scene
from numba import njit, prange

# Tamaño de la ventana
width, height = 800, 800
max_iter = 200



@njit(parallel=True, fastmath=True)
def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    result = np.zeros((height, width), dtype=np.uint8)
    pixel_width = (xmax - xmin) / width
    pixel_height = (ymax - ymin) / height

    for i in prange(width):
        for j in range(height):
            x0 = xmin + i * pixel_width
            y0 = ymin + j * pixel_height
            x, y = 0.0, 0.0
            iter_count = 0
            while x*x + y*y <= 4.0 and iter_count < max_iter:
                x, y = x*x - y*y + x0, 2*x*y + y0
                iter_count += 1
            result[j, i] = int(255 * iter_count / max_iter)
    
    return result  # ¡solo matriz 2D!



# --- Inicialización ---
canvas = scene.SceneCanvas(keys='interactive', size=(width, height), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(aspect=1)
view.camera.set_range()

# Generamos imagen inicial


gray_image = mandelbrot(-2, 1, -1.5, 1.5, width, height, max_iter)
image_data = np.stack([gray_image]*3, axis=-1)

image = scene.visuals.Image(image_data, parent=view.scene, method='subdivide', cmap='inferno')
image.transform = scene.STTransform(scale=(3/width, 3/height), translate=(-2, -1.5))

# --- Evento de zoom o movimiento ---


@canvas.events.mouse_release.connect
def update_image(event):
    print("→ Evento de interacción detectado. Programando actualización...")
    rect = view.camera.rect
    xmin, xmax = rect.left, rect.right
    ymin, ymax = rect.bottom, rect.top
    print(f"→ Cámara rect: left={xmin}, right={xmax}, bottom={ymin}, top={ymax}")
    
    # Recalcular la imagen en la nueva región


    gray_image = mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    image_data = np.stack([gray_image]*3, axis=-1)
    image.set_data(image_data)

    # Reajustar la transformación visual
    image.transform = scene.STTransform(
        scale=((xmax - xmin) / width, (ymax - ymin) / height),
        translate=(xmin, ymin)
    )
    print("✅ Imagen actualizada")

app.run()

