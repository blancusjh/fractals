import numpy as np
from vispy import app, gloo

# --- Configuración inicial ---
canvas = app.Canvas(keys='interactive', size=(800, 800), title='Mandelbrot GPU')


vertex_shader = """
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

fragment_shader = """
uniform vec2 u_resolution;
uniform vec2 u_center;
uniform float u_zoom;
uniform int u_max_iter;

vec3 colormap(float t) {
    return 0.5 + 0.5 * cos(3.0 + 2.0 * 3.14159 * vec3(t, t*1.5, t*2.0));
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_zoom + u_center;

    vec2 z = vec2(0.0);
    int i;
    for (i = 0; i < u_max_iter; ++i) {
        if (dot(z, z) > 4.0) break;
        z = vec2(
            z.x * z.x - z.y * z.y + uv.x,
            2.0 * z.x * z.y + uv.y
        );
    }

    if (i == u_max_iter) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        float magnitude = length(z);
        float mu = float(i) - log(log(magnitude)) / log(2.0);
        float t = mu / float(u_max_iter);
        vec3 color = colormap(t);
        gl_FragColor = vec4(color, 1.0);
    }
}
"""



# Programa de shaders
program = gloo.Program(vertex_shader, fragment_shader)

# Cuadrado de pantalla completa
quad = np.array([
    [-1, -1],
    [+1, -1],
    [-1, +1],
    [-1, +1],
    [+1, -1],
    [+1, +1],
], dtype=np.float32)
program['a_position'] = quad

# Parámetros iniciales
center = np.array([0.0, 0.0], dtype=np.float32)
zoom = 300.0
max_iter = 300


def framebuffer_size():
    """Return the physical pixel size used by gl_FragCoord."""
    return np.asarray(canvas.physical_size, dtype=np.float32)


def cursor_offset(pos):
    """Return the cursor offset from the framebuffer centre in GL pixels."""
    logical_size = np.asarray(canvas.size, dtype=np.float32)
    scale = framebuffer_size() / logical_size
    mouse = np.asarray(pos, dtype=np.float32) * scale
    resolution = framebuffer_size()
    return np.array([
        mouse[0] - 0.5 * resolution[0],
        0.5 * resolution[1] - mouse[1],
    ], dtype=np.float32)


@canvas.connect
def on_draw(event):
    gloo.clear('black')
    program['u_resolution'] = framebuffer_size()
    program['u_center'] = center
    program['u_zoom'] = zoom
    program['u_max_iter'] = max_iter
    program.draw('triangles')

@canvas.connect
def on_mouse_wheel(event):
    global center, zoom

    # Keep the complex-plane coordinate below the cursor fixed while zooming.
    # Events are logical top-left pixels; gl_FragCoord is physical bottom-left.
    offset = cursor_offset(event.pos)

    new_zoom = zoom * 1.1 ** event.delta[1]
    center += offset * (1.0 / zoom - 1.0 / new_zoom)
    zoom = new_zoom
    canvas.update()



# Estado de drag
dragging = False
last_mouse_pos = None

@canvas.connect
def on_mouse_press(event):
    global dragging, last_mouse_pos
    if event.button == 1:  # Botón izquierdo
        dragging = True
        last_mouse_pos = event.pos

@canvas.connect
def on_mouse_release(event):
    global dragging
    if event.button == 1:
        dragging = False

@canvas.connect
def on_mouse_move(event):
    global center, last_mouse_pos
    if dragging and event.is_dragging:
        logical_size = np.asarray(canvas.size, dtype=np.float32)
        scale = framebuffer_size() / logical_size
        dx, dy = (np.asarray(event.pos) - last_mouse_pos) * scale
        last_mouse_pos = event.pos
        center[0] -= dx / zoom
        center[1] += dy / zoom
        canvas.update()

@canvas.connect
def on_resize(event):
    gloo.set_viewport(0, 0, *event.physical_size)

canvas.show()
app.run()
