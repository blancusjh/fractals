# Fractals

A garden of escape-time fractals, open to any depth.

![An endless zoom into the Mandelbrot set, down Seahorse Valley to a tiny copy of the whole set, where it starts again.](docs/infinite_zoom.gif)

*Down Seahorse Valley to a copy of the whole set, 10¹⁵ times smaller, and round again.*

## The garden

<table>
<tr>
<td><img src="docs/garden/seahorse_valley.png" alt="Seahorse Valley"><br><sub>Seahorse Valley</sub></td>
<td><img src="docs/garden/elephant_valley.png" alt="Elephant Valley"><br><sub>Elephant Valley</sub></td>
<td><img src="docs/garden/triple_spiral.png" alt="Triple spiral"><br><sub>Triple spiral</sub></td>
</tr>
<tr>
<td><img src="docs/garden/endless_spiral.png" alt="An endless spiral"><br><sub>An endless spiral, 10⁸×</sub></td>
<td><img src="docs/garden/star.png" alt="Star"><br><sub>Star</sub></td>
<td><img src="docs/garden/tendrils.png" alt="Tendrils"><br><sub>Tendrils</sub></td>
</tr>
<tr>
<td><img src="docs/garden/minibrot.png" alt="A minibrot"><br><sub>A minibrot, 10¹⁵×</sub></td>
<td><img src="docs/garden/rabbit.png" alt="Douady's rabbit"><br><sub>Julia: Douady's rabbit</sub></td>
<td><img src="docs/garden/julia_spirals.png" alt="Julia spirals"><br><sub>Julia: spirals</sub></td>
</tr>
<tr>
<td><img src="docs/garden/tricorn.png" alt="The Tricorn"><br><sub>Tricorn</sub></td>
<td><img src="docs/garden/armada.png" alt="The Burning Ship armada"><br><sub>Burning Ship: the armada</sub></td>
<td><img src="docs/garden/lone_ship.png" alt="A lone ship"><br><sub>Burning Ship: a lone ship</sub></td>
</tr>
</table>

## Walk in

```bash
pip install -e ".[viewer,gif]"
python -m fractals view          # scroll to zoom, drag to move, 1–4 to change fractal
```

`--gpu` renders on the graphics card. `python -m fractals render` saves a
picture, and `python -m fractals zoom` saves a zoom animation; the garden and
the animation above come from [`examples/`](examples).

The original renderers are in [`examples/original/`](examples/original).
