# Understanding Gradients

[![YouTube Video](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=N6WKU3QbuR4)

A Manim animation explaining the gradient - what it is, why it exists, and what it means geometrically.

The video covers:
- Slopes in multiple directions on a 3D surface
- Why infinitely many directional slopes collapse into one vector
- How the gradient encodes all local rates of change via the dot product
- Why level curves are perpendicular to the gradient
- Partial derivatives and the directional derivative formula

## Watch

> Click on the link above to watch on YouTube

## Run the animations

```bash
manim -pql src/scenes/main_surface.py
```

## Project structure

```
src/scenes/   # Manim scene files
src/utils.py  # Shared helpers
media/        # Rendered output (auto-generated, not committed)
```
