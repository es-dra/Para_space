"""Coordinate grids and rendering."""
from __future__ import annotations

import numpy as np

from . import siren


def grid(side: int) -> np.ndarray:
    """(side*side, 2) coordinates, row-major (y then x), each axis linspace(-1, 1, side).

    Matches ``get_mgrid`` of the SIREN / NFN code: coords[:, 0] is the row (y) axis,
    coords[:, 1] the column (x) axis.
    """
    t = np.linspace(-1.0, 1.0, side)
    yy, xx = np.meshgrid(t, t, indexing="ij")
    return np.stack([yy.ravel(), xx.ravel()], axis=1)


def render(params: dict, side: int = 28, clip: bool = True) -> np.ndarray:
    """Render to (side, side, out) image. Pixel values are the network output (targets in [0, 1])."""
    out = siren.forward(params, grid(side))
    img = out.reshape(side, side, -1)
    if clip:
        img = np.clip(img, 0.0, 1.0)
    return img


def render_effective(eff: dict, side: int = 28, clip: bool = True) -> np.ndarray:
    out = siren.forward_effective(eff, grid(side))
    img = out.reshape(side, side, -1)
    return np.clip(img, 0.0, 1.0) if clip else img


def ascii_art(img: np.ndarray, levels: str = " .:-=+*#%@") -> str:
    """Grayscale (H, W) -> text, one char per pixel."""
    a = np.clip(img.squeeze(), 0, 1)
    idx = np.minimum((a * len(levels)).astype(int), len(levels) - 1)
    return "\n".join("".join(levels[i] for i in row) for row in idx)
