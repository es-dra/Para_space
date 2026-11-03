"""Edit tasks: a target-image transform, an optional closed-form oracle, and a text description.

Levels
  0  exact in weight space (geometry via layer 1, value maps via the output layer)
  1  exact up to sampling of the target (arbitrary rotation, zoom)
  2  no closed form known (morphology, blur, thresholding)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import ndimage

from . import oracle


@dataclass
class Task:
    name: str
    level: int
    description: str                                  # instruction shown to the editor
    target: Callable[[np.ndarray], np.ndarray]        # base image (H, W) -> target image (H, W)
    oracle: Optional[Callable[[dict], dict]] = None   # effective params -> edited effective params
    mask: Optional[Callable[[int], np.ndarray]] = None  # (side) -> bool mask of pixels that are defined
    tags: tuple = field(default_factory=tuple)


def _shift(img, dy, dx):
    out = np.zeros_like(img)
    H, W = img.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = img[ys, xs]
    return out


def _shift_mask(side, dy, dx):
    m = np.zeros((side, side), bool)
    m[max(dy, 0):side + min(dy, 0), max(dx, 0):side + min(dx, 0)] = True
    return m


def _zoom_img(img, factor):
    H, W = img.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    src_y = cy + (yy - cy) / factor
    src_x = cx + (xx - cx) / factor
    return ndimage.map_coordinates(img, [src_y, src_x], order=1, mode="constant", cval=0.0)


def _zoom_mask(side, factor):
    c = (side - 1) / 2.0
    yy, xx = np.mgrid[0:side, 0:side]
    sy = c + (yy - c) / factor
    sx = c + (xx - c) / factor
    return (sy >= 0) & (sy <= side - 1) & (sx >= 0) & (sx <= side - 1)


def _rotate_img(img, deg):
    return ndimage.rotate(img, deg, reshape=False, order=1, mode="constant", cval=0.0)


def _rotate_mask(side, deg):
    ones = np.ones((side, side))
    return ndimage.rotate(ones, deg, reshape=False, order=0, mode="constant", cval=0.0) > 0.5


def make_tasks(side: int = 28) -> dict:
    T = {}

    def add(t):
        T[t.name] = t

    # ---- level 0: exact ----
    add(Task("flip_lr", 0, "Mirror the picture left-to-right (what was on the left must appear on the right).",
             lambda im: im[:, ::-1].copy(), oracle.flip_lr, tags=("geometry",)))
    add(Task("flip_ud", 0, "Mirror the picture top-to-bottom.",
             lambda im: im[::-1, :].copy(), oracle.flip_ud, tags=("geometry",)))
    add(Task("rot90", 0, "Rotate the picture by 90 degrees counter-clockwise (the top edge becomes the left edge).",
             lambda im: np.rot90(im, 1).copy(), lambda e: oracle.rot90(e, 1), tags=("geometry",)))
    add(Task("transpose", 0, "Transpose the picture: swap its row and column axes (reflect it across the main diagonal).",
             lambda im: im.T.copy(), oracle.transpose, tags=("geometry",)))
    for dy, dx in [(0, 4), (3, 0), (-3, 3)]:
        name = f"shift_{dy:+d}_{dx:+d}".replace("+", "p").replace("-", "m")
        add(Task(name, 0,
                 f"Move the picture by {dy:+d} pixels vertically (positive = down) and {dx:+d} pixels horizontally "
                 f"(positive = right). Pixels that move out of the frame are lost; pixels that are uncovered should be black (0).",
                 lambda im, dy=dy, dx=dx: _shift(im, dy, dx),
                 lambda e, dy=dy, dx=dx: oracle.translate(e, dy, dx, side),
                 mask=lambda s, dy=dy, dx=dx: _shift_mask(s, dy, dx), tags=("geometry",)))
    add(Task("negative", 0, "Invert the grey values (v -> 1 - v): black becomes white and white becomes black.",
             lambda im: 1.0 - im, oracle.negative, tags=("value",)))
    add(Task("brighten_0.3", 0, "Make the picture brighter by adding 0.3 to every grey value (values are clipped to [0, 1] when rendered).",
             lambda im: np.clip(im + 0.3, 0, 1), lambda e: oracle.brightness(e, 0.3), tags=("value",)))
    add(Task("contrast_2", 0, "Double the contrast about mid-grey: v -> 0.5 + 2 (v - 0.5), clipped to [0, 1] when rendered.",
             lambda im: np.clip(0.5 + 2 * (im - 0.5), 0, 1), lambda e: oracle.contrast(e, 2.0), tags=("value",)))
    add(Task("contrast_0.5", 0, "Halve the contrast about mid-grey: v -> 0.5 + 0.5 (v - 0.5).",
             lambda im: 0.5 + 0.5 * (im - 0.5), lambda e: oracle.contrast(e, 0.5), tags=("value",)))

    # ---- level 1: exact in function space, sampled target ----
    add(Task("rotate_30", 1, "Rotate the picture by 30 degrees counter-clockwise about its centre. Uncovered corners should be black.",
             lambda im: _rotate_img(im, 30), lambda e: oracle.rotate(e, 30), mask=lambda s: _rotate_mask(s, 30), tags=("geometry",)))
    add(Task("zoom_1.5", 1, "Magnify the picture by a factor 1.5 about its centre (the outer parts leave the frame).",
             lambda im: _zoom_img(im, 1.5), lambda e: oracle.zoom(e, 1.5), tags=("geometry",)))
    add(Task("zoom_0.7", 1, "Shrink the picture to 0.7 of its size about its centre; the uncovered border should be black.",
             lambda im: _zoom_img(im, 0.7), lambda e: oracle.zoom(e, 0.7), mask=lambda s: _zoom_mask(s, 0.7), tags=("geometry",)))

    # ---- level 2: no closed form ----
    add(Task("dilate", 2, "Thicken the bright strokes: replace every grey value by the maximum over its 3 x 3 pixel neighbourhood (morphological dilation).",
             lambda im: ndimage.grey_dilation(im, size=(3, 3)), None, tags=("morphology",)))
    add(Task("erode", 2, "Thin the bright strokes: replace every grey value by the minimum over its 3 x 3 pixel neighbourhood (morphological erosion).",
             lambda im: ndimage.grey_erosion(im, size=(3, 3)), None, tags=("morphology",)))
    add(Task("blur_1", 2, "Blur the picture with a Gaussian of standard deviation 1 pixel.",
             lambda im: ndimage.gaussian_filter(im, 1.0), lambda e: oracle.lowpass_gauss(e, 1.0, side),
             tags=("filter",)))  # the oracle is exact only for shallow (one sine layer) networks; None otherwise
    add(Task("threshold", 2, "Binarise the picture: grey values above 0.5 become 1, the rest 0.",
             lambda im: (im > 0.5).astype(np.float64), None, tags=("value",)))
    return T


TASKS = make_tasks()
