"""Closed-form weight-space edits of a SIREN, in effective form (see siren.py).

Geometry acts on the first layer only: for an invertible map p -> T(p) on the coordinate
plane, the edited network g(p) = f(T(p)) is obtained from
    p @ A1 + c1  ->  T(p) @ A1 + c1.
Affine T(p) = p @ R + t gives A1' = R @ A1 and c1' = c1 + t @ A1.
Pointwise value maps act on the output layer: v' = a v + b gives A_L' = a A_L, c_L' = a c_L + b.

Coordinates: p = (y, x), y = row axis (-1 top .. +1 bottom), x = column axis (-1 left .. +1 right).
All functions return a new effective dict and never modify the input.
"""
from __future__ import annotations

import copy

import numpy as np


def _copy(eff):
    return {k: np.array(v, dtype=np.float64, copy=True) for k, v in eff.items()}


def _last(eff):
    return max(int(k[1:]) for k in eff if k.startswith("A"))


def affine_coords(eff, R, t=(0.0, 0.0)):
    """g(p) = f(p @ R + t)."""
    out = _copy(eff)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    out["c1"] = eff["c1"] + t @ eff["A1"]
    out["A1"] = R @ eff["A1"]
    return out


def translate(eff, dy_pix: float, dx_pix: float, side: int = 28):
    """Move the picture down by dy_pix pixels and right by dx_pix pixels: g(p) = f(p - d)."""
    step = 2.0 / (side - 1)
    d = np.array([dy_pix * step, dx_pix * step])
    return affine_coords(eff, np.eye(2), -d)


def flip_lr(eff):
    """Mirror left-right: g(y, x) = f(y, -x)."""
    return affine_coords(eff, np.diag([1.0, -1.0]))


def flip_ud(eff):
    """Mirror top-bottom: g(y, x) = f(-y, x)."""
    return affine_coords(eff, np.diag([-1.0, 1.0]))


def transpose(eff):
    """g(y, x) = f(x, y)."""
    return affine_coords(eff, np.array([[0.0, 1.0], [1.0, 0.0]]))


def rot90(eff, k: int = 1):
    """Counter-clockwise rotation by k * 90 degrees, matching numpy.rot90 on the rendered grid.

    np.rot90(img, 1)[i, j] = img[j, side-1-i]; in coordinates out(y, x) = in(x, -y).
    """
    k %= 4
    R1 = np.array([[0.0, -1.0], [1.0, 0.0]])  # p @ R1 = (x, -y)
    R = np.linalg.matrix_power(R1, k)
    return affine_coords(eff, R)


def rotate(eff, degrees: float):
    """Counter-clockwise rotation about the image centre by an arbitrary angle (sampled target)."""
    a = np.deg2rad(degrees)
    # out(p) = in(R^-1 p) with R the CCW rotation in (x, y); in our (y, x) row-vector convention:
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return affine_coords(eff, R)


def zoom(eff, factor: float):
    """Magnify about the centre by `factor` (>1 zooms in): g(p) = f(p / factor)."""
    return affine_coords(eff, np.eye(2) / factor)


def value_affine(eff, a: float, b: float):
    """v' = a v + b on the output."""
    out = _copy(eff)
    L = _last(eff)
    out[f"A{L}"] = a * eff[f"A{L}"]
    out[f"c{L}"] = a * eff[f"c{L}"] + b
    return out


def negative(eff):
    return value_affine(eff, -1.0, 1.0)


def brightness(eff, delta: float):
    return value_affine(eff, 1.0, delta)


def contrast(eff, gamma: float, pivot: float = 0.5):
    """v' = pivot + gamma (v - pivot)."""
    return value_affine(eff, gamma, pivot * (1.0 - gamma))


def permute_hidden(eff, layer: int, perm):
    """A function-preserving symmetry: permute the units of hidden layer `layer` (1-based)."""
    out = _copy(eff)
    perm = np.asarray(perm)
    out[f"A{layer}"] = eff[f"A{layer}"][:, perm]
    out[f"c{layer}"] = eff[f"c{layer}"][perm]
    out[f"A{layer + 1}"] = eff[f"A{layer + 1}"][perm, :]
    return out


def sign_flip_hidden(eff, layer: int, units):
    """Another symmetry of sine layers: sin(-z) = -sin(z); negate a unit's input row and output row."""
    out = _copy(eff)
    units = list(units)
    out[f"A{layer}"][:, units] *= -1
    out[f"c{layer}"][units] *= -1
    out[f"A{layer + 1}"][units, :] *= -1
    return out
