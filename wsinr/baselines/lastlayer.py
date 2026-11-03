"""Frozen-dictionary bound: keep every sine layer and refit only the linear readout (A_L, c_L)
to the target image by least squares. Any edit that touches only the output layer cannot beat
this; a gap between it and the full fine-tune measures how much the hidden dictionary must change."""
from __future__ import annotations

import numpy as np

from .. import siren
from ..render import grid


def lastlayer_lstsq(eff: dict, target: np.ndarray, mask: np.ndarray | None = None, ridge: float = 1e-6):
    side = target.shape[0]
    L = sum(1 for k in eff if k.startswith("A"))
    h = grid(side)
    for k in range(1, L):
        h = np.sin(h @ eff[f"A{k}"] + eff[f"c{k}"])
    X = np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)
    y = target.reshape(-1, 1).astype(np.float64)
    if mask is not None:
        m = mask.reshape(-1)
        X, y = X[m], y[m]
    beta = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)
    out = {k: np.array(v, dtype=np.float64, copy=True) for k, v in eff.items()}
    out[f"A{L}"] = beta[:-1]
    out[f"c{L}"] = beta[-1]
    return out
