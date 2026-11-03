"""Resolve the symmetries of a SIREN before comparing two networks weight by weight.

Hidden units of each sine layer can be permuted, and any unit's sign can be flipped
(sin(-z) = -sin(z)), without changing the function. Given a reference network, this
module finds, layer by layer, the permutation and signs that bring a second network
closest to it (Hungarian weight matching on the incoming weights, then sign matching),
and propagates them to the next layer's incoming rows. Ported from the alignment code
of the earlier Para_space project to the dict format of ``wsinr.siren``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import siren


def align(ref: dict, other: dict, sign_flips: bool = True):
    """Return (aligned copy of `other`, per-layer costs). Both dicts in ``wsinr.siren`` layout
    (W_k of shape (in, out)). The output layer is never permuted."""
    L = siren.num_matrices(ref)
    out = {k: np.array(v, dtype=np.float64, copy=True) for k, v in other.items()}
    costs = []
    for k in range(1, L):
        W_r = ref[f"W{k}"].astype(np.float64)            # (in, out): columns are units
        W_o = out[f"W{k}"]
        # Hungarian matching on incoming weight vectors (columns), including the bias.
        A = np.concatenate([W_r, ref[f"b{k}"][None, :].astype(np.float64)], axis=0).T   # (out, in+1)
        B = np.concatenate([W_o, out[f"b{k}"][None, :]], axis=0).T
        if sign_flips:
            C = np.minimum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1),
                           ((A[:, None, :] + B[None, :, :]) ** 2).sum(-1))
        else:
            C = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        rows, cols = linear_sum_assignment(C)
        perm = np.empty_like(cols)
        perm[rows] = cols                                   # unit j of ref <- unit perm[j] of other
        out[f"W{k}"] = W_o[:, perm]
        out[f"b{k}"] = out[f"b{k}"][perm]
        out[f"W{k + 1}"] = out[f"W{k + 1}"][perm, :]
        if sign_flips:
            dots = (A * np.concatenate([out[f"W{k}"], out[f"b{k}"][None, :]], axis=0).T).sum(-1)
            s = np.where(dots < 0, -1.0, 1.0)
            out[f"W{k}"] *= s[None, :]
            out[f"b{k}"] *= s
            out[f"W{k + 1}"] *= s[:, None]
        costs.append(float(np.mean((W_r - out[f"W{k}"]) ** 2)))
    return {k: v.astype(np.float32) for k, v in out.items()}, costs


def distance(ref: dict, other: dict, aligned: bool = True) -> float:
    """Relative L2 distance between flattened parameters, after alignment by default."""
    o = align(ref, other)[0] if aligned else other
    a, b = siren.flatten(ref), siren.flatten(o)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
