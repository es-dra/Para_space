"""NumPy-only batched SIREN fitting (manual backprop + Adam). Useful where PyTorch is unavailable
(e.g. a restricted sandbox) and for networks of a few thousand weights."""
from __future__ import annotations

import numpy as np

from . import siren
from .render import grid


def _init(B, dims, omega, rng):
    P = {}
    for k in range(1, len(dims)):
        fan_in = dims[k - 1]
        bound = (1.0 / fan_in) if k == 1 else np.sqrt(6.0 / fan_in) / omega
        P[f"W{k}"] = rng.uniform(-bound, bound, size=(B, fan_in, dims[k]))
        P[f"b{k}"] = rng.uniform(-1 / np.sqrt(fan_in), 1 / np.sqrt(fan_in), size=(B, dims[k]))
    return P


def _forward(P, X, omega):
    L = sum(1 for k in P if k.startswith("W"))
    hs, zs = [X], []
    h = X
    for k in range(1, L):
        z = omega * (h @ P[f"W{k}"] + P[f"b{k}"][:, None, :])
        h = np.sin(z)
        zs.append(z)
        hs.append(h)
    y = h @ P[f"W{L}"] + P[f"b{L}"][:, None, :]
    return y, hs, zs


def fit(images: np.ndarray, steps=2000, lr=1e-3, hidden=32, hidden_layers=1, omega=siren.OMEGA, seed=0,
        init: list | None = None, log_every=0, masks: np.ndarray | None = None, harmonic_penalty=0.0):
    """images: (B, H, W). Returns (list of float32 param dicts, per-network final MSE).

    harmonic_penalty: lambda (scalar or per-network array) on the mean harmonic index of the hidden sine
    layers, lambda * mean_j sum_i |omega W_k[i, j]| for k = 2..L-1. It keeps hidden units near the linear
    regime of the sine, where layer-local weight-space programs stay exact ("editable" training)."""
    B, H, W = images.shape
    N = H * W
    X = np.broadcast_to(grid(H), (B, N, 2)).astype(np.float64)
    T = images.reshape(B, N, 1).astype(np.float64)
    M = None if masks is None else masks.reshape(B, N, 1).astype(np.float64)
    rng = np.random.default_rng(seed)
    if init is None:
        dims = [2] + [hidden] * (hidden_layers + 1) + [1]
        P = _init(B, dims, omega, rng)
    else:
        P = {k: np.stack([p[k] for p in init]).astype(np.float64) for k in init[0]}
        Lm = sum(1 for k in init[0] if k.startswith("W"))
        dims = [init[0]["W1"].shape[0]] + [init[0][f"W{k}"].shape[1] for k in range(1, Lm + 1)]
    L = len(dims) - 1
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(x) for k, x in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(1, steps + 1):
        y, hs, zs = _forward(P, X, omega)
        diff = y - T
        if M is not None:
            diff = diff * M
            denom = np.maximum(M.sum(axis=(1, 2), keepdims=True), 1)
        else:
            denom = N
        g = 2 * diff / denom                      # dL/dy, L = per-network mean
        grads = {}
        for k in range(L, 0, -1):
            hprev = hs[k - 1]
            grads[f"W{k}"] = np.einsum("bni,bnj->bij", hprev, g)
            grads[f"b{k}"] = g.sum(axis=1)
            if k > 1:
                g = (g @ P[f"W{k}"].transpose(0, 2, 1)) * np.cos(zs[k - 2]) * omega
        if np.any(np.asarray(harmonic_penalty) > 0):
            lam = np.broadcast_to(np.asarray(harmonic_penalty, float), (B,))[:, None, None]
            for k in range(2, L):
                nunits = P[f"W{k}"].shape[2]
                grads[f"W{k}"] = grads[f"W{k}"] + lam * omega * np.sign(P[f"W{k}"]) / nunits
        for k in P:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            mh = m[k] / (1 - b1 ** it)
            vh = v[k] / (1 - b2 ** it)
            P[k] -= lr * mh / (np.sqrt(vh) + eps)
        if log_every and it % log_every == 0:
            per = ((y - T) ** 2).mean(axis=(1, 2))
            print(f"step {it}: mean mse {per.mean():.5f} worst {per.max():.5f}", flush=True)
    y, _, _ = _forward(P, X, omega)
    per = ((np.clip(y, 0, 1) - T) ** 2)
    per = (per * M).sum(axis=(1, 2)) / np.maximum(M.sum(axis=(1, 2)), 1) if M is not None else per.mean(axis=(1, 2))
    out = [{k: P[k][i].astype(np.float32) for k in P} for i in range(B)]
    return out, per
