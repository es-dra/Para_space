"""SIREN as a flat parameter record, with a NumPy forward pass.

Layout follows the NFN SIREN zoo (Zhou et al., 2023): one first sine layer,
``hidden_layers`` hidden sine layers, and a linear output layer; omega_0 = 30.
Parameters are kept as a dict of float32 arrays ``{"W1","b1",...,"W{L}","b{L}"}``
with ``W_k`` of shape (in, out) so that ``h = sin(omega * (h @ W + b))`` (row i
of W belongs to input/unit i, as in the Sparks prompts).

The *effective* form shown to readers and editors is
    A_k = omega * W_k, c_k = omega * b_k (sine layers),  A_L = W_L, c_L = b_L (output)
so that ``h_k = sin(h_{k-1} @ A_k + c_k)`` and ``y = h @ A_L + c_L``.
"""
from __future__ import annotations

import numpy as np

OMEGA = 30.0


def layer_names(num_matrices: int):
    return [f"W{k}" for k in range(1, num_matrices + 1)], [f"b{k}" for k in range(1, num_matrices + 1)]


def num_matrices(params: dict) -> int:
    return sum(1 for k in params if k.startswith("W"))


def shapes(params: dict) -> dict:
    return {k: tuple(v.shape) for k, v in params.items()}


def count_params(params: dict) -> int:
    return int(sum(v.size for v in params.values()))


def forward(params: dict, coords: np.ndarray, omega: float = OMEGA) -> np.ndarray:
    """coords: (N, 2) in [-1, 1]. Returns (N, out)."""
    L = num_matrices(params)
    h = coords.astype(np.float64)
    for k in range(1, L):
        h = np.sin(omega * (h @ params[f"W{k}"].astype(np.float64) + params[f"b{k}"].astype(np.float64)))
    return h @ params[f"W{L}"].astype(np.float64) + params[f"b{L}"].astype(np.float64)


def to_effective(params: dict, omega: float = OMEGA) -> dict:
    L = num_matrices(params)
    eff = {}
    for k in range(1, L + 1):
        s = omega if k < L else 1.0
        eff[f"A{k}"] = params[f"W{k}"].astype(np.float64) * s
        eff[f"c{k}"] = params[f"b{k}"].astype(np.float64) * s
    return eff


def from_effective(eff: dict, omega: float = OMEGA) -> dict:
    L = sum(1 for k in eff if k.startswith("A"))
    params = {}
    for k in range(1, L + 1):
        s = omega if k < L else 1.0
        params[f"W{k}"] = (eff[f"A{k}"] / s).astype(np.float32)
        params[f"b{k}"] = (eff[f"c{k}"] / s).astype(np.float32)
    return params


def forward_effective(eff: dict, coords: np.ndarray) -> np.ndarray:
    L = sum(1 for k in eff if k.startswith("A"))
    h = coords.astype(np.float64)
    for k in range(1, L):
        h = np.sin(h @ eff[f"A{k}"] + eff[f"c{k}"])
    return h @ eff[f"A{L}"] + eff[f"c{L}"]


def init_params(in_dim=2, hidden=32, hidden_layers=1, out_dim=1, omega=OMEGA, seed=0) -> dict:
    """SIREN initialisation (Sitzmann et al., 2020), as in the NFN zoo."""
    rng = np.random.default_rng(seed)
    dims = [in_dim] + [hidden] * (hidden_layers + 1) + [out_dim]
    params = {}
    for k in range(1, len(dims)):
        fan_in = dims[k - 1]
        if k == 1:
            bound = 1.0 / fan_in
        else:
            bound = np.sqrt(6.0 / fan_in) / omega
        params[f"W{k}"] = rng.uniform(-bound, bound, size=(fan_in, dims[k])).astype(np.float32)
        # PyTorch Linear default bias init: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        bb = 1.0 / np.sqrt(fan_in)
        params[f"b{k}"] = rng.uniform(-bb, bb, size=(dims[k],)).astype(np.float32)
    return params


def flatten(params: dict) -> np.ndarray:
    L = num_matrices(params)
    return np.concatenate([params[n].ravel() for k in range(1, L + 1) for n in (f"W{k}", f"b{k}")])


def unflatten(vec: np.ndarray, like: dict) -> dict:
    out, i = {}, 0
    L = num_matrices(like)
    for k in range(1, L + 1):
        for n in (f"W{k}", f"b{k}"):
            sz = like[n].size
            out[n] = vec[i:i + sz].reshape(like[n].shape).astype(np.float32)
            i += sz
    return out


def save_npz(path, params: dict, **meta):
    np.savez(path, **params, **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    params = {k: z[k] for k in z.files if not k.startswith("meta_")}
    meta = {k[5:]: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files if k.startswith("meta_")}
    return params, meta
