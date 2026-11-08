"""Fit many small SIRENs at once (batched over networks) with PyTorch on CPU or GPU."""
from __future__ import annotations

import math

import numpy as np
import torch

from . import siren
from .render import grid


def init_batched(B, in_dim=2, hidden=32, hidden_layers=1, out_dim=1, omega=siren.OMEGA, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dims = [in_dim] + [hidden] * (hidden_layers + 1) + [out_dim]
    P = {}
    for k in range(1, len(dims)):
        fan_in = dims[k - 1]
        bound = (1.0 / fan_in) if k == 1 else math.sqrt(6.0 / fan_in) / omega
        W = (torch.rand(B, fan_in, dims[k], generator=g) * 2 - 1) * bound
        bb = 1.0 / math.sqrt(fan_in)
        b = (torch.rand(B, dims[k], generator=g) * 2 - 1) * bb
        P[f"W{k}"] = W.to(device).requires_grad_(True)
        P[f"b{k}"] = b.to(device).requires_grad_(True)
    return P


def forward_batched(P: dict, coords: torch.Tensor, omega=siren.OMEGA) -> torch.Tensor:
    """coords: (N, 2) shared by all networks -> (B, N, out)."""
    L = sum(1 for k in P if k.startswith("W"))
    B = P["W1"].shape[0]
    h = coords.unsqueeze(0).expand(B, -1, -1)
    for k in range(1, L):
        h = torch.sin(omega * (torch.bmm(h, P[f"W{k}"]) + P[f"b{k}"].unsqueeze(1)))
    return torch.bmm(h, P[f"W{L}"]) + P[f"b{L}"].unsqueeze(1)


def fit(images: np.ndarray, steps=1000, lr=1e-3, hidden=32, hidden_layers=1, omega=siren.OMEGA, seed=0,
        device="cpu", init: dict | None = None, log_every=0, swing_bound: float = 0.0):
    """images: (B, H, W) in [0, 1]. Returns list of param dicts (numpy) and final per-network MSE."""
    B, H, W = images.shape
    assert H == W
    coords = torch.tensor(grid(H), dtype=torch.float32, device=device)
    target = torch.tensor(images.reshape(B, H * W, 1), dtype=torch.float32, device=device)
    if init is None:
        P = init_batched(B, hidden=hidden, hidden_layers=hidden_layers, omega=omega, seed=seed, device=device)
    else:
        P = {k: torch.tensor(np.stack([p[k] for p in init]), dtype=torch.float32, device=device, requires_grad=True)
             for k in init[0]}
    opt = torch.optim.Adam(list(P.values()), lr=lr)
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = forward_batched(P, coords, omega)
        per = ((pred - target) ** 2).mean(dim=(1, 2))
        per.sum().backward()
        opt.step()
        if swing_bound > 0:
            with torch.no_grad():
                L = sum(1 for k in P if k.startswith("W"))
                for k in range(2, L):
                    s_unit = omega * P[f"W{k}"].abs().sum(dim=1, keepdim=True)
                    P[f"W{k}"].mul_(torch.clamp(swing_bound / s_unit.clamp_min(1e-12), max=1.0))
        if log_every and (it + 1) % log_every == 0:
            print(f"step {it + 1}: mean mse {per.mean().item():.5f}  worst {per.max().item():.5f}", flush=True)
    with torch.no_grad():
        pred = forward_batched(P, coords, omega)
        per = ((pred - target) ** 2).mean(dim=(1, 2)).cpu().numpy()
    out = []
    for i in range(B):
        out.append({k: P[k][i].detach().cpu().numpy().astype(np.float32) for k in P})
    return out, per
