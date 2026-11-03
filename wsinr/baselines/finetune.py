"""Gradient baseline: fine-tune each INR towards its target image from its current weights."""
from __future__ import annotations

import numpy as np
import torch

from ..render import grid
from ..zoo import forward_batched


def finetune(param_dicts: list, targets: np.ndarray, steps_list=(10, 50, 200, 1000), lr=1e-3, device="cpu",
             masks: np.ndarray | None = None):
    """Returns {steps: (list of params, per-network mse)} for each checkpoint in steps_list."""
    B, H, W = targets.shape
    coords = torch.tensor(grid(H), dtype=torch.float32, device=device)
    tgt = torch.tensor(targets.reshape(B, H * W, 1), dtype=torch.float32, device=device)
    m = None
    if masks is not None:
        m = torch.tensor(masks.reshape(B, H * W, 1), dtype=torch.float32, device=device)
    P = {k: torch.tensor(np.stack([p[k] for p in param_dicts]), dtype=torch.float32, device=device, requires_grad=True)
         for k in param_dicts[0]}
    opt = torch.optim.Adam(list(P.values()), lr=lr)
    results = {}
    total = max(steps_list)
    for it in range(1, total + 1):
        opt.zero_grad(set_to_none=True)
        pred = forward_batched(P, coords)
        err = (pred - tgt) ** 2
        if m is not None:
            per = (err * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1)
        else:
            per = err.mean(dim=(1, 2))
        per.sum().backward()
        opt.step()
        if it in steps_list:
            with torch.no_grad():
                pred = forward_batched(P, coords).clamp(0, 1)
                e = (pred - tgt) ** 2
                if m is not None:
                    pm = ((e * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1)).cpu().numpy()
                else:
                    pm = e.mean(dim=(1, 2)).cpu().numpy()
            results[it] = ([{k: P[k][i].detach().cpu().numpy().astype(np.float32) for k in P} for i in range(B)], pm)
    return results
