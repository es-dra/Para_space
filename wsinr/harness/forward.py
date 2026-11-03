"""The forward-pass tool of the sandbox protocols. Each request costs one pass from the budget.

  {"type": "render", "model": "original" | "edited" | "both", "side": 28}
      renders the network(s) on the side x side grid; saves /work/fwd_<k>_<model>.npy and returns
      an ASCII preview plus summary statistics (and, for "both", the per-pixel difference statistics).
  {"type": "probe", "model": ..., "points": [[y, x], ...]}         (at most 256 points)
      returns the output value at the given coordinates and, with "activations": true, the hidden
      activations h1, h2 (saved to /work/fwd_<k>_probe.npz).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .. import siren
from ..render import ascii_art, grid


def _models(kernel, orig_eff, which):
    out = {}
    if which in ("original", "both"):
        out["original"] = orig_eff
    if which in ("edited", "both"):
        e = kernel.edited_params()
        out["edited"] = e if e is not None else orig_eff
    return out


def run_forward(req: dict, kernel, orig_eff: dict, k: int, side_default: int = 28) -> str:
    which = req.get("model", "edited")
    if which not in ("original", "edited", "both"):
        return "ERROR: model must be original, edited or both (request not counted)"
    typ = req.get("type", "render")
    nets = _models(kernel, orig_eff, which)
    lines = []
    if typ == "render":
        side = int(req.get("side", side_default))
        if not 4 <= side <= 128:
            return "ERROR: side must be between 4 and 128 (request not counted)"
        imgs = {}
        for name, eff in nets.items():
            img = siren.forward_effective(eff, grid(side)).reshape(side, side, -1)[..., 0]
            imgs[name] = img
            np.save(os.path.join(kernel.work_dir, f"fwd_{k}_{name}.npy"), img.astype(np.float32))
            c = np.clip(img, 0, 1)
            lines.append(f"[{name}] {side}x{side} rendering saved to {kernel.work_dir}/fwd_{k}_{name}.npy "
                         f"(raw output; clipped stats: mean {c.mean():.3f}, share > 0.5: {(c > 0.5).mean():.3f}, "
                         f"raw min {img.min():.2f} max {img.max():.2f})")
            lines.append(ascii_art(c))
        if len(imgs) == 2:
            d = np.clip(imgs["edited"], 0, 1) - np.clip(imgs["original"], 0, 1)
            lines.append(f"[edited - original] mean abs diff {np.abs(d).mean():.4f}, max abs diff {np.abs(d).max():.3f}, "
                         f"pixels changed by > 0.1: {(np.abs(d) > 0.1).mean():.3f}")
    elif typ == "probe":
        pts = np.asarray(req.get("points", []), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or not 1 <= len(pts) <= 256:
            return "ERROR: points must be a list of at most 256 [y, x] pairs (request not counted)"
        save = {}
        for name, eff in nets.items():
            L = sum(1 for key in eff if key.startswith("A"))
            h = pts
            acts = []
            for j in range(1, L):
                h = np.sin(h @ eff[f"A{j}"] + eff[f"c{j}"])
                acts.append(h)
            v = h @ eff[f"A{L}"] + eff[f"c{L}"]
            save[f"{name}_v"] = v
            for j, a in enumerate(acts, 1):
                save[f"{name}_h{j}"] = a
            shown = min(len(pts), 32)
            lines.append(f"[{name}] output at the first {shown} of {len(pts)} points (y, x -> v):")
            lines.append("\n".join(f"  ({p[0]:+.3f}, {p[1]:+.3f}) -> {val[0]:+.4f}" for p, val in zip(pts[:shown], v[:shown])))
        np.savez(os.path.join(kernel.work_dir, f"fwd_{k}_probe.npz"), points=pts, **save)
        if req.get("activations"):
            lines.append(f"hidden activations saved to {kernel.work_dir}/fwd_{k}_probe.npz (keys: {sorted(save)})")
    else:
        return "ERROR: type must be render or probe (request not counted)"
    return "\n".join(lines)
