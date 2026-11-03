"""Fit a zoo of SIRENs to MNIST digits (NFN configuration: 2 -> 32 -> 32 -> 1, omega 30).

Example:
  python scripts/build_zoo.py --mnist /path/to/mnist --n 200 --steps 2000 --out zoo/mnist_s32
"""
import argparse
import json
import os
import time

import numpy as np

from wsinr import siren
from wsinr.data import load_mnist
from wsinr.render import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--hidden-layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", choices=["torch", "np"], default="torch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.backend == "torch":
        from wsinr.zoo import fit
        extra = {"device": a.device}
    else:
        from wsinr.zoo_np import fit
        extra = {}

    x, y = load_mnist(a.mnist, a.split)
    idx = np.arange(a.offset, a.offset + a.n)
    imgs, labels = x[idx], y[idx]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    params, mse = fit(imgs, steps=a.steps, lr=a.lr, hidden=a.hidden, hidden_layers=a.hidden_layers,
                      seed=a.seed, log_every=max(a.steps // 10, 1), **extra)
    dt = time.time() - t0
    rows = []
    for i, (p, m) in enumerate(zip(params, mse)):
        img = render(p, 28)[..., 0]
        fit_mse = float(np.mean((img - imgs[i]) ** 2))
        name = f"{a.split}_{idx[i]:05d}"
        siren.save_npz(os.path.join(a.out, name + ".npz"), p, label=int(labels[i]), index=int(idx[i]),
                       split=a.split, fit_mse=fit_mse, steps=a.steps, lr=a.lr, seed=a.seed)
        rows.append({"name": name, "label": int(labels[i]), "fit_mse": fit_mse,
                     "psnr": float(10 * np.log10(1 / max(fit_mse, 1e-12)))})
    meta = {"config": vars(a), "n_params": siren.count_params(params[0]), "seconds": dt,
            "mean_fit_mse": float(np.mean([r["fit_mse"] for r in rows])),
            "mean_psnr": float(np.mean([r["psnr"] for r in rows])), "rows": rows}
    json.dump(meta, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print(f"{len(rows)} networks, {meta['n_params']} params each, mean fit MSE {meta['mean_fit_mse']:.5f} "
          f"(PSNR {meta['mean_psnr']:.2f} dB), {dt:.0f} s")


if __name__ == "__main__":
    main()
