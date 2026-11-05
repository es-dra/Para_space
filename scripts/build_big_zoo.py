"""Build a large SIREN zoo with the PyTorch trainer in chunks (for weight-space learning experiments).

  python scripts/build_big_zoo.py --mnist Data/mnist --split train --n 5000 --chunk 250 --out zoo/mnist_s32_train5k
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from wsinr import siren
from wsinr.data import load_mnist
from wsinr.render import render
from wsinr.zoo import fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--hidden-layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    x, y = load_mnist(a.mnist, a.split)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    t0 = time.time()
    for start in range(a.offset, a.offset + a.n, a.chunk):
        idx = np.arange(start, min(start + a.chunk, a.offset + a.n))
        params, mse = fit(x[idx], steps=a.steps, lr=a.lr, hidden=a.hidden, hidden_layers=a.hidden_layers, seed=a.seed + start)
        for i, p, m in zip(idx, params, mse):
            img = render(p, 28)[..., 0]
            fit_mse = float(np.mean((img - x[i]) ** 2))
            siren.save_npz(os.path.join(a.out, f"{a.split}_{i:05d}.npz"), p, label=int(y[i]), index=int(i), split=a.split,
                           fit_mse=fit_mse, steps=a.steps, lr=a.lr, seed=a.seed + start)
            rows.append({"name": f"{a.split}_{i:05d}", "label": int(y[i]), "fit_mse": fit_mse,
                         "psnr": float(10 * np.log10(1 / max(fit_mse, 1e-12)))})
        print(f"{start + len(idx) - a.offset}/{a.n} done, {time.time() - t0:.0f}s, chunk mean fit MSE {float(np.mean(mse)):.5f}", flush=True)
        json.dump({"config": vars(a), "n_params": siren.count_params(params[0]), "rows": rows,
                   "mean_psnr": float(np.mean([r["psnr"] for r in rows]))}, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print("zoo complete", len(rows), "networks")


if __name__ == "__main__":
    main()
