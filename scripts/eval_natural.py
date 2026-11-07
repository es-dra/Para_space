"""Stage 09: price of programmability, exact programs and anti-aliasing on 64x64 natural-image crops."""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from wsinr import programs, siren
from wsinr.render import grid
from wsinr.zoo import fit as torch_fit
from wsinr.zoo_np import fit as np_fit

SIDE = 64  # overridden by --side


def fwd(eff, coords, chunk=65536):
    """forward_effective in row chunks (a 1.25M x 1024 float64 activation matrix is 10 GB otherwise)."""
    out = np.empty((coords.shape[0], 1))
    for i in range(0, coords.shape[0], chunk):
        out[i:i + chunk] = siren.forward_effective(eff, coords[i:i + chunk])
    return out


def crops(folder, n=20, side=SIDE, min_std=0.08):
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        im = np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0
        H, W = im.shape
        for cy, cx in [(H // 2, W // 2), (H // 4, W // 4)]:
            y0, x0 = max(0, min(cy - side // 2, H - side)), max(0, min(cx - side // 2, W - side))
            c = im[y0:y0 + side, x0:x0 + side]
            if c.shape == (side, side) and c.std() > min_std:
                out.append(c)
    return np.stack(out[:n])


def render_eff(eff, side=None, clip=False):
    side = side or SIDE
    img = fwd(eff, grid(side)).reshape(side, side)
    return np.clip(img, 0, 1) if clip else img


def fine_target(eff, transfer_fn, side=None, up=8):
    side = side or SIDE
    t = np.linspace(-1, 1, side)
    step = t[1] - t[0]
    pad = 6 * step
    tf = np.arange(-1 - pad, 1 + pad + 1e-9, step / up)
    yy, xx = np.meshgrid(tf, tf, indexing="ij")
    F = fwd(eff, np.stack([yy.ravel(), xx.ravel()], 1)).reshape(len(tf), len(tf))
    Fb = transfer_fn(F, up)
    idx = np.round((t - tf[0]) / (step / up)).astype(int)
    return Fb[np.ix_(idx, idx)]


def footprint_truth(eff, factor, side=None, up=8):
    side = side or SIDE
    step = 2.0 / (side - 1)
    n_out = side // factor
    t = -1 - step / 2 + (np.arange(n_out * factor * up) + 0.5) * step / up
    yy, xx = np.meshgrid(t, t, indexing="ij")
    F = fwd(eff, np.stack([yy.ravel(), xx.ravel()], 1)).reshape(len(t), len(t))
    blk = factor * up
    truth = F.reshape(n_out, blk, n_out, blk).mean(axis=(1, 3))
    centres = -1 - step / 2 + (np.arange(n_out) + 0.5) * factor * step
    return truth, centres


def sample(eff, centres, k, factor, side=None):
    side = side or SIDE
    step = 2.0 / (side - 1)
    offs = (np.arange(k) + 0.5) / k - 0.5
    vals = np.zeros((len(centres), len(centres)))
    for oy in offs:
        for ox in offs:
            yy, xx = np.meshgrid(centres + oy * factor * step, centres + ox * factor * step, indexing="ij")
            vals += fwd(eff, np.stack([yy.ravel(), xx.ravel()], 1)).reshape(len(centres), len(centres))
    return vals / (k * k)


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--zoos", nargs="*", default=["deep_h64", "shallow_w1024", "deep_h64x3", "shallow_w2048", "deep_h64_swing2", "deep_h64_swing4"])
    ap.add_argument("--side", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=0, help="fit this many crops at a time (memory); 0 = all at once")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global SIDE
    SIDE = a.side
    torch.set_num_threads(2)
    X = crops(a.images, a.n, side=a.side)
    print("crops", X.shape, flush=True)
    specs = {"deep_h64": (64, 1, 0), "deep_h64x3": (64, 2, 0), "shallow_w1024": (1024, 0, 0), "shallow_w2048": (2048, 0, 0),
             "deep_h64_swing2": (64, 1, 2.0), "deep_h64_swing4": (64, 1, 4.0), "deep_h32x3": (32, 2, 0), "shallow_w512": (512, 0, 0)}
    os.makedirs(a.out, exist_ok=True)
    rpath = os.path.join(a.out, "results.json")
    results = json.load(open(rpath)) if os.path.exists(rpath) else {}
    for name in a.zoos:
        hidden, hl, bound = specs[name]
        t0 = time.time()
        if name in results and len(results[name].get("rows", [])) == len(X):
            print(name, "already done, skipping", flush=True)
            continue
        params = []
        ch = a.chunk or len(X)
        for s0 in range(0, len(X), ch):
            Xc = X[s0:s0 + ch]
            if bound > 0:
                pc, _ = np_fit(Xc, steps=a.steps, lr=1e-3, hidden=hidden, hidden_layers=hl, swing_bound=bound, seed=s0)
            else:
                pc, _ = torch_fit(Xc, steps=a.steps, lr=1e-3, hidden=hidden, hidden_layers=hl, seed=s0)
            params += pc
        np.savez(os.path.join(a.out, f"{name}_params.npz"), **{f"{i}_{k}": v for i, p in enumerate(params) for k, v in p.items()})
        rows = []
        for i, p in enumerate(params):
            eff = siren.to_effective(p)
            L = siren.num_matrices(p)
            r = {"psnr": psnr(render_eff(eff, clip=True), X[i]), "H": float(programs.network_harmonic_index(eff)) if L > 2 else 0.0}
            base = render_eff(eff)
            # blur 1 px and box 4 px, function-space targets
            tb = fine_target(eff, lambda F, up: ndimage.gaussian_filter(F, up, mode="nearest"))
            tx = fine_target(eff, lambda F, up: ndimage.uniform_filter(F, size=4 * up + 1, mode="nearest"))
            Gb, Gx = programs.gaussian_transfer(1.0, SIDE), programs.box_transfer(4 * (4 * 8 + 1) / (4 * 8), 4 * (4 * 8 + 1) / (4 * 8), SIDE)
            if L == 2:
                eb, ex = programs.lsi_filter(eff, Gb), programs.lsi_filter(eff, Gx)
            else:
                eb = programs.input_attenuation_blur(eff, 1.0, SIDE)
                ex = {k: np.array(v) for k, v in eff.items()}
                g = Gx(eff["A1"].T)
                ex["A2"] = eff["A2"] * np.abs(g)[:, None]
                ex["c1"] = eff["c1"] + np.angle(g)
            r["blur_ratio"] = float(np.mean((render_eff(eb) - tb) ** 2) / max(np.mean((base - tb) ** 2), 1e-12))
            r["box_ratio"] = float(np.mean((render_eff(ex) - tx) ** 2) / max(np.mean((base - tx) ** 2), 1e-12))
            # anti-aliasing x4
            truth, centres = footprint_truth(eff, 4)
            Hb = programs.box_transfer(4, 4, SIDE)
            if L == 2:
                ep = programs.lsi_filter(eff, Hb)
            else:
                ep = {k: np.array(v) for k, v in eff.items()}
                g = Hb(eff["A1"].T)
                ep["A2"] = eff["A2"] * np.abs(g)[:, None]
                ep["c1"] = eff["c1"] + np.angle(g)
            r["aa_point"] = psnr(sample(eff, centres, 1, 4), truth)
            r["aa_program"] = psnr(sample(ep, centres, 1, 4), truth)
            r["aa_ss2"] = psnr(sample(eff, centres, 2, 4), truth)
            r["aa_ss4"] = psnr(sample(eff, centres, 4, 4), truth)
            rows.append(r)
        med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        results[name] = {"params": siren.count_params(params[0]), "median": med, "rows": rows, "seconds": time.time() - t0}
        print(f"{name:16s} params {results[name]['params']:5d} " + " ".join(f"{k} {v:.3g}" for k, v in med.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
