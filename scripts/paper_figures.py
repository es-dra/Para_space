"""Paper figures from the archived study results.

  python scripts/paper_figures.py --out paper/figures
Produces: fig_width_depth.pdf (fit PSNR vs weights, both families, two datasets), fig_qualitative.pdf
(blur program on a shallow and on a deep network next to the function-space target).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from wsinr import oracle, programs, siren
from wsinr.render import render


def fig_width_depth(out):
    rows = json.load(open("studies/03-width-vs-depth/outcomes.json"))["rows"]
    fam = {"shallow": ("one sine layer + linear readout (exactly programmable)", "C2", "o"),
           "deep": ("two or three sine layers (not programmable)", "C0", "s")}
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for f, (lab, col, mk) in fam.items():
        pts = sorted((r["params"], r["psnr_median"], r["psnr_ci"]) for r in rows.values() if r["family"] == f)
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        lo = [p[1] - p[2][0] for p in pts]
        hi = [p[2][1] - p[1] for p in pts]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=mk + "-", color=col, label=lab, capsize=2, ms=4)
    fr = json.load(open("studies/03-width-vs-depth/results/fashion_replication.json"))
    ax.plot([fr["fashion_shallow384"]["params"]], [fr["fashion_shallow384"]["psnr_median"]], "o", mfc="none", color="C2", label="Fashion-MNIST, shallow")
    ax.plot([fr["fashion_d3_s16"]["params"], fr["fashion_s32"]["params"]], [fr["fashion_d3_s16"]["psnr_median"], fr["fashion_s32"]["psnr_median"]], "s", mfc="none", color="C0", label="Fashion-MNIST, deep")
    ax.set_xscale("log")
    ax.set_xlabel("weights")
    ax.set_ylabel("fit PSNR (dB), median of 20 images")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_width_depth.pdf"))
    fig.savefig(os.path.join(out, "fig_width_depth.png"), dpi=160)


def _fine_blur(eff, side=28, up=8, sigma=1.0):
    t = np.linspace(-1, 1, side)
    step = t[1] - t[0]
    pad = 6 * step
    tf = np.arange(-1 - pad, 1 + pad + 1e-9, step / up)
    yy, xx = np.meshgrid(tf, tf, indexing="ij")
    F = siren.forward_effective(eff, np.stack([yy.ravel(), xx.ravel()], 1)).reshape(len(tf), len(tf))
    Fb = ndimage.gaussian_filter(F, sigma * up, mode="nearest")
    idx = np.round((t - tf[0]) / (step / up)).astype(int)
    return Fb[np.ix_(idx, idx)]


def fig_qualitative(out, k=2):
    cases = [("one sine layer, 2-64-1", sorted(glob.glob("zoo/mnist_shallow64/*.npz"))[k], True),
             ("bounded depth 3 (H <= 2)", sorted(glob.glob("zoo/mnist_s32_swing2/*.npz"))[k], False),
             ("standard depth 3, 2-32-32-1", sorted(glob.glob("zoo/mnist_s32/*.npz"))[k], False)]
    fig, axes = plt.subplots(len(cases), 4, figsize=(7.2, 1.9 * len(cases)))
    for r, (title, f, shallow) in enumerate(cases):
        p, _ = siren.load_npz(f)
        eff = siren.to_effective(p)
        base = render(p, clip=False)[..., 0]
        tgt = _fine_blur(eff)
        e2 = oracle.lowpass_gauss(eff, 1.0) if shallow else programs.input_attenuation_blur(eff, 1.0)
        pred = render(siren.from_effective(e2), clip=False)[..., 0]
        panels = [(base, "network"), (tgt, "target: blur sigma = 1 px"), (pred, "weight-space program"), (pred - tgt, "program - target")]
        for c, (img, lab) in enumerate(panels):
            ax = axes[r, c]
            if c < 3:
                ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
                ax.set_title(f"{lab}\nMSE {np.mean((pred - tgt) ** 2):.1e}", fontsize=7)
            if c < 3:
                ax.set_title(lab, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_qualitative.pdf"))
    fig.savefig(os.path.join(out, "fig_qualitative.png"), dpi=160)


def fig_antialias(out, k=3, factor=4):
    """Downsampling x4 of a one-sine-layer SIREN: point sampling, the footprint program, 4x4 supersampling, truth."""
    import importlib
    ea = importlib.import_module("scripts.eval_antialias") if False else None
    from wsinr import programs as P
    f = sorted(glob.glob("zoo/mnist_shallow64/*.npz"))[k]
    p, _ = siren.load_npz(f)
    eff = siren.to_effective(p)
    side, up = 28, 8
    step = 2.0 / (side - 1)
    n_out = side // factor
    t = -1 - step / 2 + (np.arange(n_out * factor * up) + 0.5) * step / up
    yy, xx = np.meshgrid(t, t, indexing="ij")
    F = siren.forward_effective(eff, np.stack([yy.ravel(), xx.ravel()], 1)).reshape(len(t), len(t))
    truth = F.reshape(n_out, factor * up, n_out, factor * up).mean(axis=(1, 3))
    centres = -1 - step / 2 + (np.arange(n_out) + 0.5) * factor * step

    def samp(e, kk):
        offs = (np.arange(kk) + 0.5) / kk - 0.5
        v = np.zeros((n_out, n_out))
        for oy in offs:
            for ox in offs:
                Y, X = np.meshgrid(centres + oy * factor * step, centres + ox * factor * step, indexing="ij")
                v += siren.forward_effective(e, np.stack([Y.ravel(), X.ravel()], 1)).reshape(n_out, n_out)
        return v / kk ** 2
    ep = P.lsi_filter(eff, P.box_transfer(factor, factor))
    panels = [(render(p, 28, clip=False)[..., 0], "network (28 x 28)"), (samp(eff, 1), "point sampling, 1 spp"),
              (samp(eff, 4), "4 x 4 supersampling, 16 spp"), (samp(ep, 1), "footprint program, 1 spp"), (truth, "footprint mean (truth)")]
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.4))
    for ax, (img, lab) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if img.shape == truth.shape and "truth" not in lab:
            lab += f"\n{10 * np.log10(1 / max(np.mean((img - truth) ** 2), 1e-12)):.0f} dB"
        ax.set_title(lab, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_antialias.pdf"))
    fig.savefig(os.path.join(out, "fig_antialias.png"), dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_width_depth(a.out)
    fig_qualitative(a.out)
    fig_antialias(a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
