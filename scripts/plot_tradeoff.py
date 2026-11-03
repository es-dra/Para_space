"""Fit quality vs blur programmability across training regimes and depths (stage 04 figure).

  python scripts/plot_tradeoff.py --out studies/04-editable-training/tradeoff.png
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"standard, depth 3": "C0", "standard, depth 4": "C1", "shallow (exact)": "C2",
          "depth 3, swing bound": "C3", "depth 3, L1 penalty": "C4"}
SERIES = {
    "standard, depth 3": [("L3w8", "mnist_d3_s8"), ("L3w16", "mnist_d3_s16"), ("L3w32", "mnist_s32")],
    "standard, depth 4": [("L4w16", "mnist_d4_s16"), ("L4w32", "mnist_d4_s32")],
    "shallow (exact)": [("L2w32", "mnist_shallow32"), ("L2w64", "mnist_shallow64"), ("L2w128", "mnist_shallow128"),
                        ("L2w256", "mnist_shallow256"), ("L2w384", "mnist_shallow384"), ("L2w1024", "mnist_shallow1024")],
    "depth 3, swing bound": [("L3w32_swing0.5", "mnist_s32_swing0.5"), ("L3w32_swing1", "mnist_s32_swing1"),
                             ("L3w32_swing2", "mnist_s32_swing2"), ("L3w32_swing4", "mnist_s32_swing4")],
    "depth 3, L1 penalty": [("L3w32_lam1e-2", "mnist_s32_lam1e-2"), ("L3w32_lam1e-3", "mnist_s32_lam1e-3")],
}


def load(tag, zoo):
    f = f"results/programs_{tag}.json"
    idx = json.load(open(f"zoo/{zoo}/index.json"))
    psnr = float(np.median([r["psnr"] for r in idx["rows"][:20]]))
    if not os.path.exists(f):
        return psnr, None, None
    r = json.load(open(f))
    b = r["tasks"]["blur_1_fn"]
    prog = "exact" if "exact" in b else "input_attenuation"
    ratio = float(np.median(np.array(b[prog]) / np.maximum(np.array(b["no_edit"]), 1e-12)))
    H = float(np.median(r["harmonic_index"]))
    return psnr, ratio, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    rows = {}
    for name, items in SERIES.items():
        pts = []
        for tag, zoo in items:
            if not os.path.isdir(f"zoo/{zoo}"):
                continue
            psnr, ratio, H = load(tag, zoo)
            rows[tag] = {"series": name, "psnr": psnr, "blur_ratio": ratio, "H": H}
            if ratio is not None:
                pts.append((psnr, max(ratio, 1e-4), H, tag))
        if not pts:
            continue
        pts.sort()
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=name, color=COLORS[name])
        for p in pts:
            axes[0].annotate(p[3].replace("L3w32_", "").replace("mnist_", ""), (p[0], p[1]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        if name != "shallow (exact)":
            axes[1].plot([p[2] for p in pts], [p[1] for p in pts], "o", label=name, color=COLORS[name])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("fit PSNR (dB, median of 20 networks)")
    axes[0].set_ylabel("blur program error / no-edit error (function-space, median; floor 1e-4)")
    axes[0].axhline(1.0, color="k", lw=0.5, ls="--")
    axes[0].set_title("Programmability against fit quality")
    axes[0].legend(fontsize=7)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("network harmonic index H (median)")
    axes[1].set_ylabel("blur program error / no-edit error")
    axes[1].axhline(1.0, color="k", lw=0.5, ls="--")
    axes[1].set_title("Failure follows the harmonic index")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    json.dump(rows, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
