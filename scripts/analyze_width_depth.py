"""Stage 03: fit quality vs parameter count for shallow (exactly programmable) and deep SIREN families,
with blur programmability on the function-space target. Scores W1-W5 of studies/03-width-vs-depth/protocol.json.

  python scripts/analyze_width_depth.py --out studies/03-width-vs-depth/outcomes.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import ndimage

from wsinr import oracle, programs, siren
from wsinr.metrics import mse
from wsinr.render import render

ZOOS = {  # tag: (zoo dir, family)
    "L2w32": ("zoo/mnist_shallow32", "shallow"), "L2w64": ("zoo/mnist_shallow64", "shallow"),
    "L2w128": ("zoo/mnist_shallow128", "shallow"), "L2w256": ("zoo/mnist_shallow256", "shallow"),
    "L2w384": ("zoo/mnist_shallow384", "shallow"), "L2w1024": ("zoo/mnist_shallow1024", "shallow"),
    "L3w8": ("zoo/mnist_d3_s8", "deep"), "L3w16": ("zoo/mnist_d3_s16", "deep"), "L3w32": ("zoo/mnist_s32", "deep"),
    "L4w16": ("zoo/mnist_d4_s16", "deep"), "L4w32": ("zoo/mnist_d4_s32", "deep"),
    "L2w384_long": ("zoo/mnist_shallow384_long", "control"),
}


def fine_blur(eff, sigma_pix=1.0, side=28, up=8):
    t = np.linspace(-1, 1, side)
    step = t[1] - t[0]
    pad = 6 * step
    tf = np.arange(-1 - pad, 1 + pad + 1e-9, step / up)
    yy, xx = np.meshgrid(tf, tf, indexing="ij")
    F = siren.forward_effective(eff, np.stack([yy.ravel(), xx.ravel()], 1)).reshape(len(tf), len(tf))
    Fb = ndimage.gaussian_filter(F, sigma_pix * up, mode="nearest")
    idx = np.round((t - tf[0]) / (step / up)).astype(int)
    return Fb[np.ix_(idx, idx)]


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    v = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}
    for tag, (zdir, fam) in ZOOS.items():
        files = sorted(glob.glob(os.path.join(zdir, "*.npz")))[: a.n]
        if not files:
            continue
        psnr, prog_ratio = [], []
        for f in files:
            p, meta = siren.load_npz(f)
            psnr.append(10 * np.log10(1 / max(float(meta["fit_mse"]), 1e-12)))
            eff = siren.to_effective(p)
            tgt = fine_blur(eff)
            base = render(p, 28, clip=False)[..., 0]
            e2 = oracle.lowpass_gauss(eff, 1.0) if siren.num_matrices(p) == 2 else programs.input_attenuation_blur(eff, 1.0)
            pred = render(siren.from_effective(e2), 28, clip=False)[..., 0]
            prog_ratio.append(mse(pred, tgt) / max(mse(base, tgt), 1e-12))
            prog_abs = mse(pred, tgt)
        rows[tag] = {"family": fam, "n": len(files), "params": siren.count_params(p), "psnr_median": float(np.median(psnr)),
                     "psnr_ci": boot_ci(psnr), "blur_ratio_median": float(np.median(prog_ratio)),
                     "blur_ratio_max": float(np.max(prog_ratio)), "blur_abs_last": prog_abs,
                     "steps": int(meta.get("steps", 0))}
        print(f"{tag:12s} {fam:8s} params {rows[tag]['params']:5d} PSNR {rows[tag]['psnr_median']:5.1f} {rows[tag]['psnr_ci']}  blur ratio {rows[tag]['blur_ratio_median']:.3f}")
    P = {}
    g = lambda t: rows.get(t, {}).get("psnr_median")
    if g("L3w32") and g("L2w256") and g("L2w384"):
        P["W1_gap_at_matched_params"] = {"deep": g("L3w32"), "shallow256": g("L2w256"), "shallow384": g("L2w384"),
                                         "pass": g("L3w32") - max(g("L2w256"), g("L2w384")) > 25}
    if g("L2w256") and g("L2w1024"):
        per_doubling = (g("L2w1024") - g("L2w256")) / 2
        P["W2_shallow_saturates"] = {"gain_per_doubling_256_to_1024": per_doubling, "shallow1024": g("L2w1024"),
                                     "pass": per_doubling < 4 and g("L2w1024") < 32}
    if g("L2w384") and g("L2w384_long"):
        P["W3_optimisation_is_not_the_limit"] = {"2k": g("L2w384"), "10k": g("L2w384_long"), "pass": g("L2w384_long") - g("L2w384") < 1}
    if g("L3w16") and g("L2w1024"):
        P["W4_deep_small_beats_shallow_large"] = {"L3w16": g("L3w16"), "L2w1024": g("L2w1024"), "pass": g("L3w16") > g("L2w1024")}
    sh = [r for t, r in rows.items() if r["family"] == "shallow"]
    dp = [r for t, r in rows.items() if r["family"] == "deep"]
    if sh and dp:
        P["W5_programmability_unchanged"] = {"shallow_max_blur_abs": max(r["blur_abs_last"] for r in sh),
                                             "shallow_all_exact": all(r["blur_ratio_max"] < 1e-3 for r in sh),
                                             "deep_min_ratio": min(r["blur_ratio_median"] for r in dp),
                                             "pass": all(r["blur_ratio_max"] < 1e-3 for r in sh) and min(r["blur_ratio_median"] for r in dp) >= 0.7}
    json.dump({"rows": rows, "predictions": P}, open(a.out, "w"), indent=1)
    print(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
