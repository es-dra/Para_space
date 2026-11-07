"""Stage 08: prefiltered downsampling by weight edit vs point sampling and supersampling."""
import argparse
import glob
import json
import os

import numpy as np

from wsinr import programs, siren


def footprint_truth(eff, factor, side=28, up=8):
    """Mean of the continuous function over each output pixel footprint (factor x factor fine-grid pixels),
    computed from an up x oversampled rendering of the 28-grid domain. Output pixel centres sit at the centres
    of factor x factor blocks of the 28-grid pixels; the domain is treated as the [-1, 1] square with 28 pixels
    of width 2/27 (pixel-centre convention), extended by half a pixel at the borders."""
    step = 2.0 / (side - 1)
    n_out = side // factor
    # fine samples covering the output image area: n_out*factor pixels, up samples each
    t = -1 - step / 2 + (np.arange(n_out * factor * up) + 0.5) * step / up
    yy, xx = np.meshgrid(t, t, indexing="ij")
    F = siren.forward_effective(eff, np.stack([yy.ravel(), xx.ravel()], 1)).reshape(len(t), len(t))
    blk = factor * up
    truth = F.reshape(n_out, blk, n_out, blk).mean(axis=(1, 3))
    centres = -1 - step / 2 + (np.arange(n_out) + 0.5) * factor * step
    return truth, centres


def sample(eff, centres, k=1, factor=1, side=28):
    """k x k point samples per output pixel (k = 1: the centre), averaged."""
    step = 2.0 / (side - 1)
    offs = (np.arange(k) + 0.5) / k - 0.5
    vals = np.zeros((len(centres), len(centres)))
    for oy in offs:
        for ox in offs:
            yy, xx = np.meshgrid(centres + oy * factor * step, centres + ox * factor * step, indexing="ij")
            vals += siren.forward_effective(eff, np.stack([yy.ravel(), xx.ravel()], 1)).reshape(len(centres), len(centres))
    return vals / (k * k)


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--factors", nargs="*", type=int, default=[2, 4, 7])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.zoo, "*.npz")))[: a.n]
    res = {f: {m: [] for m in ("point", "program", "ss2", "ss4")} for f in a.factors}
    for fpath in files:
        p, _ = siren.load_npz(fpath)
        eff = siren.to_effective(p)
        L = siren.num_matrices(p)
        for factor in a.factors:
            truth, centres = footprint_truth(eff, factor)
            H = programs.box_transfer(factor, factor)
            if L == 2:
                eff_p = programs.lsi_filter(eff, H)
            else:                                        # first-order: attenuate the dictionary's input waves
                eff_p = dict(eff)
                g = H(eff["A1"].T)
                eff_p["A2"] = eff["A2"] * np.abs(g)[:, None]
                eff_p = {k: np.array(v) for k, v in eff_p.items()}
                eff_p["c1"] = eff["c1"] + np.angle(g)
            res[factor]["point"].append(psnr(sample(eff, centres, 1, factor), truth))
            res[factor]["program"].append(psnr(sample(eff_p, centres, 1, factor), truth))
            res[factor]["ss2"].append(psnr(sample(eff, centres, 2, factor), truth))
            res[factor]["ss4"].append(psnr(sample(eff, centres, 4, factor), truth))
    summary = {str(f): {m: {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}
                        for m, v in d.items()} for f, d in res.items()}
    for f, d in summary.items():
        print(f"{a.tag} x{f}: " + "  ".join(f"{m} {v['median']:.1f} dB" for m, v in d.items()))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"zoo": a.zoo, "tag": a.tag, "n": len(files), "summary": summary,
               "per_network": {str(f): d for f, d in res.items()}}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
