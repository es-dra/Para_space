"""Why does per-unit attenuation fail on deep SIRENs? Per hidden unit of the last sine layer, compare the
weight-only estimates with the truth measured from the unit's response (analysis, not the capability protocol):

  nu2_pred : 1/2 sum_i a_ij^2 |w_i|^2            (weight-only mean-square frequency)
  nu2_true : energy-weighted mean |k|^2 of the unit's 2-D spectrum on a fine grid
  spread   : the unit's spectral energy outside a band around nu_true (how far from a single wave it is)
  H_j      : harmonic index sum_i |a_ij|
  err_j    : per-unit blur error of the attenuation rule, ||blur(h_j) - g(nu_pred) h_j||^2 / ||blur(h_j)||^2

  python scripts/unit_spectra.py --zoo zoo/mnist_s32 --n 20 --sigma 1 --out results/unit_spectra_L3w32.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import ndimage, stats

from wsinr import programs, siren


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--side", type=int, default=28)
    ap.add_argument("--up", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.zoo, "*.npz")))[: a.n]
    side, up = a.side, a.up
    t = np.linspace(-1, 1, side)
    step = t[1] - t[0]
    tf = np.arange(-1, 1 + 1e-9, step / up)           # fine grid on the domain, spacing step/up
    N = len(tf)
    yy, xx = np.meshgrid(tf, tf, indexing="ij")
    coords = np.stack([yy.ravel(), xx.ravel()], 1)
    # angular frequency grid for the FFT (coordinates in [-1, 1]; wave sin(w . p) has |k| = |w|)
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=step / up)
    KY, KX = np.meshgrid(k1, k1, indexing="ij")
    K2 = KY ** 2 + KX ** 2
    sig = a.sigma * step                                  # sigma in coordinate units
    rows = []
    for f in files:
        p, meta = siren.load_npz(f)
        eff = siren.to_effective(p)
        L = siren.num_matrices(p)
        h = coords
        for k in range(1, L):
            h = np.sin(h @ eff[f"A{k}"] + eff[f"c{k}"])
        H = h.reshape(N, N, -1)                            # last sine layer activations
        nu2_pred = programs.unit_rms_frequency(eff)[-1]
        Hidx = programs.harmonic_index(eff)
        g_pred = np.exp(-0.5 * nu2_pred * sig ** 2)
        for j in range(H.shape[2]):
            u = H[:, :, j] - H[:, :, j].mean()
            U = np.fft.fft2(u)
            P = np.abs(U) ** 2
            P[0, 0] = 0
            tot = P.sum() + 1e-12
            nu2_true = float((P * K2).sum() / tot)
            band = (np.abs(np.sqrt(K2) - np.sqrt(nu2_true)) < 0.2 * np.sqrt(nu2_true) + 1e-9)
            spread = float(1 - P[band].sum() / tot)
            ub = ndimage.gaussian_filter(H[:, :, j], a.sigma * up, mode="nearest")
            err = float(np.sum((ub - g_pred[j] * H[:, :, j]) ** 2) / (np.sum(ub ** 2) + 1e-12))
            # best possible per-unit scalar (oracle attenuation for this unit)
            s_best = float(np.sum(ub * H[:, :, j]) / (np.sum(H[:, :, j] ** 2) + 1e-12))
            err_best = float(np.sum((ub - s_best * H[:, :, j]) ** 2) / (np.sum(ub ** 2) + 1e-12))
            rows.append({"net": os.path.basename(f), "unit": j, "nu2_pred": float(nu2_pred[j]), "nu2_true": nu2_true,
                         "H": float(Hidx[j]), "spread": spread, "err_rule": err, "err_best_scalar": err_best,
                         "readout_w": float(abs(eff[f"A{L}"][j, 0]))})
    R = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("net",)}
    summary = {
        "n_units": len(rows),
        "spearman_nu2_pred_vs_true": float(stats.spearmanr(R["nu2_pred"], R["nu2_true"])[0]),
        "median_ratio_nu2_true_over_pred": float(np.median(R["nu2_true"] / np.maximum(R["nu2_pred"], 1e-12))),
        "median_spread": float(np.median(R["spread"])),
        "median_err_rule": float(np.median(R["err_rule"])),
        "median_err_best_scalar": float(np.median(R["err_best_scalar"])),
        "spearman_H_vs_err_rule": float(stats.spearmanr(R["H"], R["err_rule"])[0]),
        "spearman_H_vs_spread": float(stats.spearmanr(R["H"], R["spread"])[0]),
        "spearman_spread_vs_err_best_scalar": float(stats.spearmanr(R["spread"], R["err_best_scalar"])[0]),
        "H_quantiles": [float(x) for x in np.percentile(R["H"], [5, 25, 50, 75, 95])],
    }
    json.dump({"zoo": a.zoo, "sigma": a.sigma, "summary": summary, "units": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
