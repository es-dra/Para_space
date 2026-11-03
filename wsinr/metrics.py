from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = np.asarray(a, np.float64).squeeze()
    b = np.asarray(b, np.float64).squeeze()
    if mask is None:
        return float(np.mean((a - b) ** 2))
    m = np.asarray(mask, bool).squeeze()
    return float(np.mean((a[m] - b[m]) ** 2))


def psnr(a, b, mask=None, peak: float = 1.0) -> float:
    e = mse(a, b, mask)
    return float(10 * np.log10(peak ** 2 / max(e, 1e-12)))


def param_change(orig: dict, new: dict):
    """Fraction of entries changed (beyond float32 rounding) and relative L2 change."""
    n_tot, n_chg, num, den = 0, 0, 0.0, 0.0
    for k in orig:
        a = np.asarray(orig[k], np.float64)
        b = np.asarray(new[k], np.float64)
        n_tot += a.size
        n_chg += int(np.sum(np.abs(a - b) > 1e-6 * (1 + np.abs(a))))
        num += float(np.sum((a - b) ** 2))
        den += float(np.sum(a ** 2))
    return {"frac_changed": n_chg / max(n_tot, 1), "rel_l2": float(np.sqrt(num / max(den, 1e-12)))}
