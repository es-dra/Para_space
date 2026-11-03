"""Weight-space programs for spectrum-changing operations, and weight-only diagnostics.

All functions take and return the effective form (see siren.py). Nothing here evaluates the network.
"""
from __future__ import annotations

import numpy as np

from . import siren
from .oracle import _copy, _last


def unit_rms_frequency(eff: dict) -> list[np.ndarray]:
    """Mean-square spatial frequency nu^2 of every hidden unit, layer by layer, from the weights alone.

    Layer 1: z_i = w_i . p + c_i, so |grad z_i|^2 = |w_i|^2 exactly.
    Layer k > 1: z_j = sum_i a_ij sin(z_i^{k-1}) + c_j, so grad z_j = sum_i a_ij cos(z_i) grad z_i.
    Treating the cosines as uncorrelated with mean square 1/2 gives E|grad z_j|^2 = 1/2 sum_i a_ij^2 E|grad z_i|^2.
    Returns one array per sine layer (nu^2 per unit)."""
    L = _last(eff)
    nu2 = [np.sum(eff["A1"] ** 2, axis=0)]
    for k in range(2, L):
        nu2.append(0.5 * (eff[f"A{k}"] ** 2).T @ nu2[-1])
    return nu2


def harmonic_index(eff: dict) -> np.ndarray:
    """Per unit of the last sine layer: sum_i |A_ij| (plus nothing for the bias), the amplitude of the
    argument of its sine. Below ~1 rad the unit is close to a single harmonic; several radians means a
    strongly harmonic mixture. For L = 2 every unit is a pure plane wave and the index is 0."""
    L = _last(eff)
    if L == 2:
        return np.zeros(eff["A1"].shape[1])
    return np.sum(np.abs(eff[f"A{L - 1}"]), axis=0)


def network_harmonic_index(eff: dict) -> float:
    """Readout-weighted mean of the per-unit harmonic index."""
    L = _last(eff)
    h = harmonic_index(eff)
    w = np.abs(eff[f"A{L}"][:, 0])
    return float(np.sum(w * h) / max(np.sum(w), 1e-12))


def rms_attenuation_blur(eff: dict, sigma_pix: float, side: int = 28) -> dict:
    """Gaussian blur as a per-unit attenuation of the readout by exp(-nu_j^2 sigma^2 / 2), nu_j from the
    weights. Exact for L = 2 (identical to oracle.lowpass_gauss); a heuristic for deeper networks."""
    out = _copy(eff)
    L = _last(eff)
    sigma = sigma_pix * 2.0 / (side - 1)
    nu2 = unit_rms_frequency(eff)[-1]
    out[f"A{L}"] = eff[f"A{L}"] * np.exp(-0.5 * nu2 * sigma ** 2)[:, None]
    return out


def rms_highpass(eff: dict, sigma_pix: float, side: int = 28) -> dict:
    """f - blur(f): readout rows scaled by 1 - exp(-nu^2 sigma^2 / 2); the constant term goes to zero."""
    out = _copy(eff)
    L = _last(eff)
    sigma = sigma_pix * 2.0 / (side - 1)
    nu2 = unit_rms_frequency(eff)[-1]
    g = 1.0 - np.exp(-0.5 * nu2 * sigma ** 2)
    out[f"A{L}"] = eff[f"A{L}"] * g[:, None]
    out[f"c{L}"] = np.zeros_like(eff[f"c{L}"])
    return out
