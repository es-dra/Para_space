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


def input_attenuation_blur(eff: dict, sigma_pix: float, side: int = 28) -> dict:
    """Blur by attenuating the plane-wave inputs of the first hidden sine layer:
    A2[i, :] *= exp(-|w_i|^2 sigma^2 / 2).

    Derivation: z_j = c_j + sum_i a_ij sin(theta_i) with theta_i = w_i . p + c_i. To first order in the
    swing of z_j about c_j, sin(z_j) = sin(c_j) + cos(c_j) sum_i a_ij sin(theta_i), a linear combination of
    plane waves, and the Gaussian blur of a plane wave of frequency w_i is the same wave times
    g_i = exp(-|w_i|^2 sigma^2 / 2). Replacing a_ij by a_ij g_i therefore blurs the network exactly in the
    linear regime; the error is the part of sin(z_j) carried by harmonics (controlled by the harmonic index
    sum_i |a_ij|). For L = 2 this is the exact shallow program on A2 (the readout)."""
    out = _copy(eff)
    sigma = sigma_pix * 2.0 / (side - 1)
    g = np.exp(-0.5 * np.sum(eff["A1"] ** 2, axis=0) * sigma ** 2)
    out["A2"] = eff["A2"] * g[:, None]
    return out


# ----------------------------------------------------------------------------- LSI filters on shallow SIRENs
def lsi_filter(eff: dict, transfer) -> dict:
    """Any linear shift-invariant filter on a one-sine-layer SIREN, exactly.

    Each hidden unit is a single plane wave sin(w_i . p + c_i). A real LSI filter with (complex) frequency
    response G(w) maps it to |G(w_i)| sin(w_i . p + c_i + arg G(w_i)). So the program is
        A2_i *= |G(w_i)|,   c1_i += arg G(w_i),   c2 *= G(0) (the DC gain).
    `transfer(W)` takes an array of frequency vectors (H, 2) in the (y, x) order of A1's rows and returns
    complex gains (H,). Requires L = 2; returns None otherwise. Coordinates are in the [-1, 1] units of the
    network; convert pixel sizes with 2 / (side - 1)."""
    if _last(eff) != 2:
        return None
    out = _copy(eff)
    W = eff["A1"].T                                          # (H, 2): rows are (w_y, w_x)
    G = np.asarray(transfer(W), dtype=complex)
    out["A2"] = eff["A2"] * np.abs(G)[:, None]
    out["c1"] = eff["c1"] + np.angle(G)
    G0 = complex(np.asarray(transfer(np.zeros((1, 2))))[0])
    out["c2"] = eff["c2"] * G0.real
    return out


def _pix(side):
    return 2.0 / (side - 1)


def gaussian_transfer(sigma_pix, side=28, aniso=None):
    """Isotropic Gaussian (or anisotropic with aniso = (sigma_y, sigma_x) in pixels)."""
    if aniso is None:
        s = sigma_pix * _pix(side)
        return lambda W: np.exp(-0.5 * (W ** 2).sum(1) * s ** 2)
    sy, sx = aniso[0] * _pix(side), aniso[1] * _pix(side)
    return lambda W: np.exp(-0.5 * ((W[:, 0] * sy) ** 2 + (W[:, 1] * sx) ** 2))


def box_transfer(cy_pix, cx_pix, side=28):
    """Mean over a rectangular footprint of cy x cx pixels (the pixel-footprint integral of arbitrary-scale
    SR decoders): G(w) = sinc(w_y cy / 2) sinc(w_x cx / 2) with sinc(t) = sin(t)/t."""
    cy, cx = cy_pix * _pix(side), cx_pix * _pix(side)
    return lambda W: np.sinc(W[:, 0] * cy / (2 * np.pi)) * np.sinc(W[:, 1] * cx / (2 * np.pi))


def motion_transfer(length_pix, direction=(0.0, 1.0), side=28):
    """Uniform motion blur of `length_pix` along `direction` (y, x): G(w) = sinc(w . d L / 2)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d) * length_pix * _pix(side)
    return lambda W: np.sinc((W @ d) / (2 * np.pi))


def derivative_transfer(direction=(0.0, 1.0), side=28):
    """Directional derivative in units of grey level per pixel: G(w) = i (w . d) * pixel."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d) * _pix(side)
    return lambda W: 1j * (W @ d)


def laplacian_transfer(side=28):
    """Laplacian in grey level per pixel^2: G(w) = -|w|^2 pixel^2."""
    s = _pix(side)
    return lambda W: -(W ** 2).sum(1) * s ** 2


def unsharp_transfer(sigma_pix, amount, side=28):
    """f + amount (f - gaussian_sigma * f)."""
    g = gaussian_transfer(sigma_pix, side)
    return lambda W: 1.0 + amount * (1.0 - g(W))


def dog_transfer(sigma1_pix, sigma2_pix, side=28):
    """Difference of Gaussians (band-pass)."""
    g1, g2 = gaussian_transfer(sigma1_pix, side), gaussian_transfer(sigma2_pix, side)
    return lambda W: g1(W) - g2(W)
