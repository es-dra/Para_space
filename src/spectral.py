"""
Frequency-domain analysis utilities for INR fitting dynamics.

Provides FFT-based tools for computing frequency spectra of images and
reconstruction errors. Used to track spectral bias — the phenomenon where
neural networks fit low frequencies first, then high frequencies.

All functions work with numpy arrays on CPU (no torch dependency).
"""

from typing import Optional

import numpy as np


def compute_frequency_spectrum(
    image: np.ndarray,
    n_freq_bins: int = 8,
) -> np.ndarray:
    """Compute radially-averaged frequency spectrum of a 2D image.

    Divides the 2D FFT power spectrum into concentric ring bins and
    returns the fraction of total energy in each bin. Bin 0 = lowest
    frequencies (DC neighborhood), bin N-1 = highest frequencies.

    Args:
        image: Input image of shape (C, H, W). Each channel is
               processed independently and results are averaged.
        n_freq_bins: Number of radial frequency bins.

    Returns:
        Array of shape (n_freq_bins,) with normalized energy per bin.
    """
    C, H, W = image.shape
    cy, cx = H // 2, W // 2
    max_radius = min(cy, cx)

    # Precompute radius mask grid (shared across channels)
    y, x = np.ogrid[:H, :W]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Bin boundaries
    bin_edges = np.linspace(0, max_radius, n_freq_bins + 1)

    channel_spectra = []
    for c in range(C):
        fft = np.fft.fft2(image[c])
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted) ** 2

        freq_energies = []
        for b in range(n_freq_bins):
            r_inner = bin_edges[b]
            r_outer = bin_edges[b + 1]
            mask = (radius >= r_inner) & (radius < r_outer)
            freq_energies.append(np.sum(magnitude[mask]))

        total = sum(freq_energies) + 1e-10
        channel_spectra.append([e / total for e in freq_energies])

    return np.mean(channel_spectra, axis=0)


def compute_error_spectrum(
    target: np.ndarray,
    reconstruction: np.ndarray,
    n_freq_bins: int = 8,
) -> np.ndarray:
    """Compute frequency spectrum of the reconstruction error.

    Same method as compute_frequency_spectrum but on the error map
    (target - reconstruction). This reveals which frequency bands
    the network has not yet learned to represent.

    Args:
        target: Ground truth image, shape (C, H, W).
        reconstruction: Reconstructed image, shape (C, H, W).
        n_freq_bins: Number of radial frequency bins.

    Returns:
        Array of shape (n_freq_bins,) with normalized energy of error.
    """
    error = (target - reconstruction).astype(np.float64)
    return compute_frequency_spectrum(error, n_freq_bins)


def compute_low_freq_ratio(spectrum: np.ndarray) -> float:
    """Fraction of energy in the lower half of frequency bins.

    A value close to 1.0 means error is concentrated in low frequencies
    (network hasn't learned basic structure). A value close to 0 means
    error is in high frequencies (network is refining details).

    Args:
        spectrum: Frequency spectrum array from compute_*_spectrum.

    Returns:
        Scalar in [0, 1].
    """
    n = len(spectrum)
    half = n // 2
    return float(np.sum(spectrum[:half]) / (np.sum(spectrum) + 1e-10))


def decompose_error_by_band(
    target: np.ndarray,
    reconstruction: np.ndarray,
    n_freq_bins: int = 8,
) -> dict:
    """Full spectral bias decomposition of reconstruction error.

    Returns a dict with per-band energy fractions and the low-freq ratio,
    suitable for logging and visualization.

    Args:
        target: Ground truth image, shape (C, H, W).
        reconstruction: Reconstructed image, shape (C, H, W).
        n_freq_bins: Number of radial frequency bins.

    Returns:
        dict with keys:
          - 'per_band': list of energy fractions per band
          - 'low_freq_ratio': total low-frequency fraction
          - 'n_freq_bins': number of bins used
    """
    spectrum = compute_error_spectrum(target, reconstruction, n_freq_bins)
    return {
        "per_band": spectrum.tolist(),
        "low_freq_ratio": compute_low_freq_ratio(spectrum),
        "n_freq_bins": n_freq_bins,
    }
