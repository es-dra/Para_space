"""
Image quality metrics for INR experiments.

Provides PSNR, SSIM, MSE and other metrics for evaluating
image reconstruction quality.
"""

from typing import Dict, List, Union

import numpy as np
import torch


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two images.

    Args:
        img1: First image
        img2: Second image

    Returns:
        MSE value
    """
    return np.mean((img1 - img2) ** 2)


def psnr(
    img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        img1: First image
        img2: Second image
        max_val: Maximum possible pixel value (1.0 for normalized images)

    Returns:
        PSNR in dB
    """
    mse_val = np.mean((img1 - img2) ** 2)
    if mse_val == 0:
        return float("inf")
    return 20 * np.log10(max_val / np.sqrt(mse_val))


def ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 1.0,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute Structural Similarity Index between two images.

    Simplified implementation without external dependencies.
    For full SSIM, use skimage.metrics.structural_similarity.

    Args:
        img1: First image
        img2: Second image
        max_val: Maximum possible pixel value
        window_size: Size of moving window
        k1, k2: SSIM stability constants

    Returns:
        SSIM value in [-1, 1]
    """
    from scipy.ndimage import gaussian_filter

    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2

    # Mean
    mu1 = gaussian_filter(img1, window_size / 3)
    mu2 = gaussian_filter(img2, window_size / 3)

    # Variance and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 ** 2, window_size / 3) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, window_size / 3) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, window_size / 3) - mu1_mu2

    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(np.mean(ssim_map))


def compute_psnr_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Compute PSNR between two torch tensors.

    Args:
        img1: First image tensor (C, H, W) or (H, W)
        img2: Second image tensor (C, H, W) or (H, W)
        max_val: Maximum pixel value

    Returns:
        PSNR in dB
    """
    mse_val = torch.mean((img1 - img2) ** 2).item()
    if mse_val == 0:
        return float("inf")
    return 20 * np.log10(max_val / np.sqrt(mse_val))


def compute_msssim(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 1.0,
) -> float:
    """
    Compute Multi-Scale Structural Similarity Index (MS-SSIM).

    Simplified single-scale version.

    Args:
        img1: First image
        img2: Second image
        max_val: Maximum pixel value

    Returns:
        MS-SSIM value
    """
    # Use weighted combination of SSIM at different scales
    from scipy.ndimage import gaussian_filter

    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    # Luminance term
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssims = []
    for scale in [1, 2, 4]:
        if scale > 1:
            from scipy.ndimage import zoom

            img1_s = zoom(img1, 1 / scale, order=1)
            img2_s = zoom(img2, 1 / scale, order=1)
        else:
            img1_s, img2_s = img1, img2

        mu1 = gaussian_filter(img1_s, 7)
        mu2 = gaussian_filter(img2_s, 7)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1 = gaussian_filter(img1_s ** 2, 7) - mu1_sq
        sigma2 = gaussian_filter(img2_s ** 2, 7) - mu2_sq
        sigma12 = gaussian_filter(img1_s * img2_s, 7) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1 + sigma2 + C2)
        )
        ssims.append(np.mean(ssim_map))

    # Weighted product (simplified)
    return float(np.prod(np.array(ssims) ** (1 / len(ssims))))


def evaluate_reconstruction(
    original: Union[np.ndarray, torch.Tensor],
    reconstructed: Union[np.ndarray, torch.Tensor],
    metrics: List[str] = ["psnr", "ssim", "mse"],
) -> Dict[str, float]:
    """
    Evaluate reconstruction quality with multiple metrics.

    Args:
        original: Ground truth image
        reconstructed: Reconstructed image
        metrics: List of metrics to compute ('psnr', 'ssim', 'mse')

    Returns:
        Dictionary of metric_name -> value
    """
    # Convert torch tensors to numpy
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # Handle different shapes
    if original.ndim == 3 and original.shape[0] in [1, 3]:
        # (C, H, W) -> (H, W, C)
        original = np.transpose(original, (1, 2, 0))
    if reconstructed.ndim == 3 and reconstructed.shape[0] in [1, 3]:
        reconstructed = np.transpose(reconstructed, (1, 2, 0))

    # Squeeze extra dimensions
    original = np.squeeze(original)
    reconstructed = np.squeeze(reconstructed)

    results = {}
    for metric in metrics:
        if metric == "psnr":
            results["psnr"] = psnr(original, reconstructed)
        elif metric == "ssim":
            try:
                results["ssim"] = ssim(original, reconstructed)
            except ImportError:
                results["ssim"] = ssim_simple(original, reconstructed)
        elif metric == "mse":
            results["mse"] = mse(original, reconstructed)

    return results


def ssim_simple(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 1.0,
) -> float:
    """
    Simple SSIM implementation without scipy dependency.

    Args:
        img1: First image
        img2: Second image
        max_val: Maximum pixel value

    Returns:
        SSIM value
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Mean
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    # Variance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)

    # Covariance
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return numerator / denominator


def image_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch image tensor to numpy array.

    Args:
        tensor: Shape (C, H, W) or (B, C, H, W) or (H, W)

    Returns:
        numpy array of shape (H, W, C) or (H, W)
    """
    if tensor.ndim == 4:
        # (B, C, H, W) -> take first batch
        tensor = tensor[0]
    if tensor.ndim == 3:
        # (C, H, W) -> (H, W, C)
        if tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
    return tensor.detach().cpu().numpy()


def numpy_to_image_tensor(
    array: np.ndarray, normalize: bool = True
) -> torch.Tensor:
    """
    Convert numpy image array to torch tensor.

    Args:
        array: Shape (H, W, C) or (H, W)
        normalize: Whether to normalize to [0, 1]

    Returns:
        torch.Tensor of shape (C, H, W)
    """
    if array.ndim == 2:
        # (H, W) -> (1, H, W)
        tensor = torch.from_numpy(array).unsqueeze(0)
    else:
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(array).permute(2, 0, 1)

    if normalize:
        tensor = tensor.float() / 255.0 if tensor.max() > 1.0 else tensor.float()

    return tensor
