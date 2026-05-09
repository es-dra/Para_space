#!/usr/bin/env python3
"""Stage-C pilot: local geometry vs function-response dynamics.

This is a read-only probe for existing Stage-B LIIF scratch outputs. It
reconstructs model snapshots from raw full-parameter trajectories, evaluates
function responses on the fitted image, and tests whether geometry-neighbor
patches have more similar response dynamics than shuffled controls.

Important: response evaluation uses raw full snapshots, not aligned decoder
snapshots. The current decoder is a ReLU MLP, so sign-flip alignment is not a
function-preserving transformation for response analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.Phase1_FittingDynamics.run import load_image  # noqa: E402
from experiments.config import LIIF_CONFIG, LIIF_CONFIG_REDUCED  # noqa: E402
from src.datasets import get_image_coordinates  # noqa: E402
from src.models.liif import LIIFModel  # noqa: E402


SKIP_DIRS = {"logs", "viz"}
GEOMETRY_DESCRIPTORS = (
    "rgb_grad",
    "rgb_grad_context",
    "rgb",
    "gradient",
    "structure_tensor",
    "local_spectrum",
)
RESPONSE_MODES = (
    "final_delta",
    "trajectory_delta",
    "feature_final_delta",
    "feature_trajectory_delta",
    "coord_jacobian_final_delta",
    "coord_jacobian_trajectory_delta",
)


@dataclass(frozen=True)
class PatchGrid:
    """Patch starts and centers shared by geometry and response descriptors."""

    starts: list[tuple[int, int]]
    centers: np.ndarray


@dataclass(frozen=True)
class LIIFEvaluation:
    """Reconstructed LIIF response objects for one trajectory."""

    outputs: np.ndarray
    target_hr: np.ndarray
    lr_up: np.ndarray
    summary: dict[str, Any]
    features_hr: np.ndarray | None = None
    coord_jacobians: np.ndarray | None = None


def load_summary(result_dir: Path) -> dict[str, Any]:
    """Load result summary."""
    for name in ("dynamics_summary.json", "summary.json"):
        path = result_dir / name
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
    raise FileNotFoundError(f"missing dynamics_summary.json/summary.json in {result_dir}")


def resolve_image_path_from_summary(summary: dict[str, Any], data_root: Path) -> Path:
    """Resolve the image path for a fitting-dynamics result.

    Newer summaries record the exact image path used for training. Older Stage-B
    Set5 summaries only record the image basename, so they fall back to the
    historical Data/Set5/HR convention.
    """
    candidates: list[Path] = []
    for key in ("image_path", "image_relpath"):
        recorded = summary.get(key)
        if recorded:
            path = Path(recorded)
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates.extend([path, data_root.parent / path, data_root / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = data_root / "Set5" / "HR" / str(summary["image"])
    if fallback.exists() or not candidates:
        return fallback
    return candidates[0]


def build_liif_model(summary: dict[str, Any], n_params: int, device: torch.device) -> LIIFModel:
    """Build a LIIF model whose parameter count matches the trajectory."""
    if str(summary.get("model_type", "")).lower() != "liif":
        raise ValueError("Stage-C pilot currently supports scratch LIIF outputs only")

    candidates = [
        ("LIIF_CONFIG_REDUCED", LIIF_CONFIG_REDUCED),
        ("LIIF_CONFIG", LIIF_CONFIG),
    ]
    preferred = summary.get("model_config_name")
    if preferred:
        candidates = sorted(candidates, key=lambda item: 0 if item[0] == preferred else 1)

    for _, config in candidates:
        model = LIIFModel(**config).to(device)
        total = sum(p.numel() for p in model.get_params().values())
        if total == n_params:
            return model
    raise ValueError(f"no LIIF config matches trajectory parameter count {n_params}")


def unflatten_full_state(model: LIIFModel, flat: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """Unflatten a full-model snapshot using model.get_params() order."""
    templates = model.get_params()
    total = sum(t.numel() for t in templates.values())
    if total != flat.shape[0]:
        raise ValueError(f"snapshot has {flat.shape[0]} params, model expects {total}")

    state: dict[str, torch.Tensor] = {}
    offset = 0
    for key, template in templates.items():
        n_elem = template.numel()
        arr = flat[offset:offset + n_elem].reshape(tuple(template.shape))
        state[key] = torch.from_numpy(arr).to(device=device, dtype=template.dtype)
        offset += n_elem
    return state


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute RGB PSNR for arrays in [0, 1]."""
    mse = float(np.mean((pred - target) ** 2))
    return float(-10.0 * np.log10(mse + 1e-10))


def estimate_coord_jacobian(
    model: LIIFModel,
    coords: torch.Tensor,
    lr_tensor: torch.Tensor,
    sr_scale: int,
    height: int,
    width: int,
    eps: float,
) -> np.ndarray:
    """Estimate output-coordinate Jacobian by central finite differences.

    This probes the fitted LIIF function response with respect to normalized
    output coordinates. It is intentionally a function-space diagnostic, not a
    claim about parameter-space decoder symmetries.
    """
    if eps <= 0.0:
        raise ValueError("jacobian eps must be positive")
    offsets = [
        torch.tensor([eps, 0.0], device=coords.device, dtype=coords.dtype),
        torch.tensor([-eps, 0.0], device=coords.device, dtype=coords.dtype),
        torch.tensor([0.0, eps], device=coords.device, dtype=coords.dtype),
        torch.tensor([0.0, -eps], device=coords.device, dtype=coords.dtype),
    ]
    shifted = [
        (coords + offset).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        for offset in offsets
    ]
    out_xp = model(shifted[0], lr_tensor, scale=sr_scale)
    out_xm = model(shifted[1], lr_tensor, scale=sr_scale)
    out_yp = model(shifted[2], lr_tensor, scale=sr_scale)
    out_ym = model(shifted[3], lr_tensor, scale=sr_scale)
    dx = (out_xp - out_xm) / (2.0 * eps)
    dy = (out_yp - out_ym) / (2.0 * eps)
    jac = torch.cat([dx, dy], dim=1).reshape(height, width, 6)
    return jac.detach().cpu().numpy()


def evaluate_liif_response_objects(
    result_dir: Path,
    data_root: Path,
    device: torch.device,
    collect_features: bool = False,
    collect_coord_jacobians: bool = False,
    jacobian_eps: float = 1e-3,
) -> LIIFEvaluation:
    """Reconstruct raw full snapshots and evaluate selected response objects."""
    summary = load_summary(result_dir)
    trajectory = np.load(result_dir / "trajectory.npz", allow_pickle=True)
    full_snapshots = np.asarray(trajectory["full_snapshots"], dtype=np.float32)

    image_name = summary["image"]
    image_size = int(summary.get("hr_size", 48))
    sr_scale = int(summary.get("sr_scale", 4))
    image_path = resolve_image_path_from_summary(summary, data_root)
    lr_tensor, hr_tensor = load_image(image_path, image_size=image_size, sr_scale=sr_scale)

    lr_tensor = lr_tensor.to(device)
    hr_tensor = hr_tensor.to(device)
    _, height, width = hr_tensor.shape
    coords = get_image_coordinates(height, width, normalize="center", device=device).reshape(-1, 2)
    target_flat = hr_tensor.reshape(3, -1).T

    model = build_liif_model(summary, full_snapshots.shape[1], device)
    model.eval()

    outputs = []
    psnrs = []
    features_hr = []
    coord_jacobians = []
    with torch.no_grad():
        for flat in full_snapshots:
            model.set_params(unflatten_full_state(model, flat, device))
            out = model(coords, lr_tensor, scale=sr_scale).clamp(0, 1)
            mse = F.mse_loss(out, target_flat).item()
            psnrs.append(float(-10.0 * np.log10(mse + 1e-10)))
            outputs.append(out.reshape(height, width, 3).detach().cpu().numpy())

            if collect_features:
                feat = F.interpolate(
                    model.feat,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                features_hr.append(feat.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())

            if collect_coord_jacobians:
                coord_jacobians.append(
                    estimate_coord_jacobian(
                        model,
                        coords,
                        lr_tensor,
                        sr_scale,
                        height,
                        width,
                        jacobian_eps,
                    )
                )

    target_hr = hr_tensor.permute(1, 2, 0).detach().cpu().numpy()
    lr_up_tensor = F.interpolate(
        lr_tensor.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    lr_up = lr_up_tensor.permute(1, 2, 0).detach().cpu().numpy()

    summary["_recomputed_final_psnr"] = psnrs[-1]
    summary["_recomputed_initial_psnr"] = psnrs[0]
    return LIIFEvaluation(
        outputs=np.stack(outputs, axis=0),
        target_hr=target_hr,
        lr_up=lr_up,
        summary=summary,
        features_hr=np.stack(features_hr, axis=0) if collect_features else None,
        coord_jacobians=np.stack(coord_jacobians, axis=0) if collect_coord_jacobians else None,
    )


def evaluate_liif_snapshots(
    result_dir: Path,
    data_root: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct raw full snapshots and evaluate LIIF outputs.

    Returns:
        outputs: [T, H, W, 3] model outputs in [0, 1].
        target_hr: [H, W, 3] target HR image in [0, 1].
        lr_up: [H, W, 3] bicubic-like upsampled LR input in [0, 1].
        summary: dynamics summary.
    """
    evaluation = evaluate_liif_response_objects(result_dir, data_root, device)
    return evaluation.outputs, evaluation.target_hr, evaluation.lr_up, evaluation.summary


def make_patch_grid(height: int, width: int, patch_size: int, stride: int) -> PatchGrid:
    """Create deterministic patch starts and centers."""
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if patch_size > height or patch_size > width:
        raise ValueError("patch_size must fit inside the image")

    starts: list[tuple[int, int]] = []
    centers = []
    radius = patch_size // 2
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            starts.append((y, x))
            centers.append((y + radius, x + radius))
    return PatchGrid(starts=starts, centers=np.asarray(centers, dtype=np.float64))


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalize row descriptors with stable zero-row handling."""
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _gray_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return gray-image gradients with stable handling for tiny patches."""
    if min(gray.shape) < 2:
        zeros = np.zeros_like(gray, dtype=np.float64)
        return zeros, zeros
    grad_y, grad_x = np.gradient(gray)
    return np.asarray(grad_x, dtype=np.float64), np.asarray(grad_y, dtype=np.float64)


def _local_spectrum_descriptor(gray: np.ndarray) -> np.ndarray:
    """Return a phase-free local spectrum descriptor for one gray patch."""
    centered = gray - gray.mean()
    if min(centered.shape) > 1:
        window = np.outer(np.hanning(centered.shape[0]), np.hanning(centered.shape[1]))
        if np.any(window):
            centered = centered * window
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    return np.log1p(np.abs(spectrum)).reshape(-1)


def _geometry_descriptor_for_patch(patch: np.ndarray, descriptor: str) -> np.ndarray:
    """Extract one geometry descriptor vector from an RGB patch."""
    centered_rgb = patch - patch.mean(axis=(0, 1), keepdims=True)
    gray = patch.mean(axis=2)
    grad_x, grad_y = _gray_gradients(gray)

    if descriptor == "rgb":
        return centered_rgb.reshape(-1)
    if descriptor == "gradient":
        return np.concatenate([grad_x.reshape(-1), grad_y.reshape(-1)])
    if descriptor == "structure_tensor":
        return np.concatenate(
            [
                (grad_x * grad_x).reshape(-1),
                (grad_x * grad_y).reshape(-1),
                (grad_y * grad_y).reshape(-1),
            ]
        )
    if descriptor == "local_spectrum":
        return _local_spectrum_descriptor(gray)
    if descriptor == "rgb_grad":
        return np.concatenate(
            [centered_rgb.reshape(-1), grad_x.reshape(-1), grad_y.reshape(-1)]
        )
    raise ValueError(f"unknown geometry descriptor: {descriptor}")


def _resize_patch_nearest(patch: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB patch to a square size with deterministic nearest sampling."""
    if patch.shape[0] == size and patch.shape[1] == size:
        return patch
    y_idx = np.linspace(0, patch.shape[0] - 1, size).round().astype(int)
    x_idx = np.linspace(0, patch.shape[1] - 1, size).round().astype(int)
    return patch[y_idx][:, x_idx]


def _crop_with_edge_padding(
    image: np.ndarray,
    center_y: float,
    center_x: float,
    size: int,
) -> np.ndarray:
    """Crop a square context patch around a center, padding with edge values."""
    if size <= 0:
        raise ValueError("context crop size must be positive")
    arr = np.asarray(image)
    radius = size // 2
    center_y_i = int(round(center_y))
    center_x_i = int(round(center_x))
    start_y = center_y_i - radius
    start_x = center_x_i - radius
    end_y = start_y + size
    end_x = start_x + size
    pad_top = max(0, -start_y)
    pad_left = max(0, -start_x)
    pad_bottom = max(0, end_y - arr.shape[0])
    pad_right = max(0, end_x - arr.shape[1])
    if any(v > 0 for v in (pad_top, pad_bottom, pad_left, pad_right)):
        arr = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        start_y += pad_top
        end_y += pad_top
        start_x += pad_left
        end_x += pad_left
    return arr[start_y:end_y, start_x:end_x, :]


def geometry_patch_descriptors(
    image: np.ndarray,
    grid: PatchGrid,
    patch_size: int,
    descriptor: str = "rgb_grad",
    normalize: bool = True,
) -> np.ndarray:
    """Extract patch-level geometry descriptors.

    Descriptor modes:
        rgb: mean-centered RGB patch residuals.
        gradient: grayscale dx/dy fields.
        structure_tensor: grayscale dx^2, dx*dy, dy^2 fields.
        local_spectrum: phase-free FFT magnitude of a centered gray patch.
        rgb_grad: the original Stage-C pilot descriptor, RGB + gradients.
        rgb_grad_context: rgb_grad plus a 3x context crop resized to patch size.
    """
    if descriptor not in GEOMETRY_DESCRIPTORS:
        raise ValueError(f"unknown geometry descriptor: {descriptor}")
    descriptors = []
    for (y, x), center in zip(grid.starts, grid.centers):
        patch = np.asarray(image[y:y + patch_size, x:x + patch_size], dtype=np.float64)
        if descriptor == "rgb_grad_context":
            context_size = patch_size * 3
            context = _crop_with_edge_padding(
                image,
                center_y=float(center[0]),
                center_x=float(center[1]),
                size=context_size,
            )
            context_resized = _resize_patch_nearest(context, patch_size)
            descriptors.append(
                np.concatenate(
                    [
                        _geometry_descriptor_for_patch(patch, "rgb_grad"),
                        _geometry_descriptor_for_patch(context_resized, "rgb_grad"),
                    ]
                )
            )
        else:
            descriptors.append(_geometry_descriptor_for_patch(patch, descriptor))
    stacked = np.stack(descriptors, axis=0)
    return l2_normalize_rows(stacked) if normalize else stacked


def response_patch_descriptors(
    outputs: np.ndarray,
    grid: PatchGrid,
    patch_size: int,
    mode: str,
    normalize: bool = True,
) -> np.ndarray:
    """Extract patch-level response descriptors from output snapshots."""
    if mode.endswith("_final_delta"):
        field = outputs[-1] - outputs[0]
    elif mode.endswith("_trajectory_delta"):
        field = np.diff(outputs, axis=0)
    elif mode == "final_delta":
        field = outputs[-1] - outputs[0]
    elif mode == "trajectory_delta":
        field = np.diff(outputs, axis=0)
    else:
        raise ValueError(f"unknown response mode: {mode}")

    descriptors = []
    for y, x in grid.starts:
        if field.ndim == 3:
            patch = field[y:y + patch_size, x:x + patch_size, :]
        else:
            patch = field[:, y:y + patch_size, x:x + patch_size, :]
        descriptors.append(np.asarray(patch, dtype=np.float64).reshape(-1))
    stacked = np.stack(descriptors, axis=0)
    return l2_normalize_rows(stacked) if normalize else stacked


def pairwise_distances(descriptors: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between row descriptors."""
    gram = descriptors @ descriptors.T
    sq_norms = np.sum(descriptors * descriptors, axis=1)
    dist2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    return np.sqrt(np.maximum(dist2, 0.0))


def eligible_pair_mask(centers: np.ndarray, min_spatial_distance: float) -> np.ndarray:
    """Return directed non-self pairs that pass a spatial-distance exclusion."""
    spatial = pairwise_distances(centers)
    mask = spatial >= float(min_spatial_distance)
    np.fill_diagonal(mask, False)
    return mask


def geometry_neighbor_pairs(
    geometry_dist: np.ndarray,
    eligible: np.ndarray,
    k: int,
) -> list[tuple[int, int]]:
    """Select directed k-nearest geometry neighbors under an eligibility mask."""
    pairs: list[tuple[int, int]] = []
    for i in range(geometry_dist.shape[0]):
        candidates = np.flatnonzero(eligible[i])
        if candidates.size == 0:
            continue
        order = candidates[np.argsort(geometry_dist[i, candidates], kind="mergesort")]
        for j in order[:k]:
            pairs.append((i, int(j)))
    return pairs


def mean_pair_distance(distance: np.ndarray, pairs: Sequence[tuple[int, int]]) -> float:
    """Mean distance over directed pairs."""
    if not pairs:
        return float("nan")
    return float(np.mean([distance[i, j] for i, j in pairs]))


def pair_distance_values(distance: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Return distance values over directed pairs."""
    if not pairs:
        return np.array([], dtype=np.float64)
    return np.asarray([distance[i, j] for i, j in pairs], dtype=np.float64)


def finite_percentile(values: np.ndarray, q: float) -> float:
    """Return a percentile over finite values, or NaN if none exist."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def summarize_row_norms(prefix: str, descriptors: np.ndarray) -> dict[str, float]:
    """Summarize row L2 norms for raw descriptor matrices."""
    norms = np.linalg.norm(np.asarray(descriptors, dtype=np.float64), axis=1)
    mean = float(np.mean(norms))
    std = float(np.std(norms, ddof=1)) if norms.size > 1 else 0.0
    return {
        f"{prefix}_norm_mean": mean,
        f"{prefix}_norm_std": std,
        f"{prefix}_norm_cv": float(std / mean) if mean > 0.0 else float("nan"),
        f"{prefix}_norm_p10": finite_percentile(norms, 10),
        f"{prefix}_norm_p50": finite_percentile(norms, 50),
        f"{prefix}_norm_p90": finite_percentile(norms, 90),
        f"{prefix}_near_zero_frac": float(np.mean(norms <= 1e-12)),
    }


def sample_random_pair_mean(
    response_dist: np.ndarray,
    eligible_pairs: list[tuple[int, int]],
    n_pairs: int,
    rng: np.random.Generator,
) -> float:
    """Sample random eligible pairs and return mean response distance."""
    if not eligible_pairs or n_pairs <= 0:
        return float("nan")
    replace = len(eligible_pairs) < n_pairs
    idx = rng.choice(len(eligible_pairs), size=n_pairs, replace=replace)
    return float(np.mean([response_dist[eligible_pairs[i]] for i in idx]))


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks for one-dimensional values."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation without relying on scipy.stats."""
    if len(x) < 2:
        return float("nan")
    rx = _rank_average(np.asarray(x, dtype=np.float64))
    ry = _rank_average(np.asarray(y, dtype=np.float64))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom <= 0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def one_sided_le_pvalue(observed: float, controls: list[float]) -> float:
    """Empirical p(control <= observed) with a plus-one correction."""
    finite = [x for x in controls if math.isfinite(x)]
    if not math.isfinite(observed) or not finite:
        return float("nan")
    le = sum(1 for x in finite if x <= observed)
    return float((le + 1) / (len(finite) + 1))


def analyze_geometry_response_descriptors(
    geometry_desc: np.ndarray,
    response_desc: np.ndarray,
    centers: np.ndarray,
    k: int = 5,
    min_spatial_distance: float = 8.0,
    n_shuffles: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Compare geometry-neighbor response distances to controls."""
    if geometry_desc.shape[0] != response_desc.shape[0]:
        raise ValueError("geometry and response descriptors must have the same number of patches")

    geometry_dist = pairwise_distances(geometry_desc)
    response_dist = pairwise_distances(response_desc)
    eligible = eligible_pair_mask(centers, min_spatial_distance)
    neighbor_pairs = geometry_neighbor_pairs(geometry_dist, eligible, k=k)
    eligible_pairs = list(zip(*np.nonzero(eligible)))
    if not neighbor_pairs:
        raise ValueError("no eligible geometry-neighbor pairs; lower min_spatial_distance")

    observed = mean_pair_distance(response_dist, neighbor_pairs)
    neighbor_response_values = pair_distance_values(response_dist, neighbor_pairs)
    neighbor_geometry_values = pair_distance_values(geometry_dist, neighbor_pairs)
    rng = np.random.default_rng(seed)
    random_means = []
    shuffled_means = []
    n_pairs = len(neighbor_pairs)
    for _ in range(n_shuffles):
        random_means.append(sample_random_pair_mean(response_dist, eligible_pairs, n_pairs, rng))
        perm = rng.permutation(response_dist.shape[0])
        shuffled_response_dist = response_dist[perm][:, perm]
        shuffled_means.append(mean_pair_distance(shuffled_response_dist, neighbor_pairs))

    random_mean = float(np.mean(random_means))
    shuffled_mean = float(np.mean(shuffled_means))

    upper = np.triu(eligible, k=1)
    geom_pair_dist = geometry_dist[upper]
    resp_pair_dist = response_dist[upper]

    spatial = pairwise_distances(centers)
    neighbor_spatial_values = pair_distance_values(spatial, neighbor_pairs)
    return {
        "n_patches": float(geometry_desc.shape[0]),
        "n_neighbor_pairs": float(n_pairs),
        "neighbor_response_dist": observed,
        "neighbor_response_dist_p50": finite_percentile(neighbor_response_values, 50),
        "neighbor_response_dist_p90": finite_percentile(neighbor_response_values, 90),
        "eligible_response_dist_mean": float(np.mean(resp_pair_dist)),
        "eligible_response_dist_p50": finite_percentile(resp_pair_dist, 50),
        "random_response_dist_mean": random_mean,
        "random_response_dist_std": float(np.std(random_means, ddof=1)) if n_shuffles > 1 else 0.0,
        "random_response_p_le": one_sided_le_pvalue(observed, random_means),
        "shuffled_response_dist_mean": shuffled_mean,
        "shuffled_response_dist_std": float(np.std(shuffled_means, ddof=1)) if n_shuffles > 1 else 0.0,
        "shuffled_response_p_le": one_sided_le_pvalue(observed, shuffled_means),
        "effect_vs_random": random_mean - observed,
        "effect_vs_shuffle": shuffled_mean - observed,
        "effect_vs_eligible": float(np.mean(resp_pair_dist)) - observed,
        "effect_vs_shuffle_frac": (
            (shuffled_mean - observed) / shuffled_mean if shuffled_mean > 0.0 else float("nan")
        ),
        "geometry_response_spearman": spearman_correlation(geom_pair_dist, resp_pair_dist),
        "neighbor_geometry_dist": float(np.mean(neighbor_geometry_values)),
        "neighbor_geometry_dist_p50": finite_percentile(neighbor_geometry_values, 50),
        "eligible_geometry_dist_mean": float(np.mean(geom_pair_dist)),
        "eligible_geometry_dist_p50": finite_percentile(geom_pair_dist, 50),
        "mean_neighbor_spatial_distance": float(np.mean([spatial[i, j] for i, j in neighbor_pairs])),
        "median_neighbor_spatial_distance": finite_percentile(neighbor_spatial_values, 50),
    }


def analyze_evaluated_result(
    result_dir: Path,
    target_hr: np.ndarray,
    lr_up: np.ndarray,
    summary: dict[str, Any],
    geometry_source: str,
    geometry_descriptor: str,
    response_mode: str,
    patch_size: int,
    stride: int,
    k: int,
    min_spatial_distance: float,
    n_shuffles: int,
    seed: int,
    evaluation: LIIFEvaluation | None = None,
    outputs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the Stage-C pilot probe on already evaluated LIIF outputs."""
    geometry_image = target_hr if geometry_source == "hr" else lr_up
    grid = make_patch_grid(target_hr.shape[0], target_hr.shape[1], patch_size, stride)
    if evaluation is not None:
        response_source = response_source_for_mode(evaluation, response_mode)
    elif outputs is not None:
        response_source = outputs
    else:
        raise ValueError("either evaluation or outputs must be provided")

    geometry_raw = geometry_patch_descriptors(
        geometry_image,
        grid,
        patch_size,
        descriptor=geometry_descriptor,
        normalize=False,
    )
    response_raw = response_patch_descriptors(
        response_source,
        grid,
        patch_size,
        response_mode,
        normalize=False,
    )
    geometry_desc = l2_normalize_rows(geometry_raw)
    response_desc = l2_normalize_rows(response_raw)

    metrics = analyze_geometry_response_descriptors(
        geometry_desc,
        response_desc,
        grid.centers,
        k=k,
        min_spatial_distance=min_spatial_distance,
        n_shuffles=n_shuffles,
        seed=seed,
    )

    final_psnr = float(summary.get("final_psnr", float("nan")))
    recomputed_final = float(summary["_recomputed_final_psnr"])
    row: dict[str, Any] = {
        "run": result_dir.name,
        "status": "ok",
        "image": summary.get("image", "?"),
        "seed": summary.get("seed", "?"),
        "geometry_source": geometry_source,
        "geometry_descriptor": geometry_descriptor,
        "response_mode": response_mode,
        "patch_size": patch_size,
        "stride": stride,
        "k": k,
        "min_spatial_distance": min_spatial_distance,
        "n_shuffles": n_shuffles,
        "summary_final_psnr": final_psnr,
        "recomputed_final_psnr": recomputed_final,
        "final_psnr_abs_error": abs(recomputed_final - final_psnr),
        "recomputed_initial_psnr": float(summary["_recomputed_initial_psnr"]),
    }
    row.update(metrics)
    row.update(summarize_row_norms("geometry", geometry_raw))
    row.update(summarize_row_norms("response", response_raw))
    return row


def response_requirements(response_mode: str) -> tuple[bool, bool]:
    """Return whether a response mode needs features or coordinate Jacobians."""
    if response_mode in {"feature_final_delta", "feature_trajectory_delta"}:
        return True, False
    if response_mode in {"coord_jacobian_final_delta", "coord_jacobian_trajectory_delta"}:
        return False, True
    if response_mode in {"final_delta", "trajectory_delta"}:
        return False, False
    raise ValueError(f"unknown response mode: {response_mode}")


def response_source_for_mode(evaluation: LIIFEvaluation, response_mode: str) -> np.ndarray:
    """Select evaluated response object for a response mode."""
    if response_mode in {"final_delta", "trajectory_delta"}:
        return evaluation.outputs
    if response_mode in {"feature_final_delta", "feature_trajectory_delta"}:
        if evaluation.features_hr is None:
            raise ValueError(f"{response_mode} requires feature response evaluation")
        return evaluation.features_hr
    if response_mode in {"coord_jacobian_final_delta", "coord_jacobian_trajectory_delta"}:
        if evaluation.coord_jacobians is None:
            raise ValueError(f"{response_mode} requires coordinate Jacobian evaluation")
        return evaluation.coord_jacobians
    raise ValueError(f"unknown response mode: {response_mode}")


def analyze_result_dir(
    result_dir: Path,
    data_root: Path,
    device: torch.device,
    geometry_source: str,
    geometry_descriptor: str,
    response_mode: str,
    patch_size: int,
    stride: int,
    k: int,
    min_spatial_distance: float,
    n_shuffles: int,
    seed: int,
) -> dict[str, Any]:
    """Run the Stage-C pilot probe on one LIIF result directory."""
    collect_features, collect_coord_jacobians = response_requirements(response_mode)
    evaluation = evaluate_liif_response_objects(
        result_dir,
        data_root,
        device,
        collect_features=collect_features,
        collect_coord_jacobians=collect_coord_jacobians,
    )
    return analyze_evaluated_result(
        result_dir=result_dir,
        evaluation=evaluation,
        target_hr=evaluation.target_hr,
        lr_up=evaluation.lr_up,
        summary=evaluation.summary,
        geometry_source=geometry_source,
        geometry_descriptor=geometry_descriptor,
        response_mode=response_mode,
        patch_size=patch_size,
        stride=stride,
        k=k,
        min_spatial_distance=min_spatial_distance,
        n_shuffles=n_shuffles,
        seed=seed,
    )


def scan_result_dirs(results_dir: Path) -> list[Path]:
    """Return candidate LIIF Stage-B result directories."""
    dirs = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith(".") or subdir.name in SKIP_DIRS:
            continue
        try:
            summary = load_summary(subdir)
        except FileNotFoundError:
            continue
        if str(summary.get("model_type", "")).lower() == "liif":
            dirs.append(subdir)
    return dirs


def format_float(value: Any, digits: int = 4) -> str:
    """Format compact numeric output."""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print compact pilot table."""
    headers = [
        ("run", 35),
        ("geom", 7),
        ("desc", 17),
        ("resp", 16),
        ("nbr", 8),
        ("rand", 8),
        ("shuf", 8),
        ("eff_s", 8),
        ("p_shuf", 8),
        ("rho", 8),
        ("psnr_err", 9),
    ]
    print()
    print("".join(f"{name:<{width}}" for name, width in headers))
    print("-" * sum(width for _, width in headers))
    for row in rows:
        values = {
            "run": row["run"],
            "geom": row["geometry_source"],
            "desc": row["geometry_descriptor"],
            "resp": row["response_mode"],
            "nbr": format_float(row["neighbor_response_dist"]),
            "rand": format_float(row["random_response_dist_mean"]),
            "shuf": format_float(row["shuffled_response_dist_mean"]),
            "eff_s": format_float(row["effect_vs_shuffle"]),
            "p_shuf": format_float(row["shuffled_response_p_le"]),
            "rho": format_float(row["geometry_response_spearman"]),
            "psnr_err": format_float(row["final_psnr_abs_error"], 5),
        }
        print("".join(f"{values[name]:<{width}}" for name, width in headers))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-C pilot geometry-response probe")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result_dir", action="append", default=None)
    group.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="Data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--geometry_source", choices=["hr", "lr_up"], default="lr_up")
    parser.add_argument(
        "--geometry_descriptor",
        choices=[*GEOMETRY_DESCRIPTORS, "all"],
        default="rgb_grad",
    )
    parser.add_argument("--response_mode", choices=RESPONSE_MODES, default="trajectory_delta")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_spatial_distance", type=float, default=8.0)
    parser.add_argument("--n_shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.result_dir:
        result_dirs = [Path(p) for p in args.result_dir]
    else:
        result_dirs = scan_result_dirs(Path(args.results_dir))

    geometry_descriptors = (
        list(GEOMETRY_DESCRIPTORS)
        if args.geometry_descriptor == "all"
        else [args.geometry_descriptor]
    )

    rows = []
    for idx, result_dir in enumerate(result_dirs):
        collect_features, collect_coord_jacobians = response_requirements(args.response_mode)
        evaluation = evaluate_liif_response_objects(
            result_dir,
            Path(args.data_root),
            device,
            collect_features=collect_features,
            collect_coord_jacobians=collect_coord_jacobians,
        )
        for descriptor in geometry_descriptors:
            row = analyze_evaluated_result(
                result_dir=result_dir,
                evaluation=evaluation,
                target_hr=evaluation.target_hr,
                lr_up=evaluation.lr_up,
                summary=evaluation.summary,
                geometry_source=args.geometry_source,
                geometry_descriptor=descriptor,
                response_mode=args.response_mode,
                patch_size=args.patch_size,
                stride=args.stride,
                k=args.k,
                min_spatial_distance=args.min_spatial_distance,
                n_shuffles=args.n_shuffles,
                seed=args.seed + idx,
            )
            rows.append(row)

    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_table(rows)
        print(f"\n{len(rows)} LIIF result(s) analyzed on {device}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
