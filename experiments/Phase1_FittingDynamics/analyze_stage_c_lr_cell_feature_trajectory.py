#!/usr/bin/env python3
"""Read-only C-minimal LR-cell encoder-feature trajectory audit.

This diagnostic is deliberately narrower than the old Stage-C natural-image
nearest-neighbor probes. It asks whether per-LR-cell encoder feature trajectory
complexity can be explained by local coordinate, content, and structure-tensor
geometry features.

The analysis object is the LIIF encoder feature map before feature unfolding:

    full_snapshots[t] -> rebuild LIIF -> set_params
    -> inp=(lr_tensor.unsqueeze(0)-0.5)/0.5 -> model.gen_feat(inp)
    -> feature_map[t]: [1, C, H_lr, W_lr]

Each LR cell then has a trajectory [T, C]. The main target is
orthogonal_energy_fraction. Blocked spatial CV R2 is the primary regression
metric; in-sample R2 is recorded only as a diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_b_response_audit import (  # noqa: E402
    load_summary,
)
from experiments.Phase1_FittingDynamics.analyze_stage_b_trajectory_audit import (  # noqa: E402
    spectrum_metrics,
    update_decomposition_metrics,
)
from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import (  # noqa: E402
    build_liif_model,
    resolve_image_path_from_summary,
    unflatten_full_state,
)
from experiments.Phase1_FittingDynamics.run import load_image  # noqa: E402


DEFAULT_RUN_NAMES = (
    "LIIF_reduced_baby_sr4_seed123",
    "LIIF_reduced_woman_sr4_seed42",
    "LIIF_reduced_bird_sr4_seed42",
    "LIIF_reduced_head_sr4_seed42",
)
SKIP_DIRS = {"logs", "viz"}
EPS = 1e-12

MAIN_TARGET = "orthogonal_energy_fraction"
TARGET_ROLES = {
    "orthogonal_energy_fraction": "main",
    "straightness": "auxiliary",
    "effective_rank": "auxiliary",
    "path_length": "diagnostic",
    "endpoint_norm": "diagnostic",
    "mean_step_norm": "diagnostic",
}


def normalize_liif_input(lr_tensor: torch.Tensor) -> torch.Tensor:
    """Return LIIF encoder input with the model.forward normalization applied."""
    batched = lr_tensor.unsqueeze(0) if lr_tensor.dim() == 3 else lr_tensor
    return (batched - 0.5) / 0.5


def gen_encoder_feature(model: Any, lr_tensor: torch.Tensor) -> torch.Tensor:
    """Run LIIF gen_feat through the explicit [0, 1] -> [-1, 1] path."""
    return model.gen_feat(normalize_liif_input(lr_tensor))


def lr_feature_trajectory(
    result_dir: Path,
    data_root: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct one LIIF run and return encoder feature maps.

    Returns:
        features: [T, H_lr, W_lr, C]
        lr_image: [H_lr, W_lr, 3] in [0, 1]
        hr_image: [H_hr, W_hr, 3] in [0, 1]
        lr_up: [H_hr, W_hr, 3] bilinear LR-up image in [0, 1]
        summary: run summary
    """
    summary = load_summary(result_dir)
    if str(summary.get("model_type", "")).lower() != "liif":
        raise ValueError(f"{result_dir.name} is not a LIIF run")

    trajectory = np.load(result_dir / "trajectory.npz", allow_pickle=True)
    full_snapshots = np.asarray(trajectory["full_snapshots"], dtype=np.float32)
    image_size = int(summary.get("hr_size", 48))
    sr_scale = int(summary.get("sr_scale", 4))
    image_path = resolve_image_path_from_summary(summary, data_root)
    lr_tensor, hr_tensor = load_image(image_path, image_size=image_size, sr_scale=sr_scale)
    lr_tensor = lr_tensor.to(device)
    hr_tensor = hr_tensor.to(device)

    model = build_liif_model(summary, full_snapshots.shape[1], device)
    model.eval()

    feature_maps: list[np.ndarray] = []
    with torch.no_grad():
        for flat in full_snapshots:
            model.set_params(unflatten_full_state(model, flat, device))
            feat = gen_encoder_feature(model, lr_tensor)
            feature_maps.append(feat.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())

    lr_up_tensor = F.interpolate(
        lr_tensor.unsqueeze(0),
        size=tuple(hr_tensor.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return (
        np.asarray(feature_maps, dtype=np.float64),
        lr_tensor.permute(1, 2, 0).detach().cpu().numpy().astype(np.float64),
        hr_tensor.permute(1, 2, 0).detach().cpu().numpy().astype(np.float64),
        lr_up_tensor.permute(1, 2, 0).detach().cpu().numpy().astype(np.float64),
        summary,
    )


def feature_maps_to_cell_trajectories(feature_maps: np.ndarray) -> np.ndarray:
    """Convert feature maps [T, H, W, C] to cell trajectories [H, W, T, C]."""
    arr = np.asarray(feature_maps, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError("feature_maps must have shape [T, H, W, C]")
    return np.transpose(arr, (1, 2, 0, 3))


def cell_trajectory_metric_row(trajectory: np.ndarray) -> dict[str, float]:
    """Compute complexity metrics for one cell trajectory [T, C]."""
    snapshots = np.asarray(trajectory, dtype=np.float64)
    if snapshots.ndim != 2 or snapshots.shape[0] < 3:
        raise ValueError("cell trajectory must have shape [T, C] with T >= 3")
    updates = np.diff(snapshots, axis=0)
    update_metrics = update_decomposition_metrics(snapshots)
    rank_metrics = spectrum_metrics(updates, centered=False)
    endpoint_norm = float(np.linalg.norm(snapshots[-1] - snapshots[0]))
    return {
        "orthogonal_energy_fraction": update_metrics["orthogonal_energy_fraction"],
        "straightness": update_metrics["straightness"],
        "effective_rank": rank_metrics["effective_rank"],
        "path_length": update_metrics["path_length"],
        "endpoint_norm": endpoint_norm,
        "mean_step_norm": update_metrics["mean_step_norm"],
    }


def cell_trajectory_targets(cell_trajectories: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all target vectors over an [H, W, T, C] cell trajectory grid."""
    arr = np.asarray(cell_trajectories, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError("cell_trajectories must have shape [H, W, T, C]")
    rows: list[dict[str, float]] = []
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            rows.append(cell_trajectory_metric_row(arr[y, x]))
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in TARGET_ROLES
    }


def crop_center_edge(image: np.ndarray, center_y: int, center_x: int, size: int) -> np.ndarray:
    """Crop a square LR support around one cell, using edge padding."""
    if size <= 0 or size % 2 != 1:
        raise ValueError("support size must be a positive odd integer")
    arr = np.asarray(image, dtype=np.float64)
    radius = size // 2
    padded = np.pad(
        arr,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="edge",
    )
    y = center_y + radius
    x = center_x + radius
    return padded[y - radius:y + radius + 1, x - radius:x + radius + 1, :]


def _gray_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient-x and gradient-y arrays for a grayscale image."""
    if min(gray.shape) < 2:
        zeros = np.zeros_like(gray, dtype=np.float64)
        return zeros, zeros
    grad_y, grad_x = np.gradient(gray)
    return np.asarray(grad_x, dtype=np.float64), np.asarray(grad_y, dtype=np.float64)


def structure_tensor_orientation_features(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
) -> np.ndarray:
    """Return coherence and orientation circular encoding from local gradients."""
    gx = np.asarray(grad_x, dtype=np.float64)
    gy = np.asarray(grad_y, dtype=np.float64)
    jxx = float(np.mean(gx * gx))
    jxy = float(np.mean(gx * gy))
    jyy = float(np.mean(gy * gy))
    trace = jxx + jyy
    diff = jxx - jyy
    discr = math.sqrt(diff * diff + 4.0 * jxy * jxy)
    coherence = discr / (trace + EPS)
    # cos(2 theta), sin(2 theta) for the dominant tensor orientation.
    cos2 = diff / (discr + EPS)
    sin2 = 2.0 * jxy / (discr + EPS)
    return np.asarray(
        [
            coherence,
            coherence * cos2,
            coherence * sin2,
        ],
        dtype=np.float64,
    )


def coordinate_features(height: int, width: int) -> np.ndarray:
    """Return row/col polynomial features for LR cells."""
    rows = []
    y_den = max(height - 1, 1)
    x_den = max(width - 1, 1)
    for y in range(height):
        yn = (2.0 * y / y_den) - 1.0
        for x in range(width):
            xn = (2.0 * x / x_den) - 1.0
            rows.append([yn, xn, yn * yn, xn * xn, yn * xn])
    return np.asarray(rows, dtype=np.float64)


def local_content_geometry_features(
    lr_image: np.ndarray,
    support_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return content and structure-tensor geometry features for LR cells."""
    lr = np.asarray(lr_image, dtype=np.float64)
    if lr.ndim != 3:
        raise ValueError("lr_image must have shape [H, W, C]")
    gray = lr.mean(axis=2)
    grad_x, grad_y = _gray_gradients(gray)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    content_rows = []
    geometry_rows = []
    for y in range(lr.shape[0]):
        for x in range(lr.shape[1]):
            support_gray = crop_center_edge(gray[..., None], y, x, support_size)[..., 0]
            support_gx = crop_center_edge(grad_x[..., None], y, x, support_size)[..., 0]
            support_gy = crop_center_edge(grad_y[..., None], y, x, support_size)[..., 0]
            support_grad = crop_center_edge(grad_mag[..., None], y, x, support_size)[..., 0]
            content_rows.append(
                [
                    float(np.mean(support_gray)),
                    float(np.var(support_gray)),
                    float(np.mean(support_grad)),
                ]
            )
            geometry_rows.append(structure_tensor_orientation_features(support_gx, support_gy))
    return (
        np.asarray(content_rows, dtype=np.float64),
        np.asarray(geometry_rows, dtype=np.float64),
    )


def difficulty_features(
    hr_image: np.ndarray,
    lr_up: np.ndarray,
    sr_scale: int,
) -> np.ndarray:
    """Return HR block vs LR-up residual diagnostics for each LR cell."""
    hr = np.asarray(hr_image, dtype=np.float64)
    up = np.asarray(lr_up, dtype=np.float64)
    if hr.shape != up.shape:
        raise ValueError("hr_image and lr_up must have the same shape")
    if hr.shape[0] % sr_scale != 0 or hr.shape[1] % sr_scale != 0:
        raise ValueError("HR shape must be divisible by sr_scale")
    rows = []
    for y in range(hr.shape[0] // sr_scale):
        for x in range(hr.shape[1] // sr_scale):
            residual = hr[
                y * sr_scale:(y + 1) * sr_scale,
                x * sr_scale:(x + 1) * sr_scale,
                :,
            ] - up[
                y * sr_scale:(y + 1) * sr_scale,
                x * sr_scale:(x + 1) * sr_scale,
                :,
            ]
            rows.append(
                [
                    float(np.mean(residual * residual)),
                    float(np.mean(np.abs(residual))),
                    float(np.var(residual)),
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def zscore_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score columns using train statistics only."""
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    scale = np.where(std > EPS, std, 1.0)
    return (x_train - mean) / scale, (x_test - mean) / scale


def fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit a standardized ridge regressor with an unregularized intercept."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_mean = float(np.mean(y_train))
    if x_train.shape[1] == 0:
        return np.full(x_test.shape[0], y_mean, dtype=np.float64)
    train_z, test_z = zscore_train_test(x_train, x_test)
    centered_y = y_train - y_mean
    gram = train_z.T @ train_z
    reg = np.eye(train_z.shape[1], dtype=np.float64) * float(alpha)
    try:
        coef = np.linalg.solve(gram + reg, train_z.T @ centered_y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram + reg) @ train_z.T @ centered_y
    return y_mean + test_z @ coef


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return global R2, or NaN for a constant target."""
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(pred)
    if finite.sum() < 2:
        return float("nan")
    y = y[finite]
    pred = pred[finite]
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= EPS:
        return float("nan")
    sse = float(np.sum((y - pred) ** 2))
    return float(1.0 - sse / sst)


def spatial_block_labels(height: int, width: int, block_size: int) -> np.ndarray:
    """Assign each cell to a contiguous spatial block for blocked CV."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    labels = np.zeros((height, width), dtype=np.int64)
    n_block_x = int(math.ceil(width / block_size))
    for y in range(height):
        for x in range(width):
            labels[y, x] = (y // block_size) * n_block_x + (x // block_size)
    return labels.reshape(-1)


def blocked_spatial_cv_r2(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> tuple[float, int]:
    """Return blocked spatial CV R2 using each spatial block as one test fold."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    g = np.asarray(groups)
    if x.shape[0] != y.shape[0] or y.shape[0] != g.shape[0]:
        raise ValueError("features, target, and groups must have the same row count")
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if finite.sum() < 3:
        return float("nan"), 0
    pred = np.full(y.shape[0], np.nan, dtype=np.float64)
    n_folds = 0
    for group in np.unique(g[finite]):
        test = finite & (g == group)
        train = finite & (g != group)
        if test.sum() == 0 or train.sum() < 2:
            continue
        pred[test] = fit_ridge_predict(x[train], y[train], x[test], alpha=alpha)
        n_folds += 1
    return r2_score(y[finite], pred[finite]), n_folds


def in_sample_r2(features: np.ndarray, target: np.ndarray, alpha: float) -> float:
    """Return in-sample ridge R2 as an overfit diagnostic."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if finite.sum() < 3:
        return float("nan")
    pred = fit_ridge_predict(x[finite], y[finite], x[finite], alpha=alpha)
    return r2_score(y[finite], pred)


def feature_groups_for_run(
    lr_image: np.ndarray,
    hr_image: np.ndarray,
    lr_up: np.ndarray,
    sr_scale: int,
    support_size: int,
) -> dict[str, np.ndarray]:
    """Build registered predictor groups for one run."""
    height, width = lr_image.shape[:2]
    coord = coordinate_features(height, width)
    content, geometry = local_content_geometry_features(lr_image, support_size=support_size)
    difficulty = difficulty_features(hr_image, lr_up, sr_scale=sr_scale)
    return {
        "coordinate": coord,
        "content": content,
        "geometry": geometry,
        "difficulty": difficulty,
        "coordinate_content": np.concatenate([coord, content], axis=1),
        "coordinate_content_geometry": np.concatenate([coord, content, geometry], axis=1),
        "coordinate_content_geometry_difficulty": np.concatenate(
            [coord, content, geometry, difficulty],
            axis=1,
        ),
    }


def summarize_vector(prefix: str, values: np.ndarray) -> dict[str, float]:
    """Return compact finite distribution stats."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_max": float(finite.max()),
    }


def analyze_target(
    target_name: str,
    target: np.ndarray,
    groups: dict[str, np.ndarray],
    block_labels: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    """Analyze one cell-level target against registered feature groups."""
    r2_coord, n_folds = blocked_spatial_cv_r2(
        groups["coordinate"],
        target,
        block_labels,
        alpha=alpha,
    )
    r2_base, _ = blocked_spatial_cv_r2(
        groups["coordinate_content"],
        target,
        block_labels,
        alpha=alpha,
    )
    r2_full, _ = blocked_spatial_cv_r2(
        groups["coordinate_content_geometry"],
        target,
        block_labels,
        alpha=alpha,
    )
    r2_full_diff, _ = blocked_spatial_cv_r2(
        groups["coordinate_content_geometry_difficulty"],
        target,
        block_labels,
        alpha=alpha,
    )

    base_full_ratio = (
        float(r2_base / r2_full)
        if math.isfinite(r2_base) and math.isfinite(r2_full) and r2_full > EPS
        else float("nan")
    )

    row: dict[str, Any] = {
        "target": target_name,
        "target_role": TARGET_ROLES[target_name],
        "n_cells": int(target.shape[0]),
        "n_cv_folds": int(n_folds),
        "r2_cv_coordinate": r2_coord,
        "r2_cv_coordinate_content": r2_base,
        "r2_cv_coordinate_content_geometry": r2_full,
        "delta_r2_cv_geometry_vs_coordinate_content": (
            r2_full - r2_base
            if math.isfinite(r2_full) and math.isfinite(r2_base)
            else float("nan")
        ),
        "base_to_full_r2_cv_ratio": base_full_ratio,
        "r2_cv_coordinate_content_geometry_difficulty": r2_full_diff,
        "delta_r2_cv_difficulty_vs_coordinate_content_geometry": (
            r2_full_diff - r2_full
            if math.isfinite(r2_full_diff) and math.isfinite(r2_full)
            else float("nan")
        ),
        "r2_insample_coordinate": in_sample_r2(groups["coordinate"], target, alpha=alpha),
        "r2_insample_coordinate_content": in_sample_r2(
            groups["coordinate_content"],
            target,
            alpha=alpha,
        ),
        "r2_insample_coordinate_content_geometry": in_sample_r2(
            groups["coordinate_content_geometry"],
            target,
            alpha=alpha,
        ),
    }
    row.update(summarize_vector("target", target))
    return row


def analyze_result_dir(
    result_dir: Path,
    data_root: Path,
    device: torch.device,
    support_size: int,
    block_size: int,
    ridge_alpha: float,
) -> list[dict[str, Any]]:
    """Analyze one LIIF run and return one row per registered target."""
    feature_maps, lr_image, hr_image, lr_up, summary = lr_feature_trajectory(
        result_dir,
        data_root,
        device,
    )
    cell_trajectories = feature_maps_to_cell_trajectories(feature_maps)
    targets = cell_trajectory_targets(cell_trajectories)
    sr_scale = int(summary.get("sr_scale", 4))
    groups = feature_groups_for_run(
        lr_image,
        hr_image,
        lr_up,
        sr_scale=sr_scale,
        support_size=support_size,
    )
    block_labels = spatial_block_labels(lr_image.shape[0], lr_image.shape[1], block_size)

    rows = []
    for target_name, target in targets.items():
        row = {
            "run": result_dir.name,
            "status": "ok",
            "image": summary.get("image", "?"),
            "seed": summary.get("seed", "?"),
            "model_config_name": summary.get("model_config_name", "?"),
            "sr_scale": sr_scale,
            "n_snapshots": int(feature_maps.shape[0]),
            "feature_height": int(feature_maps.shape[1]),
            "feature_width": int(feature_maps.shape[2]),
            "feature_channels": int(feature_maps.shape[3]),
            "support_size": int(support_size),
            "spatial_block_size": int(block_size),
            "ridge_alpha": float(ridge_alpha),
            "coordinate_feature_dim": int(groups["coordinate"].shape[1]),
            "content_feature_dim": int(groups["content"].shape[1]),
            "geometry_feature_dim": int(groups["geometry"].shape[1]),
            "difficulty_feature_dim": int(groups["difficulty"].shape[1]),
        }
        row.update(
            summarize_vector("difficulty_residual_mse", groups["difficulty"][:, 0])
        )
        row.update(
            analyze_target(
                target_name,
                target,
                groups,
                block_labels,
                alpha=ridge_alpha,
            )
        )
        rows.append(row)
    return rows


def finite_median(values: Iterable[float]) -> float:
    """Median over finite values."""
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def gate_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen C-minimal diagnostic gate to the main target."""
    main_rows = [row for row in rows if row.get("target") == MAIN_TARGET]
    deltas = {
        str(row["run"]): float(row["delta_r2_cv_geometry_vs_coordinate_content"])
        for row in main_rows
    }
    ratios = {
        str(row["run"]): float(row["base_to_full_r2_cv_ratio"])
        for row in main_rows
    }
    baby_woman = [
        value for run, value in deltas.items()
        if "baby" in run or "woman" in run
    ]
    bird_head = [
        value for run, value in deltas.items()
        if "bird" in run or "head" in run
    ]
    checks = {
        "baby_woman_all_delta_ge_0.03": (
            len(baby_woman) >= 2 and all(value >= 0.03 for value in baby_woman)
        ),
        "all_run_median_delta_ge_0.02": finite_median(deltas.values()) >= 0.02,
        "bird_head_not_strong_negative": (
            len(bird_head) >= 2 and all(value >= -0.01 for value in bird_head)
        ),
    }
    go_signal = all(checks.values())

    hard_stop_delta = sum(value <= 0.01 for value in deltas.values() if math.isfinite(value)) >= 3
    finite_ratios = [value for value in ratios.values() if math.isfinite(value)]
    base_explains_full = (
        len(finite_ratios) >= 3
        and sum(value >= 0.90 for value in finite_ratios) >= 3
        and all(value < 0.03 for value in deltas.values() if math.isfinite(value))
    )
    hard_stop = bool(hard_stop_delta or base_explains_full)
    if go_signal:
        verdict = "go"
    elif hard_stop:
        verdict = "hard_stop"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "go_signal": go_signal,
        "hard_stop": hard_stop,
        "checks": checks,
        "hard_stop_reasons": {
            "three_of_four_delta_le_0.01": hard_stop_delta,
            "coordinate_content_explains_full_without_delta_0.03": base_explains_full,
        },
        "main_target": MAIN_TARGET,
        "strict_delta_r2_cv_by_run": deltas,
        "base_to_full_r2_cv_ratio_by_run": ratios,
        "median_delta_r2_cv": finite_median(deltas.values()),
    }


def scan_liif_result_dirs(results_dir: Path) -> list[Path]:
    """Return LIIF result dirs under results_dir."""
    result_dirs = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        try:
            summary = load_summary(child)
        except FileNotFoundError:
            continue
        if str(summary.get("model_type", "")).lower() == "liif":
            result_dirs.append(child)
    return result_dirs


def resolve_result_dirs(args: argparse.Namespace) -> list[Path]:
    """Resolve CLI result directory selection."""
    if args.result_dir:
        return [Path(path) for path in args.result_dir]
    if args.all_liif:
        return scan_liif_result_dirs(args.results_dir)
    names = args.run_name if args.run_name else list(DEFAULT_RUN_NAMES)
    return [args.results_dir / name for name in names]


def format_float(value: Any, digits: int = 4) -> str:
    """Format floats for terminal tables."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def print_table(rows: list[dict[str, Any]], verdict: dict[str, Any]) -> None:
    """Print compact rows for the main target."""
    main_rows = [row for row in rows if row.get("target") == MAIN_TARGET]
    print("C-minimal LR-cell feature trajectory audit")
    print("run                                base_cv full_cv delta_cv diff_cv in_full")
    print("-" * 88)
    for row in main_rows:
        print(
            f"{row['run']:<35} "
            f"{format_float(row['r2_cv_coordinate_content']):<7} "
            f"{format_float(row['r2_cv_coordinate_content_geometry']):<7} "
            f"{format_float(row['delta_r2_cv_geometry_vs_coordinate_content']):<8} "
            f"{format_float(row['delta_r2_cv_difficulty_vs_coordinate_content_geometry']):<7} "
            f"{format_float(row['r2_insample_coordinate_content_geometry']):<7}"
        )
    print(f"\nGate verdict: {verdict['verdict']}")
    print(f"Median strict delta R2 CV: {format_float(verdict['median_delta_r2_cv'])}")


def write_output(output: dict[str, Any], path: Path) -> None:
    """Write JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(json_safe(output), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def json_safe(value: Any) -> Any:
    """Convert non-finite floats to null-compatible values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only C-minimal LR-cell encoder-feature trajectory audit"
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results/FittingDynamics_StageB"))
    parser.add_argument("--result_dir", action="append", default=None)
    parser.add_argument("--run_name", action="append", default=None)
    parser.add_argument("--all_liif", action="store_true")
    parser.add_argument("--data_root", type=Path, default=Path("Data"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--support_size", type=int, default=3)
    parser.add_argument("--spatial_block_size", type=int, default=3)
    parser.add_argument("--ridge_alpha", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/FittingDynamics_StageB_diagnostics/"
            "stage_c_lr_cell_feature_trajectory_c_minimal_2026-05-10.json"
        ),
    )
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    result_dirs = resolve_result_dirs(args)
    rows: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        rows.extend(
            analyze_result_dir(
                result_dir,
                data_root=args.data_root,
                device=device,
                support_size=args.support_size,
                block_size=args.spatial_block_size,
                ridge_alpha=args.ridge_alpha,
            )
        )

    verdict = gate_verdict(rows)
    output = {
        "settings": {
            "analysis": "c_minimal_lr_cell_encoder_feature_trajectory",
            "trajectory_object": "LIIF gen_feat feature_map before feat_unfold",
            "normalization": "inp=(lr_tensor.unsqueeze(0)-0.5)/0.5 before gen_feat",
            "main_target": MAIN_TARGET,
            "auxiliary_targets": [
                name for name, role in TARGET_ROLES.items() if role == "auxiliary"
            ],
            "diagnostic_targets": [
                name for name, role in TARGET_ROLES.items() if role == "diagnostic"
            ],
            "support_size": args.support_size,
            "spatial_block_size": args.spatial_block_size,
            "ridge_alpha": args.ridge_alpha,
            "device": str(device),
            "result_dirs": [str(path) for path in result_dirs],
            "feature_groups": {
                "coordinate": ["row", "col", "row2", "col2", "row_col"],
                "content": ["lr_support_mean_intensity", "lr_support_variance", "lr_support_grad_mag"],
                "geometry": ["tensor_coherence", "coherence_cos2theta", "coherence_sin2theta"],
                "difficulty": ["hr_lr_up_residual_mse", "hr_lr_up_residual_mae", "hr_lr_up_residual_var"],
            },
            "gate": {
                "go": "baby/woman delta>=0.03, median all>=0.02, bird/head>=-0.01",
                "hard_stop": "3/4 delta<=0.01, or coord+content explains >=90% of full without any delta>=0.03",
            },
        },
        "gate_verdict": verdict,
        "rows": rows,
    }
    if args.output is not None:
        write_output(output, args.output)

    if args.format == "json":
        print(json.dumps(json_safe(output), indent=2, sort_keys=True, allow_nan=False))
    elif args.format == "csv":
        keys = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(sys.stdout, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    else:
        print_table(rows, verdict)
        print(f"\n{len(result_dirs)} LIIF result(s) audited on {device}.")
        if args.output is not None:
            print(f"JSON written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
