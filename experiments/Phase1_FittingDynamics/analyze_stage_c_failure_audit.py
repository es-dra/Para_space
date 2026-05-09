#!/usr/bin/env python3
"""Stage-C pilot failure audit for geometry-response evidence.

This read-only audit reuses the Stage-C geometry-response probe and adds a
reviewer-style interpretation layer. It does not train models or modify result
directories. The goal is to separate robust evidence from small effects,
single-object positives, fitting-quality confounds, and image/content failures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import (  # noqa: E402
    GEOMETRY_DESCRIPTORS,
    RESPONSE_MODES,
    analyze_evaluated_result,
    analyze_geometry_response_descriptors,
    eligible_pair_mask,
    evaluate_liif_response_objects,
    finite_percentile,
    geometry_patch_descriptors,
    l2_normalize_rows,
    response_requirements,
    response_patch_descriptors,
    response_source_for_mode,
    scan_result_dirs,
    make_patch_grid,
    pairwise_distances,
)


DEFAULT_RESPONSE_MODES = (
    "trajectory_delta",
    "feature_trajectory_delta",
    "coord_jacobian_final_delta",
)
CONTENT_STRATA = ("gradient", "variance")
STRATUM_LABELS = ("low", "mid", "high")
SPATIAL_REGIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
CANDIDATE_ATTRIBUTE_SPECS = (
    ("geometry_distance", "lower", "top1_geometry_dist", "oracle_geometry_dist"),
    (
        "edge_orientation_agreement",
        "higher",
        "top1_edge_orientation_agreement",
        "oracle_edge_orientation_agreement",
    ),
    (
        "edge_weighted_orientation_agreement",
        "higher",
        "top1_edge_weighted_orientation_agreement",
        "oracle_edge_weighted_orientation_agreement",
    ),
    (
        "structure_coherence_absdiff",
        "lower",
        "top1_structure_coherence_absdiff",
        "oracle_structure_coherence_absdiff",
    ),
    (
        "gradient_mean_absdiff",
        "lower",
        "top1_gradient_mean_absdiff",
        "oracle_gradient_mean_absdiff",
    ),
    (
        "laplacian_abs_mean_absdiff",
        "lower",
        "top1_laplacian_abs_mean_absdiff",
        "oracle_laplacian_abs_mean_absdiff",
    ),
    (
        "patch_variance_absdiff",
        "lower",
        "top1_patch_variance_absdiff",
        "oracle_patch_variance_absdiff",
    ),
    (
        "context_lowfreq_dist",
        "lower",
        "top1_context_lowfreq_dist",
        "oracle_context_lowfreq_dist",
    ),
    ("same_spatial_region", "higher", "top1_same_spatial_region", "oracle_same_spatial_region"),
    (
        "context_descriptor_dist",
        "lower",
        "top1_rerank_descriptor_dist",
        "oracle_rerank_descriptor_dist",
    ),
)


def parse_csv_values(value: str, allowed: Iterable[str], name: str) -> list[str]:
    """Parse comma-separated CLI values and validate against allowed names."""
    allowed_set = set(allowed)
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    invalid = [item for item in values if item not in allowed_set]
    if invalid:
        raise ValueError(f"unknown {name}: {', '.join(invalid)}")
    return values


def effect_label(
    effect: float,
    frac: float,
    p_value: float,
    rho: float,
    *,
    min_effect: float = 0.03,
    min_frac: float = 0.02,
    max_p: float = 0.05,
    min_abs_rho: float = 0.05,
) -> str:
    """Classify one result row without inflating tiny positive deltas."""
    if not all(math.isfinite(x) for x in (effect, frac, p_value)):
        return "invalid"
    if effect <= 0.0:
        return "negative"
    if effect < min_effect or frac < min_frac:
        return "tiny"
    if p_value > max_p:
        return "weak_control"
    if math.isfinite(rho) and abs(rho) < min_abs_rho:
        return "local_only"
    return "support"


def row_review(row: dict[str, Any]) -> dict[str, Any]:
    """Add reviewer-facing labels and caution flags to one probe row."""
    effect = float(row["effect_vs_shuffle"])
    frac = float(row["effect_vs_shuffle_frac"])
    p_value = float(row["shuffled_response_p_le"])
    rho = float(row["geometry_response_spearman"])
    label = effect_label(effect, frac, p_value, rho)
    flags: list[str] = []
    if label in {"negative", "tiny", "weak_control"}:
        flags.append("does_not_support_claim")
    if label == "local_only":
        flags.append("nearest_neighbor_only_not_global")
    if math.isfinite(rho) and abs(rho) < 0.05:
        flags.append("spearman_near_zero")
    if effect > 0.0 and frac < 0.02:
        flags.append("fractional_gain_too_small")
    if float(row.get("summary_final_psnr", float("nan"))) < 25.0:
        flags.append("low_final_psnr_confound")
    if float(row.get("final_psnr_abs_error", 0.0)) > 1e-4:
        flags.append("response_reconstruction_check_failed")
    return {
        "run": row["run"],
        "image": row["image"],
        "seed": row["seed"],
        "response_mode": row["response_mode"],
        "effect_vs_shuffle": effect,
        "effect_vs_shuffle_frac": frac,
        "shuffled_response_p_le": p_value,
        "geometry_response_spearman": rho,
        "summary_final_psnr": float(row["summary_final_psnr"]),
        "recomputed_initial_psnr": float(row["recomputed_initial_psnr"]),
        "response_norm_mean": float(row["response_norm_mean"]),
        "neighbor_response_dist": float(row["neighbor_response_dist"]),
        "shuffled_response_dist_mean": float(row["shuffled_response_dist_mean"]),
        "label": label,
        "flags": flags,
    }


def patch_content_metrics(image: np.ndarray, patch_size: int, stride: int) -> dict[str, float]:
    """Summarize patch-level local content from an HR or upsampled LR image."""
    grid = make_patch_grid(image.shape[0], image.shape[1], patch_size, stride)
    gray = np.asarray(image, dtype=np.float64).mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    descriptors = geometry_patch_descriptors(
        image,
        grid,
        patch_size,
        descriptor="rgb_grad",
        normalize=False,
    )
    geom_norms = np.linalg.norm(descriptors, axis=1)
    patch_grad = []
    patch_var = []
    for y, x in grid.starts:
        g_patch = grad_mag[y:y + patch_size, x:x + patch_size]
        i_patch = image[y:y + patch_size, x:x + patch_size]
        patch_grad.append(float(np.mean(g_patch)))
        patch_var.append(float(np.var(i_patch)))
    patch_grad_arr = np.asarray(patch_grad, dtype=np.float64)
    patch_var_arr = np.asarray(patch_var, dtype=np.float64)
    return {
        "content_n_patches": float(len(grid.starts)),
        "content_grad_mean": float(np.mean(patch_grad_arr)),
        "content_grad_p10": finite_percentile(patch_grad_arr, 10),
        "content_grad_p50": finite_percentile(patch_grad_arr, 50),
        "content_grad_p90": finite_percentile(patch_grad_arr, 90),
        "content_var_mean": float(np.mean(patch_var_arr)),
        "content_var_p10": finite_percentile(patch_var_arr, 10),
        "content_var_p50": finite_percentile(patch_var_arr, 50),
        "content_var_p90": finite_percentile(patch_var_arr, 90),
        "content_geom_norm_mean": float(np.mean(geom_norms)),
        "content_geom_norm_p50": finite_percentile(geom_norms, 50),
    }


def patch_content_arrays(
    image: np.ndarray,
    patch_size: int,
    stride: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return patch grid plus per-patch gradient and variance diagnostics."""
    grid = make_patch_grid(image.shape[0], image.shape[1], patch_size, stride)
    gray = np.asarray(image, dtype=np.float64).mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    patch_grad = []
    patch_var = []
    for y, x in grid.starts:
        g_patch = grad_mag[y:y + patch_size, x:x + patch_size]
        i_patch = image[y:y + patch_size, x:x + patch_size]
        patch_grad.append(float(np.mean(g_patch)))
        patch_var.append(float(np.var(i_patch)))
    return (
        grid,
        np.asarray(patch_grad, dtype=np.float64),
        np.asarray(patch_var, dtype=np.float64),
    )


def rank_fraction(values: np.ndarray) -> np.ndarray:
    """Return deterministic percentile ranks in [0, 1] for one-dimensional values."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("rank values must be one-dimensional")
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    if arr.size == 1:
        return np.asarray([0.0], dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and arr[order[end]] == arr[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks / float(arr.size - 1)


def tercile_masks(values: np.ndarray) -> dict[str, np.ndarray]:
    """Split patch scalar values into deterministic low/mid/high rank terciles."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("tercile values must be one-dimensional")
    order = np.argsort(arr, kind="mergesort")
    groups = np.array_split(order, 3)
    masks = {}
    for label, group in zip(STRATUM_LABELS, groups):
        mask = np.zeros(arr.shape[0], dtype=bool)
        mask[group] = True
        masks[label] = mask
    return masks


def spatial_region_labels(centers: np.ndarray) -> np.ndarray:
    """Assign patch centers to deterministic image quadrants."""
    arr = np.asarray(centers, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("centers must have shape [n, 2]")
    mid_y = float(np.median(arr[:, 0]))
    mid_x = float(np.median(arr[:, 1]))
    labels = []
    for y, x in arr:
        vertical = "top" if y <= mid_y else "bottom"
        horizontal = "left" if x <= mid_x else "right"
        labels.append(f"{vertical}_{horizontal}")
    return np.asarray(labels, dtype=object)


def response_distance_percentiles(
    response_dist: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Percentile-rank each directed response distance within its eligible row."""
    dist = np.asarray(response_dist, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if dist.shape != mask.shape or dist.ndim != 2:
        raise ValueError("response distance and eligible mask must have matching square shape")
    percentiles = np.full(dist.shape, np.nan, dtype=np.float64)
    for i in range(dist.shape[0]):
        candidates = np.flatnonzero(mask[i])
        if candidates.size == 0:
            continue
        ranks = rank_fraction(dist[i, candidates])
        percentiles[i, candidates] = ranks
    return percentiles


def geometry_topk_indices(
    geometry_dist: np.ndarray,
    eligible: np.ndarray,
    k: int,
) -> list[np.ndarray]:
    """Return directed geometry top-k candidate indices for every anchor patch."""
    if k <= 0:
        raise ValueError("k must be positive")
    dist = np.asarray(geometry_dist, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if dist.shape != mask.shape or dist.ndim != 2:
        raise ValueError("geometry distance and eligible mask must have matching square shape")
    rows = []
    for i in range(dist.shape[0]):
        candidates = np.flatnonzero(mask[i])
        if candidates.size == 0:
            rows.append(np.asarray([], dtype=np.int64))
            continue
        order = candidates[np.argsort(dist[i, candidates], kind="mergesort")]
        rows.append(order[:k].astype(np.int64, copy=False))
    return rows


def patch_failure_rows(
    geometry_desc: np.ndarray,
    response_desc: np.ndarray,
    centers: np.ndarray,
    patch_grad: np.ndarray,
    patch_var: np.ndarray,
    patch_size: int,
    *,
    k: int,
    min_spatial_distance: float,
    response_norm_values: np.ndarray | None = None,
    rerank_desc: np.ndarray | None = None,
    candidate_attrs: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Build per-patch diagnostics for geometry-neighbor response failures.

    The rows are diagnostic only: high percentiles mean the geometry-nearest
    candidate is response-distant compared with other eligible candidates for
    the same anchor, not that the patch is intrinsically bad.
    """
    geometry = np.asarray(geometry_desc, dtype=np.float64)
    response = np.asarray(response_desc, dtype=np.float64)
    centers_arr = np.asarray(centers, dtype=np.float64)
    if geometry.shape[0] != response.shape[0] or geometry.shape[0] != centers_arr.shape[0]:
        raise ValueError("geometry, response, and centers must have the same patch count")
    if geometry.shape[0] != len(patch_grad) or geometry.shape[0] != len(patch_var):
        raise ValueError("content arrays must have the same patch count as descriptors")
    attrs = candidate_attrs or {}
    for name, values in attrs.items():
        arr = np.asarray(values)
        if arr.shape[0] != geometry.shape[0]:
            raise ValueError(f"candidate attribute {name} must have the same patch count")
    rerank_dist = None
    if rerank_desc is not None:
        rerank = np.asarray(rerank_desc, dtype=np.float64)
        if rerank.shape[0] != geometry.shape[0]:
            raise ValueError("rerank descriptors must have the same patch count")
        rerank_dist = pairwise_distances(rerank)

    geometry_dist = pairwise_distances(geometry)
    response_dist = pairwise_distances(response)
    eligible = eligible_pair_mask(centers_arr, min_spatial_distance)
    topk = geometry_topk_indices(geometry_dist, eligible, k=k)
    response_percentiles = response_distance_percentiles(response_dist, eligible)
    if response_norm_values is None:
        response_norms = np.linalg.norm(response, axis=1)
    else:
        response_norms = np.asarray(response_norm_values, dtype=np.float64)
        if response_norms.shape[0] != geometry.shape[0]:
            raise ValueError("response norm values must have the same patch count")
    response_norm_rank = rank_fraction(response_norms)
    grad_rank = rank_fraction(patch_grad)
    var_rank = rank_fraction(patch_var)
    regions = spatial_region_labels(centers_arr)

    rows: list[dict[str, Any]] = []
    radius = patch_size // 2
    for anchor, candidates in enumerate(topk):
        if candidates.size == 0:
            continue
        top1 = int(candidates[0])
        topk_response = response_dist[anchor, candidates]
        topk_geometry = geometry_dist[anchor, candidates]
        topk_percentiles = response_percentiles[anchor, candidates]
        oracle_rank0 = int(np.argmin(topk_response))
        oracle_index = int(candidates[oracle_rank0])
        oracle_percentile = float(topk_percentiles[oracle_rank0])
        top1_percentile = float(response_percentiles[anchor, top1])
        oracle_gain = top1_percentile - oracle_percentile
        rerank_fields: dict[str, Any] = {}
        if rerank_dist is not None:
            rerank_candidate_dist = rerank_dist[anchor, candidates]
            rerank_rank0 = int(np.argmin(rerank_candidate_dist))
            rerank_index = int(candidates[rerank_rank0])
            rerank_percentile = float(response_percentiles[anchor, rerank_index])
            rerank_gain = top1_percentile - rerank_percentile
            oracle_descriptor_dist = float(rerank_dist[anchor, oracle_index])
            top1_descriptor_dist = float(rerank_dist[anchor, top1])
            rerank_order = np.argsort(rerank_candidate_dist, kind="mergesort")
            rerank_positions = np.empty(candidates.size, dtype=np.float64)
            rerank_positions[rerank_order] = np.arange(1, candidates.size + 1, dtype=np.float64)
            rerank_fields = {
                "rerank_index": rerank_index,
                "rerank_geometry_rank": float(rerank_rank0 + 1),
                "top1_rerank_descriptor_rank": float(rerank_positions[0]),
                "rerank_center_y": float(centers_arr[rerank_index, 0]),
                "rerank_center_x": float(centers_arr[rerank_index, 1]),
                "rerank_geometry_dist": float(geometry_dist[anchor, rerank_index]),
                "rerank_descriptor_dist": float(rerank_dist[anchor, rerank_index]),
                "rerank_response_dist": float(response_dist[anchor, rerank_index]),
                "rerank_response_percentile": rerank_percentile,
                "rerank_response_percentile_gain": float(rerank_gain),
                "rerank_oracle_percentile_gap": float(rerank_percentile - oracle_percentile),
                "oracle_rerank_descriptor_rank": float(rerank_positions[oracle_rank0]),
                "oracle_rerank_descriptor_dist": oracle_descriptor_dist,
                "top1_rerank_descriptor_dist": top1_descriptor_dist,
                "oracle_minus_top1_rerank_descriptor_dist": float(
                    oracle_descriptor_dist - top1_descriptor_dist
                ),
            }
        candidate_fields: dict[str, Any] = {}
        if attrs:
            edge_unit = np.asarray(attrs["edge_orientation_unit"], dtype=np.float64)
            edge_strength = np.asarray(attrs["edge_strength"], dtype=np.float64)
            coherence = np.asarray(attrs["structure_coherence"], dtype=np.float64)
            gradient_mean = np.asarray(attrs["gradient_mean"], dtype=np.float64)
            laplacian_abs = np.asarray(attrs["laplacian_abs_mean"], dtype=np.float64)
            variance = np.asarray(attrs["patch_variance"], dtype=np.float64)
            lowfreq = np.asarray(attrs["context_lowfreq"], dtype=np.float64)
            attr_regions = np.asarray(attrs["spatial_region"], dtype=object)

            def orientation_agreement(a: int, b: int) -> float:
                return float(np.dot(edge_unit[a], edge_unit[b]))

            def weighted_orientation_agreement(a: int, b: int) -> float:
                weight = math.sqrt(max(0.0, float(coherence[a])) * max(0.0, float(coherence[b])))
                return float(orientation_agreement(a, b) * weight)

            def lowfreq_dist(a: int, b: int) -> float:
                return float(np.linalg.norm(lowfreq[a] - lowfreq[b]))

            for prefix, candidate in (("top1", top1), ("oracle", oracle_index)):
                candidate_fields.update(
                    {
                        f"{prefix}_edge_orientation_agreement": orientation_agreement(anchor, candidate),
                        f"{prefix}_edge_weighted_orientation_agreement": (
                            weighted_orientation_agreement(anchor, candidate)
                        ),
                        f"{prefix}_structure_coherence_absdiff": float(
                            abs(coherence[anchor] - coherence[candidate])
                        ),
                        f"{prefix}_gradient_mean_absdiff": float(
                            abs(gradient_mean[anchor] - gradient_mean[candidate])
                        ),
                        f"{prefix}_laplacian_abs_mean_absdiff": float(
                            abs(laplacian_abs[anchor] - laplacian_abs[candidate])
                        ),
                        f"{prefix}_patch_variance_absdiff": float(
                            abs(variance[anchor] - variance[candidate])
                        ),
                        f"{prefix}_context_lowfreq_dist": lowfreq_dist(anchor, candidate),
                        f"{prefix}_same_spatial_region": float(
                            attr_regions[anchor] == attr_regions[candidate]
                        ),
                    }
                )
            candidate_fields.update(
                {
                    "anchor_edge_strength": float(edge_strength[anchor]),
                    "anchor_structure_coherence": float(coherence[anchor]),
                    "top1_edge_strength": float(edge_strength[top1]),
                    "oracle_edge_strength": float(edge_strength[oracle_index]),
                }
            )
        if topk_geometry[0] <= 1e-12:
            topk_to_top1_ratio = 1.0 if topk_geometry[-1] <= 1e-12 else float("inf")
            top2_to_top1_ratio = 1.0 if topk_geometry.size > 1 and topk_geometry[1] <= 1e-12 else float("inf")
        else:
            topk_to_top1_ratio = float(topk_geometry[-1] / topk_geometry[0])
            top2_to_top1_ratio = (
                float(topk_geometry[1] / topk_geometry[0])
                if topk_geometry.size > 1 else float("nan")
            )
        n_ties_1pct = float(np.sum(topk_geometry <= topk_geometry[0] * 1.01 + 1e-12))
        rows.append(
            {
                "patch_index": int(anchor),
                "center_y": float(centers_arr[anchor, 0]),
                "center_x": float(centers_arr[anchor, 1]),
                "start_y": float(centers_arr[anchor, 0] - radius),
                "start_x": float(centers_arr[anchor, 1] - radius),
                "spatial_region": str(regions[anchor]),
                "patch_grad": float(patch_grad[anchor]),
                "patch_var": float(patch_var[anchor]),
                "patch_grad_rank": float(grad_rank[anchor]),
                "patch_var_rank": float(var_rank[anchor]),
                "response_norm": float(response_norms[anchor]),
                "response_norm_rank": float(response_norm_rank[anchor]),
                "top1_index": top1,
                "top1_center_y": float(centers_arr[top1, 0]),
                "top1_center_x": float(centers_arr[top1, 1]),
                "top1_geometry_dist": float(geometry_dist[anchor, top1]),
                "top1_response_dist": float(response_dist[anchor, top1]),
                "top1_response_percentile": top1_percentile,
                "top1_response_norm": float(response_norms[top1]),
                "top1_response_norm_rank": float(response_norm_rank[top1]),
                "top1_response_norm_rank_absdiff": float(
                    abs(response_norm_rank[anchor] - response_norm_rank[top1])
                ),
                "oracle_index": oracle_index,
                "oracle_geometry_rank": float(oracle_rank0 + 1),
                "oracle_center_y": float(centers_arr[oracle_index, 0]),
                "oracle_center_x": float(centers_arr[oracle_index, 1]),
                "oracle_geometry_dist": float(geometry_dist[anchor, oracle_index]),
                "oracle_response_dist": float(response_dist[anchor, oracle_index]),
                "oracle_response_percentile": oracle_percentile,
                "oracle_response_norm": float(response_norms[oracle_index]),
                "oracle_response_norm_rank": float(response_norm_rank[oracle_index]),
                "oracle_response_norm_rank_absdiff": float(
                    abs(response_norm_rank[anchor] - response_norm_rank[oracle_index])
                ),
                "oracle_response_percentile_gain": float(oracle_gain),
                "topk_geometry_dist_mean": float(np.mean(topk_geometry)),
                "topk_geometry_dist_std": (
                    float(np.std(topk_geometry, ddof=1)) if topk_geometry.size > 1 else 0.0
                ),
                "topk_response_dist_mean": float(np.mean(topk_response)),
                "topk_response_dist_std": (
                    float(np.std(topk_response, ddof=1)) if topk_response.size > 1 else 0.0
                ),
                "topk_response_percentile_mean": float(np.nanmean(topk_percentiles)),
                "topk_response_percentile_p90": finite_percentile(topk_percentiles, 90),
                "geometry_topk_to_top1_ratio": topk_to_top1_ratio,
                "geometry_top2_to_top1_ratio": top2_to_top1_ratio,
                "geometry_top2_minus_top1": (
                    float(topk_geometry[1] - topk_geometry[0])
                    if topk_geometry.size > 1 else float("nan")
                ),
                "geometry_ambiguity_ratio": topk_to_top1_ratio,
                "geometry_ambiguity_abs": float(topk_geometry[-1] - topk_geometry[0]),
                "n_geometry_ties_1pct": n_ties_1pct,
                "topk_size": float(candidates.size),
                **rerank_fields,
                **candidate_fields,
            }
        )
    return rows


def summarize_patch_failures(
    rows: list[dict[str, Any]],
    *,
    worst_percentile: float = 0.8,
    nonunique_ratio: float = 1.05,
) -> dict[str, Any]:
    """Summarize per-patch failure rows into compact reviewer diagnostics."""
    if not rows:
        return {
            "n_patch_rows": 0.0,
            "worst_response_percentile_threshold": worst_percentile,
            "nonunique_geometry_ratio_threshold": nonunique_ratio,
            "has_patch_rerank": False,
            "worst_response_frac": float("nan"),
            "nonunique_geometry_frac": float("nan"),
        }

    top1_pct = np.asarray([row["top1_response_percentile"] for row in rows], dtype=np.float64)
    oracle_pct = np.asarray([row["oracle_response_percentile"] for row in rows], dtype=np.float64)
    oracle_gain = np.asarray(
        [row["oracle_response_percentile_gain"] for row in rows],
        dtype=np.float64,
    )
    oracle_rank = np.asarray([row["oracle_geometry_rank"] for row in rows], dtype=np.float64)
    topk_pct_mean = np.asarray(
        [row["topk_response_percentile_mean"] for row in rows],
        dtype=np.float64,
    )
    topk_to_top1_ratio = np.asarray(
        [row["geometry_topk_to_top1_ratio"] for row in rows],
        dtype=np.float64,
    )
    top2_to_top1_ratio = np.asarray(
        [row["geometry_top2_to_top1_ratio"] for row in rows],
        dtype=np.float64,
    )
    top2_minus_top1 = np.asarray(
        [row["geometry_top2_minus_top1"] for row in rows],
        dtype=np.float64,
    )
    tie_counts = np.asarray([row["n_geometry_ties_1pct"] for row in rows], dtype=np.float64)
    response_norm_rank = np.asarray([row["response_norm_rank"] for row in rows], dtype=np.float64)
    grad_rank = np.asarray([row["patch_grad_rank"] for row in rows], dtype=np.float64)
    var_rank = np.asarray([row["patch_var_rank"] for row in rows], dtype=np.float64)
    worst_mask = top1_pct >= worst_percentile
    nonunique_mask = tie_counts >= 2.0
    top2_nonunique_mask = top2_to_top1_ratio <= nonunique_ratio
    oracle_help_mask = oracle_gain >= 0.2
    oracle_fixes_worst_mask = worst_mask & (oracle_pct < worst_percentile)
    has_rerank = all("rerank_response_percentile" in row for row in rows)

    summary: dict[str, Any] = {
        "n_patch_rows": float(len(rows)),
        "worst_response_percentile_threshold": float(worst_percentile),
        "nonunique_geometry_ratio_threshold": float(nonunique_ratio),
        "has_patch_rerank": has_rerank,
        "top1_response_percentile_mean": float(np.nanmean(top1_pct)),
        "top1_response_percentile_p50": finite_percentile(top1_pct, 50),
        "top1_response_percentile_p90": finite_percentile(top1_pct, 90),
        "topk_response_percentile_mean": float(np.nanmean(topk_pct_mean)),
        "oracle_response_percentile_mean": float(np.nanmean(oracle_pct)),
        "oracle_response_percentile_p50": finite_percentile(oracle_pct, 50),
        "oracle_response_percentile_p90": finite_percentile(oracle_pct, 90),
        "oracle_response_percentile_gain_mean": float(np.nanmean(oracle_gain)),
        "oracle_response_percentile_gain_p90": finite_percentile(oracle_gain, 90),
        "oracle_geometry_rank_mean": float(np.nanmean(oracle_rank)),
        "oracle_geometry_rank_p50": finite_percentile(oracle_rank, 50),
        "oracle_help_frac": float(np.mean(oracle_help_mask)),
        "oracle_fixes_worst_frac": float(np.mean(oracle_fixes_worst_mask)),
        "worst_response_frac": float(np.mean(worst_mask)),
        "worst_response_grad_rank_mean": (
            float(np.mean(grad_rank[worst_mask])) if np.any(worst_mask) else float("nan")
        ),
        "worst_response_var_rank_mean": (
            float(np.mean(var_rank[worst_mask])) if np.any(worst_mask) else float("nan")
        ),
        "worst_response_norm_rank_mean": (
            float(np.mean(response_norm_rank[worst_mask])) if np.any(worst_mask) else float("nan")
        ),
        "geometry_topk_to_top1_ratio_p10": finite_percentile(topk_to_top1_ratio, 10),
        "geometry_topk_to_top1_ratio_p50": finite_percentile(topk_to_top1_ratio, 50),
        "geometry_top2_to_top1_ratio_p10": finite_percentile(top2_to_top1_ratio, 10),
        "geometry_top2_to_top1_ratio_p50": finite_percentile(top2_to_top1_ratio, 50),
        "geometry_top2_minus_top1_p50": finite_percentile(top2_minus_top1, 50),
        "geometry_ties_1pct_mean": float(np.mean(tie_counts)),
        "geometry_ties_1pct_p90": finite_percentile(tie_counts, 90),
        "nonunique_geometry_frac": float(np.mean(nonunique_mask)),
        "top2_nonunique_geometry_frac": float(np.mean(top2_nonunique_mask)),
        "worst_and_nonunique_frac": float(np.mean(worst_mask & nonunique_mask)),
        "worst_and_top2_nonunique_frac": float(np.mean(worst_mask & top2_nonunique_mask)),
    }

    if has_rerank:
        rerank_pct = np.asarray(
            [row["rerank_response_percentile"] for row in rows],
            dtype=np.float64,
        )
        rerank_gain = np.asarray(
            [row["rerank_response_percentile_gain"] for row in rows],
            dtype=np.float64,
        )
        rerank_rank = np.asarray(
            [row["rerank_geometry_rank"] for row in rows],
            dtype=np.float64,
        )
        rerank_oracle_gap = np.asarray(
            [row["rerank_oracle_percentile_gap"] for row in rows],
            dtype=np.float64,
        )
        oracle_rerank_rank = np.asarray(
            [row["oracle_rerank_descriptor_rank"] for row in rows],
            dtype=np.float64,
        )
        oracle_minus_top1_rerank_dist = np.asarray(
            [row["oracle_minus_top1_rerank_descriptor_dist"] for row in rows],
            dtype=np.float64,
        )
        rerank_help_mask = rerank_gain >= 0.2
        rerank_fixes_worst_mask = worst_mask & (rerank_pct < worst_percentile)
        oracle_is_rerank_best_mask = oracle_rerank_rank <= 1.0
        oracle_rerank_farther_mask = oracle_minus_top1_rerank_dist > 0.0
        summary.update(
            {
                "rerank_response_percentile_mean": float(np.nanmean(rerank_pct)),
                "rerank_response_percentile_p50": finite_percentile(rerank_pct, 50),
                "rerank_response_percentile_p90": finite_percentile(rerank_pct, 90),
                "rerank_response_percentile_gain_mean": float(np.nanmean(rerank_gain)),
                "rerank_recovered_oracle_gain_frac_mean": (
                    float(np.nanmean(rerank_gain) / np.nanmean(oracle_gain))
                    if abs(float(np.nanmean(oracle_gain))) > 1e-12 else float("nan")
                ),
                "rerank_response_percentile_gain_p90": finite_percentile(rerank_gain, 90),
                "rerank_geometry_rank_mean": float(np.nanmean(rerank_rank)),
                "rerank_geometry_rank_p50": finite_percentile(rerank_rank, 50),
                "rerank_help_frac": float(np.mean(rerank_help_mask)),
                "rerank_fixes_worst_frac": float(np.mean(rerank_fixes_worst_mask)),
                "rerank_worsens_frac": float(np.mean(rerank_gain < -0.05)),
                "rerank_oracle_percentile_gap_mean": float(np.nanmean(rerank_oracle_gap)),
                "rerank_oracle_percentile_gap_p90": finite_percentile(rerank_oracle_gap, 90),
                "oracle_rerank_descriptor_rank_mean": float(np.nanmean(oracle_rerank_rank)),
                "oracle_rerank_descriptor_rank_p50": finite_percentile(oracle_rerank_rank, 50),
                "oracle_is_rerank_best_frac": float(np.mean(oracle_is_rerank_best_mask)),
                "oracle_rerank_farther_than_top1_frac": float(np.mean(oracle_rerank_farther_mask)),
                "oracle_minus_top1_rerank_descriptor_dist_mean": float(
                    np.nanmean(oracle_minus_top1_rerank_dist)
                ),
                "oracle_minus_top1_rerank_descriptor_dist_p50": finite_percentile(
                    oracle_minus_top1_rerank_dist,
                    50,
                ),
            }
        )

    regions = sorted({str(row["spatial_region"]) for row in rows})
    for region in regions:
        region_mask = np.asarray([row["spatial_region"] == region for row in rows], dtype=bool)
        summary[f"region_{region}_frac"] = float(np.mean(region_mask))
        summary[f"region_{region}_worst_frac"] = (
            float(np.mean(worst_mask[region_mask])) if np.any(region_mask) else float("nan")
        )
    return summary


def directed_attribute_advantage(
    rows: list[dict[str, Any]],
    *,
    min_oracle_gain: float = 0.2,
) -> dict[str, Any]:
    """Summarize whether oracle candidates differ from top1 in frozen attributes."""
    valid_rows = [
        row for row in rows
        if row.get("oracle_response_percentile_gain", 0.0) >= min_oracle_gain
    ]
    summary: dict[str, Any] = {
        "candidate_attribute_rows": float(len(rows)),
        "candidate_attribute_min_oracle_gain": float(min_oracle_gain),
        "candidate_attribute_oracle_help_rows": float(len(valid_rows)),
        "candidate_attribute_oracle_help_frac": (
            float(len(valid_rows) / len(rows)) if rows else float("nan")
        ),
    }
    if not valid_rows:
        return summary

    for name, direction, top1_key, oracle_key in CANDIDATE_ATTRIBUTE_SPECS:
        if top1_key not in valid_rows[0] or oracle_key not in valid_rows[0]:
            continue
        top1_values = np.asarray([row[top1_key] for row in valid_rows], dtype=np.float64)
        oracle_values = np.asarray([row[oracle_key] for row in valid_rows], dtype=np.float64)
        diff = oracle_values - top1_values
        if direction == "lower":
            advantage = diff < 0.0
            signed_advantage = -diff
        else:
            advantage = diff > 0.0
            signed_advantage = diff
        summary.update(
            {
                f"{name}_top1_mean": float(np.nanmean(top1_values)),
                f"{name}_oracle_mean": float(np.nanmean(oracle_values)),
                f"{name}_oracle_minus_top1_mean": float(np.nanmean(diff)),
                f"{name}_oracle_better_frac": float(np.mean(advantage)),
                f"{name}_signed_advantage_mean": float(np.nanmean(signed_advantage)),
                f"{name}_signed_advantage_p50": finite_percentile(signed_advantage, 50),
            }
        )
    return summary


def select_worst_patch_rows(
    rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Select the highest response-mismatch patch rows with stable tie-breaking."""
    if limit <= 0:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["top1_response_percentile"]),
            -float(row["top1_response_dist"]),
            int(row["patch_index"]),
        ),
    )
    return ordered[:limit]


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an RGB float image in [0, 1] or [0, 255] to uint8."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected an RGB image with shape [H, W, 3]")
    arr = arr.astype(np.float64)
    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def crop_center_with_padding(image: np.ndarray, center_y: float, center_x: float, size: int) -> np.ndarray:
    """Extract a square RGB crop centered at a patch center, padding at borders."""
    if size <= 0:
        raise ValueError("crop size must be positive")
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


def context_lowfreq_descriptor(crop: np.ndarray, bins: int = 4) -> np.ndarray:
    """Return a small response-blind low-frequency layout descriptor for RGB context."""
    arr = np.asarray(crop, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected an RGB context crop")
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    gray = arr.mean(axis=2)
    y_edges = np.linspace(0, gray.shape[0], bins + 1).round().astype(int)
    x_edges = np.linspace(0, gray.shape[1], bins + 1).round().astype(int)
    values = []
    for yi in range(bins):
        for xi in range(bins):
            block = gray[y_edges[yi]:y_edges[yi + 1], x_edges[xi]:x_edges[xi + 1]]
            values.append(float(np.mean(block)) if block.size else 0.0)
    vec = np.asarray(values, dtype=np.float64)
    vec = vec - float(np.mean(vec))
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


def patch_candidate_attribute_arrays(
    image: np.ndarray,
    grid: Any,
    patch_grad: np.ndarray,
    patch_var: np.ndarray,
    patch_size: int,
) -> dict[str, np.ndarray]:
    """Compute pre-registered response-blind attributes for candidate comparison."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    gray = arr.mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    grad_yy, _ = np.gradient(grad_y)
    _, grad_xx = np.gradient(grad_x)
    laplacian = grad_xx + grad_yy

    edge_units = []
    edge_strength = []
    coherence = []
    laplacian_abs = []
    lowfreq = []
    context_size = patch_size * 3
    for (y, x), center in zip(grid.starts, grid.centers):
        gx_patch = grad_x[y:y + patch_size, x:x + patch_size]
        gy_patch = grad_y[y:y + patch_size, x:x + patch_size]
        gmag_patch = grad_mag[y:y + patch_size, x:x + patch_size]
        jxx = float(np.mean(gx_patch * gx_patch))
        jyy = float(np.mean(gy_patch * gy_patch))
        jxy = float(np.mean(gx_patch * gy_patch))
        trace = jxx + jyy
        anisotropy = math.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy)
        theta = 0.5 * math.atan2(2.0 * jxy, jxx - jyy)
        edge_units.append([math.cos(2.0 * theta), math.sin(2.0 * theta)])
        edge_strength.append(float(np.mean(gmag_patch)))
        coherence.append(float(anisotropy / (trace + 1e-12)))
        lap_patch = laplacian[y:y + patch_size, x:x + patch_size]
        laplacian_abs.append(float(np.mean(np.abs(lap_patch))))
        context = crop_center_with_padding(
            arr,
            center_y=float(center[0]),
            center_x=float(center[1]),
            size=context_size,
        )
        lowfreq.append(context_lowfreq_descriptor(context))

    return {
        "edge_orientation_unit": np.asarray(edge_units, dtype=np.float64),
        "edge_strength": np.asarray(edge_strength, dtype=np.float64),
        "structure_coherence": np.asarray(coherence, dtype=np.float64),
        "gradient_mean": np.asarray(patch_grad, dtype=np.float64),
        "laplacian_abs_mean": np.asarray(laplacian_abs, dtype=np.float64),
        "patch_variance": np.asarray(patch_var, dtype=np.float64),
        "context_lowfreq": np.asarray(lowfreq, dtype=np.float64),
        "spatial_region": spatial_region_labels(np.asarray(grid.centers, dtype=np.float64)),
    }


def _resize_nearest(crop: np.ndarray, scale: int) -> Any:
    """Resize an RGB crop with nearest-neighbor interpolation."""
    from PIL import Image

    if scale <= 0:
        raise ValueError("patch crop scale must be positive")
    image = Image.fromarray(image_to_uint8(crop))
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def render_patch_crop_grid(
    image: np.ndarray,
    rows: list[dict[str, Any]],
    patch_size: int,
    *,
    scale: int = 14,
    context_factor: int = 3,
) -> Any:
    """Render diagnostic anchor/neighbor patch crops for manual inspection."""
    from PIL import Image, ImageDraw, ImageFont

    if not rows:
        raise ValueError("at least one patch row is required")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    context_size = max(patch_size, patch_size * context_factor)
    cell_size = context_size * scale
    patch_cell_size = patch_size * scale
    text_width = 360
    row_pad = 10
    header_h = 34
    row_h = cell_size + row_pad
    width = cell_size * 3 + patch_cell_size * 2 + text_width + row_pad * 7
    height = header_h + row_h * len(rows) + row_pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = f"{rows[0]['run']} | {rows[0]['response_mode']} | diagnostic patch crops"
    draw.text((row_pad, 10), title, fill=(0, 0, 0), font=font)

    context_radius = context_size // 2
    patch_radius = patch_size // 2
    rect_start = (context_radius - patch_radius) * scale
    rect_end = (context_radius + patch_radius + 1) * scale - 1

    for row_index, row in enumerate(rows):
        y0 = header_h + row_index * row_h
        anchor_context = crop_center_with_padding(
            image,
            float(row["center_y"]),
            float(row["center_x"]),
            context_size,
        )
        neighbor_context = crop_center_with_padding(
            image,
            float(row["top1_center_y"]),
            float(row["top1_center_x"]),
            context_size,
        )
        anchor_patch = crop_center_with_padding(
            image,
            float(row["center_y"]),
            float(row["center_x"]),
            patch_size,
        )
        neighbor_patch = crop_center_with_padding(
            image,
            float(row["top1_center_y"]),
            float(row["top1_center_x"]),
            patch_size,
        )
        diff_patch = np.abs(anchor_patch.astype(np.float64) - neighbor_patch.astype(np.float64))
        if diff_patch.max(initial=0.0) > 1.5:
            diff_patch = diff_patch / 255.0
        diff_patch = np.clip(diff_patch * 2.0, 0.0, 1.0)

        x = row_pad
        for crop, label, box_color in (
            (anchor_context, "anchor ctx", (220, 0, 0)),
            (anchor_patch, "anchor", None),
            (neighbor_context, "neighbor ctx", (0, 80, 220)),
            (neighbor_patch, "neighbor", None),
            (diff_patch, "abs diff x2", None),
        ):
            resized = _resize_nearest(crop, scale)
            canvas.paste(resized, (x, y0))
            if box_color is not None:
                box = [x + rect_start, y0 + rect_start, x + rect_end, y0 + rect_end]
                draw.rectangle(box, outline=box_color, width=2)
            draw.text((x, y0 + resized.height + 1), label, fill=(0, 0, 0), font=font)
            x += resized.width + row_pad

        lines = [
            f"rank {row['worst_rank']}  patch {row['patch_index']} -> {row['top1_index']}",
            f"center ({row['center_y']:.0f},{row['center_x']:.0f}) -> "
            f"({row['top1_center_y']:.0f},{row['top1_center_x']:.0f})",
            f"resp_pct {row['top1_response_percentile']:.3f}  "
            f"resp_dist {row['top1_response_dist']:.4f}",
            f"geom_dist {row['top1_geometry_dist']:.4f}  "
            f"top2/top1 {row.get('geometry_top2_to_top1_ratio', float('nan')):.3f}",
            f"grad_r {row['patch_grad_rank']:.3f}  var_r {row['patch_var_rank']:.3f}  "
            f"norm_r {row['response_norm_rank']:.3f}",
            f"region {row['spatial_region']}",
        ]
        tx = x
        for line_index, line in enumerate(lines):
            draw.text((tx, y0 + line_index * 14), line, fill=(0, 0, 0), font=font)
    return canvas


def write_patch_crop_grids(
    output_dir: Path,
    result_dir: Path,
    evaluation: Any,
    geometry_source: str,
    patch_size: int,
    worst_rows: list[dict[str, Any]],
    *,
    scale: int,
) -> list[dict[str, Any]]:
    """Write diagnostic patch crop grids grouped by response mode."""
    if not worst_rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    image = evaluation.target_hr if geometry_source == "hr" else evaluation.lr_up
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in worst_rows:
        grouped[str(row["response_mode"])].append(row)

    files = []
    for response_mode, rows in sorted(grouped.items()):
        grid = render_patch_crop_grid(image, rows, patch_size, scale=scale)
        path = output_dir / f"{result_dir.name}_{response_mode}_patch_crop_diagnostic.png"
        grid.save(path)
        files.append(
            {
                "run": result_dir.name,
                "response_mode": response_mode,
                "path": str(path),
                "n_rows": float(len(rows)),
                "geometry_source": geometry_source,
                "patch_size": float(patch_size),
                "scale": float(scale),
            }
        )
    return files


def analyze_content_stratum(
    geometry_desc: np.ndarray,
    response_desc: np.ndarray,
    centers: np.ndarray,
    mask: np.ndarray,
    *,
    k: int,
    min_spatial_distance: float,
    n_shuffles: int,
    seed: int,
    min_patches: int = 12,
) -> dict[str, Any]:
    """Analyze one content stratum or return an explicit insufficiency row."""
    mask = np.asarray(mask, dtype=bool)
    n_patches = int(np.sum(mask))
    if n_patches < min_patches:
        return {"status": "insufficient_patches", "n_patches": float(n_patches)}
    try:
        metrics = analyze_geometry_response_descriptors(
            geometry_desc[mask],
            response_desc[mask],
            centers[mask],
            k=min(k, max(1, n_patches - 1)),
            min_spatial_distance=min_spatial_distance,
            n_shuffles=n_shuffles,
            seed=seed,
        )
    except ValueError as exc:
        return {
            "status": "insufficient_pairs",
            "n_patches": float(n_patches),
            "reason": str(exc),
        }
    metrics["status"] = "ok"
    metrics["n_patches"] = float(n_patches)
    return metrics


def content_stratified_reviews(
    result_dir: Path,
    evaluation: Any,
    geometry_source: str,
    geometry_descriptor: str,
    response_modes: list[str],
    patch_size: int,
    stride: int,
    k: int,
    min_spatial_distance: float,
    n_shuffles: int,
    seed: int,
    min_patches: int,
) -> list[dict[str, Any]]:
    """Run diagnostic geometry-response audit inside gradient/variance strata."""
    geometry_image = evaluation.target_hr if geometry_source == "hr" else evaluation.lr_up
    grid, patch_grad, patch_var = patch_content_arrays(geometry_image, patch_size, stride)
    geometry_raw = geometry_patch_descriptors(
        geometry_image,
        grid,
        patch_size,
        descriptor=geometry_descriptor,
        normalize=False,
    )
    geometry_desc = l2_normalize_rows(geometry_raw)
    strata = {
        "gradient": tercile_masks(patch_grad),
        "variance": tercile_masks(patch_var),
    }

    rows: list[dict[str, Any]] = []
    for mode_index, response_mode in enumerate(response_modes):
        response_source = response_source_for_mode(evaluation, response_mode)
        response_raw = response_patch_descriptors(
            response_source,
            grid,
            patch_size,
            response_mode,
            normalize=False,
        )
        response_desc = l2_normalize_rows(response_raw)
        for stratum_index, stratum_name in enumerate(CONTENT_STRATA):
            for label_index, label_name in enumerate(STRATUM_LABELS):
                metrics = analyze_content_stratum(
                    geometry_desc,
                    response_desc,
                    grid.centers,
                    strata[stratum_name][label_name],
                    k=k,
                    min_spatial_distance=min_spatial_distance,
                    n_shuffles=n_shuffles,
                    seed=seed + mode_index * 1009 + stratum_index * 101 + label_index,
                    min_patches=min_patches,
                )
                row: dict[str, Any] = {
                    "run": result_dir.name,
                    "image": evaluation.summary.get("image", "?"),
                    "seed": evaluation.summary.get("seed", "?"),
                    "response_mode": response_mode,
                    "stratum": stratum_name,
                    "stratum_label": label_name,
                    "status": metrics.pop("status"),
                }
                row.update(metrics)
                if row["status"] == "ok":
                    row["label"] = effect_label(
                        float(row["effect_vs_shuffle"]),
                        float(row["effect_vs_shuffle_frac"]),
                        float(row["shuffled_response_p_le"]),
                        float(row["geometry_response_spearman"]),
                    )
                else:
                    row["label"] = row["status"]
                rows.append(row)
    return rows


def patch_level_failure_diagnostics(
    result_dir: Path,
    evaluation: Any,
    geometry_source: str,
    geometry_descriptor: str,
    response_modes: list[str],
    patch_size: int,
    stride: int,
    k: int,
    min_spatial_distance: float,
    top_n: int,
    rerank_descriptor: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run patch-level localization and non-unique-neighbor diagnostics."""
    geometry_image = evaluation.target_hr if geometry_source == "hr" else evaluation.lr_up
    grid, patch_grad, patch_var = patch_content_arrays(geometry_image, patch_size, stride)
    candidate_attrs = patch_candidate_attribute_arrays(
        geometry_image,
        grid,
        patch_grad,
        patch_var,
        patch_size,
    )
    geometry_raw = geometry_patch_descriptors(
        geometry_image,
        grid,
        patch_size,
        descriptor=geometry_descriptor,
        normalize=False,
    )
    geometry_desc = l2_normalize_rows(geometry_raw)
    rerank_desc = None
    if rerank_descriptor is not None:
        rerank_raw = geometry_patch_descriptors(
            geometry_image,
            grid,
            patch_size,
            descriptor=rerank_descriptor,
            normalize=False,
        )
        rerank_desc = l2_normalize_rows(rerank_raw)

    summaries: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    for response_mode in response_modes:
        response_source = response_source_for_mode(evaluation, response_mode)
        response_raw = response_patch_descriptors(
            response_source,
            grid,
            patch_size,
            response_mode,
            normalize=False,
        )
        response_desc = l2_normalize_rows(response_raw)
        response_norms = np.linalg.norm(response_raw, axis=1)
        rows = patch_failure_rows(
            geometry_desc,
            response_desc,
            grid.centers,
            patch_grad,
            patch_var,
            patch_size,
            k=k,
            min_spatial_distance=min_spatial_distance,
            response_norm_values=response_norms,
            rerank_desc=rerank_desc,
            candidate_attrs=candidate_attrs,
        )
        summary = {
            "run": result_dir.name,
            "image": evaluation.summary.get("image", "?"),
            "seed": evaluation.summary.get("seed", "?"),
            "response_mode": response_mode,
            "patch_rerank_descriptor": rerank_descriptor,
        }
        summary.update(summarize_patch_failures(rows))
        summary.update(directed_attribute_advantage(rows))
        summaries.append(summary)

        for rank, row in enumerate(select_worst_patch_rows(rows, top_n), start=1):
            out_row = {
                "run": result_dir.name,
                "image": evaluation.summary.get("image", "?"),
                "seed": evaluation.summary.get("seed", "?"),
                "response_mode": response_mode,
                "worst_rank": rank,
            }
            out_row.update(row)
            worst_rows.append(out_row)
    return summaries, worst_rows


def low_frequency_summary(summary: dict[str, Any]) -> dict[str, float]:
    """Extract final and trend diagnostics from stored low-frequency ratios."""
    values = summary.get("freq_ratios", [])
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "lowfreq_initial": float("nan"),
            "lowfreq_final": float("nan"),
            "lowfreq_min": float("nan"),
            "lowfreq_rebound": float("nan"),
        }
    return {
        "lowfreq_initial": float(arr[0]),
        "lowfreq_final": float(arr[-1]),
        "lowfreq_min": float(np.min(arr)),
        "lowfreq_rebound": float(arr[-1] - np.min(arr)),
    }


def aggregate_image_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-response labels into one reviewer summary per image/seed."""
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        grouped[(str(review["image"]), review["seed"])].append(review)

    rows = []
    for (image, seed), group in sorted(grouped.items()):
        labels = [item["label"] for item in group]
        effects = [float(item["effect_vs_shuffle"]) for item in group]
        support_count = sum(label == "support" for label in labels)
        negative_count = sum(label == "negative" for label in labels)
        tiny_count = sum(label == "tiny" for label in labels)
        if support_count == len(group):
            verdict = "consistent_support"
        elif support_count > 0 and negative_count == 0:
            verdict = "mixed_weak_support"
        elif support_count > 0:
            verdict = "object_dependent"
        else:
            verdict = "failure_or_unresolved"
        rows.append(
            {
                "image": image,
                "seed": seed,
                "n_response_modes": len(group),
                "support_count": support_count,
                "negative_count": negative_count,
                "tiny_count": tiny_count,
                "min_effect": float(np.min(effects)),
                "max_effect": float(np.max(effects)),
                "mean_effect": float(np.mean(effects)),
                "verdict": verdict,
            }
        )
    return rows


def aggregate_stratified_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate content-stratified rows into reviewer-readable diagnostics."""
    grouped: dict[tuple[str, Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        grouped[
            (
                str(review["image"]),
                review["seed"],
                str(review["stratum"]),
                str(review["stratum_label"]),
            )
        ].append(review)

    rows = []
    for (image, seed, stratum, stratum_label), group in sorted(grouped.items()):
        ok_rows = [item for item in group if item["status"] == "ok"]
        labels = [str(item["label"]) for item in ok_rows]
        effects = [float(item["effect_vs_shuffle"]) for item in ok_rows]
        support_count = sum(label == "support" for label in labels)
        negative_count = sum(label == "negative" for label in labels)
        tiny_count = sum(label == "tiny" for label in labels)
        local_only_count = sum(label == "local_only" for label in labels)
        weak_control_count = sum(label == "weak_control" for label in labels)
        if not ok_rows:
            verdict = "insufficient"
            min_effect = max_effect = mean_effect = float("nan")
            n_patches = float("nan")
        else:
            n_patches = float(ok_rows[0]["n_patches"])
            min_effect = float(np.min(effects))
            max_effect = float(np.max(effects))
            mean_effect = float(np.mean(effects))
            if support_count == len(ok_rows):
                verdict = "consistent_stratum_support"
            elif support_count > 0 and negative_count == 0:
                verdict = "mixed_stratum_support"
            elif support_count > 0:
                verdict = "object_dependent"
            elif negative_count > 0:
                verdict = "no_support_or_negative"
            else:
                verdict = "no_support"
        rows.append(
            {
                "image": image,
                "seed": seed,
                "stratum": stratum,
                "stratum_label": stratum_label,
                "n_response_modes": len(group),
                "n_ok": len(ok_rows),
                "n_patches": n_patches,
                "support_count": support_count,
                "negative_count": negative_count,
                "tiny_count": tiny_count,
                "local_only_count": local_only_count,
                "weak_control_count": weak_control_count,
                "min_effect": min_effect,
                "max_effect": max_effect,
                "mean_effect": mean_effect,
                "verdict": verdict,
            }
        )
    return rows


def audit_result_dir(
    result_dir: Path,
    data_root: Path,
    device: torch.device,
    geometry_source: str,
    geometry_descriptor: str,
    response_modes: list[str],
    patch_size: int,
    stride: int,
    k: int,
    min_spatial_distance: float,
    n_shuffles: int,
    seed: int,
    content_stratification: bool = False,
    min_stratum_patches: int = 12,
    patch_failure_diagnostics: bool = False,
    patch_failure_top_n: int = 8,
    patch_rerank_descriptor: str | None = None,
    patch_crop_dir: Path | None = None,
    patch_crop_scale: int = 14,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate one result directory once, then audit multiple response modes."""
    needs_features = any(response_requirements(mode)[0] for mode in response_modes)
    needs_coord_jacobians = any(response_requirements(mode)[1] for mode in response_modes)
    evaluation = evaluate_liif_response_objects(
        result_dir,
        data_root,
        device,
        collect_features=needs_features,
        collect_coord_jacobians=needs_coord_jacobians,
    )

    reviews = []
    for mode_index, response_mode in enumerate(response_modes):
        row = analyze_evaluated_result(
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
            seed=seed + mode_index,
        )
        reviews.append(row_review(row))

    content_image = evaluation.target_hr if geometry_source == "hr" else evaluation.lr_up
    diagnostics: dict[str, Any] = {
        "run": result_dir.name,
        "image": evaluation.summary.get("image", "?"),
        "seed": evaluation.summary.get("seed", "?"),
        "summary_final_psnr": float(evaluation.summary.get("final_psnr", float("nan"))),
        "recomputed_initial_psnr": float(evaluation.summary["_recomputed_initial_psnr"]),
        "recomputed_final_psnr": float(evaluation.summary["_recomputed_final_psnr"]),
    }
    diagnostics.update(low_frequency_summary(evaluation.summary))
    diagnostics.update(patch_content_metrics(content_image, patch_size, stride))
    stratified: list[dict[str, Any]] = []
    if content_stratification:
        stratified = content_stratified_reviews(
            result_dir=result_dir,
            evaluation=evaluation,
            geometry_source=geometry_source,
            geometry_descriptor=geometry_descriptor,
            response_modes=response_modes,
            patch_size=patch_size,
            stride=stride,
            k=k,
            min_spatial_distance=min_spatial_distance,
            n_shuffles=n_shuffles,
            seed=seed + 7919,
            min_patches=min_stratum_patches,
        )
    patch_summaries: list[dict[str, Any]] = []
    worst_patches: list[dict[str, Any]] = []
    if patch_failure_diagnostics:
        patch_summaries, worst_patches = patch_level_failure_diagnostics(
            result_dir=result_dir,
            evaluation=evaluation,
            geometry_source=geometry_source,
            geometry_descriptor=geometry_descriptor,
            response_modes=response_modes,
            patch_size=patch_size,
            stride=stride,
            k=k,
            min_spatial_distance=min_spatial_distance,
            top_n=patch_failure_top_n,
            rerank_descriptor=patch_rerank_descriptor,
        )
    patch_crop_files: list[dict[str, Any]] = []
    if patch_crop_dir is not None:
        if not patch_failure_diagnostics:
            raise ValueError("--patch_crop_dir requires --patch_failure_diagnostics")
        patch_crop_files = write_patch_crop_grids(
            patch_crop_dir,
            result_dir,
            evaluation,
            geometry_source,
            patch_size,
            worst_patches,
            scale=patch_crop_scale,
        )
    return reviews, diagnostics, stratified, patch_summaries, worst_patches, patch_crop_files


def print_summary(image_rows: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> None:
    """Print reviewer-oriented tables."""
    print("\nImage-level reviewer summary")
    print("image       seed modes support neg tiny min_eff max_eff mean_eff verdict")
    print("-" * 86)
    for row in image_rows:
        print(
            f"{row['image']:<11} {row['seed']!s:<4} {row['n_response_modes']:<5} "
            f"{row['support_count']:<7} {row['negative_count']:<3} {row['tiny_count']:<4} "
            f"{row['min_effect']:<7.4f} {row['max_effect']:<7.4f} "
            f"{row['mean_effect']:<8.4f} {row['verdict']}"
        )

    print("\nResponse-level audit")
    print("run                                response                 effect  frac    p       rho     label")
    print("-" * 105)
    for row in reviews:
        print(
            f"{row['run']:<35} {row['response_mode']:<24} "
            f"{row['effect_vs_shuffle']:<7.4f} {row['effect_vs_shuffle_frac']:<7.4f} "
            f"{row['shuffled_response_p_le']:<7.4f} {row['geometry_response_spearman']:<7.4f} "
            f"{row['label']}"
        )


def print_stratified_summary(stratified: list[dict[str, Any]]) -> None:
    """Print a compact diagnostic table for content-stratified rows."""
    if not stratified:
        return
    print("\nContent-stratified diagnostic rows")
    print("run                                response                 stratum  bin   n     effect  frac    p       label")
    print("-" * 111)
    for row in stratified:
        if row["status"] != "ok":
            effect = frac = p_value = "nan"
        else:
            effect = f"{row['effect_vs_shuffle']:<7.4f}"
            frac = f"{row['effect_vs_shuffle_frac']:<7.4f}"
            p_value = f"{row['shuffled_response_p_le']:<7.4f}"
        print(
            f"{row['run']:<35} {row['response_mode']:<24} "
            f"{row['stratum']:<8} {row['stratum_label']:<5} "
            f"{row['n_patches']:<5.0f} {effect:<7} {frac:<7} {p_value:<7} "
            f"{row['label']}"
        )


def print_stratified_aggregate(stratified_rows: list[dict[str, Any]]) -> None:
    """Print aggregate support counts for content strata."""
    if not stratified_rows:
        return
    print("\nContent-stratified reviewer summary")
    print("image       seed stratum  bin   ok/modes support neg tiny local mean_eff verdict")
    print("-" * 95)
    for row in stratified_rows:
        mean_eff = "nan" if math.isnan(row["mean_effect"]) else f"{row['mean_effect']:.4f}"
        print(
            f"{row['image']:<11} {row['seed']!s:<4} {row['stratum']:<8} "
            f"{row['stratum_label']:<5} {row['n_ok']}/{row['n_response_modes']:<5} "
            f"{row['support_count']:<7} {row['negative_count']:<3} "
            f"{row['tiny_count']:<4} {row['local_only_count']:<5} "
            f"{mean_eff:<8} {row['verdict']}"
        )


def print_patch_failure_summary(rows: list[dict[str, Any]]) -> None:
    """Print patch-level non-unique-neighbor and response-mismatch diagnostics."""
    if not rows:
        return
    print("\nPatch-level failure diagnostic summary")
    has_rerank = any(bool(row.get("has_patch_rerank", False)) for row in rows)
    if has_rerank:
        print(
            "run                                response                 top1  rerank rgain rfix  "
            "oracle ogain ofix  or_ctx or_far worst  top2   norm_w region_worst"
        )
        print("-" * 163)
    else:
        print("run                                response                 top1  oracle gain  fix   worst  ties2  top2   norm_w region_worst")
        print("-" * 124)
    for row in rows:
        region_keys = [
            key for key in sorted(row)
            if key.startswith("region_") and key.endswith("_worst_frac")
        ]
        region_values = [
            (key[len("region_"):-len("_worst_frac")], float(row[key]))
            for key in region_keys
            if math.isfinite(float(row[key]))
        ]
        worst_region = "n/a"
        if region_values:
            region, value = max(region_values, key=lambda item: item[1])
            worst_region = f"{region}:{value:.2f}"
        if has_rerank:
            if row.get("has_patch_rerank", False):
                rerank_pct = f"{row['rerank_response_percentile_mean']:<6.3f}"
                rerank_gain = f"{row['rerank_response_percentile_gain_mean']:<5.3f}"
                rerank_fix = f"{row['rerank_fixes_worst_frac']:<5.3f}"
            else:
                rerank_pct = "nan   "
                rerank_gain = "nan  "
                rerank_fix = "nan  "
            print(
                f"{row['run']:<35} {row['response_mode']:<24} "
                f"{row['top1_response_percentile_mean']:<6.3f} "
                f"{rerank_pct} "
                f"{rerank_gain} "
                f"{rerank_fix} "
                f"{row['oracle_response_percentile_mean']:<6.3f} "
                f"{row['oracle_response_percentile_gain_mean']:<5.3f} "
                f"{row['oracle_fixes_worst_frac']:<5.3f} "
                f"{row['oracle_is_rerank_best_frac']:<6.3f} "
                f"{row['oracle_rerank_farther_than_top1_frac']:<6.3f} "
                f"{row['worst_response_frac']:<6.3f} "
                f"{row['top2_nonunique_geometry_frac']:<6.3f} "
                f"{row['worst_response_norm_rank_mean']:<6.3f} "
                f"{worst_region}"
            )
        else:
            print(
                f"{row['run']:<35} {row['response_mode']:<24} "
                f"{row['top1_response_percentile_mean']:<8.3f} "
                f"{row['oracle_response_percentile_mean']:<6.3f} "
                f"{row['oracle_response_percentile_gain_mean']:<5.3f} "
                f"{row['oracle_fixes_worst_frac']:<5.3f} "
                f"{row['worst_response_frac']:<6.3f} "
                f"{row['nonunique_geometry_frac']:<7.3f} "
                f"{row['top2_nonunique_geometry_frac']:<6.3f} "
                f"{row['worst_response_norm_rank_mean']:<6.3f} "
                f"{worst_region}"
            )


def print_worst_patch_rows(rows: list[dict[str, Any]]) -> None:
    """Print worst patch rows for manual localization."""
    if not rows:
        return
    print("\nWorst patch-level geometry-neighbor mismatches")
    has_rerank = any("rerank_response_percentile" in row for row in rows)
    if has_rerank:
        print(
            "run                                response                 rank idx  cy   cx   region       "
            "nbr  resp_pct rerank rerank_r rgain oracle oracle_r ogain geom_dist grad_r var_r norm_r"
        )
        print("-" * 184)
    else:
        print("run                                response                 rank idx  cy   cx   region       nbr  ny   nx   resp_pct oracle oracle_r gain  geom_dist grad_r var_r norm_r")
        print("-" * 166)
    for row in rows:
        if has_rerank:
            if "rerank_response_percentile" in row:
                rerank_pct = f"{row['rerank_response_percentile']:<6.3f}"
                rerank_rank = f"{row['rerank_geometry_rank']:<8.0f}"
                rerank_gain = f"{row['rerank_response_percentile_gain']:<6.3f}"
            else:
                rerank_pct = "nan   "
                rerank_rank = "nan     "
                rerank_gain = "nan   "
            print(
                f"{row['run']:<35} {row['response_mode']:<24} "
                f"{row['worst_rank']:<4} {row['patch_index']:<4} "
                f"{row['center_y']:<4.0f} {row['center_x']:<4.0f} "
                f"{row['spatial_region']:<12} {row['top1_index']:<4} "
                f"{row['top1_response_percentile']:<8.3f} "
                f"{rerank_pct} "
                f"{rerank_rank} "
                f"{rerank_gain} "
                f"{row['oracle_response_percentile']:<6.3f} "
                f"{row['oracle_geometry_rank']:<8.0f} "
                f"{row['oracle_response_percentile_gain']:<6.3f} "
                f"{row['top1_geometry_dist']:<9.4f} "
                f"{row['patch_grad_rank']:<6.3f} "
                f"{row['patch_var_rank']:<5.3f} "
                f"{row['response_norm_rank']:<6.3f}"
            )
        else:
            print(
                f"{row['run']:<35} {row['response_mode']:<24} "
                f"{row['worst_rank']:<4} {row['patch_index']:<4} "
                f"{row['center_y']:<4.0f} {row['center_x']:<4.0f} "
                f"{row['spatial_region']:<12} {row['top1_index']:<4} "
                f"{row['top1_center_y']:<4.0f} {row['top1_center_x']:<4.0f} "
                f"{row['top1_response_percentile']:<8.3f} "
                f"{row['oracle_response_percentile']:<6.3f} "
                f"{row['oracle_geometry_rank']:<8.0f} "
                f"{row['oracle_response_percentile_gain']:<6.3f} "
                f"{row['top1_geometry_dist']:<9.4f} "
                f"{row['patch_grad_rank']:<6.3f} "
                f"{row['patch_var_rank']:<5.3f} "
                f"{row['response_norm_rank']:<6.3f}"
            )


def print_candidate_attribute_summary(rows: list[dict[str, Any]]) -> None:
    """Print oracle-vs-top1 response-blind candidate attribute diagnostics."""
    if not rows:
        return
    print("\nCandidate-level oracle-vs-top1 attribute diagnostic")
    print("run                                response                 attr                         help_n better  signed  top1    oracle  direction")
    print("-" * 131)
    for row in rows:
        help_n = int(row.get("candidate_attribute_oracle_help_rows", 0.0))
        if help_n <= 0:
            continue
        for name, direction, _, _ in CANDIDATE_ATTRIBUTE_SPECS:
            better_key = f"{name}_oracle_better_frac"
            if better_key not in row:
                continue
            print(
                f"{row['run']:<35} {row['response_mode']:<24} "
                f"{name:<28} {help_n:<6d} "
                f"{row[better_key]:<7.3f} "
                f"{row[f'{name}_signed_advantage_mean']:<7.4f} "
                f"{row[f'{name}_top1_mean']:<7.4f} "
                f"{row[f'{name}_oracle_mean']:<7.4f} "
                f"{direction}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Stage-C failure audit")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result_dir", action="append", default=None)
    group.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="Data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--geometry_source", choices=["hr", "lr_up"], default="lr_up")
    parser.add_argument(
        "--geometry_descriptor",
        choices=GEOMETRY_DESCRIPTORS,
        default="rgb_grad",
    )
    parser.add_argument(
        "--response_modes",
        type=str,
        default=",".join(DEFAULT_RESPONSE_MODES),
        help="Comma-separated response modes to audit",
    )
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_spatial_distance", type=float, default=8.0)
    parser.add_argument("--n_shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--content_stratification",
        action="store_true",
        help="Add diagnostic gradient/variance tercile rows; read-only and not a claim search.",
    )
    parser.add_argument(
        "--min_stratum_patches",
        type=int,
        default=12,
        help="Minimum patch count required before a content stratum is evaluated.",
    )
    parser.add_argument(
        "--show_stratified_rows",
        action="store_true",
        help="Print detailed per-response content-stratum rows in table mode.",
    )
    parser.add_argument(
        "--patch_failure_diagnostics",
        action="store_true",
        help=(
            "Add diagnostic patch-level geometry-neighbor mismatch and "
            "non-unique-neighbor rows; read-only and not a claim search."
        ),
    )
    parser.add_argument(
        "--patch_failure_top_n",
        type=int,
        default=8,
        help="Number of worst patch mismatch rows to retain per run/response mode.",
    )
    parser.add_argument(
        "--patch_rerank_descriptor",
        choices=GEOMETRY_DESCRIPTORS,
        default=None,
        help=(
            "Optional response-blind reranker for geometry top-k candidates. "
            "The main geometry descriptor still selects top-k; this descriptor "
            "only chooses one candidate inside that set."
        ),
    )
    parser.add_argument(
        "--show_worst_patches",
        action="store_true",
        help="Print worst patch mismatch rows in table mode.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path for writing the full audit output JSON.",
    )
    parser.add_argument(
        "--patch_crop_dir",
        type=str,
        default=None,
        help="Optional directory for diagnostic worst-patch crop PNG grids.",
    )
    parser.add_argument(
        "--patch_crop_scale",
        type=int,
        default=14,
        help="Nearest-neighbor display scale for diagnostic patch crop grids.",
    )
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    response_modes = parse_csv_values(args.response_modes, RESPONSE_MODES, "response mode")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.result_dir:
        result_dirs = [Path(path) for path in args.result_dir]
    else:
        result_dirs = scan_result_dirs(Path(args.results_dir))

    reviews: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    stratified: list[dict[str, Any]] = []
    patch_diagnostic_summaries: list[dict[str, Any]] = []
    worst_patch_rows: list[dict[str, Any]] = []
    patch_crop_files: list[dict[str, Any]] = []
    for index, result_dir in enumerate(result_dirs):
        (
            run_reviews,
            run_diagnostics,
            run_stratified,
            run_patch_diagnostics,
            run_worst_patches,
            run_patch_crop_files,
        ) = audit_result_dir(
            result_dir=result_dir,
            data_root=Path(args.data_root),
            device=device,
            geometry_source=args.geometry_source,
            geometry_descriptor=args.geometry_descriptor,
            response_modes=response_modes,
            patch_size=args.patch_size,
            stride=args.stride,
            k=args.k,
            min_spatial_distance=args.min_spatial_distance,
            n_shuffles=args.n_shuffles,
            seed=args.seed + index * 1009,
            content_stratification=args.content_stratification,
            min_stratum_patches=args.min_stratum_patches,
            patch_failure_diagnostics=args.patch_failure_diagnostics,
            patch_failure_top_n=args.patch_failure_top_n,
            patch_rerank_descriptor=args.patch_rerank_descriptor,
            patch_crop_dir=Path(args.patch_crop_dir) if args.patch_crop_dir else None,
            patch_crop_scale=args.patch_crop_scale,
        )
        reviews.extend(run_reviews)
        diagnostics.append(run_diagnostics)
        stratified.extend(run_stratified)
        patch_diagnostic_summaries.extend(run_patch_diagnostics)
        worst_patch_rows.extend(run_worst_patches)
        patch_crop_files.extend(run_patch_crop_files)

    image_rows = aggregate_image_reviews(reviews)
    stratified_rows = aggregate_stratified_reviews(stratified)
    output = {
        "settings": {
            "geometry_source": args.geometry_source,
            "geometry_descriptor": args.geometry_descriptor,
            "response_modes": response_modes,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "k": args.k,
            "min_spatial_distance": args.min_spatial_distance,
            "n_shuffles": args.n_shuffles,
            "seed": args.seed,
            "content_stratification": args.content_stratification,
            "min_stratum_patches": args.min_stratum_patches,
            "show_stratified_rows": args.show_stratified_rows,
            "patch_failure_diagnostics": args.patch_failure_diagnostics,
            "patch_failure_top_n": args.patch_failure_top_n,
            "patch_rerank_descriptor": args.patch_rerank_descriptor,
            "show_worst_patches": args.show_worst_patches,
            "patch_crop_dir": args.patch_crop_dir,
            "patch_crop_scale": args.patch_crop_scale,
        },
        "image_summary": image_rows,
        "response_reviews": reviews,
        "run_diagnostics": diagnostics,
        "content_stratified_summary": stratified_rows,
        "content_stratified_reviews": stratified,
        "patch_failure_summary": patch_diagnostic_summaries,
        "worst_patch_rows": worst_patch_rows,
        "patch_crop_files": patch_crop_files,
    }

    if args.format == "json":
        rendered = json.dumps(output, indent=2, sort_keys=True)
        print(rendered)
    else:
        print_summary(image_rows, reviews)
        print_stratified_aggregate(stratified_rows)
        if args.show_stratified_rows:
            print_stratified_summary(stratified)
        print_patch_failure_summary(patch_diagnostic_summaries)
        print_candidate_attribute_summary(patch_diagnostic_summaries)
        if args.show_worst_patches:
            print_worst_patch_rows(worst_patch_rows)
        if patch_crop_files:
            print("\nPatch crop diagnostic files")
            for item in patch_crop_files:
                print(f"{item['run']} {item['response_mode']}: {item['path']}")
        print(f"\n{len(result_dirs)} LIIF result(s) audited on {device}.")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(output, f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
