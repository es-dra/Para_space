#!/usr/bin/env python3
"""Stage-C response-object audit for geometry-response failures.

This read-only diagnostic asks whether the current output trajectory response
object mixes amplitude, direction, or time phases in a way that could explain
bird/head failures. It does not train models, change result directories, tune
geometry descriptors, or use response-label oracles as predictors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_c_failure_audit import (  # noqa: E402
    effect_label,
    patch_content_arrays,
    patch_failure_rows,
    summarize_patch_failures,
)
from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import (  # noqa: E402
    GEOMETRY_DESCRIPTORS,
    analyze_geometry_response_descriptors,
    eligible_pair_mask,
    evaluate_liif_response_objects,
    finite_percentile,
    geometry_patch_descriptors,
    l2_normalize_rows,
    make_patch_grid,
    pairwise_distances,
    scan_result_dirs,
    spearman_correlation,
    summarize_row_norms,
)


@dataclass(frozen=True)
class ResponseObjectVariant:
    """Pre-registered output response object variant."""

    name: str
    base: str
    transform: str
    temporal_window: str
    description: str


RESPONSE_OBJECT_VARIANTS: tuple[ResponseObjectVariant, ...] = (
    ResponseObjectVariant(
        name="trajectory_unit",
        base="trajectory_delta",
        transform="unit_direction",
        temporal_window="full",
        description="Current primary output trajectory direction descriptor.",
    ),
    ResponseObjectVariant(
        name="trajectory_raw",
        base="trajectory_delta",
        transform="raw",
        temporal_window="full",
        description="Full output trajectory descriptor with amplitude retained.",
    ),
    ResponseObjectVariant(
        name="trajectory_norm",
        base="trajectory_delta",
        transform="norm_only",
        temporal_window="full",
        description="Scalar full-trajectory patch response magnitude only.",
    ),
    ResponseObjectVariant(
        name="final_unit",
        base="final_delta",
        transform="unit_direction",
        temporal_window="endpoint",
        description="Endpoint output delta direction descriptor.",
    ),
    ResponseObjectVariant(
        name="final_raw",
        base="final_delta",
        transform="raw",
        temporal_window="endpoint",
        description="Endpoint output delta descriptor with amplitude retained.",
    ),
    ResponseObjectVariant(
        name="final_norm",
        base="final_delta",
        transform="norm_only",
        temporal_window="endpoint",
        description="Scalar endpoint patch response magnitude only.",
    ),
    ResponseObjectVariant(
        name="trajectory_early_unit",
        base="trajectory_delta",
        transform="unit_direction",
        temporal_window="early",
        description="Early third of output trajectory direction descriptor.",
    ),
    ResponseObjectVariant(
        name="trajectory_mid_unit",
        base="trajectory_delta",
        transform="unit_direction",
        temporal_window="mid",
        description="Middle third of output trajectory direction descriptor.",
    ),
    ResponseObjectVariant(
        name="trajectory_late_unit",
        base="trajectory_delta",
        transform="unit_direction",
        temporal_window="late",
        description="Late third of output trajectory direction descriptor.",
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in RESPONSE_OBJECT_VARIANTS}
DEFAULT_VARIANTS = ",".join(variant.name for variant in RESPONSE_OBJECT_VARIANTS)


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


def _trajectory_window_indices(n_deltas: int, window: str) -> np.ndarray:
    """Return deterministic time-delta indices for a temporal window."""
    if n_deltas <= 0:
        raise ValueError("trajectory response needs at least two snapshots")
    if window == "full":
        return np.arange(n_deltas, dtype=np.int64)
    groups = np.array_split(np.arange(n_deltas, dtype=np.int64), 3)
    mapping = {"early": 0, "mid": 1, "late": 2}
    if window not in mapping:
        raise ValueError(f"unknown trajectory window: {window}")
    selected = groups[mapping[window]]
    if selected.size == 0:
        raise ValueError(f"not enough trajectory deltas for {window} window")
    return selected


def response_field_for_variant(outputs: np.ndarray, variant: ResponseObjectVariant) -> np.ndarray:
    """Extract the output response field for one pre-registered variant."""
    arr = np.asarray(outputs, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError("outputs must have shape [T, H, W, C]")
    if arr.shape[0] < 2:
        raise ValueError("response audit requires at least two snapshots")

    if variant.base == "final_delta":
        return arr[-1] - arr[0]
    if variant.base == "trajectory_delta":
        delta = np.diff(arr, axis=0)
        indices = _trajectory_window_indices(delta.shape[0], variant.temporal_window)
        return delta[indices]
    raise ValueError(f"unknown response base: {variant.base}")


def patch_descriptors_from_field(field: np.ndarray, grid: Any, patch_size: int) -> np.ndarray:
    """Flatten patch response fields into row descriptors."""
    arr = np.asarray(field, dtype=np.float64)
    descriptors = []
    for y, x in grid.starts:
        if arr.ndim == 3:
            patch = arr[y:y + patch_size, x:x + patch_size, :]
        elif arr.ndim == 4:
            patch = arr[:, y:y + patch_size, x:x + patch_size, :]
        else:
            raise ValueError("response field must have shape [H,W,C] or [T,H,W,C]")
        descriptors.append(np.asarray(patch, dtype=np.float64).reshape(-1))
    return np.stack(descriptors, axis=0)


def transform_response_descriptors(raw: np.ndarray, transform: str) -> np.ndarray:
    """Apply the pre-registered descriptor transform."""
    arr = np.asarray(raw, dtype=np.float64)
    if transform == "raw":
        return arr
    if transform == "unit_direction":
        return l2_normalize_rows(arr)
    if transform == "norm_only":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return np.log(norms + 1e-12)
    raise ValueError(f"unknown response descriptor transform: {transform}")


def response_variant_descriptors(
    outputs: np.ndarray,
    grid: Any,
    patch_size: int,
    variant: ResponseObjectVariant,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed and raw patch descriptors for one response variant."""
    field = response_field_for_variant(outputs, variant)
    raw = patch_descriptors_from_field(field, grid, patch_size)
    return transform_response_descriptors(raw, variant.transform), raw


def norm_failure_coupling(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize whether patch failures track response norm or norm gaps."""
    if not rows:
        return {
            "norm_failure_spearman": float("nan"),
            "norm_gap_failure_spearman": float("nan"),
            "top1_norm_gap_mean": float("nan"),
            "oracle_norm_gap_mean": float("nan"),
        }
    failure_pct = np.asarray([row["top1_response_percentile"] for row in rows], dtype=np.float64)
    norm_rank = np.asarray([row["response_norm_rank"] for row in rows], dtype=np.float64)
    top1_gap = np.asarray(
        [row["top1_response_norm_rank_absdiff"] for row in rows],
        dtype=np.float64,
    )
    oracle_gap = np.asarray(
        [row["oracle_response_norm_rank_absdiff"] for row in rows],
        dtype=np.float64,
    )
    return {
        "norm_failure_spearman": spearman_correlation(norm_rank, failure_pct),
        "norm_gap_failure_spearman": spearman_correlation(top1_gap, failure_pct),
        "top1_norm_gap_mean": float(np.nanmean(top1_gap)),
        "top1_norm_gap_p90": finite_percentile(top1_gap, 90),
        "oracle_norm_gap_mean": float(np.nanmean(oracle_gap)),
        "oracle_minus_top1_norm_gap_mean": float(np.nanmean(oracle_gap - top1_gap)),
    }


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


def directed_pair_percentile_matrix(distance: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Convert each directed eligible row of a distance matrix to percentile ranks."""
    dist = np.asarray(distance, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if dist.shape != mask.shape or dist.ndim != 2:
        raise ValueError("distance and eligible mask must have matching square shape")
    out = np.full(dist.shape, np.nan, dtype=np.float64)
    for i in range(dist.shape[0]):
        candidates = np.flatnonzero(mask[i])
        if candidates.size == 0:
            continue
        out[i, candidates] = rank_fraction(dist[i, candidates])
    return out


def temporal_consistency_metrics(
    outputs: np.ndarray,
    grid: Any,
    patch_size: int,
    centers: np.ndarray,
    *,
    min_spatial_distance: float,
) -> dict[str, float]:
    """Check whether early/mid/late response ordering is stable over time."""
    segment_names = ("trajectory_early_unit", "trajectory_mid_unit", "trajectory_late_unit")
    eligible = eligible_pair_mask(centers, min_spatial_distance)
    upper = np.triu(eligible, k=1)
    percentile_mats = []
    upper_distances = []
    for name in segment_names:
        desc, _ = response_variant_descriptors(outputs, grid, patch_size, VARIANT_BY_NAME[name])
        dist = pairwise_distances(desc)
        percentile_mats.append(directed_pair_percentile_matrix(dist, eligible))
        upper_distances.append(dist[upper])

    pairwise_corrs = []
    for i in range(len(upper_distances)):
        for j in range(i + 1, len(upper_distances)):
            pairwise_corrs.append(spearman_correlation(upper_distances[i], upper_distances[j]))

    directed_consistency = []
    for i in range(eligible.shape[0]):
        candidates = np.flatnonzero(eligible[i])
        if candidates.size == 0:
            continue
        values = np.stack([mat[i, candidates] for mat in percentile_mats], axis=0)
        if values.shape[1] == 0:
            continue
        directed_consistency.append(float(np.nanmean(np.nanstd(values, axis=0))))

    return {
        "temporal_segment_pair_spearman_mean": (
            float(np.nanmean(pairwise_corrs)) if pairwise_corrs else float("nan")
        ),
        "temporal_segment_pair_spearman_min": (
            float(np.nanmin(pairwise_corrs)) if pairwise_corrs else float("nan")
        ),
        "temporal_directed_percentile_std_mean": (
            float(np.nanmean(directed_consistency)) if directed_consistency else float("nan")
        ),
    }


def audit_response_objects_for_evaluation(
    result_dir: Path,
    evaluation: Any,
    *,
    geometry_source: str,
    geometry_descriptor: str,
    variant_names: list[str],
    patch_size: int,
    stride: int,
    k: int,
    min_spatial_distance: float,
    n_shuffles: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Audit pre-registered response object variants for one evaluated run."""
    geometry_image = evaluation.target_hr if geometry_source == "hr" else evaluation.lr_up
    grid = make_patch_grid(geometry_image.shape[0], geometry_image.shape[1], patch_size, stride)
    _, patch_grad, patch_var = patch_content_arrays(geometry_image, patch_size, stride)
    geometry_raw = geometry_patch_descriptors(
        geometry_image,
        grid,
        patch_size,
        descriptor=geometry_descriptor,
        normalize=False,
    )
    geometry_desc = l2_normalize_rows(geometry_raw)

    rows: list[dict[str, Any]] = []
    for variant_index, variant_name in enumerate(variant_names):
        variant = VARIANT_BY_NAME[variant_name]
        response_desc, response_raw = response_variant_descriptors(
            evaluation.outputs,
            grid,
            patch_size,
            variant,
        )
        metrics = analyze_geometry_response_descriptors(
            geometry_desc,
            response_desc,
            grid.centers,
            k=k,
            min_spatial_distance=min_spatial_distance,
            n_shuffles=n_shuffles,
            seed=seed + variant_index * 1009,
        )
        patch_rows = patch_failure_rows(
            geometry_desc,
            response_desc,
            grid.centers,
            patch_grad,
            patch_var,
            patch_size,
            k=k,
            min_spatial_distance=min_spatial_distance,
            response_norm_values=np.linalg.norm(response_raw, axis=1),
        )
        patch_summary = summarize_patch_failures(patch_rows)
        norm_coupling = norm_failure_coupling(patch_rows)
        label = effect_label(
            float(metrics["effect_vs_shuffle"]),
            float(metrics["effect_vs_shuffle_frac"]),
            float(metrics["shuffled_response_p_le"]),
            float(metrics["geometry_response_spearman"]),
        )
        row: dict[str, Any] = {
            "run": result_dir.name,
            "image": evaluation.summary.get("image", "?"),
            "seed": evaluation.summary.get("seed", "?"),
            "status": "ok",
            "geometry_source": geometry_source,
            "geometry_descriptor": geometry_descriptor,
            "response_variant": variant.name,
            "response_base": variant.base,
            "response_transform": variant.transform,
            "temporal_window": variant.temporal_window,
            "variant_description": variant.description,
            "patch_size": patch_size,
            "stride": stride,
            "k": k,
            "min_spatial_distance": min_spatial_distance,
            "n_shuffles": n_shuffles,
            "label": label,
        }
        row.update(metrics)
        row.update(summarize_row_norms("response_raw", response_raw))
        for key, value in patch_summary.items():
            row[f"patch_{key}"] = value
        row.update(norm_coupling)
        rows.append(row)
    return rows


def temporal_consistency_row(
    result_dir: Path,
    evaluation: Any,
    *,
    geometry_source: str,
    patch_size: int,
    stride: int,
    min_spatial_distance: float,
) -> dict[str, Any]:
    """Return one temporal consistency diagnostic row for a run."""
    geometry_image = evaluation.target_hr if geometry_source == "hr" else evaluation.lr_up
    grid = make_patch_grid(geometry_image.shape[0], geometry_image.shape[1], patch_size, stride)
    row: dict[str, Any] = {
        "run": result_dir.name,
        "image": evaluation.summary.get("image", "?"),
        "seed": evaluation.summary.get("seed", "?"),
        "geometry_source": geometry_source,
        "patch_size": patch_size,
        "stride": stride,
        "min_spatial_distance": min_spatial_distance,
    }
    row.update(
        temporal_consistency_metrics(
            evaluation.outputs,
            grid,
            patch_size,
            grid.centers,
            min_spatial_distance=min_spatial_distance,
        )
    )
    return row


def aggregate_response_object_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate variant rows into conservative reviewer-facing summaries."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["response_variant"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for variant, group in sorted(grouped.items()):
        labels = [str(row["label"]) for row in group]
        effects = np.asarray([row["effect_vs_shuffle"] for row in group], dtype=np.float64)
        effect_fracs = np.asarray(
            [row["effect_vs_shuffle_frac"] for row in group],
            dtype=np.float64,
        )
        patch_top1 = np.asarray(
            [row["patch_top1_response_percentile_mean"] for row in group],
            dtype=np.float64,
        )
        failure_rows = [
            row for row in group
            if "bird" in str(row["run"]) or "head" in str(row["run"])
        ]
        guardrail_rows = [
            row for row in group
            if "baby" in str(row["run"]) or "woman" in str(row["run"])
        ]
        failure_labels = [str(row["label"]) for row in failure_rows]
        guardrail_labels = [str(row["label"]) for row in guardrail_rows]
        failure_effects = np.asarray(
            [row["effect_vs_shuffle"] for row in failure_rows],
            dtype=np.float64,
        )
        failure_effect_fracs = np.asarray(
            [row["effect_vs_shuffle_frac"] for row in failure_rows],
            dtype=np.float64,
        )
        guardrail_effects = np.asarray(
            [row["effect_vs_shuffle"] for row in guardrail_rows],
            dtype=np.float64,
        )
        guardrail_effect_fracs = np.asarray(
            [row["effect_vs_shuffle_frac"] for row in guardrail_rows],
            dtype=np.float64,
        )
        failure_patch = np.asarray(
            [row["patch_top1_response_percentile_mean"] for row in failure_rows],
            dtype=np.float64,
        )
        guardrail_patch = np.asarray(
            [row["patch_top1_response_percentile_mean"] for row in guardrail_rows],
            dtype=np.float64,
        )
        failure_support_count = sum(label in {"support", "local_only"} for label in failure_labels)
        guardrail_negative_count = sum(label in {"negative", "tiny"} for label in guardrail_labels)
        if failure_rows and failure_support_count == len(failure_rows) and guardrail_negative_count == 0:
            verdict = "candidate_response_redefinition"
        elif failure_support_count > 0:
            verdict = "partial_or_image_specific"
        elif any(label == "negative" for label in failure_labels):
            verdict = "does_not_resolve_failure"
        else:
            verdict = "no_clear_support"
        summaries.append(
            {
                "response_variant": variant,
                "n_runs": float(len(group)),
                "support_or_local_count": float(sum(label in {"support", "local_only"} for label in labels)),
                "negative_or_tiny_count": float(sum(label in {"negative", "tiny"} for label in labels)),
                "effect_vs_shuffle_mean": float(np.nanmean(effects)),
                "effect_vs_shuffle_frac_mean": float(np.nanmean(effect_fracs)),
                "effect_vs_shuffle_min": float(np.nanmin(effects)),
                "effect_vs_shuffle_max": float(np.nanmax(effects)),
                "patch_top1_response_percentile_mean": float(np.nanmean(patch_top1)),
                "failure_n_runs": float(len(failure_rows)),
                "failure_support_or_local_count": float(failure_support_count),
                "failure_effect_mean": (
                    float(np.nanmean(failure_effects)) if failure_effects.size else float("nan")
                ),
                "failure_effect_frac_mean": (
                    float(np.nanmean(failure_effect_fracs))
                    if failure_effect_fracs.size else float("nan")
                ),
                "failure_patch_top1_response_percentile_mean": (
                    float(np.nanmean(failure_patch)) if failure_patch.size else float("nan")
                ),
                "guardrail_n_runs": float(len(guardrail_rows)),
                "guardrail_negative_or_tiny_count": float(guardrail_negative_count),
                "guardrail_effect_mean": (
                    float(np.nanmean(guardrail_effects)) if guardrail_effects.size else float("nan")
                ),
                "guardrail_effect_frac_mean": (
                    float(np.nanmean(guardrail_effect_fracs))
                    if guardrail_effect_fracs.size else float("nan")
                ),
                "guardrail_patch_top1_response_percentile_mean": (
                    float(np.nanmean(guardrail_patch)) if guardrail_patch.size else float("nan")
                ),
                "verdict": verdict,
            }
        )
    return summaries


def summarize_temporal_consistency(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize temporal consistency differences between failure and guardrail runs."""
    failure = [
        row for row in rows
        if "bird" in str(row["run"]) or "head" in str(row["run"])
    ]
    guardrail = [
        row for row in rows
        if "baby" in str(row["run"]) or "woman" in str(row["run"])
    ]

    def mean_key(group: list[dict[str, Any]], key: str) -> float:
        values = np.asarray([row[key] for row in group], dtype=np.float64)
        return float(np.nanmean(values)) if values.size else float("nan")

    return {
        "failure_segment_pair_spearman_mean": mean_key(
            failure,
            "temporal_segment_pair_spearman_mean",
        ),
        "guardrail_segment_pair_spearman_mean": mean_key(
            guardrail,
            "temporal_segment_pair_spearman_mean",
        ),
        "failure_directed_percentile_std_mean": mean_key(
            failure,
            "temporal_directed_percentile_std_mean",
        ),
        "guardrail_directed_percentile_std_mean": mean_key(
            guardrail,
            "temporal_directed_percentile_std_mean",
        ),
    }


def print_audit_table(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    temporal_rows: list[dict[str, Any]],
) -> None:
    """Print compact response-object audit tables."""
    print("\nResponse-object audit rows")
    print("run                                variant                label       eff_shuf  rho      patch_pct")
    print("-" * 98)
    for row in rows:
        print(
            f"{row['run']:<35} {row['response_variant']:<22} {row['label']:<11} "
            f"{row['effect_vs_shuffle']:<9.4f} {row['geometry_response_spearman']:<8.4f} "
            f"{row['patch_top1_response_percentile_mean']:<8.4f}"
        )

    print("\nResponse-object variant summary")
    print("variant                fail_frac fail_patch guard_frac guard_patch verdict")
    print("-" * 90)
    for row in summaries:
        print(
            f"{row['response_variant']:<22} {row['failure_effect_frac_mean']:<9.4f} "
            f"{row['failure_patch_top1_response_percentile_mean']:<10.4f} "
            f"{row['guardrail_effect_frac_mean']:<10.4f} "
            f"{row['guardrail_patch_top1_response_percentile_mean']:<11.4f} "
            f"{row['verdict']}"
        )

    print("\nTemporal consistency rows")
    print("run                                seg_rho  seg_rho_min  directed_std")
    print("-" * 78)
    for row in temporal_rows:
        print(
            f"{row['run']:<35} {row['temporal_segment_pair_spearman_mean']:<8.4f} "
            f"{row['temporal_segment_pair_spearman_min']:<12.4f} "
            f"{row['temporal_directed_percentile_std_mean']:<8.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Stage-C response-object audit")
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
        "--response_object_variants",
        type=str,
        default=DEFAULT_VARIANTS,
        help="Comma-separated pre-registered response-object variants to audit.",
    )
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_spatial_distance", type=float, default=8.0)
    parser.add_argument("--n_shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    variant_names = parse_csv_values(
        args.response_object_variants,
        VARIANT_BY_NAME,
        "response object variant",
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.result_dir:
        result_dirs = [Path(path) for path in args.result_dir]
    else:
        result_dirs = scan_result_dirs(Path(args.results_dir))

    rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    for index, result_dir in enumerate(result_dirs):
        evaluation = evaluate_liif_response_objects(
            result_dir,
            Path(args.data_root),
            device,
            collect_features=False,
            collect_coord_jacobians=False,
        )
        rows.extend(
            audit_response_objects_for_evaluation(
                result_dir,
                evaluation,
                geometry_source=args.geometry_source,
                geometry_descriptor=args.geometry_descriptor,
                variant_names=variant_names,
                patch_size=args.patch_size,
                stride=args.stride,
                k=args.k,
                min_spatial_distance=args.min_spatial_distance,
                n_shuffles=args.n_shuffles,
                seed=args.seed + index * 10007,
            )
        )
        temporal_rows.append(
            temporal_consistency_row(
                result_dir,
                evaluation,
                geometry_source=args.geometry_source,
                patch_size=args.patch_size,
                stride=args.stride,
                min_spatial_distance=args.min_spatial_distance,
            )
        )

    summaries = aggregate_response_object_audit(rows)
    temporal_summary = summarize_temporal_consistency(temporal_rows)
    output = {
        "settings": {
            "geometry_source": args.geometry_source,
            "geometry_descriptor": args.geometry_descriptor,
            "response_object_variants": variant_names,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "k": args.k,
            "min_spatial_distance": args.min_spatial_distance,
            "n_shuffles": args.n_shuffles,
            "seed": args.seed,
        },
        "variant_summary": summaries,
        "temporal_consistency_summary": temporal_summary,
        "temporal_consistency_rows": temporal_rows,
        "response_object_rows": rows,
    }

    if args.format == "json":
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print_audit_table(rows, summaries, temporal_rows)
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
