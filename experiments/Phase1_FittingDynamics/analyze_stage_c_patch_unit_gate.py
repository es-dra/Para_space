#!/usr/bin/env python3
"""Stage-C patch-unit gate for LIIF-aware matching.

This read-only gate tests one pre-registered patch-unit change: replacing the
current HR 7x7 appearance patch with an LR feature-cell/query-support unit.
It does not train models, add geometry descriptors, rerank candidates, or use
response-label oracle information as a predictor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_c_failure_audit import (  # noqa: E402
    effect_label,
    patch_content_arrays,
    patch_failure_rows,
    summarize_patch_failures,
    tercile_masks,
)
from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import (  # noqa: E402
    analyze_geometry_response_descriptors,
    eligible_pair_mask,
    evaluate_liif_response_objects,
    finite_percentile,
    geometry_neighbor_pairs,
    geometry_patch_descriptors,
    l2_normalize_rows,
    make_patch_grid,
    pair_distance_values,
    pairwise_distances,
    resolve_image_path_from_summary,
    response_patch_descriptors,
    scan_result_dirs,
)
from experiments.Phase1_FittingDynamics.run import load_image  # noqa: E402


@dataclass(frozen=True)
class UnitDescriptors:
    """Geometry/response descriptors and diagnostics for one patch unit."""

    name: str
    geometry_desc: np.ndarray
    response_desc: np.ndarray
    response_raw: np.ndarray
    centers: np.ndarray
    patch_grad: np.ndarray
    patch_var: np.ndarray
    content_control_desc: np.ndarray
    coordinate_desc: np.ndarray
    patch_size_for_rows: int


def load_lr_image_from_summary(summary: dict[str, Any], data_root: Path, device: torch.device) -> np.ndarray:
    """Load the LR input image used by the LIIF run as an HWC array."""
    image_name = str(summary["image"])
    image_size = int(summary.get("hr_size", 48))
    sr_scale = int(summary.get("sr_scale", 4))
    image_path = resolve_image_path_from_summary(summary, data_root)
    lr_tensor, _ = load_image(image_path, image_size=image_size, sr_scale=sr_scale)
    return lr_tensor.to(device).permute(1, 2, 0).detach().cpu().numpy()


def crop_center_edge(image: np.ndarray, center_y: int, center_x: int, size: int) -> np.ndarray:
    """Extract a square crop centered on an integer coordinate with edge padding."""
    if size <= 0 or size % 2 != 1:
        raise ValueError("LR support size must be a positive odd integer")
    arr = np.asarray(image, dtype=np.float64)
    radius = size // 2
    padded = np.pad(arr, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    y = center_y + radius
    x = center_x + radius
    return padded[y - radius:y + radius + 1, x - radius:x + radius + 1, :]


def rgb_grad_descriptor_for_crop(crop: np.ndarray) -> np.ndarray:
    """Return a fixed mean-centered RGB + grayscale gradient descriptor."""
    arr = np.asarray(crop, dtype=np.float64)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    centered_rgb = arr - np.mean(arr, axis=(0, 1), keepdims=True)
    gray = arr.mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    return np.concatenate([centered_rgb.reshape(-1), grad_x.reshape(-1), grad_y.reshape(-1)])


def zscore_columns(values: np.ndarray) -> np.ndarray:
    """Z-score columns, keeping deterministic zeros for constant columns."""
    arr = np.asarray(values, dtype=np.float64)
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    return (arr - mean) / np.where(std > 1e-12, std, 1.0)


def lr_cell_unit_descriptors(
    lr_image: np.ndarray,
    outputs: np.ndarray,
    *,
    sr_scale: int,
    support_size: int = 3,
) -> UnitDescriptors:
    """Build the pre-registered LR-cell/query-support unit descriptors."""
    lr = np.asarray(lr_image, dtype=np.float64)
    out = np.asarray(outputs, dtype=np.float64)
    if out.ndim != 4:
        raise ValueError("outputs must have shape [T, H, W, C]")
    if lr.shape[0] * sr_scale != out.shape[1] or lr.shape[1] * sr_scale != out.shape[2]:
        raise ValueError("LR image and output dimensions are inconsistent with sr_scale")

    delta = np.diff(out, axis=0)
    geometry = []
    response = []
    centers = []
    grad_values = []
    var_values = []
    content = []
    gray = lr.mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    for y in range(lr.shape[0]):
        for x in range(lr.shape[1]):
            crop = crop_center_edge(lr, y, x, support_size)
            geometry.append(rgb_grad_descriptor_for_crop(crop))
            hr_y = y * sr_scale
            hr_x = x * sr_scale
            block = delta[:, hr_y:hr_y + sr_scale, hr_x:hr_x + sr_scale, :]
            response.append(block.reshape(-1))
            centers.append((hr_y + (sr_scale - 1) / 2.0, hr_x + (sr_scale - 1) / 2.0))
            grad_crop = crop_center_edge(grad_mag[..., None], y, x, support_size)[..., 0]
            grad_mean = float(np.mean(grad_crop))
            patch_var = float(np.var(crop))
            patch_mean = float(np.mean(crop))
            grad_values.append(grad_mean)
            var_values.append(patch_var)
            content.append((patch_mean, patch_var, grad_mean))

    response_raw = np.asarray(response, dtype=np.float64)
    centers_arr = np.asarray(centers, dtype=np.float64)
    return UnitDescriptors(
        name="lr_cell_query_support",
        geometry_desc=l2_normalize_rows(np.asarray(geometry, dtype=np.float64)),
        response_desc=l2_normalize_rows(response_raw),
        response_raw=response_raw,
        centers=centers_arr,
        patch_grad=np.asarray(grad_values, dtype=np.float64),
        patch_var=np.asarray(var_values, dtype=np.float64),
        content_control_desc=l2_normalize_rows(zscore_columns(np.asarray(content, dtype=np.float64))),
        coordinate_desc=l2_normalize_rows(zscore_columns(centers_arr)),
        patch_size_for_rows=sr_scale,
    )


def hr7_baseline_unit_descriptors(lr_up: np.ndarray, outputs: np.ndarray) -> UnitDescriptors:
    """Build the existing HR 7x7/stride4 baseline unit descriptors."""
    patch_size = 7
    stride = 4
    grid, patch_grad, patch_var = patch_content_arrays(lr_up, patch_size, stride)
    geometry_raw = geometry_patch_descriptors(
        lr_up,
        grid,
        patch_size,
        descriptor="rgb_grad",
        normalize=False,
    )
    response_raw = response_patch_descriptors(
        outputs,
        grid,
        patch_size,
        mode="trajectory_delta",
        normalize=False,
    )
    content = np.stack(
        [
            np.asarray([
                float(np.mean(lr_up[y:y + patch_size, x:x + patch_size, :]))
                for y, x in grid.starts
            ]),
            patch_var,
            patch_grad,
        ],
        axis=1,
    )
    return UnitDescriptors(
        name="hr7_rgb_grad_baseline",
        geometry_desc=l2_normalize_rows(geometry_raw),
        response_desc=l2_normalize_rows(response_raw),
        response_raw=response_raw,
        centers=grid.centers,
        patch_grad=patch_grad,
        patch_var=patch_var,
        content_control_desc=l2_normalize_rows(zscore_columns(content)),
        coordinate_desc=l2_normalize_rows(zscore_columns(grid.centers)),
        patch_size_for_rows=patch_size,
    )


def _content_group_labels(content_desc: np.ndarray) -> np.ndarray:
    """Assign each unit to a deterministic mean/variance/gradient tercile group."""
    arr = np.asarray(content_desc, dtype=np.float64)
    groups = []
    for col in range(arr.shape[1]):
        masks = tercile_masks(arr[:, col])
        labels = np.empty(arr.shape[0], dtype=np.int64)
        for idx, name in enumerate(("low", "mid", "high")):
            labels[masks[name]] = idx
        groups.append(labels)
    return np.stack(groups, axis=1)


def matched_random_response_mean(
    response_dist: np.ndarray,
    centers: np.ndarray,
    pairs: list[tuple[int, int]],
    eligible: np.ndarray,
    *,
    rng: np.random.Generator,
    mode: str,
    content_labels: np.ndarray | None = None,
    spatial_bandwidth: float = 4.0,
    n_repeats: int = 50,
) -> float:
    """Sample response distances from a matched random control."""
    if not pairs:
        return float("nan")
    spatial = pairwise_distances(centers)
    means = []
    for _ in range(n_repeats):
        sampled = []
        for anchor, neighbor in pairs:
            candidates = np.flatnonzero(eligible[anchor])
            if candidates.size == 0:
                continue
            if mode == "spatial":
                target_dist = spatial[anchor, neighbor]
                diff = np.abs(spatial[anchor, candidates] - target_dist)
                pool = candidates[diff <= spatial_bandwidth]
                if pool.size == 0:
                    order = np.argsort(diff, kind="mergesort")
                    pool = candidates[order[: min(10, candidates.size)]]
            elif mode == "content":
                if content_labels is None:
                    raise ValueError("content labels are required for content-matched control")
                same = np.all(content_labels[candidates] == content_labels[anchor], axis=1)
                pool = candidates[same]
                if pool.size == 0:
                    pool = candidates
            else:
                raise ValueError(f"unknown matched random mode: {mode}")
            sampled.append(float(response_dist[anchor, int(rng.choice(pool))]))
        if sampled:
            means.append(float(np.mean(sampled)))
    return float(np.mean(means)) if means else float("nan")


def analyze_case(
    unit: UnitDescriptors,
    geometry_desc: np.ndarray,
    *,
    analysis_name: str,
    run: str,
    image: str,
    seed_value: Any,
    k: int,
    min_spatial_distance: float,
    n_shuffles: int,
    seed: int,
    include_matched_controls: bool,
) -> dict[str, Any]:
    """Analyze one geometry descriptor against the unit response descriptors."""
    metrics = analyze_geometry_response_descriptors(
        geometry_desc,
        unit.response_desc,
        unit.centers,
        k=k,
        min_spatial_distance=min_spatial_distance,
        n_shuffles=n_shuffles,
        seed=seed,
    )
    rows = patch_failure_rows(
        geometry_desc,
        unit.response_desc,
        unit.centers,
        unit.patch_grad,
        unit.patch_var,
        unit.patch_size_for_rows,
        k=k,
        min_spatial_distance=min_spatial_distance,
        response_norm_values=np.linalg.norm(unit.response_raw, axis=1),
    )
    patch_summary = summarize_patch_failures(rows)
    label = effect_label(
        float(metrics["effect_vs_shuffle"]),
        float(metrics["effect_vs_shuffle_frac"]),
        float(metrics["shuffled_response_p_le"]),
        float(metrics["geometry_response_spearman"]),
    )
    out: dict[str, Any] = {
        "run": run,
        "image": image,
        "seed": seed_value,
        "unit": unit.name,
        "analysis": analysis_name,
        "response_mode": "trajectory_delta",
        "label": label,
        "n_units": float(unit.centers.shape[0]),
    }
    out.update(metrics)
    for key, value in patch_summary.items():
        out[f"patch_{key}"] = value

    if include_matched_controls:
        geometry_dist = pairwise_distances(geometry_desc)
        response_dist = pairwise_distances(unit.response_desc)
        eligible = eligible_pair_mask(unit.centers, min_spatial_distance)
        pairs = geometry_neighbor_pairs(geometry_dist, eligible, k=k)
        rng = np.random.default_rng(seed + 7919)
        spatial_mean = matched_random_response_mean(
            response_dist,
            unit.centers,
            pairs,
            eligible,
            rng=rng,
            mode="spatial",
        )
        content_mean = matched_random_response_mean(
            response_dist,
            unit.centers,
            pairs,
            eligible,
            rng=rng,
            mode="content",
            content_labels=_content_group_labels(unit.content_control_desc),
        )
        neighbor = float(metrics["neighbor_response_dist"])
        out.update(
            {
                "spatial_matched_random_response_dist_mean": spatial_mean,
                "effect_vs_spatial_matched_random": spatial_mean - neighbor,
                "content_matched_random_response_dist_mean": content_mean,
                "effect_vs_content_matched_random": content_mean - neighbor,
            }
        )
    return out


def analyze_unit(
    unit: UnitDescriptors,
    *,
    run: str,
    image: str,
    seed_value: Any,
    k: int,
    min_spatial_distance: float,
    n_shuffles: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Analyze a patch unit and its weak controls."""
    rng = np.random.default_rng(seed + 17)
    shuffled_geometry = unit.geometry_desc[rng.permutation(unit.geometry_desc.shape[0])]
    cases = [
        ("geometry", unit.geometry_desc, True),
        ("coordinate_only", unit.coordinate_desc, False),
        ("content_intensity_only", unit.content_control_desc, False),
        ("geometry_shuffled", shuffled_geometry, False),
    ]
    return [
        analyze_case(
            unit,
            desc,
            analysis_name=name,
            run=run,
            image=image,
            seed_value=seed_value,
            k=k,
            min_spatial_distance=min_spatial_distance,
            n_shuffles=n_shuffles,
            seed=seed + idx * 1009,
            include_matched_controls=matched,
        )
        for idx, (name, desc, matched) in enumerate(cases)
    ]


def gate_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen LR-cell gate standards."""
    lr_rows = [
        row for row in rows
        if row["unit"] == "lr_cell_query_support" and row["analysis"] == "geometry"
    ]
    controls = {
        (row["run"], row["analysis"]): row
        for row in rows
        if row["unit"] == "lr_cell_query_support" and row["analysis"] in {
            "coordinate_only",
            "content_intensity_only",
        }
    }
    failure_runs = [row for row in lr_rows if "bird" in row["run"] or "head" in row["run"]]
    guardrail_runs = [row for row in lr_rows if "baby" in row["run"] or "woman" in row["run"]]
    checks: dict[str, bool] = {}
    for row in failure_runs:
        prefix = row["run"]
        checks[f"{prefix}_support_label"] = row["label"] == "support"
        checks[f"{prefix}_effect_frac_ge_0.05"] = float(row["effect_vs_shuffle_frac"]) >= 0.05
        checks[f"{prefix}_p_le_0.05"] = float(row["shuffled_response_p_le"]) <= 0.05
        checks[f"{prefix}_patch_pct_le_0.40"] = (
            float(row["patch_top1_response_percentile_mean"]) <= 0.40
        )
        checks[f"{prefix}_worst_frac_le_0.18"] = float(row["patch_worst_response_frac"]) <= 0.18
    for row in guardrail_runs:
        prefix = row["run"]
        checks[f"{prefix}_guard_effect_frac_ge_0.03"] = (
            float(row["effect_vs_shuffle_frac"]) >= 0.03
        )
        checks[f"{prefix}_guard_patch_pct_le_0.38"] = (
            float(row["patch_top1_response_percentile_mean"]) <= 0.38
        )
        checks[f"{prefix}_guard_not_negative"] = row["label"] != "negative"
    for row in lr_rows:
        coord = controls.get((row["run"], "coordinate_only"))
        content = controls.get((row["run"], "content_intensity_only"))
        if coord is not None:
            checks[f"{row['run']}_better_than_coordinate"] = (
                float(row["patch_top1_response_percentile_mean"])
                <= float(coord["patch_top1_response_percentile_mean"]) - 0.05
                or float(row["effect_vs_shuffle_frac"])
                >= float(coord["effect_vs_shuffle_frac"]) + 0.03
            )
        if content is not None:
            checks[f"{row['run']}_better_than_content"] = (
                float(row["patch_top1_response_percentile_mean"])
                <= float(content["patch_top1_response_percentile_mean"]) - 0.05
                or float(row["effect_vs_shuffle_frac"])
                >= float(content["effect_vs_shuffle_frac"]) + 0.03
            )
    passed = bool(checks) and all(checks.values())
    if passed:
        verdict = "pass"
    elif any("bird" in key or "head" in key for key, ok in checks.items() if not ok):
        verdict = "fail"
    else:
        verdict = "guardrail_or_control_fail"
    return {
        "verdict": verdict,
        "passed": passed,
        "checks": checks,
        "n_checks": float(len(checks)),
        "n_failed_checks": float(sum(not ok for ok in checks.values())),
    }


def print_rows(rows: list[dict[str, Any]], verdict: dict[str, Any]) -> None:
    """Print compact gate tables."""
    print("\nPatch-unit gate rows")
    print("run                                unit                    analysis               label       eff_frac patch_pct worst")
    print("-" * 118)
    for row in rows:
        print(
            f"{row['run']:<35} {row['unit']:<23} {row['analysis']:<22} "
            f"{row['label']:<11} {row['effect_vs_shuffle_frac']:<8.4f} "
            f"{row['patch_top1_response_percentile_mean']:<9.4f} "
            f"{row['patch_worst_response_frac']:<6.4f}"
        )
    print(f"\nGate verdict: {verdict['verdict']} ({int(verdict['n_failed_checks'])}/{int(verdict['n_checks'])} failed checks)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Stage-C patch-unit gate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result_dir", action="append", default=None)
    group.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="Data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_spatial_distance", type=float, default=8.0)
    parser.add_argument("--n_shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if args.result_dir:
        result_dirs = [Path(path) for path in args.result_dir]
    else:
        result_dirs = scan_result_dirs(Path(args.results_dir))

    rows: list[dict[str, Any]] = []
    for index, result_dir in enumerate(result_dirs):
        evaluation = evaluate_liif_response_objects(result_dir, Path(args.data_root), device)
        sr_scale = int(evaluation.summary.get("sr_scale", 4))
        lr_image = load_lr_image_from_summary(evaluation.summary, Path(args.data_root), device)
        units = [
            hr7_baseline_unit_descriptors(evaluation.lr_up, evaluation.outputs),
            lr_cell_unit_descriptors(lr_image, evaluation.outputs, sr_scale=sr_scale),
        ]
        for unit in units:
            rows.extend(
                analyze_unit(
                    unit,
                    run=result_dir.name,
                    image=evaluation.summary.get("image", "?"),
                    seed_value=evaluation.summary.get("seed", "?"),
                    k=args.k,
                    min_spatial_distance=args.min_spatial_distance,
                    n_shuffles=args.n_shuffles,
                    seed=args.seed + index * 10007,
                )
            )

    verdict = gate_verdict(rows)
    output = {
        "settings": {
            "patch_unit_gate": "lr_cell_query_support",
            "baseline_unit": "hr7_rgb_grad_baseline",
            "response_mode": "trajectory_delta",
            "lr_support_size": 3,
            "hr_response_block": "sr_scale x sr_scale",
            "k": args.k,
            "min_spatial_distance": args.min_spatial_distance,
            "n_shuffles": args.n_shuffles,
            "seed": args.seed,
        },
        "gate_verdict": verdict,
        "rows": rows,
    }
    if args.format == "json":
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print_rows(rows, verdict)
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
