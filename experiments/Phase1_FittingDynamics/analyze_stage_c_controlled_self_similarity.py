#!/usr/bin/env python3
"""Stage-C controlled self-similarity sanity gate.

This read-only diagnostic asks a narrower question than natural-image Stage C:
if an image contains known repeated local structure, does the fitting-dynamics
response probe recover lower response distance inside those known duplicate
groups? A failure here points to the probe/response/training path itself; a
pass here with natural-image failures points back to natural-image matching,
semantics, context, or model-internal unit definition.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import (  # noqa: E402
    analyze_geometry_response_descriptors,
    eligible_pair_mask,
    evaluate_liif_response_objects,
    finite_percentile,
    geometry_neighbor_pairs,
    geometry_patch_descriptors,
    l2_normalize_rows,
    make_patch_grid,
    mean_pair_distance,
    one_sided_le_pvalue,
    pair_distance_values,
    pairwise_distances,
    response_patch_descriptors,
    sample_random_pair_mean,
    scan_result_dirs,
    spearman_correlation,
)
from experiments.Phase1_FittingDynamics.analyze_stage_c_patch_unit_gate import zscore_columns  # noqa: E402

NEGATIVE_CONTROL_ROLES = {"negative_nonperiodic_texture", "negative_independent_tiles"}


def load_metadata_file(path: Path) -> dict[str, Any]:
    """Load the controlled-image metadata document."""
    with path.open("r") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata file is not a JSON object: {path}")
    return metadata


def load_metadata(path: Path) -> dict[str, Any]:
    """Load controlled-image metadata keyed by image name."""
    metadata = load_metadata_file(path)
    images = metadata.get("images", metadata)
    if not isinstance(images, dict):
        raise ValueError(f"metadata file has no image mapping: {path}")
    return images


def read_rgb_image(path: Path) -> np.ndarray:
    """Read an RGB image as float64 in [0, 1]."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"failed to read image: {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float64) / 255.0


def lr_up_from_hr(image: np.ndarray, sr_scale: int = 4) -> np.ndarray:
    """Mimic the Stage-B x4 LR input upsampled back to HR size."""
    height, width = image.shape[:2]
    lr = cv2.resize(
        image,
        (width // sr_scale, height // sr_scale),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(lr, (width, height), interpolation=cv2.INTER_LINEAR)


def patch_group_labels(starts: Sequence[tuple[int, int]], tile_size: int) -> np.ndarray:
    """Assign patch starts to known duplicate groups by relative tile offset."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    return np.asarray([f"{y % tile_size:03d}_{x % tile_size:03d}" for y, x in starts])


def known_group_pairs(
    labels: np.ndarray,
    eligible: np.ndarray,
    min_group_count: int,
) -> list[tuple[int, int]]:
    """Return directed eligible pairs sharing a known duplicate-group label."""
    counts = Counter(labels.tolist())
    pairs: list[tuple[int, int]] = []
    for i, label in enumerate(labels):
        if counts[label] < min_group_count:
            continue
        candidates = np.flatnonzero(eligible[i] & (labels == label))
        for j in candidates:
            pairs.append((i, int(j)))
    return pairs


def duplicate_hit_at_k(
    geometry_dist: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    *,
    k: int,
    min_group_count: int,
) -> float:
    """Fraction of geometry top-k directed pairs that hit a known duplicate group."""
    pairs = geometry_neighbor_pairs(geometry_dist, eligible, k=k)
    if not pairs:
        return float("nan")
    counts = Counter(labels.tolist())
    hits = [
        labels[i] == labels[j] and counts[labels[i]] >= min_group_count
        for i, j in pairs
    ]
    return float(np.mean(hits))


def directed_response_percentiles(
    response_dist: np.ndarray,
    eligible: np.ndarray,
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Percentile of each directed pair response distance among anchor-eligible responses."""
    values = []
    for anchor, neighbor in pairs:
        candidates = np.flatnonzero(eligible[anchor])
        if candidates.size <= 1:
            continue
        candidate_values = response_dist[anchor, candidates]
        # Use the best-case tie convention for controlled duplicates: exact
        # repeated response patches should receive percentile 0, not be
        # penalized by the number of identical duplicate ties.
        rank = float(np.mean(candidate_values < response_dist[anchor, neighbor]))
        values.append(rank)
    return np.asarray(values, dtype=np.float64)


def geometry_top1_response_percentile_summary(
    geometry_desc: np.ndarray,
    response_desc: np.ndarray,
    centers: np.ndarray,
    min_spatial_distance: float,
) -> dict[str, float]:
    """Summarize response percentiles for geometry top-1 neighbors."""
    geometry_dist = pairwise_distances(geometry_desc)
    response_dist = pairwise_distances(response_desc)
    eligible = eligible_pair_mask(centers, min_spatial_distance)
    pairs = geometry_neighbor_pairs(geometry_dist, eligible, k=1)
    values = directed_response_percentiles(response_dist, eligible, pairs)
    return {
        "patch_top1_response_percentile_mean": float(np.mean(values)) if values.size else float("nan"),
        "patch_top1_response_percentile_p90": finite_percentile(values, 90),
        "patch_worst_response_frac": float(np.mean(values >= 0.8)) if values.size else float("nan"),
    }


def known_group_metrics(
    geometry_desc: np.ndarray,
    response_desc: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    min_group_count: int,
    min_spatial_distance: float,
    n_shuffles: int,
    seed: int,
) -> dict[str, float]:
    """Compare known duplicate-group response distances to random/shuffled controls."""
    geometry_dist = pairwise_distances(geometry_desc)
    response_dist = pairwise_distances(response_desc)
    eligible = eligible_pair_mask(centers, min_spatial_distance)
    pairs = known_group_pairs(labels, eligible, min_group_count)
    eligible_pairs = list(zip(*np.nonzero(eligible)))
    if not pairs:
        raise ValueError("no eligible known-group pairs; lower min_group_count/min_spatial_distance")

    observed = mean_pair_distance(response_dist, pairs)
    geometry_observed = mean_pair_distance(geometry_dist, pairs)
    rng = np.random.default_rng(seed)
    random_means = []
    shuffled_means = []
    for _ in range(n_shuffles):
        random_means.append(sample_random_pair_mean(response_dist, eligible_pairs, len(pairs), rng))
        perm = rng.permutation(response_dist.shape[0])
        shuffled_dist = response_dist[perm][:, perm]
        shuffled_means.append(mean_pair_distance(shuffled_dist, pairs))

    random_mean = float(np.mean(random_means))
    shuffled_mean = float(np.mean(shuffled_means))
    pair_percentiles = directed_response_percentiles(response_dist, eligible, pairs)
    upper = np.triu(eligible, k=1)
    eligible_geometry = geometry_dist[upper]
    eligible_response = response_dist[upper]
    group_counts = Counter(labels.tolist())
    retained_groups = [count for count in group_counts.values() if count >= min_group_count]
    response_values = pair_distance_values(response_dist, pairs)
    geometry_values = pair_distance_values(geometry_dist, pairs)

    return {
        "n_groups_total": float(len(group_counts)),
        "n_groups_retained": float(len(retained_groups)),
        "n_known_group_pairs": float(len(pairs)),
        f"duplicate_hit_at_{k}": duplicate_hit_at_k(
            geometry_dist,
            labels,
            eligible,
            k=k,
            min_group_count=min_group_count,
        ),
        "known_group_response_dist": observed,
        "known_group_response_dist_p50": finite_percentile(response_values, 50),
        "known_group_response_dist_p90": finite_percentile(response_values, 90),
        "random_response_dist_mean": random_mean,
        "shuffled_response_dist_mean": shuffled_mean,
        "shuffled_response_p_le": one_sided_le_pvalue(observed, shuffled_means),
        "effect_vs_random": random_mean - observed,
        "effect_vs_shuffle": shuffled_mean - observed,
        "effect_vs_shuffle_frac": (
            (shuffled_mean - observed) / shuffled_mean if shuffled_mean > 0.0 else float("nan")
        ),
        "known_group_response_percentile_mean": float(np.mean(pair_percentiles)),
        "known_group_response_percentile_p90": finite_percentile(pair_percentiles, 90),
        "known_group_geometry_dist": geometry_observed,
        "known_group_geometry_dist_p50": finite_percentile(geometry_values, 50),
        "eligible_geometry_dist_mean": float(np.mean(eligible_geometry)),
        "known_group_geometry_effect_frac": (
            (float(np.mean(eligible_geometry)) - geometry_observed) / float(np.mean(eligible_geometry))
            if float(np.mean(eligible_geometry)) > 0.0 else float("nan")
        ),
        "eligible_response_dist_mean": float(np.mean(eligible_response)),
        "geometry_response_spearman": spearman_correlation(eligible_geometry, eligible_response),
    }


def patch_content_descriptor(image: np.ndarray, starts: Sequence[tuple[int, int]], patch_size: int) -> np.ndarray:
    """Build a weak content descriptor from mean intensity, variance, and gradient."""
    gray = np.asarray(image, dtype=np.float64).mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    values = []
    for y, x in starts:
        patch = np.asarray(image[y:y + patch_size, x:x + patch_size, :], dtype=np.float64)
        grad_patch = grad_mag[y:y + patch_size, x:x + patch_size]
        values.append((float(np.mean(patch)), float(np.var(patch)), float(np.mean(grad_patch))))
    return l2_normalize_rows(zscore_columns(np.asarray(values, dtype=np.float64)))


def coordinate_descriptor(centers: np.ndarray) -> np.ndarray:
    """Coordinate-only weak control descriptor."""
    return l2_normalize_rows(zscore_columns(np.asarray(centers, dtype=np.float64)))


def standard_analysis_rows(
    geometry_desc: np.ndarray,
    response_desc: np.ndarray,
    centers: np.ndarray,
    content_desc: np.ndarray,
    *,
    k: int,
    min_spatial_distance: float,
    n_shuffles: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run standard geometry nearest-neighbor analysis and weak controls."""
    rng = np.random.default_rng(seed + 377)
    cases = [
        ("geometry", geometry_desc),
        ("coordinate_only", coordinate_descriptor(centers)),
        ("content_intensity_only", content_desc),
        ("geometry_shuffled", geometry_desc[rng.permutation(geometry_desc.shape[0])]),
    ]
    rows = []
    for index, (name, desc) in enumerate(cases):
        metrics = analyze_geometry_response_descriptors(
            desc,
            response_desc,
            centers,
            k=k,
            min_spatial_distance=min_spatial_distance,
            n_shuffles=n_shuffles,
            seed=seed + index * 1009,
        )
        row: dict[str, Any] = {"analysis": name}
        row.update(metrics)
        row.update(
            geometry_top1_response_percentile_summary(
                desc,
                response_desc,
                centers,
                min_spatial_distance,
            )
        )
        rows.append(row)
    return rows


def _role_supports_known_groups(row: dict[str, Any]) -> bool:
    return (
        float(row["effect_vs_shuffle_frac"]) >= 0.10
        and float(row["shuffled_response_p_le"]) <= 0.05
        and float(row["known_group_response_percentile_mean"]) <= 0.35
    )


def _synthetic_positive_support(row: dict[str, Any]) -> bool:
    hit_key = f"duplicate_hit_at_{int(row['k'])}"
    return (
        float(row.get(hit_key, float("nan"))) >= 0.95
        and float(row["effect_vs_shuffle_frac"]) >= 0.35
        and float(row["shuffled_response_p_le"]) <= 0.005
        and float(row["geometry_response_spearman"]) >= 0.50
        and float(row["known_group_response_percentile_mean"]) <= 0.05
    )


def _fitted_positive_support(row: dict[str, Any]) -> bool:
    geom = next((r for r in row["standard_rows"] if r["analysis"] == "geometry"), {})
    return (
        float(row.get("summary_final_psnr", float("nan"))) >= 35.0
        and float(row.get("final_psnr_abs_error", float("nan"))) <= 1e-4
        and float(row["effect_vs_shuffle_frac"]) >= 0.12
        and float(row["shuffled_response_p_le"]) <= 0.01
        and float(row["geometry_response_spearman"]) >= 0.10
        and float(geom.get("patch_top1_response_percentile_mean", float("nan"))) <= 0.25
        and float(geom.get("patch_worst_response_frac", float("nan"))) <= 0.08
    )


def gate_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply controlled sanity-gate standards."""
    checks: dict[str, bool] = {}
    synthetic_mode = all(row.get("gate_tier") == "synthetic_response_smoke" for row in rows)
    positive_rows = [r for r in rows if r["metadata_role"] == "positive_known_duplicate"]
    negative_rows = [r for r in rows if r["metadata_role"] in NEGATIVE_CONTROL_ROLES]
    if synthetic_mode:
        for row in positive_rows:
            prefix = row["run"]
            checks[f"{prefix}_synthetic_known_duplicate_support"] = _synthetic_positive_support(row)
        for row in negative_rows:
            prefix = row["run"]
            checks[f"{prefix}_synthetic_nonrepeat_not_supported"] = (
                float(row["effect_vs_shuffle_frac"]) <= 0.03
                or float(row["shuffled_response_p_le"]) > 0.05
            )
        passed = bool(checks) and all(checks.values())
        return {
            "verdict": "pass_synthetic_response_smoke" if passed else "fail_synthetic_response_smoke",
            "passed": passed,
            "checks": checks,
            "n_checks": float(len(checks)),
            "n_failed_checks": float(sum(not ok for ok in checks.values())),
        }

    standard_by_run = {
        row["run"]: row.get("standard_rows", [])
        for row in rows
    }
    for row in positive_rows:
        prefix = row["run"]
        checks[f"{prefix}_fitted_known_duplicate_support"] = _fitted_positive_support(row)
        checks[f"{prefix}_known_group_geometry_effect_ge_0.25"] = (
            float(row["known_group_geometry_effect_frac"]) >= 0.25
        )
        geom = next((r for r in standard_by_run[prefix] if r["analysis"] == "geometry"), None)
        coord = next((r for r in standard_by_run[prefix] if r["analysis"] == "coordinate_only"), None)
        if geom is not None:
            checks[f"{prefix}_standard_geometry_effect_ge_0.05"] = (
                float(geom["effect_vs_shuffle_frac"]) >= 0.05
                and float(geom["shuffled_response_p_le"]) <= 0.05
            )
        if geom is not None and coord is not None:
            checks[f"{prefix}_geometry_beats_coordinate"] = (
                float(geom["effect_vs_shuffle_frac"]) >= float(coord["effect_vs_shuffle_frac"]) + 0.03
                or float(geom["patch_top1_response_percentile_mean"])
                <= float(coord["patch_top1_response_percentile_mean"]) - 0.05
            )
        content = next(
            (r for r in standard_by_run[prefix] if r["analysis"] == "content_intensity_only"),
            None,
        )
        if geom is not None and content is not None:
            checks[f"{prefix}_geometry_beats_content"] = (
                float(geom["effect_vs_shuffle_frac"]) >= float(content["effect_vs_shuffle_frac"]) + 0.03
                or float(geom["patch_top1_response_percentile_mean"])
                <= float(content["patch_top1_response_percentile_mean"]) - 0.05
            )
    for row in negative_rows:
        prefix = row["run"]
        checks[f"{prefix}_known_group_not_supported"] = not _fitted_positive_support(row)

    positive_known_fail = any(
        key.endswith("_fitted_known_duplicate_support") and not ok for key, ok in checks.items()
    )
    negative_confound = any(
        key.endswith("_known_group_not_supported") and not ok for key, ok in checks.items()
    )
    standard_matching_fail = any(
        key.endswith("_standard_geometry_effect_ge_0.05") and not ok for key, ok in checks.items()
    )
    content_confound = any(
        key.endswith("_geometry_beats_content") and not ok for key, ok in checks.items()
    )
    passed = bool(checks) and all(checks.values())
    if passed:
        verdict = "pass_controlled_probe"
    elif positive_known_fail and content_confound:
        verdict = "fail_known_duplicate_stability_and_content_confound"
    elif positive_known_fail:
        verdict = "fail_probe_on_known_duplicates"
    elif negative_confound:
        verdict = "fail_group_or_position_confound"
    elif standard_matching_fail:
        verdict = "fail_geometry_matching_despite_known_duplicates"
    elif content_confound:
        verdict = "fail_geometry_content_dissociation"
    else:
        verdict = "fail_controlled_gate"
    return {
        "verdict": verdict,
        "passed": passed,
        "checks": checks,
        "n_checks": float(len(checks)),
        "n_failed_checks": float(sum(not ok for ok in checks.values())),
    }


def synthetic_outputs_from_image(image: np.ndarray) -> np.ndarray:
    """Create a deterministic synthetic output trajectory from an RGB image."""
    field = np.asarray(image, dtype=np.float64)
    return np.stack(
        [
            np.zeros_like(field),
            0.5 * field,
            field,
        ],
        axis=0,
    )


def analyze_synthetic_image(
    image_name: str,
    metadata: dict[str, Any],
    *,
    image_root: Path,
    geometry_source: str,
    geometry_descriptor: str,
    response_mode: str,
    patch_size: int,
    stride: int,
    k: int,
    min_spatial_distance: float,
    min_group_count: int,
    n_shuffles: int,
    seed: int,
) -> dict[str, Any]:
    """Analyze one controlled image with injected synthetic response fields."""
    target_hr = read_rgb_image(image_root / image_name)
    lr_up = lr_up_from_hr(target_hr)
    geometry_image = target_hr if geometry_source == "hr" else lr_up
    response_outputs = synthetic_outputs_from_image(target_hr)
    tile_size = int(metadata["tile_size"])
    grid = make_patch_grid(target_hr.shape[0], target_hr.shape[1], patch_size, stride)
    labels = patch_group_labels(grid.starts, tile_size)
    geometry_raw = geometry_patch_descriptors(
        geometry_image,
        grid,
        patch_size,
        descriptor=geometry_descriptor,
        normalize=False,
    )
    response_raw = response_patch_descriptors(
        response_outputs,
        grid,
        patch_size,
        response_mode,
        normalize=False,
    )
    geometry_desc = l2_normalize_rows(geometry_raw)
    response_desc = l2_normalize_rows(response_raw)
    metrics = known_group_metrics(
        geometry_desc,
        response_desc,
        grid.centers,
        labels,
        k=k,
        min_group_count=min_group_count,
        min_spatial_distance=min_spatial_distance,
        n_shuffles=n_shuffles,
        seed=seed,
    )
    content_desc = patch_content_descriptor(geometry_image, grid.starts, patch_size)
    standards = standard_analysis_rows(
        geometry_desc,
        response_desc,
        grid.centers,
        content_desc,
        k=k,
        min_spatial_distance=min_spatial_distance,
        n_shuffles=n_shuffles,
        seed=seed + 1901,
    )
    row: dict[str, Any] = {
        "run": f"synthetic_{Path(image_name).stem}",
        "image": image_name,
        "seed": metadata.get("seed", "?"),
        "gate_tier": "synthetic_response_smoke",
        "metadata_role": metadata["role"],
        "geometry_source": geometry_source,
        "geometry_descriptor": geometry_descriptor,
        "response_mode": response_mode,
        "patch_size": patch_size,
        "stride": stride,
        "k": k,
        "tile_size": tile_size,
        "min_group_count": min_group_count,
        "min_spatial_distance": min_spatial_distance,
        "standard_rows": standards,
    }
    row.update(metrics)
    return row


def analyze_synthetic_smoke(
    metadata_doc: dict[str, Any],
    *,
    metadata_path: Path,
    geometry_source: str,
    geometry_descriptor: str,
    response_mode: str,
    patch_size: int,
    stride: int,
    k: int,
    min_spatial_distance: float,
    min_group_count: int,
    n_shuffles: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the no-training Tier-0 synthetic response smoke gate."""
    images = metadata_doc.get("images", metadata_doc)
    image_root = Path(metadata_doc.get("output_root", metadata_path.parent))
    if not image_root.is_absolute():
        image_root = metadata_path.parent / image_root
        if not image_root.exists():
            image_root = Path(metadata_doc.get("output_root", "."))
    rows = []
    for index, (image_name, metadata) in enumerate(sorted(images.items())):
        rows.append(
            analyze_synthetic_image(
                image_name,
                metadata,
                image_root=image_root,
                geometry_source=geometry_source,
                geometry_descriptor=geometry_descriptor,
                response_mode=response_mode,
                patch_size=patch_size,
                stride=stride,
                k=k,
                min_spatial_distance=min_spatial_distance,
                min_group_count=min_group_count,
                n_shuffles=n_shuffles,
                seed=seed + index * 10007,
            )
        )
    return rows


def analyze_result_dir(
    result_dir: Path,
    metadata_by_image: dict[str, Any],
    *,
    data_root: Path,
    device: torch.device,
    geometry_source: str,
    geometry_descriptor: str,
    response_mode: str,
    patch_size: int,
    stride: int,
    k: int,
    min_spatial_distance: float,
    min_group_count: int,
    n_shuffles: int,
    seed: int,
) -> dict[str, Any]:
    """Analyze one controlled LIIF fitting run."""
    evaluation = evaluate_liif_response_objects(result_dir, data_root, device)
    image_name = str(evaluation.summary["image"])
    if image_name not in metadata_by_image:
        raise ValueError(f"missing controlled metadata for image {image_name}")
    metadata = metadata_by_image[image_name]
    tile_size = int(metadata["tile_size"])
    geometry_image = evaluation.target_hr if geometry_source == "hr" else evaluation.lr_up
    grid = make_patch_grid(evaluation.target_hr.shape[0], evaluation.target_hr.shape[1], patch_size, stride)
    labels = patch_group_labels(grid.starts, tile_size)
    geometry_raw = geometry_patch_descriptors(
        geometry_image,
        grid,
        patch_size,
        descriptor=geometry_descriptor,
        normalize=False,
    )
    response_raw = response_patch_descriptors(
        evaluation.outputs,
        grid,
        patch_size,
        response_mode,
        normalize=False,
    )
    geometry_desc = l2_normalize_rows(geometry_raw)
    response_desc = l2_normalize_rows(response_raw)
    metrics = known_group_metrics(
        geometry_desc,
        response_desc,
        grid.centers,
        labels,
        k=k,
        min_group_count=min_group_count,
        min_spatial_distance=min_spatial_distance,
        n_shuffles=n_shuffles,
        seed=seed,
    )
    content_desc = patch_content_descriptor(geometry_image, grid.starts, patch_size)
    standards = standard_analysis_rows(
        geometry_desc,
        response_desc,
        grid.centers,
        content_desc,
        k=k,
        min_spatial_distance=min_spatial_distance,
        n_shuffles=n_shuffles,
        seed=seed + 1901,
    )
    row: dict[str, Any] = {
        "run": result_dir.name,
        "image": image_name,
        "seed": evaluation.summary.get("seed", "?"),
        "metadata_role": metadata["role"],
        "geometry_source": geometry_source,
        "geometry_descriptor": geometry_descriptor,
        "response_mode": response_mode,
        "patch_size": patch_size,
        "stride": stride,
        "k": k,
        "tile_size": tile_size,
        "min_group_count": min_group_count,
        "min_spatial_distance": min_spatial_distance,
        "summary_final_psnr": float(evaluation.summary.get("final_psnr", float("nan"))),
        "recomputed_final_psnr": float(evaluation.summary["_recomputed_final_psnr"]),
        "final_psnr_abs_error": abs(
            float(evaluation.summary["_recomputed_final_psnr"])
            - float(evaluation.summary.get("final_psnr", float("nan")))
        ),
        "standard_rows": standards,
    }
    row.update(metrics)
    return row


def format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[dict[str, Any]], verdict: dict[str, Any]) -> None:
    """Print compact controlled-gate rows."""
    print("\nControlled self-similarity gate rows")
    print("run                                      role                          dup@k   known_eff p_shuf pct    rho    std_geom")
    print("-" * 124)
    for row in rows:
        geom = next((r for r in row["standard_rows"] if r["analysis"] == "geometry"), {})
        hit_key = f"duplicate_hit_at_{int(row['k'])}"
        print(
            f"{row['run']:<40} {row['metadata_role']:<29} "
            f"{format_float(row.get(hit_key, float('nan'))):<7} "
            f"{format_float(row['effect_vs_shuffle_frac']):<9} "
            f"{format_float(row['shuffled_response_p_le']):<7} "
            f"{format_float(row['known_group_response_percentile_mean']):<7} "
            f"{format_float(row['geometry_response_spearman']):<7} "
            f"{format_float(geom.get('effect_vs_shuffle_frac', float('nan'))):<8}"
        )
    print(
        f"\nGate verdict: {verdict['verdict']} "
        f"({int(verdict['n_failed_checks'])}/{int(verdict['n_checks'])} failed checks)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only controlled Stage-C self-similarity gate")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--result_dir", action="append", default=None)
    group.add_argument("--results_dir", type=str, default=None)
    parser.add_argument(
        "--synthetic_response_smoke",
        action="store_true",
        help="Run Tier-0 no-training synthetic response smoke using controlled images.",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        default="Data/ControlledSelfSimilarity/controlled_self_similarity_metadata.json",
    )
    parser.add_argument("--data_root", type=str, default="Data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--geometry_source", choices=["hr", "lr_up"], default="lr_up")
    parser.add_argument("--geometry_descriptor", type=str, default="rgb_grad")
    parser.add_argument("--response_mode", choices=["trajectory_delta", "final_delta"], default="trajectory_delta")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_spatial_distance", type=float, default=8.0)
    parser.add_argument("--min_group_count", type=int, default=3)
    parser.add_argument("--n_shuffles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    metadata_path = Path(args.metadata_json)
    metadata_doc = load_metadata_file(metadata_path)
    metadata_by_image = metadata_doc.get("images", metadata_doc)
    if args.synthetic_response_smoke and (args.result_dir or args.results_dir):
        parser.error("--synthetic_response_smoke cannot be combined with --result_dir/--results_dir")
    if not args.synthetic_response_smoke and not (args.result_dir or args.results_dir):
        parser.error("one of --synthetic_response_smoke, --result_dir, or --results_dir is required")

    if args.synthetic_response_smoke:
        rows = analyze_synthetic_smoke(
            metadata_doc,
            metadata_path=metadata_path,
            geometry_source=args.geometry_source,
            geometry_descriptor=args.geometry_descriptor,
            response_mode=args.response_mode,
            patch_size=args.patch_size,
            stride=args.stride,
            k=args.k,
            min_spatial_distance=args.min_spatial_distance,
            min_group_count=args.min_group_count,
            n_shuffles=args.n_shuffles,
            seed=args.seed,
        )
    elif args.result_dir:
        result_dirs = [Path(path) for path in args.result_dir]
        rows = [
            analyze_result_dir(
                result_dir,
                metadata_by_image,
                data_root=Path(args.data_root),
                device=device,
                geometry_source=args.geometry_source,
                geometry_descriptor=args.geometry_descriptor,
                response_mode=args.response_mode,
                patch_size=args.patch_size,
                stride=args.stride,
                k=args.k,
                min_spatial_distance=args.min_spatial_distance,
                min_group_count=args.min_group_count,
                n_shuffles=args.n_shuffles,
                seed=args.seed + index * 10007,
            )
            for index, result_dir in enumerate(result_dirs)
        ]
    else:
        result_dirs = scan_result_dirs(Path(args.results_dir))
        rows = [
            analyze_result_dir(
                result_dir,
                metadata_by_image,
                data_root=Path(args.data_root),
                device=device,
                geometry_source=args.geometry_source,
                geometry_descriptor=args.geometry_descriptor,
                response_mode=args.response_mode,
                patch_size=args.patch_size,
                stride=args.stride,
                k=args.k,
                min_spatial_distance=args.min_spatial_distance,
                min_group_count=args.min_group_count,
                n_shuffles=args.n_shuffles,
                seed=args.seed + index * 10007,
            )
            for index, result_dir in enumerate(result_dirs)
        ]
    verdict = gate_verdict(rows)
    output = {
        "settings": {
            "metadata_json": args.metadata_json,
            "geometry_source": args.geometry_source,
            "geometry_descriptor": args.geometry_descriptor,
            "response_mode": args.response_mode,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "k": args.k,
            "min_spatial_distance": args.min_spatial_distance,
            "min_group_count": args.min_group_count,
            "n_shuffles": args.n_shuffles,
            "seed": args.seed,
            "synthetic_response_smoke": args.synthetic_response_smoke,
        },
        "gate_verdict": verdict,
        "rows": rows,
    }
    if args.format == "json":
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print_table(rows, verdict)
        print(f"\n{len(rows)} controlled LIIF result(s) audited on {device}.")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(output, f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
