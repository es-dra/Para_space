"""Tests for Stage-C response-object audit helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import make_patch_grid
from experiments.Phase1_FittingDynamics.analyze_stage_c_response_object_audit import (
    VARIANT_BY_NAME,
    _trajectory_window_indices,
    patch_descriptors_from_field,
    response_field_for_variant,
    response_variant_descriptors,
    temporal_consistency_metrics,
    transform_response_descriptors,
)
from experiments.Phase1_FittingDynamics.analyze_stage_c_failure_audit import patch_failure_rows


def test_trajectory_window_indices_split_deltas_deterministically():
    assert _trajectory_window_indices(5, "full").tolist() == [0, 1, 2, 3, 4]
    assert _trajectory_window_indices(5, "early").tolist() == [0, 1]
    assert _trajectory_window_indices(5, "mid").tolist() == [2, 3]
    assert _trajectory_window_indices(5, "late").tolist() == [4]


def test_response_field_for_variant_selects_temporal_segments():
    outputs = np.zeros((7, 2, 2, 1), dtype=np.float64)
    for t in range(outputs.shape[0]):
        outputs[t, :, :, 0] = float(t)

    full = response_field_for_variant(outputs, VARIANT_BY_NAME["trajectory_unit"])
    early = response_field_for_variant(outputs, VARIANT_BY_NAME["trajectory_early_unit"])
    mid = response_field_for_variant(outputs, VARIANT_BY_NAME["trajectory_mid_unit"])
    late = response_field_for_variant(outputs, VARIANT_BY_NAME["trajectory_late_unit"])
    final = response_field_for_variant(outputs, VARIANT_BY_NAME["final_raw"])

    assert full.shape[0] == 6
    assert early.shape[0] == 2
    assert mid.shape[0] == 2
    assert late.shape[0] == 2
    assert np.allclose(final, 6.0)


def test_norm_only_transform_uses_log_response_norm():
    raw = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)

    transformed = transform_response_descriptors(raw, "norm_only")

    assert transformed.shape == (2, 1)
    assert transformed[:, 0].tolist() == pytest.approx(
        [np.log(5.0 + 1e-12), np.log(2.0 + 1e-12)]
    )


def test_response_variant_descriptors_preserve_raw_and_transformed_shapes():
    outputs = np.zeros((3, 4, 4, 1), dtype=np.float64)
    outputs[1, :, :, 0] = 1.0
    outputs[2, :, :, 0] = 3.0
    grid = make_patch_grid(4, 4, patch_size=2, stride=2)

    desc, raw = response_variant_descriptors(
        outputs,
        grid,
        patch_size=2,
        variant=VARIANT_BY_NAME["trajectory_raw"],
    )

    assert raw.shape == (4, 8)
    assert desc.shape == raw.shape
    assert np.linalg.norm(desc, axis=1).mean() > 0.0


def test_patch_descriptors_from_field_accepts_final_and_trajectory_fields():
    grid = make_patch_grid(4, 4, patch_size=2, stride=2)
    final = np.ones((4, 4, 3), dtype=np.float64)
    trajectory = np.ones((2, 4, 4, 3), dtype=np.float64)

    final_desc = patch_descriptors_from_field(final, grid, patch_size=2)
    trajectory_desc = patch_descriptors_from_field(trajectory, grid, patch_size=2)

    assert final_desc.shape == (4, 12)
    assert trajectory_desc.shape == (4, 24)


def test_patch_failure_rows_report_top1_and_oracle_norm_gap():
    geometry = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float64,
    )
    response = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    centers = np.array(
        [
            [1.0, 1.0],
            [1.0, 9.0],
            [9.0, 1.0],
            [9.0, 9.0],
        ],
        dtype=np.float64,
    )
    rows = patch_failure_rows(
        geometry,
        response,
        centers,
        patch_grad=np.array([0.2, 0.5, 0.7, 0.9], dtype=np.float64),
        patch_var=np.array([0.1, 0.4, 0.6, 0.8], dtype=np.float64),
        patch_size=3,
        k=2,
        min_spatial_distance=0.0,
        response_norm_values=np.array([1.0, 4.0, 2.0, 3.0], dtype=np.float64),
    )

    row0 = [row for row in rows if row["patch_index"] == 0][0]

    assert "top1_response_norm_rank_absdiff" in row0
    assert "oracle_response_norm_rank_absdiff" in row0
    assert row0["top1_response_norm_rank_absdiff"] > row0["oracle_response_norm_rank_absdiff"]


def test_temporal_consistency_is_higher_for_consistent_segment_ordering():
    outputs_consistent = np.zeros((4, 6, 6, 1), dtype=np.float64)
    outputs_inconsistent = np.zeros_like(outputs_consistent)
    grid = make_patch_grid(6, 6, patch_size=3, stride=3)
    centers = grid.centers

    patch_values_consistent = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    patch_values_a = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    patch_values_b = np.array([4.0, 2.0, 1.0, 0.0], dtype=np.float64)
    patch_values_c = np.array([0.0, 4.0, 1.0, 2.0], dtype=np.float64)
    inconsistent_segments = [patch_values_a, patch_values_b, patch_values_c]

    for patch_idx, (y, x) in enumerate(grid.starts):
        value = patch_values_consistent[patch_idx]
        outputs_consistent[1:, y:y + 3, x:x + 3, 0] = np.cumsum([value, value, value])[:, None, None]
        cumulative = 0.0
        for t, seg_values in enumerate(inconsistent_segments, start=1):
            cumulative += seg_values[patch_idx]
            outputs_inconsistent[t, y:y + 3, x:x + 3, 0] = cumulative

    consistent = temporal_consistency_metrics(
        outputs_consistent,
        grid,
        patch_size=3,
        centers=centers,
        min_spatial_distance=0.0,
    )
    inconsistent = temporal_consistency_metrics(
        outputs_inconsistent,
        grid,
        patch_size=3,
        centers=centers,
        min_spatial_distance=0.0,
    )

    assert consistent["temporal_segment_pair_spearman_mean"] > inconsistent[
        "temporal_segment_pair_spearman_mean"
    ]
    assert consistent["temporal_directed_percentile_std_mean"] < inconsistent[
        "temporal_directed_percentile_std_mean"
    ]
