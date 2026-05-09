"""Tests for the Stage-C LIIF-aware patch-unit gate helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import (
    eligible_pair_mask,
    pairwise_distances,
)
from experiments.Phase1_FittingDynamics.analyze_stage_c_patch_unit_gate import (
    crop_center_edge,
    gate_verdict,
    lr_cell_unit_descriptors,
    matched_random_response_mean,
    rgb_grad_descriptor_for_crop,
)


def test_crop_center_edge_keeps_all_lr_cells_with_padding():
    image = np.arange(3 * 3 * 1, dtype=np.float64).reshape(3, 3, 1)

    crop = crop_center_edge(image, center_y=0, center_x=0, size=3)

    assert crop.shape == (3, 3, 1)
    assert crop[0, 0, 0] == image[0, 0, 0]
    assert crop[1, 1, 0] == image[0, 0, 0]


def test_rgb_grad_descriptor_for_crop_is_response_blind_and_nonempty():
    crop = np.zeros((3, 3, 3), dtype=np.float64)
    crop[:, 1:, 0] = 1.0

    desc = rgb_grad_descriptor_for_crop(crop)

    assert desc.ndim == 1
    assert desc.size == 3 * 3 * 3 + 3 * 3 * 2
    assert np.linalg.norm(desc) > 0.0


def test_lr_cell_unit_descriptors_map_lr_cells_to_hr_blocks():
    lr = np.zeros((3, 3, 3), dtype=np.float64)
    lr[..., 0] = np.arange(3, dtype=np.float64)[None, :]
    outputs = np.zeros((3, 12, 12, 1), dtype=np.float64)
    outputs[1] = 1.0
    outputs[2] = 3.0

    unit = lr_cell_unit_descriptors(lr, outputs, sr_scale=4, support_size=3)

    assert unit.name == "lr_cell_query_support"
    assert unit.geometry_desc.shape[0] == 9
    assert unit.response_raw.shape == (9, 2 * 4 * 4 * 1)
    assert unit.centers[0].tolist() == pytest.approx([1.5, 1.5])
    assert unit.centers[-1].tolist() == pytest.approx([9.5, 9.5])
    assert unit.patch_size_for_rows == 4


def test_matched_random_response_mean_returns_finite_controls():
    centers = np.array(
        [
            [0.0, 0.0],
            [0.0, 4.0],
            [4.0, 0.0],
            [4.0, 4.0],
        ],
        dtype=np.float64,
    )
    response_desc = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    response_dist = pairwise_distances(response_desc)
    eligible = eligible_pair_mask(centers, min_spatial_distance=1.0)
    pairs = [(0, 1), (1, 3), (2, 3)]
    labels = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )

    spatial = matched_random_response_mean(
        response_dist,
        centers,
        pairs,
        eligible,
        rng=np.random.default_rng(0),
        mode="spatial",
        n_repeats=3,
    )
    content = matched_random_response_mean(
        response_dist,
        centers,
        pairs,
        eligible,
        rng=np.random.default_rng(0),
        mode="content",
        content_labels=labels,
        n_repeats=3,
    )

    assert np.isfinite(spatial)
    assert np.isfinite(content)


def test_gate_verdict_requires_failure_runs_and_controls_to_pass():
    base = {
        "unit": "lr_cell_query_support",
        "analysis": "geometry",
        "label": "support",
        "effect_vs_shuffle_frac": 0.08,
        "shuffled_response_p_le": 0.01,
        "patch_top1_response_percentile_mean": 0.35,
        "patch_worst_response_frac": 0.10,
    }
    rows = []
    for run in [
        "LIIF_reduced_bird_sr4_seed42",
        "LIIF_reduced_head_sr4_seed42",
        "LIIF_reduced_baby_sr4_seed123",
        "LIIF_reduced_woman_sr4_seed42",
    ]:
        rows.append({"run": run, **base})
        rows.append(
            {
                "run": run,
                "unit": "lr_cell_query_support",
                "analysis": "coordinate_only",
                "effect_vs_shuffle_frac": 0.01,
                "patch_top1_response_percentile_mean": 0.45,
            }
        )
        rows.append(
            {
                "run": run,
                "unit": "lr_cell_query_support",
                "analysis": "content_intensity_only",
                "effect_vs_shuffle_frac": 0.01,
                "patch_top1_response_percentile_mean": 0.45,
            }
        )

    verdict = gate_verdict(rows)

    assert verdict["verdict"] == "pass"
    assert verdict["passed"] is True

    rows[0]["label"] = "tiny"
    failed = gate_verdict(rows)

    assert failed["verdict"] == "fail"
    assert failed["passed"] is False
