"""Tests for the Stage-C geometry-response pilot probe."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import (
    GEOMETRY_DESCRIPTORS,
    LIIFEvaluation,
    analyze_evaluated_result,
    analyze_geometry_response_descriptors,
    geometry_patch_descriptors,
    geometry_neighbor_pairs,
    l2_normalize_rows,
    make_patch_grid,
    pairwise_distances,
    response_patch_descriptors,
    response_requirements,
    response_source_for_mode,
    spearman_correlation,
)


def test_geometry_neighbor_pairs_selects_nearest_eligible_neighbors():
    descriptors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
        ]
    )
    dist = pairwise_distances(l2_normalize_rows(descriptors))
    eligible = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(eligible, False)

    pairs = geometry_neighbor_pairs(dist, eligible, k=1)

    assert pairs[0] == (0, 1)
    assert pairs[1] == (1, 0)
    assert pairs[2] in {(2, 0), (2, 1)}


def test_analyze_geometry_response_detects_matched_clusters():
    geometry = l2_normalize_rows(
        np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
                [-1.0, 0.0],
                [-0.95, -0.05],
            ]
        )
    )
    response = l2_normalize_rows(
        np.array(
            [
                [1.0, 0.0],
                [0.98, 0.02],
                [0.0, 1.0],
                [0.02, 0.98],
                [-1.0, 0.0],
                [-0.98, -0.02],
            ]
        )
    )
    centers = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [0.0, 100.0],
            [100.0, 100.0],
            [50.0, 0.0],
            [50.0, 100.0],
        ]
    )

    result = analyze_geometry_response_descriptors(
        geometry,
        response,
        centers,
        k=1,
        min_spatial_distance=0.0,
        n_shuffles=32,
        seed=1,
    )

    assert result["effect_vs_shuffle"] > 0.0
    assert result["neighbor_response_dist"] < result["shuffled_response_dist_mean"]
    assert result["geometry_response_spearman"] > 0.5


def test_response_patch_descriptors_supports_final_and_trajectory_delta():
    outputs = np.zeros((3, 6, 6, 1), dtype=np.float64)
    outputs[1, 1:3, 1:3, 0] = 1.0
    outputs[2, 1:3, 1:3, 0] = 3.0
    grid = make_patch_grid(height=6, width=6, patch_size=3, stride=3)

    final_desc = response_patch_descriptors(outputs, grid, patch_size=3, mode="final_delta")
    trajectory_desc = response_patch_descriptors(
        outputs,
        grid,
        patch_size=3,
        mode="trajectory_delta",
    )

    assert final_desc.shape == (4, 9)
    assert trajectory_desc.shape == (4, 18)
    assert np.linalg.norm(final_desc[0]) > 0.0
    assert np.linalg.norm(trajectory_desc[0]) > 0.0


def test_response_patch_descriptors_supports_feature_and_jacobian_modes():
    values = np.zeros((3, 6, 6, 2), dtype=np.float64)
    values[1, 1:3, 1:3, :] = 1.0
    values[2, 1:3, 1:3, :] = 2.0
    grid = make_patch_grid(height=6, width=6, patch_size=3, stride=3)

    feature_final = response_patch_descriptors(
        values,
        grid,
        patch_size=3,
        mode="feature_final_delta",
    )
    jac_traj = response_patch_descriptors(
        values,
        grid,
        patch_size=3,
        mode="coord_jacobian_trajectory_delta",
    )

    assert feature_final.shape == (4, 18)
    assert jac_traj.shape == (4, 36)
    assert np.linalg.norm(feature_final[0]) > 0.0
    assert np.linalg.norm(jac_traj[0]) > 0.0


def test_geometry_patch_descriptors_supports_ablation_modes():
    rng = np.random.default_rng(0)
    image = rng.random((6, 6, 3), dtype=np.float64)
    grid = make_patch_grid(height=6, width=6, patch_size=3, stride=3)
    expected_dims = {
        "rgb": 27,
        "gradient": 18,
        "structure_tensor": 27,
        "local_spectrum": 9,
        "rgb_grad": 45,
        "rgb_grad_context": 90,
    }

    for descriptor in GEOMETRY_DESCRIPTORS:
        desc = geometry_patch_descriptors(image, grid, patch_size=3, descriptor=descriptor)
        assert desc.shape == (4, expected_dims[descriptor])
        assert np.allclose(np.linalg.norm(desc, axis=1), 1.0)

    with pytest.raises(ValueError, match="unknown geometry descriptor"):
        geometry_patch_descriptors(image, grid, patch_size=3, descriptor="not_a_descriptor")


def test_rgb_grad_context_descriptor_uses_surrounding_context():
    image = np.zeros((9, 9, 3), dtype=np.float64)
    image[:, :3, 0] = 1.0
    image[3:6, 3:6, 1] = 1.0
    grid = make_patch_grid(height=9, width=9, patch_size=3, stride=3)

    local = geometry_patch_descriptors(
        image,
        grid,
        patch_size=3,
        descriptor="rgb_grad",
        normalize=False,
    )
    context = geometry_patch_descriptors(
        image,
        grid,
        patch_size=3,
        descriptor="rgb_grad_context",
        normalize=False,
    )

    assert context.shape == (9, local.shape[1] * 2)
    assert np.allclose(context[:, : local.shape[1]], local)
    assert np.linalg.norm(context[:, local.shape[1]:], axis=1).max() > 0.0


def test_analyze_evaluated_result_records_descriptor_and_norm_diagnostics():
    yy, xx = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    target = np.stack([xx, yy, xx + yy], axis=2).astype(np.float64) / 10.0
    outputs = np.stack(
        [
            np.zeros_like(target),
            target * 0.5,
            target,
        ],
        axis=0,
    )
    summary = {
        "image": "synthetic.png",
        "seed": 0,
        "final_psnr": 12.0,
        "_recomputed_final_psnr": 12.0,
        "_recomputed_initial_psnr": 3.0,
    }

    row = analyze_evaluated_result(
        result_dir=Path("synthetic_run"),
        target_hr=target,
        lr_up=target,
        summary=summary,
        geometry_source="lr_up",
        geometry_descriptor="structure_tensor",
        response_mode="trajectory_delta",
        patch_size=3,
        stride=3,
        k=1,
        min_spatial_distance=0.0,
        n_shuffles=8,
        seed=2,
        outputs=outputs,
    )

    assert row["geometry_descriptor"] == "structure_tensor"
    assert row["geometry_norm_mean"] > 0.0
    assert row["response_norm_mean"] > 0.0
    assert "effect_vs_eligible" in row
    assert "neighbor_response_dist_p90" in row


def test_response_source_selection_and_requirements():
    outputs = np.ones((2, 4, 4, 3), dtype=np.float64)
    features = np.ones((2, 4, 4, 5), dtype=np.float64) * 2.0
    jacobians = np.ones((2, 4, 4, 6), dtype=np.float64) * 3.0
    evaluation = LIIFEvaluation(
        outputs=outputs,
        target_hr=np.zeros((4, 4, 3), dtype=np.float64),
        lr_up=np.zeros((4, 4, 3), dtype=np.float64),
        summary={"_recomputed_final_psnr": 1.0, "_recomputed_initial_psnr": 0.0},
        features_hr=features,
        coord_jacobians=jacobians,
    )

    assert response_requirements("trajectory_delta") == (False, False)
    assert response_requirements("feature_final_delta") == (True, False)
    assert response_requirements("coord_jacobian_trajectory_delta") == (False, True)
    assert response_source_for_mode(evaluation, "final_delta") is outputs
    assert response_source_for_mode(evaluation, "feature_trajectory_delta") is features
    assert response_source_for_mode(evaluation, "coord_jacobian_final_delta") is jacobians

    with pytest.raises(ValueError, match="requires feature response"):
        response_source_for_mode(
            LIIFEvaluation(outputs, evaluation.target_hr, evaluation.lr_up, evaluation.summary),
            "feature_final_delta",
        )


def test_analyze_evaluated_result_supports_feature_response_object():
    yy, xx = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    target = np.stack([xx, yy, xx + yy], axis=2).astype(np.float64) / 10.0
    features = np.stack([target, target * 1.5, target * 2.0], axis=0)
    evaluation = LIIFEvaluation(
        outputs=np.zeros((3, 6, 6, 3), dtype=np.float64),
        target_hr=target,
        lr_up=target,
        summary={
            "image": "synthetic.png",
            "seed": 0,
            "final_psnr": 12.0,
            "_recomputed_final_psnr": 12.0,
            "_recomputed_initial_psnr": 3.0,
        },
        features_hr=features,
    )

    row = analyze_evaluated_result(
        result_dir=Path("synthetic_run"),
        target_hr=target,
        lr_up=target,
        summary=evaluation.summary,
        geometry_source="lr_up",
        geometry_descriptor="rgb",
        response_mode="feature_trajectory_delta",
        patch_size=3,
        stride=3,
        k=1,
        min_spatial_distance=0.0,
        n_shuffles=8,
        seed=2,
        evaluation=evaluation,
    )

    assert row["response_mode"] == "feature_trajectory_delta"
    assert row["response_norm_mean"] > 0.0


def test_spearman_correlation_handles_monotone_values():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    z = np.array([40.0, 30.0, 20.0, 10.0])

    assert spearman_correlation(x, y) == pytest.approx(1.0)
    assert spearman_correlation(x, z) == pytest.approx(-1.0)
