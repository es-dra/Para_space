"""Tests for reviewer-style Stage-C failure audit helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_c_failure_audit import (
    aggregate_image_reviews,
    aggregate_stratified_reviews,
    analyze_content_stratum,
    context_lowfreq_descriptor,
    directed_attribute_advantage,
    effect_label,
    geometry_topk_indices,
    crop_center_with_padding,
    image_to_uint8,
    patch_content_metrics,
    patch_candidate_attribute_arrays,
    patch_failure_rows,
    rank_fraction,
    render_patch_crop_grid,
    response_distance_percentiles,
    select_worst_patch_rows,
    spatial_region_labels,
    summarize_patch_failures,
    tercile_masks,
    row_review,
)
from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import make_patch_grid


def test_effect_label_rejects_tiny_positive_effects():
    assert effect_label(effect=0.001, frac=0.001, p_value=0.005, rho=0.2) == "tiny"
    assert effect_label(effect=-0.01, frac=-0.01, p_value=0.99, rho=0.0) == "negative"
    assert effect_label(effect=0.08, frac=0.06, p_value=0.2, rho=0.2) == "weak_control"
    assert effect_label(effect=0.08, frac=0.06, p_value=0.01, rho=0.01) == "local_only"
    assert effect_label(effect=0.08, frac=0.06, p_value=0.01, rho=0.1) == "support"


def test_row_review_flags_low_psnr_and_fractional_gain():
    row = {
        "run": "LIIF_reduced_bird_sr4_seed42",
        "image": "bird.png",
        "seed": 42,
        "response_mode": "coord_jacobian_final_delta",
        "effect_vs_shuffle": 0.005,
        "effect_vs_shuffle_frac": 0.001,
        "shuffled_response_p_le": 0.01,
        "geometry_response_spearman": 0.01,
        "summary_final_psnr": 21.5,
        "recomputed_initial_psnr": 10.0,
        "response_norm_mean": 1.0,
        "neighbor_response_dist": 1.0,
        "shuffled_response_dist_mean": 1.005,
        "final_psnr_abs_error": 0.0,
    }

    review = row_review(row)

    assert review["label"] == "tiny"
    assert "does_not_support_claim" in review["flags"]
    assert "fractional_gain_too_small" in review["flags"]
    assert "low_final_psnr_confound" in review["flags"]


def test_aggregate_image_reviews_marks_object_dependent_results():
    reviews = [
        {
            "image": "head.png",
            "seed": 42,
            "label": "negative",
            "effect_vs_shuffle": -0.01,
        },
        {
            "image": "head.png",
            "seed": 42,
            "label": "support",
            "effect_vs_shuffle": 0.06,
        },
    ]

    rows = aggregate_image_reviews(reviews)

    assert len(rows) == 1
    assert rows[0]["verdict"] == "object_dependent"
    assert rows[0]["support_count"] == 1
    assert rows[0]["negative_count"] == 1


def test_aggregate_stratified_reviews_keeps_content_bins_separate():
    reviews = [
        {
            "image": "bird.png",
            "seed": 42,
            "stratum": "gradient",
            "stratum_label": "high",
            "status": "ok",
            "label": "negative",
            "effect_vs_shuffle": -0.02,
            "n_patches": 40.0,
        },
        {
            "image": "bird.png",
            "seed": 42,
            "stratum": "gradient",
            "stratum_label": "high",
            "status": "ok",
            "label": "tiny",
            "effect_vs_shuffle": 0.01,
            "n_patches": 40.0,
        },
        {
            "image": "bird.png",
            "seed": 42,
            "stratum": "gradient",
            "stratum_label": "low",
            "status": "ok",
            "label": "support",
            "effect_vs_shuffle": 0.08,
            "n_patches": 41.0,
        },
    ]

    rows = aggregate_stratified_reviews(reviews)

    high = [row for row in rows if row["stratum_label"] == "high"][0]
    low = [row for row in rows if row["stratum_label"] == "low"][0]
    assert high["verdict"] == "no_support_or_negative"
    assert high["negative_count"] == 1
    assert low["verdict"] == "consistent_stratum_support"
    assert low["support_count"] == 1


def test_patch_content_metrics_reports_gradient_and_variance():
    yy, xx = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8), indexing="ij")
    image = np.stack([xx, yy, xx + yy], axis=2)

    metrics = patch_content_metrics(image, patch_size=4, stride=4)

    assert metrics["content_n_patches"] == 4.0
    assert metrics["content_grad_mean"] > 0.0
    assert metrics["content_var_mean"] > 0.0
    assert metrics["content_geom_norm_mean"] > 0.0


def test_crop_center_with_padding_handles_borders():
    image = np.arange(4 * 4 * 3, dtype=np.float64).reshape(4, 4, 3) / 100.0

    crop = crop_center_with_padding(image, center_y=0, center_x=0, size=3)

    assert crop.shape == (3, 3, 3)
    assert np.allclose(crop[0, 0], image[0, 0])


def test_render_patch_crop_grid_returns_image():
    image = np.zeros((8, 8, 3), dtype=np.float64)
    image[1:4, 1:4, 0] = 1.0
    rows = [
        {
            "run": "synthetic_run",
            "response_mode": "trajectory_delta",
            "worst_rank": 1,
            "patch_index": 0,
            "top1_index": 3,
            "center_y": 2.0,
            "center_x": 2.0,
            "top1_center_y": 6.0,
            "top1_center_x": 6.0,
            "top1_response_percentile": 1.0,
            "top1_response_dist": 1.2,
            "top1_geometry_dist": 0.3,
            "geometry_top2_to_top1_ratio": 1.1,
            "patch_grad_rank": 0.5,
            "patch_var_rank": 0.6,
            "response_norm_rank": 0.7,
            "spatial_region": "top_left",
        }
    ]

    rendered = render_patch_crop_grid(image, rows, patch_size=3, scale=2)

    assert rendered.size[0] > 0
    assert rendered.size[1] > 0
    assert image_to_uint8(image).dtype == np.uint8


def test_context_lowfreq_descriptor_is_mean_centered_and_unit_scaled():
    crop = np.zeros((6, 6, 3), dtype=np.float64)
    crop[:3, :3, :] = 1.0

    desc = context_lowfreq_descriptor(crop, bins=3)

    assert desc.shape == (9,)
    assert np.mean(desc) == pytest.approx(0.0)
    assert np.linalg.norm(desc) == pytest.approx(1.0)


def test_patch_candidate_attribute_arrays_are_response_blind_patch_features():
    yy, xx = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9), indexing="ij")
    image = np.stack([xx, yy, xx * 0.5 + yy * 0.5], axis=2)
    grid = make_patch_grid(9, 9, patch_size=3, stride=3)
    patch_grad = np.linspace(0.1, 0.9, len(grid.starts))
    patch_var = np.linspace(0.2, 1.0, len(grid.starts))

    attrs = patch_candidate_attribute_arrays(image, grid, patch_grad, patch_var, patch_size=3)

    assert attrs["edge_orientation_unit"].shape == (len(grid.starts), 2)
    assert attrs["context_lowfreq"].shape[0] == len(grid.starts)
    assert attrs["gradient_mean"].tolist() == pytest.approx(patch_grad.tolist())
    assert attrs["patch_variance"].tolist() == pytest.approx(patch_var.tolist())
    assert set(np.unique(attrs["spatial_region"])) <= {
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    }


def test_tercile_masks_are_deterministic_and_cover_all_patches():
    values = np.arange(10, dtype=np.float64)

    masks = tercile_masks(values)

    assert set(masks) == {"low", "mid", "high"}
    combined = masks["low"].astype(int) + masks["mid"].astype(int) + masks["high"].astype(int)
    assert np.all(combined == 1)
    assert np.flatnonzero(masks["low"]).tolist() == [0, 1, 2, 3]
    assert np.flatnonzero(masks["mid"]).tolist() == [4, 5, 6]
    assert np.flatnonzero(masks["high"]).tolist() == [7, 8, 9]


def test_rank_fraction_handles_ties_deterministically():
    ranks = rank_fraction(np.array([2.0, 1.0, 1.0, 4.0]))

    assert ranks.tolist() == [pytest.approx(2.0 / 3.0), pytest.approx(1.0 / 6.0), pytest.approx(1.0 / 6.0), 1.0]


def test_spatial_region_labels_split_patch_centers():
    centers = np.array(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    labels = spatial_region_labels(centers)

    assert labels.tolist() == ["top_left", "top_right", "bottom_left", "bottom_right"]


def test_response_distance_percentiles_are_rowwise():
    response_dist = np.array(
        [
            [0.0, 5.0, 1.0],
            [5.0, 0.0, 2.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    eligible = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(eligible, False)

    pct = response_distance_percentiles(response_dist, eligible)

    assert np.isnan(pct[0, 0])
    assert pct[0, 1] == pytest.approx(1.0)
    assert pct[0, 2] == pytest.approx(0.0)


def test_geometry_topk_indices_uses_eligible_sorted_order():
    geometry_dist = np.array(
        [
            [0.0, 0.2, 0.1, 0.3],
            [0.2, 0.0, 0.4, 0.1],
            [0.1, 0.4, 0.0, 0.2],
            [0.3, 0.1, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    eligible = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(eligible, False)
    eligible[0, 2] = False

    topk = geometry_topk_indices(geometry_dist, eligible, k=2)

    assert topk[0].tolist() == [1, 3]
    assert topk[1].tolist() == [3, 0]


def test_patch_failure_rows_identify_geometry_neighbor_response_mismatch():
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
    patch_grad = np.array([0.2, 0.5, 0.7, 0.9], dtype=np.float64)
    patch_var = np.array([0.1, 0.4, 0.6, 0.8], dtype=np.float64)

    rows = patch_failure_rows(
        geometry,
        response,
        centers,
        patch_grad,
        patch_var,
        patch_size=3,
        k=2,
        min_spatial_distance=0.0,
        response_norm_values=np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float64),
    )
    worst = select_worst_patch_rows(rows, limit=1)[0]
    summary = summarize_patch_failures(rows, worst_percentile=0.8, nonunique_ratio=1.05)

    assert len(rows) == 4
    assert worst["patch_index"] == 0
    assert worst["top1_index"] == 1
    assert worst["top1_response_percentile"] == pytest.approx(1.0)
    assert worst["oracle_index"] == 2
    assert worst["oracle_geometry_rank"] == pytest.approx(2.0)
    assert worst["oracle_response_percentile"] < worst["top1_response_percentile"]
    assert summary["worst_response_frac"] > 0.0
    assert summary["oracle_response_percentile_mean"] < summary["top1_response_percentile_mean"]
    assert summary["oracle_response_percentile_gain_mean"] > 0.0
    assert summary["oracle_fixes_worst_frac"] > 0.0
    assert "region_top_left_worst_frac" in summary


def test_patch_failure_rows_support_response_blind_reranking():
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
    rerank = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.98, 0.02],
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
    patch_grad = np.array([0.2, 0.5, 0.7, 0.9], dtype=np.float64)
    patch_var = np.array([0.1, 0.4, 0.6, 0.8], dtype=np.float64)

    rows = patch_failure_rows(
        geometry,
        response,
        centers,
        patch_grad,
        patch_var,
        patch_size=3,
        k=2,
        min_spatial_distance=0.0,
        rerank_desc=rerank,
    )
    worst = select_worst_patch_rows(rows, limit=1)[0]
    summary = summarize_patch_failures(rows, worst_percentile=0.8, nonunique_ratio=1.05)

    assert worst["patch_index"] == 0
    assert worst["top1_index"] == 1
    assert worst["rerank_index"] == 2
    assert worst["rerank_geometry_rank"] == pytest.approx(2.0)
    assert worst["rerank_response_percentile"] < worst["top1_response_percentile"]
    assert summary["has_patch_rerank"] is True
    assert summary["rerank_response_percentile_mean"] < summary["top1_response_percentile_mean"]
    assert summary["rerank_response_percentile_gain_mean"] > 0.0
    assert summary["rerank_recovered_oracle_gain_frac_mean"] > 0.0
    assert summary["rerank_fixes_worst_frac"] > 0.0
    assert summary["rerank_oracle_percentile_gap_mean"] >= 0.0
    assert summary["oracle_is_rerank_best_frac"] > 0.0
    assert summary["oracle_rerank_farther_than_top1_frac"] < 1.0


def test_patch_failure_rows_add_candidate_attribute_differences():
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
    patch_grad = np.array([0.2, 0.9, 0.25, 0.8], dtype=np.float64)
    patch_var = np.array([0.1, 0.8, 0.15, 0.7], dtype=np.float64)
    attrs = {
        "edge_orientation_unit": np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "edge_strength": np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float64),
        "structure_coherence": np.array([0.9, 0.2, 0.85, 0.1], dtype=np.float64),
        "gradient_mean": patch_grad,
        "laplacian_abs_mean": np.array([0.3, 0.8, 0.35, 0.7], dtype=np.float64),
        "patch_variance": patch_var,
        "context_lowfreq": np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "spatial_region": np.array(["top_left", "top_right", "top_left", "bottom_right"]),
    }

    rows = patch_failure_rows(
        geometry,
        response,
        centers,
        patch_grad,
        patch_var,
        patch_size=3,
        k=2,
        min_spatial_distance=0.0,
        candidate_attrs=attrs,
    )
    worst = select_worst_patch_rows(rows, limit=1)[0]
    summary = directed_attribute_advantage(rows, min_oracle_gain=0.2)

    assert worst["patch_index"] == 0
    assert worst["oracle_index"] == 2
    assert worst["oracle_edge_orientation_agreement"] > worst["top1_edge_orientation_agreement"]
    assert worst["oracle_gradient_mean_absdiff"] < worst["top1_gradient_mean_absdiff"]
    assert worst["oracle_same_spatial_region"] == 1.0
    assert summary["candidate_attribute_oracle_help_rows"] > 0.0
    assert summary["edge_orientation_agreement_oracle_better_frac"] > 0.0
    assert summary["gradient_mean_absdiff_oracle_better_frac"] > 0.0


def test_patch_failure_summary_counts_any_near_tie_candidate_as_nonunique():
    rows = [
        {
            "top1_response_percentile": 0.9,
            "oracle_response_percentile": 0.2,
            "oracle_response_percentile_gain": 0.7,
            "oracle_geometry_rank": 2.0,
            "topk_response_percentile_mean": 0.8,
            "geometry_topk_to_top1_ratio": 2.0,
            "geometry_top2_to_top1_ratio": 1.005,
            "geometry_top2_minus_top1": 0.001,
            "n_geometry_ties_1pct": 2.0,
            "response_norm_rank": 0.5,
            "patch_grad_rank": 0.6,
            "patch_var_rank": 0.7,
            "spatial_region": "top_left",
        },
        {
            "top1_response_percentile": 0.1,
            "oracle_response_percentile": 0.1,
            "oracle_response_percentile_gain": 0.0,
            "oracle_geometry_rank": 1.0,
            "topk_response_percentile_mean": 0.2,
            "geometry_topk_to_top1_ratio": 1.02,
            "geometry_top2_to_top1_ratio": 1.2,
            "geometry_top2_minus_top1": 0.2,
            "n_geometry_ties_1pct": 1.0,
            "response_norm_rank": 0.2,
            "patch_grad_rank": 0.3,
            "patch_var_rank": 0.4,
            "spatial_region": "bottom_right",
        },
    ]

    summary = summarize_patch_failures(rows, worst_percentile=0.8, nonunique_ratio=1.05)

    assert summary["nonunique_geometry_frac"] == pytest.approx(0.5)
    assert summary["top2_nonunique_geometry_frac"] == pytest.approx(0.5)
    assert summary["worst_and_nonunique_frac"] == pytest.approx(0.5)


def test_analyze_content_stratum_requires_enough_patches():
    geometry = np.eye(5, dtype=np.float64)
    response = np.eye(5, dtype=np.float64)
    centers = np.stack([np.arange(5), np.zeros(5)], axis=1).astype(np.float64)
    mask = np.array([True, True, True, False, False])

    row = analyze_content_stratum(
        geometry,
        response,
        centers,
        mask,
        k=2,
        min_spatial_distance=1.0,
        n_shuffles=5,
        seed=0,
        min_patches=4,
    )

    assert row["status"] == "insufficient_patches"
    assert row["n_patches"] == 3.0
