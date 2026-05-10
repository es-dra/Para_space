import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_c_lr_cell_feature_trajectory import (
    blocked_spatial_cv_r2,
    cell_trajectory_metric_row,
    cell_trajectory_targets,
    coordinate_features,
    feature_maps_to_cell_trajectories,
    fit_ridge_predict,
    gen_encoder_feature,
    local_content_geometry_features,
    normalize_liif_input,
    spatial_block_labels,
    structure_tensor_orientation_features,
)


class CaptureGenFeat:
    def __init__(self):
        self.seen = None

    def gen_feat(self, inp):
        self.seen = inp.detach().clone()
        return torch.ones(1, 32, 2, 2)


def test_gen_feat_uses_explicit_liif_normalization_path():
    lr = torch.tensor(
        [
            [[0.0, 0.5], [1.0, 0.25]],
            [[0.25, 0.5], [0.75, 1.0]],
            [[1.0, 0.0], [0.5, 0.25]],
        ],
        dtype=torch.float32,
    )
    model = CaptureGenFeat()

    feat = gen_encoder_feature(model, lr)

    assert feat.shape == (1, 32, 2, 2)
    assert torch.allclose(model.seen, (lr.unsqueeze(0) - 0.5) / 0.5)
    assert torch.allclose(normalize_liif_input(lr), model.seen)


def test_feature_maps_to_cell_trajectories_shape_and_cell_order():
    feature_maps = np.arange(3 * 2 * 4 * 5, dtype=np.float64).reshape(3, 2, 4, 5)

    cells = feature_maps_to_cell_trajectories(feature_maps)

    assert cells.shape == (2, 4, 3, 5)
    np.testing.assert_array_equal(cells[1, 2], feature_maps[:, 1, 2, :])


def test_cell_trajectory_metrics_distinguish_line_from_bend():
    line = np.stack(
        [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([3.0, 0.0]),
        ]
    )
    bend = np.stack(
        [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([2.0, 1.0]),
        ]
    )

    line_metrics = cell_trajectory_metric_row(line)
    bend_metrics = cell_trajectory_metric_row(bend)

    assert line_metrics["orthogonal_energy_fraction"] == pytest.approx(0.0)
    assert line_metrics["straightness"] == pytest.approx(1.0)
    assert bend_metrics["orthogonal_energy_fraction"] > line_metrics["orthogonal_energy_fraction"]
    assert bend_metrics["straightness"] < line_metrics["straightness"]


def test_cell_trajectory_targets_returns_one_value_per_cell():
    feature_maps = np.zeros((4, 2, 3, 2), dtype=np.float64)
    for t in range(4):
        feature_maps[t, :, :, 0] = t
        feature_maps[t, :, :, 1] = np.arange(6).reshape(2, 3)

    targets = cell_trajectory_targets(feature_maps_to_cell_trajectories(feature_maps))

    assert targets["orthogonal_energy_fraction"].shape == (6,)
    assert targets["straightness"].shape == (6,)
    assert np.all(np.isfinite(targets["path_length"]))


def test_structure_tensor_orientation_features_encode_dominant_axis():
    gx = np.ones((3, 3), dtype=np.float64)
    gy = np.zeros((3, 3), dtype=np.float64)

    feat_x = structure_tensor_orientation_features(gx, gy)
    feat_y = structure_tensor_orientation_features(gy, gx)

    assert feat_x[0] > 0.99
    assert feat_x[1] > 0.99
    assert abs(feat_x[2]) < 1e-9
    assert feat_y[0] > 0.99
    assert feat_y[1] < -0.99
    assert abs(feat_y[2]) < 1e-9


def test_local_content_geometry_features_shapes_and_gradient_signal():
    lr = np.zeros((4, 4, 3), dtype=np.float64)
    lr[..., 0] = np.linspace(0.0, 1.0, 4)[None, :]

    content, geometry = local_content_geometry_features(lr, support_size=3)

    assert content.shape == (16, 3)
    assert geometry.shape == (16, 3)
    assert np.max(content[:, 2]) > 0.0
    assert np.max(np.abs(geometry[:, 1])) > 0.0


def test_blocked_spatial_cv_regression_increment_detects_geometry_signal():
    height = 4
    width = 4
    coord = coordinate_features(height, width)
    content = np.zeros((height * width, 1), dtype=np.float64)
    geometry = np.tile(np.array([[1.0], [-1.0], [1.0], [-1.0]]), (height, 1)).reshape(-1, 1)
    target = geometry[:, 0]
    block_labels = spatial_block_labels(height, width, block_size=2)

    base = np.concatenate([coord, content], axis=1)
    full = np.concatenate([coord, content, geometry], axis=1)
    r2_base, n_folds = blocked_spatial_cv_r2(base, target, block_labels, alpha=1e-6)
    r2_full, _ = blocked_spatial_cv_r2(full, target, block_labels, alpha=1e-6)

    assert n_folds == 4
    assert r2_full - r2_base > 0.5


def test_fit_ridge_predict_handles_constant_columns():
    x_train = np.ones((4, 2), dtype=np.float64)
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    x_test = np.ones((2, 2), dtype=np.float64)

    pred = fit_ridge_predict(x_train, y_train, x_test, alpha=1e-6)

    np.testing.assert_allclose(pred, np.array([2.5, 2.5]))
