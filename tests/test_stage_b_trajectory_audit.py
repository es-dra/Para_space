import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_b_trajectory_audit import (
    analyze_result_dir,
    analyze_space,
    endpoint_line_metrics,
    slice_update_energy,
    spectrum_metrics,
    update_decomposition_metrics,
)
from experiments.Phase1_FittingDynamics.analyze_stage_b_trajectory_audit import (
    AnalysisSpace,
)


def test_endpoint_line_metrics_are_zero_for_straight_line():
    snapshots = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )

    metrics = endpoint_line_metrics(snapshots)
    updates = update_decomposition_metrics(snapshots)

    assert metrics["endpoint_line_rms_rel"] == 0.0
    assert metrics["endpoint_line_max_rel"] == 0.0
    assert updates["straightness"] == 1.0
    assert updates["orthogonal_energy_fraction"] == 0.0


def test_endpoint_line_metrics_detect_curved_path():
    snapshots = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )

    metrics = endpoint_line_metrics(snapshots)
    updates = update_decomposition_metrics(snapshots)

    assert metrics["endpoint_line_rms_rel"] > 0.0
    assert metrics["endpoint_line_max_rel"] > 0.0
    assert updates["orthogonal_energy_fraction"] > 0.0
    assert updates["straightness"] < 1.0


def test_spectrum_metrics_effective_rank_for_two_directions():
    updates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    metrics = spectrum_metrics(updates, centered=False)

    assert np.isclose(metrics["pc1_pct"], 50.0)
    assert np.isclose(metrics["effective_rank"], 2.0)
    assert np.isclose(metrics["participation_rank"], 2.0)
    assert metrics["rank90"] == 2.0


def test_slice_update_energy_identifies_top_slice():
    snapshots = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    slices = (("a", 0, 1), ("b", 1, 3))

    metrics = slice_update_energy(snapshots, slices)

    assert metrics["top_path_energy_name"] == "a"
    assert metrics["top_final_energy_name"] == "a"
    assert metrics["top_path_energy_fraction"] > 0.5


def test_analyze_space_reports_expected_keys():
    snapshots = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [2.0, 0.1],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )
    space = AnalysisSpace("synthetic", "snapshots", snapshots, (("all", 0, 2),))

    row = analyze_space(space, snapshot_steps=np.arange(4), n_controls=4, seed=7)

    assert row["space"] == "synthetic"
    assert row["n_snapshots"] == 4
    assert row["n_params"] == 2
    assert "endpoint_line_rms_rel" in row
    assert "update_effective_rank" in row
    assert "rw_endpoint_line_rms_rel_mean" in row


def test_analyze_result_dir_returns_all_conditional_spaces(tmp_path):
    result_dir = tmp_path / "LIIF_synthetic"
    result_dir.mkdir()
    n_snapshots = 4

    np.savez(
        result_dir / "trajectory.npz",
        full_snapshots=np.zeros((n_snapshots, 138531), dtype=np.float32),
        enc_snapshots=np.zeros((n_snapshots, 84128), dtype=np.float32),
        dec_snapshots=np.zeros((n_snapshots, 54403), dtype=np.float32),
        dec_snapshots_aligned=np.zeros((n_snapshots, 54403), dtype=np.float32),
        snapshot_steps=np.arange(n_snapshots),
    )
    (result_dir / "dynamics_summary.json").write_text(
        json.dumps(
            {
                "model_type": "liif",
                "image": "synthetic.png",
                "seed": 42,
                "n_params": 138531,
                "n_encoder_params": 84128,
                "n_decoder_params": 54403,
                "total_steps": 10,
                "n_snapshots": n_snapshots,
                "final_psnr": 10.0,
                "snapshot_steps": list(range(n_snapshots)),
                "model_config_name": "LIIF_CONFIG_REDUCED",
            }
        )
    )

    rows = analyze_result_dir(result_dir, n_controls=0, seed=1)

    assert [row["space"] for row in rows] == [
        "full",
        "encoder",
        "decoder_raw",
        "decoder_aligned",
    ]
