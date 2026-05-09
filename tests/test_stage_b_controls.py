"""Tests for read-only Stage-B trajectory control analysis."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_b_controls import (
    analyze_result_dir,
    analyze_snapshots,
    make_norm_matched_random_walk,
    make_permuted_update_trajectory,
    pc1_percent,
    step_norms,
    trajectory_straightness,
)


def test_pc1_percent_is_one_hundred_for_straight_line():
    snapshots = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )

    assert pc1_percent(snapshots) == 100.0
    assert trajectory_straightness(snapshots) == 1.0


def test_norm_matched_random_walk_preserves_step_norm_schedule():
    norms = np.array([1.0, 2.0, 0.5])
    rng = np.random.default_rng(123)

    walk = make_norm_matched_random_walk(norms, n_params=16, rng=rng)

    np.testing.assert_allclose(step_norms(walk), norms, rtol=1e-12, atol=1e-12)
    assert walk.shape == (4, 16)


def test_permuted_update_trajectory_preserves_start_endpoint_and_step_norms():
    snapshots = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [4.0, 2.0],
            [4.0, 6.0],
        ]
    )
    rng = np.random.default_rng(5)

    permuted = make_permuted_update_trajectory(snapshots, rng)

    np.testing.assert_allclose(permuted[0], snapshots[0])
    np.testing.assert_allclose(permuted[-1], snapshots[-1])
    np.testing.assert_allclose(np.sort(step_norms(permuted)), np.sort(step_norms(snapshots)))


def test_analyze_snapshots_is_deterministic_for_fixed_seed():
    snapshots = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [2.1, 0.3, 0.0],
            [3.0, 0.4, 0.1],
            [4.2, 0.5, 0.1],
        ]
    )

    first = analyze_snapshots(snapshots, n_controls=8, seed=99)
    second = analyze_snapshots(snapshots, n_controls=8, seed=99)

    assert first == second
    assert first["n_snapshots"] == 5.0
    assert first["n_params"] == 3.0
    assert np.isfinite(first["rw_pc1_mean"])
    assert 0.0 <= first["rw_pc1_p_ge"] <= 1.0


def test_analyze_result_dir_uses_decoder_space_for_conditional_outputs(tmp_path):
    result_dir = tmp_path / "LIIF_synthetic"
    result_dir.mkdir()
    n_snapshots = 4

    np.savez(
        result_dir / "trajectory.npz",
        full_snapshots=np.zeros((n_snapshots, 8), dtype=np.float32),
        enc_snapshots=np.zeros((n_snapshots, 3), dtype=np.float32),
        dec_snapshots=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dec_snapshots_aligned=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float32,
        ),
        snapshot_steps=np.arange(n_snapshots),
        losses=np.zeros(10, dtype=np.float32),
        psnrs=np.zeros(n_snapshots - 1, dtype=np.float32),
        freq_ratios=np.zeros(n_snapshots - 1, dtype=np.float32),
        target_spectrum=np.zeros(8, dtype=np.float32),
    )
    (result_dir / "dynamics_summary.json").write_text(
        json.dumps(
            {
                "model_type": "liif",
                "image": "synthetic.png",
                "seed": 42,
                "n_params": 8,
                "total_steps": 10,
                "n_snapshots": n_snapshots,
                "final_psnr": 10.0,
                "final_loss": 0.1,
                "snapshot_steps": list(range(n_snapshots)),
                "model_config_name": "LIIF_CONFIG_REDUCED",
            }
        )
    )

    row = analyze_result_dir(result_dir, n_controls=4, seed=1)

    assert row["status"] == "ok"
    assert row["family"] == "conditional"
    assert row["primary_space"] == "decoder"
    assert row["raw_key"] == "dec_snapshots"
    assert row["aligned_key"] == "dec_snapshots_aligned"
    assert row["pc1_pct"] == 100.0
