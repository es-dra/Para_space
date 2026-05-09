"""Tests for the read-only fitting-dynamics output checker."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.check_outputs import validate_result_dir


def _write_siren_output(result_dir: Path, n_snapshots: int = 4):
    result_dir.mkdir()
    np.savez(
        result_dir / "trajectory.npz",
        full_snapshots=np.zeros((n_snapshots, 8), dtype=np.float32),
        full_snapshots_aligned=np.zeros((n_snapshots, 8), dtype=np.float32),
        snapshot_steps=np.arange(n_snapshots),
        losses=np.arange(10, dtype=np.float32),
        psnrs=np.arange(n_snapshots - 1, dtype=np.float32),
        freq_ratios=np.zeros(n_snapshots - 1, dtype=np.float32),
        target_spectrum=np.zeros(8, dtype=np.float32),
        grad_norms=np.zeros(n_snapshots - 1, dtype=np.float32),
    )
    with (result_dir / "dynamics_summary.json").open("w") as f:
        json.dump(
            {
                "model_type": "siren",
                "image": "synthetic.png",
                "seed": 42,
                "n_params": 8,
                "total_steps": 10,
                "n_snapshots": n_snapshots,
                "final_psnr": float(n_snapshots - 2),
                "final_loss": 9.0,
                "snapshot_steps": list(range(n_snapshots)),
            },
            f,
        )


def test_validate_result_dir_accepts_synthetic_siren_output(tmp_path):
    result_dir = tmp_path / "SIREN_smoke"
    _write_siren_output(result_dir)

    ok, messages = validate_result_dir(result_dir)
    assert ok
    assert any("trajectory schema: OK" in msg for msg in messages)
    assert any("trajectory shapes: OK" in msg for msg in messages)
    assert any("summary schema: OK" in msg for msg in messages)
    assert any("summary consistency: OK" in msg for msg in messages)


def test_validate_result_dir_reports_missing_trajectory(tmp_path):
    result_dir = tmp_path / "missing"
    result_dir.mkdir()
    ok, messages = validate_result_dir(result_dir)
    assert not ok
    assert any("缺少 trajectory.npz" in msg for msg in messages)


def test_validate_result_dir_detects_bad_shape(tmp_path):
    result_dir = tmp_path / "bad_shape"
    result_dir.mkdir()
    np.savez(
        result_dir / "trajectory.npz",
        full_snapshots=np.zeros((3, 8), dtype=np.float32),
        snapshot_steps=np.arange(4),
        losses=np.zeros(10, dtype=np.float32),
        psnrs=np.zeros(3, dtype=np.float32),
        freq_ratios=np.zeros(3, dtype=np.float32),
        target_spectrum=np.zeros(8, dtype=np.float32),
    )
    ok, messages = validate_result_dir(result_dir, model_family="siren", require_summary=False)
    assert not ok
    assert any("trajectory shapes: FAIL" in msg for msg in messages)


def test_validate_result_dir_requires_summary_by_default(tmp_path):
    result_dir = tmp_path / "no_summary"
    result_dir.mkdir()
    np.savez(
        result_dir / "trajectory.npz",
        full_snapshots=np.zeros((4, 8), dtype=np.float32),
        snapshot_steps=np.arange(4),
        losses=np.zeros(10, dtype=np.float32),
        psnrs=np.zeros(3, dtype=np.float32),
        freq_ratios=np.zeros(3, dtype=np.float32),
        target_spectrum=np.zeros(8, dtype=np.float32),
    )
    ok, messages = validate_result_dir(result_dir)
    assert not ok
    assert any("缺少 dynamics_summary.json" in msg for msg in messages)


def test_validate_result_dir_detects_summary_n_snapshot_mismatch(tmp_path):
    result_dir = tmp_path / "bad_summary"
    _write_siren_output(result_dir)
    summary_path = result_dir / "dynamics_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["n_snapshots"] = 999
    summary_path.write_text(json.dumps(summary))

    ok, messages = validate_result_dir(result_dir)
    assert not ok
    assert any("n_snapshots" in msg for msg in messages)


def test_validate_result_dir_detects_summary_final_metric_mismatch(tmp_path):
    result_dir = tmp_path / "bad_final_metric"
    _write_siren_output(result_dir)
    summary_path = result_dir / "dynamics_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["final_psnr"] = 123.0
    summary_path.write_text(json.dumps(summary))

    ok, messages = validate_result_dir(result_dir)
    assert not ok
    assert any("final_psnr" in msg for msg in messages)


def test_validate_result_dir_accepts_finetune_style_rounded_final_psnr(tmp_path):
    result_dir = tmp_path / "rounded_final_psnr"
    _write_siren_output(result_dir)
    summary_path = result_dir / "dynamics_summary.json"
    summary = json.loads(summary_path.read_text())
    # run_finetune.py rounds final_psnr to two decimals in the summary.
    summary["final_psnr"] = summary["final_psnr"] + 0.004
    summary_path.write_text(json.dumps(summary))

    ok, messages = validate_result_dir(result_dir)
    assert ok
    assert any("summary consistency: OK" in msg for msg in messages)


def test_validate_result_dir_warns_when_too_few_snapshots_for_analysis(tmp_path):
    result_dir = tmp_path / "two_snapshots"
    _write_siren_output(result_dir, n_snapshots=2)

    ok, messages = validate_result_dir(result_dir)
    assert ok
    assert any("fewer than 3 snapshots" in msg for msg in messages)
