"""Tests for non-invasive trajectory schema validation utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trajectory_schema import (
    CONDITIONAL_REQUIRED_KEYS,
    SIREN_REQUIRED_KEYS,
    SUMMARY_COMMON_KEYS,
    SUMMARY_CONDITIONAL_KEYS,
    validate_summary_schema,
    validate_trajectory_schema,
    validate_trajectory_shapes,
)


def test_validate_siren_schema_accepts_required_keys():
    result = validate_trajectory_schema(SIREN_REQUIRED_KEYS, model_family="siren")
    assert result.ok
    assert result.missing_keys == ()


def test_validate_siren_schema_reports_missing_key():
    keys = set(SIREN_REQUIRED_KEYS)
    keys.remove("full_snapshots")
    result = validate_trajectory_schema(keys, model_family="siren")
    assert not result.ok
    assert result.missing_keys == ("full_snapshots",)


def test_validate_conditional_schema_accepts_required_keys():
    result = validate_trajectory_schema(
        CONDITIONAL_REQUIRED_KEYS,
        model_family="conditional",
    )
    assert result.ok
    assert result.missing_keys == ()


def test_validate_conditional_schema_reports_missing_encoder_snapshot():
    keys = set(CONDITIONAL_REQUIRED_KEYS)
    keys.remove("enc_snapshots")
    result = validate_trajectory_schema(keys, model_family="liif")
    assert not result.ok
    assert result.missing_keys == ("enc_snapshots",)


def test_validate_schema_rejects_unknown_family():
    with pytest.raises(ValueError):
        validate_trajectory_schema(SIREN_REQUIRED_KEYS, model_family="unknown")


def test_validate_summary_schema_common():
    result = validate_summary_schema(SUMMARY_COMMON_KEYS)
    assert result.ok


def test_validate_summary_schema_conditional_requires_sr_fields():
    common_only = set(SUMMARY_COMMON_KEYS)
    result = validate_summary_schema(common_only, conditional=True)
    assert not result.ok
    assert set(result.missing_keys) == set(SUMMARY_CONDITIONAL_KEYS)


def test_validate_disallow_extra_keys():
    keys = set(SIREN_REQUIRED_KEYS)
    keys.add("unexpected_debug_key")
    result = validate_trajectory_schema(
        keys,
        model_family="siren",
        allow_extra=False,
    )
    assert not result.ok
    assert result.unexpected_keys == ("unexpected_debug_key",)


def _fake_siren_trajectory(n_snapshots=4, n_params=8):
    return {
        "full_snapshots": np.zeros((n_snapshots, n_params), dtype=np.float32),
        "full_snapshots_aligned": np.zeros((n_snapshots, n_params), dtype=np.float32),
        "snapshot_steps": np.arange(n_snapshots),
        "losses": np.zeros(10, dtype=np.float32),
        "psnrs": np.zeros(n_snapshots - 1, dtype=np.float32),
        "freq_ratios": np.zeros(n_snapshots - 1, dtype=np.float32),
        "target_spectrum": np.zeros(8, dtype=np.float32),
        "grad_norms": np.zeros(n_snapshots - 1, dtype=np.float32),
    }


def _fake_conditional_trajectory(n_snapshots=5):
    return {
        "full_snapshots": np.zeros((n_snapshots, 12), dtype=np.float32),
        "enc_snapshots": np.zeros((n_snapshots, 7), dtype=np.float32),
        "dec_snapshots": np.zeros((n_snapshots, 5), dtype=np.float32),
        "dec_snapshots_aligned": np.zeros((n_snapshots, 5), dtype=np.float32),
        "snapshot_steps": np.arange(n_snapshots),
        "losses": np.zeros(20, dtype=np.float32),
        "psnrs": np.zeros(n_snapshots - 1, dtype=np.float32),
        "freq_ratios": np.zeros(n_snapshots - 1, dtype=np.float32),
        "target_spectrum": np.zeros(8, dtype=np.float32),
    }


def test_validate_siren_trajectory_shapes_accepts_existing_metric_convention():
    result = validate_trajectory_shapes(_fake_siren_trajectory(), model_family="siren")
    assert result.ok
    assert result.errors == ()


def test_validate_conditional_trajectory_shapes_accepts_existing_metric_convention():
    result = validate_trajectory_shapes(
        _fake_conditional_trajectory(),
        model_family="conditional",
    )
    assert result.ok
    assert result.errors == ()


def test_validate_trajectory_shapes_detects_snapshot_length_mismatch():
    trajectory = _fake_siren_trajectory(n_snapshots=4)
    trajectory["full_snapshots"] = np.zeros((3, 8), dtype=np.float32)
    result = validate_trajectory_shapes(trajectory, model_family="siren")
    assert not result.ok
    assert any("full_snapshots first dimension" in err for err in result.errors)


def test_validate_trajectory_shapes_detects_aligned_shape_mismatch():
    trajectory = _fake_conditional_trajectory(n_snapshots=5)
    trajectory["dec_snapshots_aligned"] = np.zeros((5, 6), dtype=np.float32)
    result = validate_trajectory_shapes(trajectory, model_family="conditional")
    assert not result.ok
    assert any("dec_snapshots_aligned shape" in err for err in result.errors)


def test_validate_trajectory_shapes_detects_metric_length_mismatch():
    trajectory = _fake_siren_trajectory(n_snapshots=4)
    trajectory["psnrs"] = np.zeros(99, dtype=np.float32)
    result = validate_trajectory_shapes(trajectory, model_family="siren")
    assert not result.ok
    assert any("psnrs length" in err for err in result.errors)
