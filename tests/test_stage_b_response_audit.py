import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_b_response_audit import (
    infer_family,
    response_audit_metrics,
    state_with_parts,
)


def test_infer_family_prefers_siren_model_type():
    assert infer_family({"model_type": "siren"}, {"enc_snapshots", "dec_snapshots"}) == "siren"


def test_infer_family_detects_conditional_keys():
    assert infer_family({}, {"full_snapshots", "enc_snapshots", "dec_snapshots"}) == "conditional"


def test_response_audit_metrics_for_straight_response_line():
    responses = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    row = response_audit_metrics(responses, snapshot_steps=np.arange(4), n_controls=0, seed=0)

    assert row["endpoint_line_rms_rel"] == 0.0
    assert row["straightness"] == 1.0
    assert row["orthogonal_energy_fraction"] == 0.0
    assert np.isclose(row["response_update_effective_rank"], 1.0)


def test_response_audit_metrics_detects_curved_response_path():
    responses = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )

    row = response_audit_metrics(responses, snapshot_steps=np.arange(4), n_controls=4, seed=0)

    assert row["endpoint_line_rms_rel"] > 0.0
    assert row["straightness"] < 1.0
    assert row["orthogonal_energy_fraction"] > 0.0
    assert "rw_endpoint_line_rms_rel_mean" in row


def test_state_with_parts_copies_only_matching_prefixes():
    base = {
        "encoder.a": torch.tensor([1.0]),
        "decoder.b": torch.tensor([2.0]),
        "other": torch.tensor([3.0]),
    }
    donor = {
        "encoder.a": torch.tensor([10.0]),
        "decoder.b": torch.tensor([20.0]),
        "other": torch.tensor([30.0]),
    }

    out = state_with_parts(base, donor, ("decoder.",))

    assert torch.equal(out["encoder.a"], torch.tensor([1.0]))
    assert torch.equal(out["decoder.b"], torch.tensor([20.0]))
    assert torch.equal(out["other"], torch.tensor([3.0]))
