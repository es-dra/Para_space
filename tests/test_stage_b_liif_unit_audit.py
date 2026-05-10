import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_b_liif_unit_audit import (
    deterministic_query_indices,
    local_ensemble_decoder_objects,
    normalize_liif_input,
)
from src.models.liif import LIIFModel


def test_deterministic_query_indices_uses_full_grid_when_under_limit():
    idx = deterministic_query_indices(5, 10)
    np.testing.assert_array_equal(idx, np.arange(5))


def test_deterministic_query_indices_spreads_subset():
    idx = deterministic_query_indices(10, 4)
    np.testing.assert_array_equal(idx, np.array([0, 3, 6, 9]))


def test_normalize_liif_input_matches_forward_convention():
    lr = torch.tensor([[[0.0, 0.5], [1.0, 0.25]]], dtype=torch.float32)

    normalized = normalize_liif_input(lr)

    assert normalized.shape == (1, 1, 2, 2)
    assert torch.allclose(normalized, (lr.unsqueeze(0) - 0.5) / 0.5)


def test_local_ensemble_decoder_objects_shapes():
    model = LIIFModel(
        n_feats=4,
        n_resblocks=1,
        decoder_hidden=8,
        decoder_layers=1,
        local_ensemble=True,
        feat_unfold=True,
        cell_decode=True,
    )
    image = torch.rand(1, 3, 4, 4)
    coords = torch.tensor(
        [
            [-0.5, -0.5],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    cell = torch.full((2, 2), 0.25, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        model.gen_feat(image)
        decoder_inputs, local_outputs, aggregate = local_ensemble_decoder_objects(
            model,
            coords,
            cell,
        )

    assert decoder_inputs.shape == (4, 2, 40)
    assert local_outputs.shape == (4, 2, 3)
    assert aggregate.shape == (2, 3)
    assert torch.all(aggregate >= 0.0)
    assert torch.all(aggregate <= 1.0)
