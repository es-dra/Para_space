"""Tests for controlled Stage-C self-similarity sanity-gate helpers."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_c_controlled_self_similarity import (
    analyze_synthetic_smoke,
    gate_verdict,
    patch_group_labels,
)
from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import (
    make_patch_grid,
    resolve_image_path_from_summary,
)
from experiments.Phase1_FittingDynamics.generate_controlled_self_similarity_images import (
    generate_images,
)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    assert image is not None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def test_controlled_image_generator_writes_repeat_and_nonrepeat_tiles(tmp_path):
    output_root = tmp_path / "Controlled" / "HR"

    metadata = generate_images(output_root, seed=0, image_size=48, tile_size=12)

    periodic = _read_rgb(output_root / "css_periodic12.png")
    nonrepeat = _read_rgb(output_root / "css_nonrepeat12.png")
    assert sorted(metadata) == ["css_nonrepeat12.png", "css_periodic12.png"]
    assert periodic.shape == (48, 48, 3)
    assert nonrepeat.shape == (48, 48, 3)
    assert np.array_equal(periodic[:12, :12], periodic[12:24, 12:24])
    assert not np.array_equal(nonrepeat[:12, :12], nonrepeat[12:24, 12:24])
    assert metadata["css_periodic12.png"]["tile_size"] == 12


def test_patch_group_labels_follow_tile_phase():
    grid = make_patch_grid(48, 48, patch_size=7, stride=4)

    labels = patch_group_labels(grid.starts, tile_size=12)

    assert labels[0] == labels[3]
    assert labels[0] != labels[1]
    assert len(set(labels.tolist())) == 9


def test_synthetic_smoke_separates_periodic_from_nonrepeat(tmp_path):
    output_root = tmp_path / "Controlled" / "HR"
    metadata = generate_images(output_root, seed=0, image_size=48, tile_size=12)
    metadata_doc = {
        "output_root": str(output_root),
        "images": metadata,
    }

    rows = analyze_synthetic_smoke(
        metadata_doc,
        metadata_path=tmp_path / "manifest.json",
        geometry_source="lr_up",
        geometry_descriptor="rgb_grad",
        response_mode="trajectory_delta",
        patch_size=7,
        stride=4,
        k=5,
        min_spatial_distance=8.0,
        min_group_count=3,
        n_shuffles=32,
        seed=0,
    )

    periodic = next(row for row in rows if row["metadata_role"] == "positive_known_duplicate")
    nonrepeat = next(row for row in rows if row["metadata_role"] == "negative_nonperiodic_texture")
    assert periodic["duplicate_hit_at_5"] >= 0.95
    assert periodic["known_group_response_dist"] < 1e-7
    assert periodic["known_group_response_percentile_mean"] <= 0.05
    assert periodic["effect_vs_shuffle_frac"] > nonrepeat["effect_vs_shuffle_frac"]
    assert gate_verdict(rows)["verdict"] in {
        "pass_synthetic_response_smoke",
        "fail_synthetic_response_smoke",
    }


def test_resolve_image_path_prefers_recorded_paths(tmp_path):
    image_path = tmp_path / "controlled.png"
    image_path.write_bytes(b"not-an-image")

    resolved_abs = resolve_image_path_from_summary(
        {"image": "controlled.png", "image_path": str(image_path)},
        Path("Data"),
    )
    resolved_rel = resolve_image_path_from_summary(
        {"image": "controlled.png", "image_relpath": str(image_path)},
        Path("Data"),
    )

    assert resolved_abs == image_path
    assert resolved_rel == image_path


def test_resolve_image_path_falls_back_from_stale_absolute_to_relpath(tmp_path):
    data_root = tmp_path / "Data"
    image_path = data_root / "ControlledSelfSimilarity" / "HR" / "controlled.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not-an-image")

    resolved = resolve_image_path_from_summary(
        {
            "image": "controlled.png",
            "image_path": "/stale/workspace/Data/ControlledSelfSimilarity/HR/controlled.png",
            "image_relpath": "Data/ControlledSelfSimilarity/HR/controlled.png",
        },
        data_root,
    )

    assert resolved == image_path
