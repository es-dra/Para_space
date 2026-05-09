#!/usr/bin/env python3
"""Generate controlled self-similarity images for Stage-C sanity gates.

The generated images are diagnostic inputs, not benchmark data. They are used
to test whether the fitting-dynamics response probe can recover known repeated
local structure before interpreting natural-image failure cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def make_base_tile(tile_size: int, seed: int) -> np.ndarray:
    """Create one deterministic RGB tile with edges and textured structure."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, tile_size, endpoint=False),
        np.linspace(0.0, 1.0, tile_size, endpoint=False),
        indexing="ij",
    )
    tile = np.zeros((tile_size, tile_size, 3), dtype=np.float64)
    tile[..., 0] = 0.45 * xx + 0.25 * np.sin(2 * np.pi * (2.0 * yy + 0.25))
    tile[..., 1] = 0.45 * yy + 0.25 * np.cos(2 * np.pi * (2.0 * xx + 0.15))
    tile[..., 2] = 0.30 * (xx + yy) + 0.20 * np.sin(2 * np.pi * (xx + yy))

    tile[2:5, :, 0] += 0.45
    tile[:, 9:12, 1] += 0.40
    tile[8:14, 3:9, 2] += 0.35
    diag = np.abs(xx - yy) < (1.5 / tile_size)
    tile[diag, :] += np.array([0.20, -0.10, 0.25])
    tile += rng.normal(0.0, 0.035, size=tile.shape)
    return np.clip(_normalize01(tile), 0.0, 1.0)


def make_repeat_tile_image(tile_size: int, repeat: int, seed: int) -> np.ndarray:
    """Tile one base patch exactly across the image."""
    tile = make_base_tile(tile_size, seed)
    return np.tile(tile, (repeat, repeat, 1))


def make_nonperiodic_control_image(image_size: int, seed: int) -> np.ndarray:
    """Create a non-periodic texture control without tile-boundary phase cues."""
    rng = np.random.default_rng(seed + 1009)
    noise = rng.normal(0.0, 1.0, size=(image_size, image_size, 3))
    smooth = np.zeros_like(noise)
    for channel in range(3):
        smooth[..., channel] = cv2.GaussianBlur(noise[..., channel], (9, 9), 0)

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, image_size, endpoint=False),
        np.linspace(0.0, 1.0, image_size, endpoint=False),
        indexing="ij",
    )
    smooth[..., 0] += 0.25 * xx
    smooth[..., 1] += 0.25 * yy
    smooth[..., 2] += 0.15 * np.sin(2 * np.pi * (1.7 * xx + 0.6 * yy))
    return np.clip(_normalize01(smooth), 0.0, 1.0)


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB image in [0, 1] to PNG."""
    arr = np.clip(np.rint(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"failed to write {path}")


def build_metadata(
    *,
    image_name: str,
    role: str,
    image_size: int,
    tile_size: int,
    repeat: int,
    seed: int,
    description: str,
) -> dict[str, Any]:
    """Return metadata consumed by the controlled sanity-gate analyzer."""
    return {
        "image": image_name,
        "role": role,
        "image_size": image_size,
        "tile_size": tile_size,
        "repeat": repeat,
        "seed": seed,
        "known_grouping": "patch_start_mod_tile_size",
        "description": description,
    }


def generate_images(output_root: Path, seed: int, image_size: int, tile_size: int) -> dict[str, Any]:
    """Generate controlled images and return metadata keyed by image name."""
    if image_size % tile_size != 0:
        raise ValueError("image_size must be divisible by tile_size")
    repeat = image_size // tile_size
    specs = [
        (
            f"css_periodic{tile_size}.png",
            "positive_known_duplicate",
            make_repeat_tile_image(tile_size, repeat, seed),
            f"One {tile_size}x{tile_size} tile repeated exactly on a {repeat}x{repeat} grid.",
        ),
        (
            f"css_nonrepeat{tile_size}.png",
            "negative_nonperiodic_texture",
            make_nonperiodic_control_image(image_size, seed),
            "A non-periodic smooth random texture with no intentional tile-phase duplicates.",
        ),
    ]
    metadata: dict[str, Any] = {}
    for image_name, role, image, description in specs:
        write_rgb_png(output_root / image_name, image)
        metadata[image_name] = build_metadata(
            image_name=image_name,
            role=role,
            image_size=image_size,
            tile_size=tile_size,
            repeat=repeat,
            seed=seed,
            description=description,
        )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate controlled self-similarity images for Stage-C diagnostics."
    )
    parser.add_argument("--output_root", type=str, default="Data/ControlledSelfSimilarity/HR")
    parser.add_argument(
        "--metadata_json",
        type=str,
        default="Data/ControlledSelfSimilarity/controlled_self_similarity_metadata.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=48)
    parser.add_argument("--tile_size", type=int, default=12)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    metadata = generate_images(output_root, args.seed, args.image_size, args.tile_size)
    metadata_path = Path(args.metadata_json)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as f:
        json.dump(
            {
                "metadata_version": 1,
                "output_root": str(output_root),
                "images": metadata,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    print(f"Wrote {len(metadata)} controlled image(s) to {output_root}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
