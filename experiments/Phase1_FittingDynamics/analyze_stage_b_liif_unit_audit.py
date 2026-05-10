#!/usr/bin/env python3
"""Read-only Stage B LIIF internal-unit trajectory audit.

This script is upstream of Stage C geometry matching. It asks whether a more
LIIF-specific function-space unit has cleaner dynamics than raw output
responses:

- encoder_feature_lr: LR encoder feature field after `gen_feat`;
- decoder_input: query-conditioned local-ensemble decoder inputs;
- local_decoder_output: per-candidate decoder outputs before local-ensemble sum;
- sampled_output: final aggregated output over the same sampled query set.

No patch geometry, nearest-neighbor matching, or response-label oracle is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_b_response_audit import (
    load_summary,
    response_audit_metrics,
)
from experiments.Phase1_FittingDynamics.analyze_stage_c_geometry_response import (
    build_liif_model,
    resolve_image_path_from_summary,
    unflatten_full_state,
)
from experiments.Phase1_FittingDynamics.run import load_image
from src.datasets import get_image_coordinates
from src.models.liif import LIIFModel, make_coord


SKIP_DIRS = {"logs", "viz"}


def deterministic_query_indices(n_queries: int, max_queries: int | None) -> np.ndarray:
    """Return deterministic query indices spread over the flattened grid."""
    if max_queries is None or max_queries <= 0 or max_queries >= n_queries:
        return np.arange(n_queries, dtype=np.int64)
    return np.linspace(0, n_queries - 1, max_queries, dtype=np.int64)


def normalize_liif_input(lr_tensor: torch.Tensor) -> torch.Tensor:
    """Return LIIF encoder input with the model.forward normalization applied."""
    batched = lr_tensor.unsqueeze(0) if lr_tensor.dim() == 3 else lr_tensor
    return (batched - 0.5) / 0.5


def local_ensemble_decoder_objects(
    model: LIIFModel,
    coords: torch.Tensor,
    cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return decoder inputs, local decoder outputs, and aggregate output.

    The implementation mirrors `LIIFModel.query_rgb`, but exposes the internal
    objects needed for trajectory auditing. Returned values are in query order:

    - decoder_inputs: [K, N, D]
    - local_outputs: [K, N, 3] in [0, 1]
    - aggregate_output: [N, 3] in [0, 1]
    """
    feat = model.feat
    batch, channels, height, width = feat.shape
    if batch != 1:
        raise ValueError("this audit expects a single-image LIIF batch")

    if model.feat_unfold:
        feat = F.unfold(feat, 3, padding=1).view(batch, channels * 9, height, width)

    if model.local_ensemble:
        vx_lst, vy_lst = [-1, 1], [-1, 1]
        eps_shift = 1e-6
    else:
        vx_lst, vy_lst, eps_shift = [0], [0], 0.0

    rx = 2 / height / 2
    ry = 2 / width / 2
    feat_coord = (
        make_coord((height, width), flatten=False)
        .to(feat.device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .expand(batch, 2, height, width)
    )

    coord_batched = coords.unsqueeze(0)
    cell_batched = cell.unsqueeze(0)
    decoder_inputs: list[torch.Tensor] = []
    local_outputs: list[torch.Tensor] = []
    raw_outputs: list[torch.Tensor] = []
    areas: list[torch.Tensor] = []

    for vx in vx_lst:
        for vy in vy_lst:
            coord_ = coord_batched.clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_ = coord_.clamp(-1 + 1e-6, 1 - 1e-6)

            q_feat = F.grid_sample(
                feat,
                coord_.flip(-1).unsqueeze(1),
                mode="nearest",
                align_corners=False,
            )[:, :, 0, :].permute(0, 2, 1)

            q_coord = F.grid_sample(
                feat_coord,
                coord_.flip(-1).unsqueeze(1),
                mode="nearest",
                align_corners=False,
            )[:, :, 0, :].permute(0, 2, 1)

            rel_coord = coord_batched - q_coord
            rel_coord[:, :, 0] *= height
            rel_coord[:, :, 1] *= width
            inp = torch.cat([q_feat, rel_coord], dim=-1)

            if model.cell_decode and cell is not None:
                rel_cell = cell_batched.clone()
                rel_cell[:, :, 0] *= height
                rel_cell[:, :, 1] *= width
                inp = torch.cat([inp, rel_cell], dim=-1)

            pred = model.decoder(inp.reshape(inp.shape[1], -1)).reshape(1, inp.shape[1], -1)
            decoder_inputs.append(inp.squeeze(0))
            raw_outputs.append(pred.squeeze(0))
            local_outputs.append((pred.squeeze(0) * 0.5 + 0.5).clamp(0, 1))

            area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
            areas.append(area.squeeze(0) + 1e-9)

    tot_area = torch.stack(areas).sum(dim=0)
    if model.local_ensemble:
        areas[0], areas[3] = areas[3], areas[0]
        areas[1], areas[2] = areas[2], areas[1]

    aggregate_raw = torch.zeros_like(raw_outputs[0])
    for pred_raw, area in zip(raw_outputs, areas):
        aggregate_raw += pred_raw * (area / tot_area).unsqueeze(-1)

    aggregate = (aggregate_raw * 0.5 + 0.5).clamp(0, 1)
    return torch.stack(decoder_inputs), torch.stack(local_outputs), aggregate


def evaluate_liif_units(
    result_dir: Path,
    data_root: Path,
    device: torch.device,
    max_queries: int | None,
) -> dict[str, np.ndarray]:
    """Evaluate internal-unit trajectories for one LIIF run."""
    summary = load_summary(result_dir)
    trajectory = np.load(result_dir / "trajectory.npz", allow_pickle=True)
    full_snapshots = np.asarray(trajectory["full_snapshots"], dtype=np.float32)
    if str(summary.get("model_type", "")).lower() != "liif":
        raise ValueError(f"{result_dir.name} is not a LIIF run")

    image_size = int(summary.get("hr_size", 48))
    sr_scale = int(summary.get("sr_scale", 4))
    image_path = resolve_image_path_from_summary(summary, data_root)
    lr_tensor, hr_tensor = load_image(image_path, image_size=image_size, sr_scale=sr_scale)
    lr_tensor = lr_tensor.to(device)
    hr_tensor = hr_tensor.to(device)
    _, height, width = hr_tensor.shape
    coords_all = get_image_coordinates(height, width, normalize="center", device=device).reshape(-1, 2)
    indices = deterministic_query_indices(coords_all.shape[0], max_queries)
    coords = coords_all[torch.from_numpy(indices).to(device=device)]
    cell = torch.tensor(
        [2 / height, 2 / width],
        device=device,
        dtype=torch.float32,
    ).view(1, 2).expand(coords.shape[0], 2)

    model = build_liif_model(summary, full_snapshots.shape[1], device)
    model.eval()

    encoder_feature_lr: list[np.ndarray] = []
    decoder_input: list[np.ndarray] = []
    local_decoder_output: list[np.ndarray] = []
    sampled_output: list[np.ndarray] = []
    with torch.no_grad():
        for flat in full_snapshots:
            model.set_params(unflatten_full_state(model, flat, device))
            model.gen_feat(normalize_liif_input(lr_tensor))
            decoder_inputs, local_outputs, aggregate = local_ensemble_decoder_objects(
                model,
                coords,
                cell,
            )
            encoder_feature_lr.append(model.feat.detach().cpu().numpy().reshape(-1))
            decoder_input.append(decoder_inputs.detach().cpu().numpy().reshape(-1))
            local_decoder_output.append(local_outputs.detach().cpu().numpy().reshape(-1))
            sampled_output.append(aggregate.detach().cpu().numpy().reshape(-1))

    return {
        "encoder_feature_lr": np.asarray(encoder_feature_lr, dtype=np.float64),
        "decoder_input": np.asarray(decoder_input, dtype=np.float64),
        "local_decoder_output": np.asarray(local_decoder_output, dtype=np.float64),
        "sampled_output": np.asarray(sampled_output, dtype=np.float64),
    }


def analyze_result_dir(
    result_dir: Path,
    data_root: Path,
    device: torch.device,
    n_controls: int,
    seed: int,
    max_queries: int | None,
) -> list[dict[str, Any]]:
    """Analyze all internal units for one LIIF result directory."""
    summary = load_summary(result_dir)
    if str(summary.get("model_type", "")).lower() != "liif":
        return []
    trajectory = np.load(result_dir / "trajectory.npz", allow_pickle=True)
    snapshot_steps = (
        np.asarray(trajectory["snapshot_steps"], dtype=np.float64)
        if "snapshot_steps" in trajectory.files
        else None
    )
    unit_trajectories = evaluate_liif_units(result_dir, data_root, device, max_queries)

    rows: list[dict[str, Any]] = []
    for unit_name, responses in unit_trajectories.items():
        row: dict[str, Any] = {
            "run": result_dir.name,
            "status": "ok",
            "unit": unit_name,
            "image": summary.get("image", "?"),
            "seed": summary.get("seed", "?"),
            "n_snapshots": int(responses.shape[0]),
            "unit_dim": int(responses.shape[1]),
            "max_queries": int(max_queries or 0),
        }
        row.update(
            response_audit_metrics(
                responses,
                snapshot_steps=snapshot_steps,
                n_controls=n_controls,
                seed=seed,
            )
        )
        rows.append(row)
    return rows


def scan_results(
    results_dir: Path,
    data_root: Path,
    device: torch.device,
    n_controls: int,
    seed: int,
    max_queries: int | None,
) -> list[dict[str, Any]]:
    """Analyze LIIF result directories under results_dir."""
    rows: list[dict[str, Any]] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        rows.extend(
            analyze_result_dir(
                child,
                data_root=data_root,
                device=device,
                n_controls=n_controls,
                seed=seed,
                max_queries=max_queries,
            )
        )
    return rows


def format_float(value: Any, digits: int = 3) -> str:
    """Format floats for compact tables."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print compact terminal table."""
    columns = [
        ("run", "run"),
        ("unit", "unit"),
        ("pc1", "response_snapshot_pc1_pct"),
        ("line", "endpoint_line_rms_rel"),
        ("perm", "permuted_endpoint_line_rms_rel_mean"),
        ("rw", "rw_endpoint_line_rms_rel_mean"),
        ("straight", "straightness"),
        ("orthE", "orthogonal_energy_fraction"),
        ("erank", "response_update_effective_rank"),
        ("dim", "unit_dim"),
    ]
    widths = {label: max(8, len(label)) for label, _ in columns}
    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        formatted: dict[str, str] = {}
        for label, key in columns:
            value = row.get(key, "")
            text = str(value) if key in {"run", "unit"} else format_float(value)
            formatted[label] = text
            widths[label] = max(widths[label], len(text))
        formatted_rows.append(formatted)
    print(" ".join(label.ljust(widths[label]) for label, _ in columns))
    print(" ".join("-" * widths[label] for label, _ in columns))
    for formatted in formatted_rows:
        print(" ".join(formatted[label].ljust(widths[label]) for label, _ in columns))


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    """Write JSON or CSV based on suffix."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        keys = sorted({key for row in rows for key in row.keys()})
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with output.open("w") as f:
            json.dump(rows, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Stage B LIIF internal-unit trajectory audit"
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results/FittingDynamics_StageB"))
    parser.add_argument("--data_root", type=Path, default=Path("Data"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_controls", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_queries", type=int, default=1024)
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = scan_results(
        args.results_dir,
        data_root=args.data_root,
        device=torch.device(args.device),
        n_controls=args.n_controls,
        seed=args.seed,
        max_queries=args.max_queries,
    )
    if args.output is not None:
        write_rows(rows, args.output)

    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.format == "csv":
        keys = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(sys.stdout, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
