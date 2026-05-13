#!/usr/bin/env python3
"""Read-only Stage B-response function-space trajectory audit.

This script audits saved Stage-B fitting trajectories in output-response space.
It is intentionally upstream of Stage C geometry matching: before asking
whether image geometry explains response changes, first check whether the
response trajectory object itself has non-trivial structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.Phase1_FittingDynamics.analyze_stage_b_trajectory_audit import (
    control_metrics,
    endpoint_line_metrics,
    spectrum_metrics,
    update_decomposition_metrics,
)
from experiments.Phase1_FittingDynamics.run import load_image as load_liif_image
from experiments.Phase1_FittingDynamics.run_siren import load_image as load_siren_image
from experiments.config import LIIF_CONFIG, LIIF_CONFIG_REDUCED, SIREN_CONFIG
from src.datasets import get_image_coordinates
from src.models.liif import LIIFModel
from src.siren import SIREN


SKIP_DIRS = {"logs", "viz"}


@dataclass(frozen=True)
class ResponseEvaluation:
    """Response vectors and auxiliary fields for one run."""

    responses: np.ndarray
    psnrs: np.ndarray
    target: np.ndarray
    hybrid: dict[str, float]


def load_summary(result_dir: Path) -> dict[str, Any]:
    """Load result summary."""
    for name in ("dynamics_summary.json", "summary.json"):
        path = result_dir / name
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
    raise FileNotFoundError(f"missing dynamics_summary.json/summary.json in {result_dir}")


def infer_family(summary: dict[str, Any], keys: set[str]) -> str:
    """Infer SIREN-like or conditional LIIF-like output."""
    model_type = str(summary.get("model_type", summary.get("model", ""))).lower()
    if model_type == "siren":
        return "siren"
    if {"enc_snapshots", "dec_snapshots"}.issubset(keys):
        return "conditional"
    if model_type in {"liif", "lte", "conditional"}:
        return "conditional"
    return "siren"


def compute_psnr_vector(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR for flattened RGB responses in [0, 1]."""
    mse = float(np.mean((pred - target) ** 2))
    return float(-10.0 * np.log10(mse + 1e-10))


def resolve_image_path_from_summary(summary: dict[str, Any], data_root: Path) -> Path:
    """Resolve the image path for a fitting-dynamics result."""
    candidates: list[Path] = []
    for key in ("image_path", "image_relpath"):
        recorded = summary.get(key)
        if recorded:
            path = Path(recorded)
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates.extend([path, data_root.parent / path, data_root / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = data_root / "Set5" / "HR" / str(summary["image"])
    if fallback.exists() or not candidates:
        return fallback
    return candidates[0]


def build_liif_model(summary: dict[str, Any], n_params: int, device: torch.device) -> LIIFModel:
    """Build a LIIF model whose parameter count matches the trajectory."""
    if str(summary.get("model_type", "")).lower() != "liif":
        raise ValueError("Stage B-response audit currently supports scratch LIIF outputs only")

    candidates = [
        ("LIIF_CONFIG_REDUCED", LIIF_CONFIG_REDUCED),
        ("LIIF_CONFIG", LIIF_CONFIG),
    ]
    preferred = summary.get("model_config_name")
    if preferred:
        candidates = sorted(candidates, key=lambda item: 0 if item[0] == preferred else 1)

    for _, config in candidates:
        model = LIIFModel(**config).to(device)
        total = sum(p.numel() for p in model.get_params().values())
        if total == n_params:
            return model
    raise ValueError(f"no LIIF config matches trajectory parameter count {n_params}")


def unflatten_full_state(
    model: LIIFModel,
    flat: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Unflatten a full-model snapshot using model.get_params() order."""
    templates = model.get_params()
    total = sum(t.numel() for t in templates.values())
    if total != flat.shape[0]:
        raise ValueError(f"snapshot has {flat.shape[0]} params, model expects {total}")

    state: dict[str, torch.Tensor] = {}
    offset = 0
    for key, template in templates.items():
        n_elem = template.numel()
        arr = flat[offset:offset + n_elem].reshape(tuple(template.shape))
        state[key] = torch.from_numpy(arr).to(device=device, dtype=template.dtype)
        offset += n_elem
    return state


def unflatten_siren_state(model: SIREN, flat: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """Unflatten a SIREN full snapshot using get_params order."""
    templates = model.get_params()
    total = sum(t.numel() for t in templates.values())
    if total != flat.shape[0]:
        raise ValueError(f"snapshot has {flat.shape[0]} params, model expects {total}")
    state: dict[str, torch.Tensor] = {}
    offset = 0
    for key, template in templates.items():
        n_elem = template.numel()
        arr = flat[offset:offset + n_elem].reshape(tuple(template.shape))
        state[key] = torch.from_numpy(arr).to(device=device, dtype=template.dtype)
        offset += n_elem
    return state


def evaluate_siren_responses(
    result_dir: Path,
    summary: dict[str, Any],
    data_root: Path,
    device: torch.device,
) -> ResponseEvaluation:
    """Evaluate SIREN output response vectors at raw full snapshots."""
    trajectory = np.load(result_dir / "trajectory.npz", allow_pickle=True)
    snapshots = np.asarray(trajectory["full_snapshots"], dtype=np.float32)
    image_size = int(summary.get("image_size", 48))
    image_path = resolve_image_path_from_summary(summary, data_root)
    target_tensor = load_siren_image(image_path, image_size=image_size).to(device)
    _, height, width = target_tensor.shape
    coords = get_image_coordinates(height, width, normalize="center", device=device).reshape(-1, 2)
    target = target_tensor.permute(1, 2, 0).reshape(-1, 3).detach().cpu().numpy()

    cfg = dict(SIREN_CONFIG)
    cfg.update(summary.get("siren_config", {}) or {})
    model = SIREN(**cfg).to(device)
    model.eval()

    responses = []
    psnrs = []
    with torch.no_grad():
        for flat in snapshots:
            model.set_params(unflatten_siren_state(model, flat, device))
            out = model(coords).clamp(0, 1).detach().cpu().numpy()
            responses.append(out.reshape(-1))
            psnrs.append(compute_psnr_vector(out, target))

    return ResponseEvaluation(
        responses=np.asarray(responses, dtype=np.float64),
        psnrs=np.asarray(psnrs, dtype=np.float64),
        target=target.reshape(-1),
        hybrid={},
    )


def state_with_parts(
    base: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Copy selected state_dict prefixes from donor onto base."""
    out = {k: v.clone() for k, v in base.items()}
    for key, value in donor.items():
        if key.startswith(prefixes):
            out[key] = value.clone()
    return out


def liif_output_vector(
    model: LIIFModel,
    state: dict[str, torch.Tensor],
    coords: torch.Tensor,
    lr_tensor: torch.Tensor,
    scale: int,
) -> np.ndarray:
    """Evaluate a LIIF state and return flattened RGB output."""
    model.set_params(state)
    with torch.no_grad():
        out = model(coords, lr_tensor, scale=scale).clamp(0, 1)
    return out.detach().cpu().numpy().reshape(-1)


def evaluate_liif_responses(
    result_dir: Path,
    summary: dict[str, Any],
    data_root: Path,
    device: torch.device,
) -> ResponseEvaluation:
    """Evaluate LIIF output response vectors at raw full snapshots."""
    trajectory = np.load(result_dir / "trajectory.npz", allow_pickle=True)
    full_snapshots = np.asarray(trajectory["full_snapshots"], dtype=np.float32)
    image_size = int(summary.get("hr_size", 48))
    sr_scale = int(summary.get("sr_scale", 4))
    image_path = resolve_image_path_from_summary(summary, data_root)
    lr_tensor, hr_tensor = load_liif_image(image_path, image_size=image_size, sr_scale=sr_scale)
    lr_tensor = lr_tensor.to(device)
    hr_tensor = hr_tensor.to(device)
    _, height, width = hr_tensor.shape
    coords = get_image_coordinates(height, width, normalize="center", device=device).reshape(-1, 2)
    target = hr_tensor.permute(1, 2, 0).reshape(-1, 3).detach().cpu().numpy()

    model = build_liif_model(summary, full_snapshots.shape[1], device)
    model.eval()

    states = [unflatten_full_state(model, flat, device) for flat in full_snapshots]
    responses = []
    psnrs = []
    for state in states:
        out = liif_output_vector(model, state, coords, lr_tensor, sr_scale)
        responses.append(out)
        psnrs.append(compute_psnr_vector(out.reshape(-1, 3), target))

    initial = np.asarray(responses[0], dtype=np.float64)
    final = np.asarray(responses[-1], dtype=np.float64)
    full_delta = final - initial
    full_norm = float(np.linalg.norm(full_delta))

    enc_state = state_with_parts(states[0], states[-1], ("encoder.",))
    dec_state = state_with_parts(states[0], states[-1], ("decoder.",))
    enc_only = liif_output_vector(model, enc_state, coords, lr_tensor, sr_scale).astype(np.float64)
    dec_only = liif_output_vector(model, dec_state, coords, lr_tensor, sr_scale).astype(np.float64)
    enc_delta = enc_only - initial
    dec_delta = dec_only - initial
    residual = full_delta - enc_delta - dec_delta

    denom = full_norm + 1e-12
    hybrid = {
        "hybrid_full_delta_norm": full_norm,
        "hybrid_encoder_only_delta_norm_ratio": float(np.linalg.norm(enc_delta) / denom),
        "hybrid_decoder_only_delta_norm_ratio": float(np.linalg.norm(dec_delta) / denom),
        "hybrid_interaction_delta_norm_ratio": float(np.linalg.norm(residual) / denom),
        "hybrid_encoder_decoder_cosine": float(
            np.dot(enc_delta, dec_delta)
            / ((np.linalg.norm(enc_delta) * np.linalg.norm(dec_delta)) + 1e-12)
        ),
    }

    return ResponseEvaluation(
        responses=np.asarray(responses, dtype=np.float64),
        psnrs=np.asarray(psnrs, dtype=np.float64),
        target=target.reshape(-1),
        hybrid=hybrid,
    )


def response_audit_metrics(
    responses: np.ndarray,
    snapshot_steps: np.ndarray | None,
    n_controls: int,
    seed: int,
) -> dict[str, float]:
    """Compute trajectory audit metrics for response vectors."""
    updates = np.diff(responses, axis=0)
    uncentered = spectrum_metrics(updates, centered=False)
    centered = spectrum_metrics(updates, centered=True)
    row: dict[str, float] = {
        "response_dim": float(responses.shape[1]),
        "response_snapshot_pc1_pct": spectrum_metrics(responses, centered=True)["pc1_pct"],
        "response_update_pc1_pct": uncentered["pc1_pct"],
        "response_update_effective_rank": uncentered["effective_rank"],
        "response_centered_update_pc1_pct": centered["pc1_pct"],
        "response_centered_update_effective_rank": centered["effective_rank"],
    }
    row.update(endpoint_line_metrics(responses, snapshot_steps))
    row.update(update_decomposition_metrics(responses))
    row.update(control_metrics(responses, n_controls=n_controls, seed=seed))
    return row


def analyze_result_dir(
    result_dir: Path,
    data_root: Path,
    device: torch.device,
    n_controls: int,
    seed: int,
) -> dict[str, Any]:
    """Analyze one Stage-B result directory in response space."""
    trajectory_path = result_dir / "trajectory.npz"
    if not trajectory_path.exists():
        return {"run": result_dir.name, "status": "missing trajectory"}
    summary = load_summary(result_dir)
    trajectory = np.load(trajectory_path, allow_pickle=True)
    family = infer_family(summary, set(trajectory.files))
    snapshot_steps = (
        np.asarray(trajectory["snapshot_steps"], dtype=np.float64)
        if "snapshot_steps" in trajectory.files
        else None
    )

    if family == "conditional":
        evaluation = evaluate_liif_responses(result_dir, summary, data_root, device)
    else:
        evaluation = evaluate_siren_responses(result_dir, summary, data_root, device)

    row: dict[str, Any] = {
        "run": result_dir.name,
        "status": "ok",
        "family": family,
        "model_type": summary.get("model_type", summary.get("model", "?")),
        "image": summary.get("image", "?"),
        "seed": summary.get("seed", "?"),
        "summary_final_psnr": summary.get("final_psnr", "?"),
        "response_initial_psnr": float(evaluation.psnrs[0]),
        "response_final_psnr": float(evaluation.psnrs[-1]),
        "response_final_psnr_abs_error": float(
            abs(float(summary.get("final_psnr", evaluation.psnrs[-1])) - float(evaluation.psnrs[-1]))
        ),
        "n_snapshots": int(evaluation.responses.shape[0]),
    }
    row.update(
        response_audit_metrics(
            evaluation.responses,
            snapshot_steps=snapshot_steps,
            n_controls=n_controls,
            seed=seed,
        )
    )
    row.update(evaluation.hybrid)
    return row


def scan_results(
    results_dir: Path,
    data_root: Path,
    device: torch.device,
    n_controls: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Analyze every result directory under results_dir."""
    rows: list[dict[str, Any]] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        rows.append(
            analyze_result_dir(
                child,
                data_root=data_root,
                device=device,
                n_controls=n_controls,
                seed=seed,
            )
        )
    return rows


def format_float(value: Any, digits: int = 3) -> str:
    """Format table floats compactly."""
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
        ("family", "family"),
        ("pc1", "response_snapshot_pc1_pct"),
        ("line", "endpoint_line_rms_rel"),
        ("perm", "permuted_endpoint_line_rms_rel_mean"),
        ("rw", "rw_endpoint_line_rms_rel_mean"),
        ("straight", "straightness"),
        ("orthE", "orthogonal_energy_fraction"),
        ("erank", "response_update_effective_rank"),
        ("psnr_err", "response_final_psnr_abs_error"),
        ("dec_only", "hybrid_decoder_only_delta_norm_ratio"),
        ("enc_only", "hybrid_encoder_only_delta_norm_ratio"),
    ]
    widths = {label: max(8, len(label)) for label, _ in columns}
    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        formatted: dict[str, str] = {}
        for label, key in columns:
            value = row.get(key, "")
            text = str(value) if key in {"run", "family"} else format_float(value)
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
        description="Read-only Stage B-response function-space trajectory audit"
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results/FittingDynamics_StageB"))
    parser.add_argument("--data_root", type=Path, default=Path("Data"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_controls", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = scan_results(
        args.results_dir,
        data_root=args.data_root,
        device=device,
        n_controls=args.n_controls,
        seed=args.seed,
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
