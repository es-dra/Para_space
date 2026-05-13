#!/usr/bin/env python3
"""Read-only Stage B-prime trajectory-object audit.

This script audits saved fitting-dynamics trajectories. It does not train
models and does not modify existing result directories.

The goal is narrower than a paper claim: determine whether the saved snapshot
and update sequences show structure beyond simple endpoint drift, step-norm
schedule, smooth optimization continuity, and parameterization-sensitive PCA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.config import LIIF_CONFIG, LIIF_CONFIG_REDUCED, SIREN_CONFIG
from src.alignment import get_decoder_state_dict, get_encoder_state_dict
from src.models.liif import LIIFModel
from src.siren import SIREN


SKIP_DIRS = {"logs", "viz"}
EPS = 1e-12


@dataclass(frozen=True)
class AnalysisSpace:
    """One parameter space to audit for a result directory."""

    name: str
    key: str
    snapshots: np.ndarray
    slices: tuple[tuple[str, int, int], ...] = ()


def load_summary(result_dir: Path) -> dict[str, Any]:
    """Load the dynamics summary if present."""
    for name in ("dynamics_summary.json", "summary.json"):
        path = result_dir / name
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
    return {}


def stable_seed(base_seed: int, name: str) -> int:
    """Derive a deterministic per-run seed."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((base_seed + offset) % (2**32 - 1))


def flatten_slices(state: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    """Return flattened [start, end) slices using the dict's iteration order."""
    slices: list[tuple[str, int, int]] = []
    offset = 0
    for name, value in state.items():
        n_elem = int(value.numel())
        slices.append((name, offset, offset + n_elem))
        offset += n_elem
    return tuple(slices)


def build_siren_slices(summary: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    """Build SIREN flattened parameter slices."""
    cfg = dict(SIREN_CONFIG)
    cfg.update(summary.get("siren_config", {}) or {})
    model = SIREN(**cfg)
    return flatten_slices(model.get_params())


def liif_config_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Select the LIIF config matching a saved run."""
    config_name = str(summary.get("model_config_name", "LIIF_CONFIG_REDUCED"))
    if config_name == "LIIF_CONFIG":
        return dict(LIIF_CONFIG)
    return dict(LIIF_CONFIG_REDUCED)


def build_liif_slices(summary: dict[str, Any], space: str) -> tuple[tuple[str, int, int], ...]:
    """Build LIIF flattened parameter slices for full, encoder, or decoder."""
    model = LIIFModel(**liif_config_from_summary(summary))
    if space == "full":
        return flatten_slices(model.state_dict())
    if space == "encoder":
        return flatten_slices(get_encoder_state_dict(model))
    if space == "decoder":
        return flatten_slices(get_decoder_state_dict(model))
    raise ValueError(f"unknown LIIF space: {space}")


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


def select_analysis_spaces(data: Any, summary: dict[str, Any]) -> list[AnalysisSpace]:
    """Select all parameter spaces that should be audited."""
    keys = set(data.files)
    family = infer_family(summary, keys)
    spaces: list[AnalysisSpace] = []
    if family == "conditional":
        if "full_snapshots" in keys:
            spaces.append(
                AnalysisSpace(
                    "full",
                    "full_snapshots",
                    np.asarray(data["full_snapshots"], dtype=np.float64),
                    build_liif_slices(summary, "full"),
                )
            )
        if "enc_snapshots" in keys:
            spaces.append(
                AnalysisSpace(
                    "encoder",
                    "enc_snapshots",
                    np.asarray(data["enc_snapshots"], dtype=np.float64),
                    build_liif_slices(summary, "encoder"),
                )
            )
        if "dec_snapshots" in keys:
            dec_slices = build_liif_slices(summary, "decoder")
            spaces.append(
                AnalysisSpace(
                    "decoder_raw",
                    "dec_snapshots",
                    np.asarray(data["dec_snapshots"], dtype=np.float64),
                    dec_slices,
                )
            )
            if "dec_snapshots_aligned" in keys:
                spaces.append(
                    AnalysisSpace(
                        "decoder_aligned",
                        "dec_snapshots_aligned",
                        np.asarray(data["dec_snapshots_aligned"], dtype=np.float64),
                        dec_slices,
                    )
                )
    else:
        if "full_snapshots" in keys:
            spaces.append(
                AnalysisSpace(
                    "full_raw",
                    "full_snapshots",
                    np.asarray(data["full_snapshots"], dtype=np.float64),
                    build_siren_slices(summary),
                )
            )
        if "full_snapshots_aligned" in keys:
            spaces.append(
                AnalysisSpace(
                    "full_aligned",
                    "full_snapshots_aligned",
                    np.asarray(data["full_snapshots_aligned"], dtype=np.float64),
                    build_siren_slices(summary),
                )
            )
    return spaces


def update_vectors(snapshots: np.ndarray) -> np.ndarray:
    """Return consecutive update vectors."""
    return np.diff(snapshots, axis=0)


def step_norms(snapshots: np.ndarray) -> np.ndarray:
    """Return per-update L2 norms."""
    return np.linalg.norm(update_vectors(snapshots), axis=1)


def pca_pc1_percent(rows: np.ndarray) -> float:
    """Return PC1 explained variance percentage for row vectors."""
    if rows.ndim != 2 or rows.shape[0] < 3:
        return float("nan")
    centered = rows - rows.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigvals = np.maximum(np.linalg.eigvalsh(gram), 0.0)[::-1]
    total = float(eigvals.sum())
    if total <= 0.0:
        return float("nan")
    return float(eigvals[0] / total * 100.0)


def spectrum_metrics(rows: np.ndarray, centered: bool) -> dict[str, float]:
    """Return low-sample rank diagnostics for a row-vector matrix."""
    if rows.ndim != 2 or rows.shape[0] < 2:
        return {
            "pc1_pct": float("nan"),
            "effective_rank": float("nan"),
            "participation_rank": float("nan"),
            "rank90": float("nan"),
        }
    x = rows - rows.mean(axis=0, keepdims=True) if centered else rows
    gram = x @ x.T
    eigvals = np.maximum(np.linalg.eigvalsh(gram), 0.0)[::-1]
    total = float(eigvals.sum())
    if total <= 0.0:
        return {
            "pc1_pct": float("nan"),
            "effective_rank": float("nan"),
            "participation_rank": float("nan"),
            "rank90": float("nan"),
        }
    probs = eigvals[eigvals > 0.0] / total
    entropy = float(-(probs * np.log(probs + EPS)).sum())
    cumsum = np.cumsum(eigvals) / total
    return {
        "pc1_pct": float(eigvals[0] / total * 100.0),
        "effective_rank": float(np.exp(entropy)),
        "participation_rank": float(total * total / (np.square(eigvals).sum() + EPS)),
        "rank90": float(np.searchsorted(cumsum, 0.90) + 1),
    }


def consecutive_update_cosine(snapshots: np.ndarray) -> float:
    """Mean cosine between consecutive update vectors."""
    updates = update_vectors(snapshots)
    if updates.shape[0] < 2:
        return float("nan")
    a = updates[:-1]
    b = updates[1:]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > 0.0
    if not np.any(valid):
        return float("nan")
    dots = np.einsum("ij,ij->i", a[valid], b[valid])
    return float(np.mean(dots / denom[valid]))


def endpoint_line_metrics(
    snapshots: np.ndarray,
    snapshot_steps: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure how close snapshots are to the endpoint line."""
    start = snapshots[0]
    endpoint = snapshots[-1] - start
    endpoint_norm = float(np.linalg.norm(endpoint))
    if endpoint_norm <= 0.0:
        return {
            "endpoint_norm": endpoint_norm,
            "endpoint_line_rms_rel": float("nan"),
            "endpoint_line_max_rel": float("nan"),
            "time_line_rms_rel": float("nan"),
            "time_line_max_rel": float("nan"),
            "projection_t_min": float("nan"),
            "projection_t_max": float("nan"),
            "projection_monotonic_violations": float("nan"),
        }

    deltas = snapshots - start[None, :]
    denom = endpoint_norm * endpoint_norm
    t = (deltas @ endpoint) / denom
    best_line = start[None, :] + t[:, None] * endpoint[None, :]
    residual_norms = np.linalg.norm(snapshots - best_line, axis=1)
    interior = residual_norms[1:-1] if residual_norms.size > 2 else residual_norms

    if snapshot_steps is not None and len(snapshot_steps) == snapshots.shape[0]:
        steps = np.asarray(snapshot_steps, dtype=np.float64)
        span = float(steps[-1] - steps[0])
        if span > 0.0:
            alpha = (steps - steps[0]) / span
        else:
            alpha = np.linspace(0.0, 1.0, snapshots.shape[0])
    else:
        alpha = np.linspace(0.0, 1.0, snapshots.shape[0])
    time_line = start[None, :] + alpha[:, None] * endpoint[None, :]
    time_residual_norms = np.linalg.norm(snapshots - time_line, axis=1)
    time_interior = (
        time_residual_norms[1:-1]
        if time_residual_norms.size > 2
        else time_residual_norms
    )

    return {
        "endpoint_norm": endpoint_norm,
        "endpoint_line_rms_rel": float(np.sqrt(np.mean(np.square(interior))) / endpoint_norm),
        "endpoint_line_max_rel": float(np.max(residual_norms) / endpoint_norm),
        "time_line_rms_rel": float(np.sqrt(np.mean(np.square(time_interior))) / endpoint_norm),
        "time_line_max_rel": float(np.max(time_residual_norms) / endpoint_norm),
        "projection_t_min": float(np.min(t)),
        "projection_t_max": float(np.max(t)),
        "projection_monotonic_violations": float(np.sum(np.diff(t) < -1e-8)),
    }


def update_decomposition_metrics(snapshots: np.ndarray) -> dict[str, float]:
    """Decompose updates relative to the final displacement direction."""
    updates = update_vectors(snapshots)
    norms = np.linalg.norm(updates, axis=1)
    path_length = float(norms.sum())
    endpoint = snapshots[-1] - snapshots[0]
    endpoint_norm = float(np.linalg.norm(endpoint))
    total_update_energy = float(np.square(norms).sum())
    if endpoint_norm <= 0.0 or path_length <= 0.0 or total_update_energy <= 0.0:
        return {
            "path_length": path_length,
            "straightness": float("nan"),
            "path_excess_ratio": float("nan"),
            "parallel_energy_fraction": float("nan"),
            "orthogonal_energy_fraction": float("nan"),
            "negative_progress_fraction": float("nan"),
            "mean_step_norm": float("nan"),
            "step_norm_cv": float("nan"),
            "step_cosine": float("nan"),
        }
    unit = endpoint / endpoint_norm
    parallel = updates @ unit
    parallel_energy = float(np.square(parallel).sum())
    parallel_fraction = parallel_energy / total_update_energy
    return {
        "path_length": path_length,
        "straightness": endpoint_norm / path_length,
        "path_excess_ratio": path_length / endpoint_norm,
        "parallel_energy_fraction": parallel_fraction,
        "orthogonal_energy_fraction": max(0.0, 1.0 - parallel_fraction),
        "negative_progress_fraction": float(np.mean(parallel < 0.0)),
        "mean_step_norm": float(norms.mean()),
        "step_norm_cv": float(norms.std(ddof=0) / (norms.mean() + EPS)),
        "step_cosine": consecutive_update_cosine(snapshots),
    }


def make_permuted_update_trajectory(
    snapshots: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute update order while preserving start, endpoint, and step norms."""
    updates = update_vectors(snapshots)
    permuted = updates[rng.permutation(updates.shape[0])]
    return np.vstack([snapshots[:1], snapshots[0] + np.cumsum(permuted, axis=0)])


def make_norm_matched_random_walk(
    norms: np.ndarray,
    n_params: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a random walk with the same update-norm schedule."""
    updates = np.zeros((len(norms), n_params), dtype=np.float64)
    nonzero = norms > 0.0
    if np.any(nonzero):
        directions = rng.standard_normal((int(nonzero.sum()), n_params))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        updates[nonzero] = directions * norms[nonzero, None]
    return np.vstack([np.zeros((1, n_params), dtype=np.float64), np.cumsum(updates, axis=0)])


def summarize_distribution(values: list[float]) -> dict[str, float]:
    """Summarize finite values."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def one_sided_le_pvalue(observed: float, controls: list[float]) -> float:
    """Empirical p(control <= observed), plus-one corrected."""
    finite = [x for x in controls if math.isfinite(x)]
    if not math.isfinite(observed) or not finite:
        return float("nan")
    le = sum(1 for x in finite if x <= observed)
    return float((le + 1) / (len(finite) + 1))


def one_sided_ge_pvalue(observed: float, controls: list[float]) -> float:
    """Empirical p(control >= observed), plus-one corrected."""
    finite = [x for x in controls if math.isfinite(x)]
    if not math.isfinite(observed) or not finite:
        return float("nan")
    ge = sum(1 for x in finite if x >= observed)
    return float((ge + 1) / (len(finite) + 1))


def control_metrics(
    snapshots: np.ndarray,
    n_controls: int,
    seed: int,
) -> dict[str, float]:
    """Compare endpoint-line and path metrics against simple controls."""
    if n_controls <= 0:
        return {}
    rng = np.random.default_rng(seed)
    norms = step_norms(snapshots)
    n_params = snapshots.shape[1]

    perm_line: list[float] = []
    perm_straightness: list[float] = []
    perm_parallel: list[float] = []
    rw_line: list[float] = []
    rw_straightness: list[float] = []
    rw_parallel: list[float] = []

    for _ in range(n_controls):
        permuted = make_permuted_update_trajectory(snapshots, rng)
        perm_line.append(endpoint_line_metrics(permuted)["endpoint_line_rms_rel"])
        perm_update = update_decomposition_metrics(permuted)
        perm_straightness.append(perm_update["straightness"])
        perm_parallel.append(perm_update["parallel_energy_fraction"])

        random_walk = make_norm_matched_random_walk(norms, n_params, rng)
        rw_line.append(endpoint_line_metrics(random_walk)["endpoint_line_rms_rel"])
        rw_update = update_decomposition_metrics(random_walk)
        rw_straightness.append(rw_update["straightness"])
        rw_parallel.append(rw_update["parallel_energy_fraction"])

    observed_line = endpoint_line_metrics(snapshots)["endpoint_line_rms_rel"]
    observed_update = update_decomposition_metrics(snapshots)
    observed_straightness = observed_update["straightness"]
    observed_parallel = observed_update["parallel_energy_fraction"]

    perm_line_dist = summarize_distribution(perm_line)
    perm_straight_dist = summarize_distribution(perm_straightness)
    perm_parallel_dist = summarize_distribution(perm_parallel)
    rw_line_dist = summarize_distribution(rw_line)
    rw_straight_dist = summarize_distribution(rw_straightness)
    rw_parallel_dist = summarize_distribution(rw_parallel)

    return {
        "permuted_endpoint_line_rms_rel_mean": perm_line_dist["mean"],
        "permuted_endpoint_line_rms_rel_p50": perm_line_dist["p50"],
        "permuted_endpoint_line_rms_rel_p_le": one_sided_le_pvalue(observed_line, perm_line),
        "permuted_straightness_mean": perm_straight_dist["mean"],
        "permuted_straightness_p_ge": one_sided_ge_pvalue(observed_straightness, perm_straightness),
        "permuted_parallel_energy_fraction_mean": perm_parallel_dist["mean"],
        "permuted_parallel_energy_fraction_p_ge": one_sided_ge_pvalue(observed_parallel, perm_parallel),
        "rw_endpoint_line_rms_rel_mean": rw_line_dist["mean"],
        "rw_endpoint_line_rms_rel_p50": rw_line_dist["p50"],
        "rw_endpoint_line_rms_rel_p_le": one_sided_le_pvalue(observed_line, rw_line),
        "rw_straightness_mean": rw_straight_dist["mean"],
        "rw_straightness_p_ge": one_sided_ge_pvalue(observed_straightness, rw_straightness),
        "rw_parallel_energy_fraction_mean": rw_parallel_dist["mean"],
        "rw_parallel_energy_fraction_p_ge": one_sided_ge_pvalue(observed_parallel, rw_parallel),
    }


def slice_update_energy(
    snapshots: np.ndarray,
    slices: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    """Compute update and endpoint energy concentration by parameter slice."""
    if not slices:
        return {
            "top_path_energy_name": "",
            "top_path_energy_fraction": float("nan"),
            "top_final_energy_name": "",
            "top_final_energy_fraction": float("nan"),
            "encoder_path_energy_fraction": float("nan"),
            "decoder_path_energy_fraction": float("nan"),
        }
    updates = update_vectors(snapshots)
    final_delta = snapshots[-1] - snapshots[0]
    total_path_energy = float(np.square(updates).sum())
    total_final_energy = float(np.square(final_delta).sum())
    path_rows: list[tuple[str, float]] = []
    final_rows: list[tuple[str, float]] = []
    encoder_path = 0.0
    decoder_path = 0.0
    for name, start, end in slices:
        path_energy = float(np.square(updates[:, start:end]).sum())
        final_energy = float(np.square(final_delta[start:end]).sum())
        path_fraction = path_energy / total_path_energy if total_path_energy > 0.0 else float("nan")
        final_fraction = final_energy / total_final_energy if total_final_energy > 0.0 else float("nan")
        path_rows.append((name, path_fraction))
        final_rows.append((name, final_fraction))
        if name.startswith("encoder."):
            encoder_path += path_energy
        if name.startswith("decoder."):
            decoder_path += path_energy
    path_rows.sort(key=lambda item: item[1], reverse=True)
    final_rows.sort(key=lambda item: item[1], reverse=True)
    return {
        "top_path_energy_name": path_rows[0][0],
        "top_path_energy_fraction": path_rows[0][1],
        "top_final_energy_name": final_rows[0][0],
        "top_final_energy_fraction": final_rows[0][1],
        "encoder_path_energy_fraction": (
            encoder_path / total_path_energy if total_path_energy > 0.0 and encoder_path > 0.0 else float("nan")
        ),
        "decoder_path_energy_fraction": (
            decoder_path / total_path_energy if total_path_energy > 0.0 and decoder_path > 0.0 else float("nan")
        ),
    }


def analyze_space(
    space: AnalysisSpace,
    snapshot_steps: np.ndarray | None,
    n_controls: int,
    seed: int,
) -> dict[str, Any]:
    """Analyze one parameter space."""
    snapshots = np.asarray(space.snapshots, dtype=np.float64)
    if snapshots.ndim != 2 or snapshots.shape[0] < 3:
        raise ValueError(f"{space.name} snapshots must be 2D with >=3 rows, got {snapshots.shape}")
    updates = update_vectors(snapshots)
    uncentered = spectrum_metrics(updates, centered=False)
    centered = spectrum_metrics(updates, centered=True)
    row: dict[str, Any] = {
        "space": space.name,
        "snapshot_key": space.key,
        "n_snapshots": int(snapshots.shape[0]),
        "n_params": int(snapshots.shape[1]),
        "snapshot_pc1_pct": pca_pc1_percent(snapshots),
        "update_pc1_pct": uncentered["pc1_pct"],
        "update_effective_rank": uncentered["effective_rank"],
        "update_participation_rank": uncentered["participation_rank"],
        "update_rank90": uncentered["rank90"],
        "centered_update_pc1_pct": centered["pc1_pct"],
        "centered_update_effective_rank": centered["effective_rank"],
        "centered_update_participation_rank": centered["participation_rank"],
        "centered_update_rank90": centered["rank90"],
    }
    row.update(endpoint_line_metrics(snapshots, snapshot_steps))
    row.update(update_decomposition_metrics(snapshots))
    row.update(slice_update_energy(snapshots, space.slices))
    row.update(control_metrics(snapshots, n_controls=n_controls, seed=seed))
    return row


def analyze_result_dir(result_dir: Path, n_controls: int, seed: int) -> list[dict[str, Any]]:
    """Analyze all available spaces for one result directory."""
    trajectory_path = result_dir / "trajectory.npz"
    if not trajectory_path.exists():
        return [{"run": result_dir.name, "status": "missing trajectory"}]
    summary = load_summary(result_dir)
    data = np.load(trajectory_path, allow_pickle=True)
    snapshot_steps = (
        np.asarray(data["snapshot_steps"], dtype=np.float64)
        if "snapshot_steps" in data.files
        else None
    )
    rows: list[dict[str, Any]] = []
    for space in select_analysis_spaces(data, summary):
        row = {
            "run": result_dir.name,
            "status": "ok",
            "family": infer_family(summary, set(data.files)),
            "model_type": summary.get("model_type", summary.get("model", "?")),
            "image": summary.get("image", "?"),
            "seed": summary.get("seed", "?"),
            "final_psnr": summary.get("final_psnr", "?"),
            "model_config_name": summary.get("model_config_name", ""),
        }
        row.update(
            analyze_space(
                space,
                snapshot_steps=snapshot_steps,
                n_controls=n_controls,
                seed=stable_seed(seed, result_dir.name + ":" + space.name),
            )
        )
        rows.append(row)
    return rows


def scan_results(results_dir: Path, n_controls: int, seed: int) -> list[dict[str, Any]]:
    """Analyze every result directory under results_dir."""
    rows: list[dict[str, Any]] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        rows.extend(analyze_result_dir(child, n_controls=n_controls, seed=seed))
    return rows


def format_float(value: Any, digits: int = 4) -> str:
    """Format table floats compactly."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print a compact table for terminal review."""
    columns = [
        ("run", "run"),
        ("space", "space"),
        ("pc1", "snapshot_pc1_pct"),
        ("line_rms", "endpoint_line_rms_rel"),
        ("time_rms", "time_line_rms_rel"),
        ("straight", "straightness"),
        ("orthE", "orthogonal_energy_fraction"),
        ("upd_erank", "update_effective_rank"),
        ("cent_erank", "centered_update_effective_rank"),
        ("top_path", "top_path_energy_fraction"),
        ("rw_line", "rw_endpoint_line_rms_rel_mean"),
        ("perm_line", "permuted_endpoint_line_rms_rel_mean"),
    ]
    widths = {label: max(len(label), 10) for label, _ in columns}
    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        formatted: dict[str, str] = {}
        for label, key in columns:
            value = row.get(key, "")
            if key in {"run", "space"}:
                text = str(value)
            else:
                text = format_float(value, 3)
            formatted[label] = text
            widths[label] = max(widths[label], len(text))
        formatted_rows.append(formatted)
    print(" ".join(label.ljust(widths[label]) for label, _ in columns))
    print(" ".join("-" * widths[label] for label, _ in columns))
    for formatted in formatted_rows:
        print(" ".join(formatted[label].ljust(widths[label]) for label, _ in columns))


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    """Write rows as JSON or CSV based on suffix."""
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
        description="Read-only Stage B-prime trajectory-object audit"
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results/FittingDynamics_StageB"))
    parser.add_argument("--n_controls", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = scan_results(args.results_dir, n_controls=args.n_controls, seed=args.seed)
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
