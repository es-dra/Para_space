#!/usr/bin/env python3
"""Read-only Stage-B trajectory control analysis.

This script analyzes existing fitting-dynamics outputs and compares observed
trajectory structure against lightweight controls. It does not launch training
jobs and does not modify experiment artifacts.
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


SKIP_DIRS = {"logs", "viz"}


@dataclass(frozen=True)
class SnapshotSelection:
    """Raw and aligned snapshots selected for the primary analysis space."""

    family: str
    raw_key: str
    aligned_key: str
    raw: np.ndarray
    aligned: np.ndarray


def load_summary(result_dir: Path) -> dict[str, Any]:
    """Load a dynamics summary if one exists."""
    for name in ("dynamics_summary.json", "summary.json"):
        path = result_dir / name
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
    return {}


def infer_family(summary: dict[str, Any], keys: set[str]) -> str:
    """Infer whether a trajectory is SIREN-like or conditional LIIF/LTE-like."""
    model_type = str(summary.get("model_type", summary.get("model", ""))).lower()
    if model_type == "siren":
        return "siren"
    if {"enc_snapshots", "dec_snapshots"}.issubset(keys):
        return "conditional"
    if model_type in {"liif", "lte", "conditional", "pretrained_liif", "liif_eq"}:
        return "conditional"
    return "siren"


def select_primary_snapshots(data: Any, summary: dict[str, Any]) -> SnapshotSelection:
    """Select the primary trajectory space for Stage-B controls.

    SIREN uses full parameters. Conditional LIIF/LTE runs use decoder
    parameters because Stage C will initially probe decoder/response behavior
    rather than flattened full-model vectors.
    """
    keys = set(data.files)
    family = infer_family(summary, keys)
    if family == "conditional":
        raw_key = "dec_snapshots"
        aligned_key = "dec_snapshots_aligned" if "dec_snapshots_aligned" in keys else raw_key
    else:
        raw_key = "full_snapshots"
        aligned_key = "full_snapshots_aligned" if "full_snapshots_aligned" in keys else raw_key
    return SnapshotSelection(
        family=family,
        raw_key=raw_key,
        aligned_key=aligned_key,
        raw=np.asarray(data[raw_key], dtype=np.float64),
        aligned=np.asarray(data[aligned_key], dtype=np.float64),
    )


def pca_explained_percent(snapshots: np.ndarray) -> np.ndarray:
    """Return PCA explained variance percentages for snapshot rows.

    The computation uses the snapshot Gram matrix, which is efficient for the
    current setting where there are few snapshots but many parameters.
    """
    if snapshots.ndim != 2:
        raise ValueError(f"snapshots must be 2D, got shape {snapshots.shape}")
    if snapshots.shape[0] < 3:
        return np.array([], dtype=np.float64)

    centered = snapshots - snapshots.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.maximum(eigvals, 0.0)[::-1]
    total = float(eigvals.sum())
    if total <= 0.0:
        return np.array([], dtype=np.float64)
    return eigvals / total * 100.0


def pc1_percent(snapshots: np.ndarray) -> float:
    """Return PC1 explained variance percentage, or NaN if undefined."""
    explained = pca_explained_percent(snapshots)
    if explained.size == 0:
        return float("nan")
    return float(explained[0])


def update_vectors(snapshots: np.ndarray) -> np.ndarray:
    """Return consecutive update vectors."""
    if snapshots.ndim != 2:
        raise ValueError(f"snapshots must be 2D, got shape {snapshots.shape}")
    return np.diff(snapshots, axis=0)


def consecutive_update_cosine(snapshots: np.ndarray) -> float:
    """Mean cosine similarity between consecutive update vectors."""
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


def trajectory_straightness(snapshots: np.ndarray) -> float:
    """Endpoint distance divided by cumulative path length."""
    updates = update_vectors(snapshots)
    if updates.size == 0:
        return float("nan")
    path = float(np.linalg.norm(updates, axis=1).sum())
    if path <= 0.0:
        return float("nan")
    endpoint = float(np.linalg.norm(snapshots[-1] - snapshots[0]))
    return endpoint / path


def step_norms(snapshots: np.ndarray) -> np.ndarray:
    """Return norms of consecutive update vectors."""
    return np.linalg.norm(update_vectors(snapshots), axis=1)


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
        direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / direction_norms
        updates[nonzero] = directions * norms[nonzero, None]
    return np.vstack([np.zeros((1, n_params), dtype=np.float64), np.cumsum(updates, axis=0)])


def make_iid_gaussian_snapshots(
    n_snapshots: int,
    n_params: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create iid Gaussian snapshots with matching snapshot count and dimension."""
    return rng.standard_normal((n_snapshots, n_params))


def make_permuted_update_trajectory(
    snapshots: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute observed update order while preserving endpoint and path length."""
    updates = update_vectors(snapshots)
    permuted = updates[rng.permutation(updates.shape[0])]
    return np.vstack([snapshots[:1], snapshots[0] + np.cumsum(permuted, axis=0)])


def summarize_distribution(values: list[float]) -> dict[str, float]:
    """Summarize a control distribution while ignoring NaN values."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def one_sided_ge_pvalue(observed: float, controls: list[float]) -> float:
    """Empirical p(control >= observed) with a plus-one correction."""
    if not math.isfinite(observed):
        return float("nan")
    finite = [x for x in controls if math.isfinite(x)]
    if not finite:
        return float("nan")
    ge = sum(1 for x in finite if x >= observed)
    return float((ge + 1) / (len(finite) + 1))


def z_score(observed: float, dist: dict[str, float]) -> float:
    """Return z score of observed against a summarized control distribution."""
    std = dist["std"]
    if not math.isfinite(observed) or not math.isfinite(std) or std <= 0.0:
        return float("nan")
    return float((observed - dist["mean"]) / std)


def analyze_snapshots(
    snapshots: np.ndarray,
    n_controls: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Analyze one aligned trajectory and its controls."""
    snapshots = np.asarray(snapshots, dtype=np.float64)
    n_snapshots, n_params = snapshots.shape
    norms = step_norms(snapshots)
    observed_pc1 = pc1_percent(snapshots)
    observed_cosine = consecutive_update_cosine(snapshots)
    observed_straightness = trajectory_straightness(snapshots)

    rng = np.random.default_rng(seed)
    rw_pc1: list[float] = []
    rw_cosine: list[float] = []
    rw_straightness: list[float] = []
    iid_pc1: list[float] = []
    permuted_pc1: list[float] = []
    permuted_cosine: list[float] = []

    for _ in range(n_controls):
        random_walk = make_norm_matched_random_walk(norms, n_params, rng)
        rw_pc1.append(pc1_percent(random_walk))
        rw_cosine.append(consecutive_update_cosine(random_walk))
        rw_straightness.append(trajectory_straightness(random_walk))

        iid_pc1.append(pc1_percent(make_iid_gaussian_snapshots(n_snapshots, n_params, rng)))

        permuted = make_permuted_update_trajectory(snapshots, rng)
        permuted_pc1.append(pc1_percent(permuted))
        permuted_cosine.append(consecutive_update_cosine(permuted))

    rw_pc1_dist = summarize_distribution(rw_pc1)
    rw_cosine_dist = summarize_distribution(rw_cosine)
    rw_straightness_dist = summarize_distribution(rw_straightness)
    iid_pc1_dist = summarize_distribution(iid_pc1)
    permuted_pc1_dist = summarize_distribution(permuted_pc1)
    permuted_cosine_dist = summarize_distribution(permuted_cosine)

    return {
        "n_snapshots": float(n_snapshots),
        "n_params": float(n_params),
        "pc1_pct": observed_pc1,
        "straightness": observed_straightness,
        "step_cosine": observed_cosine,
        "path_length": float(norms.sum()),
        "endpoint_norm": float(np.linalg.norm(snapshots[-1] - snapshots[0])),
        "mean_step_norm": float(norms.mean()) if norms.size else float("nan"),
        "rw_pc1_mean": rw_pc1_dist["mean"],
        "rw_pc1_std": rw_pc1_dist["std"],
        "rw_pc1_p95": rw_pc1_dist["p95"],
        "rw_pc1_z": z_score(observed_pc1, rw_pc1_dist),
        "rw_pc1_p_ge": one_sided_ge_pvalue(observed_pc1, rw_pc1),
        "rw_straightness_mean": rw_straightness_dist["mean"],
        "rw_step_cosine_mean": rw_cosine_dist["mean"],
        "iid_pc1_mean": iid_pc1_dist["mean"],
        "iid_pc1_std": iid_pc1_dist["std"],
        "iid_pc1_p95": iid_pc1_dist["p95"],
        "iid_pc1_z": z_score(observed_pc1, iid_pc1_dist),
        "iid_pc1_p_ge": one_sided_ge_pvalue(observed_pc1, iid_pc1),
        "permuted_pc1_mean": permuted_pc1_dist["mean"],
        "permuted_pc1_std": permuted_pc1_dist["std"],
        "permuted_pc1_p95": permuted_pc1_dist["p95"],
        "permuted_pc1_z": z_score(observed_pc1, permuted_pc1_dist),
        "permuted_pc1_p_ge": one_sided_ge_pvalue(observed_pc1, permuted_pc1),
        "permuted_step_cosine_mean": permuted_cosine_dist["mean"],
        "permuted_step_cosine_std": permuted_cosine_dist["std"],
        "permuted_step_cosine_p_ge": one_sided_ge_pvalue(observed_cosine, permuted_cosine),
    }


def stable_seed(base_seed: int, name: str) -> int:
    """Derive a deterministic per-run seed from a base seed and run name."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((base_seed + offset) % (2**32 - 1))


def analyze_result_dir(result_dir: Path, n_controls: int, seed: int) -> dict[str, Any]:
    """Analyze one result directory."""
    summary = load_summary(result_dir)
    trajectory_path = result_dir / "trajectory.npz"
    if not trajectory_path.exists():
        return {"run": result_dir.name, "status": "missing trajectory"}

    data = np.load(trajectory_path, allow_pickle=True)
    selection = select_primary_snapshots(data, summary)
    analysis = analyze_snapshots(
        selection.aligned,
        n_controls=n_controls,
        seed=stable_seed(seed, result_dir.name),
    )

    raw_pc1 = pc1_percent(selection.raw)
    aligned_pc1 = analysis["pc1_pct"]
    row: dict[str, Any] = {
        "run": result_dir.name,
        "status": "ok",
        "family": selection.family,
        "primary_space": "decoder" if selection.family == "conditional" else "full",
        "raw_key": selection.raw_key,
        "aligned_key": selection.aligned_key,
        "pc1_raw_pct": raw_pc1,
        "pc1_aligned_pct": aligned_pc1,
        "pc1_alignment_delta": aligned_pc1 - raw_pc1,
        "model_type": summary.get("model_type", summary.get("model", "?")),
        "image": summary.get("image", "?"),
        "seed": summary.get("seed", "?"),
        "final_psnr": summary.get("final_psnr", "?"),
        "model_config_name": summary.get("model_config_name", ""),
    }
    row.update(analysis)

    if {"enc_snapshots", "dec_snapshots"}.issubset(set(data.files)):
        enc = np.asarray(data["enc_snapshots"], dtype=np.float64)
        dec = selection.aligned
        enc_update = float(np.linalg.norm(enc[-1] - enc[0]))
        dec_update = float(np.linalg.norm(dec[-1] - dec[0]))
        row["enc_update_norm"] = enc_update
        row["dec_update_norm"] = dec_update
        row["enc_dec_update_ratio"] = enc_update / dec_update if dec_update > 0 else float("nan")
    else:
        row["enc_update_norm"] = float("nan")
        row["dec_update_norm"] = float("nan")
        row["enc_dec_update_ratio"] = float("nan")

    return row


def scan_results(results_dir: Path, n_controls: int, seed: int) -> list[dict[str, Any]]:
    """Analyze every experiment directory under results_dir."""
    if not results_dir.exists():
        raise FileNotFoundError(f"results directory not found: {results_dir}")
    rows: list[dict[str, Any]] = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith(".") or subdir.name in SKIP_DIRS:
            continue
        rows.append(analyze_result_dir(subdir, n_controls=n_controls, seed=seed))
    return rows


def format_float(value: Any, digits: int = 3) -> str:
    """Format numbers for compact tables."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print a compact control-analysis table."""
    headers = [
        ("run", 34),
        ("pc1", 7),
        ("rw_mu", 7),
        ("rw_z", 7),
        ("rw_p>=", 7),
        ("perm_mu", 8),
        ("perm_p>=", 8),
        ("cos", 7),
        ("perm_cos", 9),
        ("cos_p>=", 8),
        ("straight", 9),
        ("enc/dec", 8),
    ]
    print()
    print("".join(f"{name:<{width}}" for name, width in headers))
    print("-" * sum(width for _, width in headers))
    for row in rows:
        values = {
            "run": row.get("run", "?"),
            "pc1": format_float(row.get("pc1_pct"), 2),
            "rw_mu": format_float(row.get("rw_pc1_mean"), 2),
            "rw_z": format_float(row.get("rw_pc1_z"), 2),
            "rw_p>=": format_float(row.get("rw_pc1_p_ge"), 3),
            "perm_mu": format_float(row.get("permuted_pc1_mean"), 2),
            "perm_p>=": format_float(row.get("permuted_pc1_p_ge"), 3),
            "cos": format_float(row.get("step_cosine"), 3),
            "perm_cos": format_float(row.get("permuted_step_cosine_mean"), 3),
            "cos_p>=": format_float(row.get("permuted_step_cosine_p_ge"), 3),
            "straight": format_float(row.get("straightness"), 3),
            "enc/dec": format_float(row.get("enc_dec_update_ratio"), 3),
        }
        print("".join(f"{values[name]:<{width}}" for name, width in headers))


def print_csv(rows: list[dict[str, Any]]) -> None:
    """Print rows as CSV."""
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Stage-B trajectory control analysis")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/FittingDynamics_StageB",
        help="Directory containing Stage-B experiment result subdirectories",
    )
    parser.add_argument(
        "--n_controls",
        type=int,
        default=100,
        help="Number of Monte-Carlo controls per run",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for controls")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table")
    args = parser.parse_args()

    rows = scan_results(Path(args.results_dir), n_controls=args.n_controls, seed=args.seed)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.format == "csv":
        print_csv(rows)
    else:
        print_table(rows)
        print(f"\n{len(rows)} experiment(s) analyzed with {args.n_controls} controls each.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
