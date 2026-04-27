#!/usr/bin/env python3
"""
Phase 1 Results Aggregation & Validation

Scans all experiment outputs under results/FittingDynamics/, extracts key
metrics from dynamics_summary.json and trajectory NPZ files, and generates
a comparison table.

Usage:
    python experiments/Phase1_FittingDynamics/aggregate_results.py \
        [--results_dir results/FittingDynamics] [--format table|csv]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_pc1_ratio(trajectory_path: str) -> dict:
    """Load NPZ and compute PC1 ratio from flattened snapshots."""
    try:
        data = np.load(trajectory_path, allow_pickle=True)
        # Prefer aligned snapshots
        if "full_snapshots_aligned" in data:
            snapshots = data["full_snapshots_aligned"]
        elif "dec_snapshots_aligned" in data:
            snapshots = data["dec_snapshots_aligned"]
        else:
            snapshots = data["full_snapshots"]

        delta = snapshots - snapshots[0]
        centered = delta - delta.mean(axis=0)
        from scipy.linalg import svd
        U, S, Vt = svd(centered, full_matrices=False)
        ev = S ** 2
        ev_ratio = ev / ev.sum()

        n = snapshots.shape[0]
        random_baseline = 100.0 / (n - 1) if n > 1 else 0

        return {
            "n_snapshots": n,
            "n_params": snapshots.shape[1],
            "pc1_pct": float(ev_ratio[0] * 100),
            "pc2_pct": float(ev_ratio[1] * 100) if len(ev_ratio) > 1 else 0,
            "pc1_baseline_pct": float(random_baseline),
            "pc1_ratio": float(ev_ratio[0] * 100 / random_baseline) if random_baseline > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def scan_results(results_dir: str) -> list:
    """Scan results directory for experiment outputs."""
    entries = []
    base = Path(results_dir)
    if not base.exists():
        print(f"Warning: results directory not found: {results_dir}")
        return entries

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue

        # Load summary
        summary = None
        for sname in ["dynamics_summary.json", "summary.json"]:
            sp = subdir / sname
            if sp.exists():
                with open(sp) as f:
                    summary = json.load(f)
                break

        # Load trajectory metrics
        traj_path = subdir / "trajectory.npz"
        metrics = {}
        if traj_path.exists():
            metrics = compute_pc1_ratio(str(traj_path))

        entries.append({
            "dirname": subdir.name,
            "path": str(subdir),
            "summary": summary or {},
            "metrics": metrics,
        })

    return entries


def format_entry(entry: dict) -> dict:
    """Extract flat display fields from a scan entry."""
    s = entry["summary"]
    m = entry["metrics"]

    model_type = s.get("model_type", s.get("model", "?"))
    image = s.get("image", "?")
    sr_scale = s.get("sr_scale", s.get("scale", 0))
    mode = s.get("mode", f"SR x{sr_scale}" if sr_scale else "Self-Recon")

    return {
        "model": f"{model_type} ({mode})",
        "image": image,
        "final_psnr": f'{s.get("final_psnr", "?"):.1f}' if isinstance(s.get("final_psnr"), (int, float)) else "?",
        "final_loss": f'{s.get("final_loss", "?"):.4e}' if isinstance(s.get("final_loss"), (int, float)) else "?",
        "n_snapshots": m.get("n_snapshots", s.get("n_snapshots", "?")),
        "pc1_pct": f'{m.get("pc1_pct", "?"):.1f}' if isinstance(m.get("pc1_pct"), float) else "?",
        "pc1_ratio": f'{m.get("pc1_ratio", "?"):.1f}x' if isinstance(m.get("pc1_ratio"), float) else "?",
        "random_bsl": f'{m.get("pc1_baseline_pct", "?"):.1f}%' if isinstance(m.get("pc1_baseline_pct"), float) else "?",
        "alignment": s.get("alignment", "none"),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate Phase 1 results")
    parser.add_argument("--results_dir", type=str,
                        default="results/FittingDynamics")
    parser.add_argument("--format", type=str, default="table",
                        choices=["table", "csv"])
    args = parser.parse_args()

    entries = scan_results(args.results_dir)
    if not entries:
        print("No results found.")
        return

    rows = [format_entry(e) for e in entries]

    if args.format == "csv":
        keys = ["model", "image", "final_psnr", "final_loss",
                "pc1_pct", "pc1_ratio", "random_bsl"]
        print(",".join(keys))
        for r in rows:
            print(",".join(str(r.get(k, "?")) for k in keys))
    else:
        # Table format
        print()
        print(f"{'Experiment':<35} {'Model':<30} {'PSNR':>6} {'PC1%':>7} "
              f"{'PC1/BSL':>8} {'Align':<12}")
        print("-" * 100)
        for r in rows:
            print(f"{r['image']:<35} {r['model']:<30} {r['final_psnr']:>6} "
                  f"{r['pc1_pct']:>7} {r['pc1_ratio']:>8} {r['alignment']:<12}")

        # Summary
        done = len(entries)
        with_pc1 = sum(1 for r in rows if r["pc1_pct"] != "?")
        print(f"\n{done} experiment(s) found, {with_pc1} with PCA data.")

        # Check for validated experiments
        validated = sum(1 for r in rows if r["alignment"] != "none")
        if validated < done:
            print(f"⚠  {done - validated} experiment(s) missing alignment "
                  f"(experiments from before the alignment integration)")


if __name__ == "__main__":
    main()
