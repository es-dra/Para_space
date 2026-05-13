#!/usr/bin/env python3
"""Validate fitting-dynamics output directories.

This is a read-only Stage-B helper. It does not modify experiment outputs and
is safe to run on existing results directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trajectory_schema import (  # noqa: E402
    validate_summary_schema,
    validate_trajectory_schema,
    validate_trajectory_shapes,
)


SIREN_NAMES = {"siren"}
CONDITIONAL_NAMES = {
    "conditional",
    "liif",
    "lte",
}


def infer_model_family(summary: dict[str, Any], trajectory_keys: set[str]) -> str:
    """Infer schema family from summary and trajectory keys."""
    model_type = str(summary.get("model_type", "")).lower()
    if model_type in SIREN_NAMES:
        return "siren"
    if model_type in CONDITIONAL_NAMES:
        return "conditional"
    if {"enc_snapshots", "dec_snapshots"}.issubset(trajectory_keys):
        return "conditional"
    return "siren"


def load_summary(result_dir: Path) -> dict[str, Any]:
    """Load dynamics summary if present."""
    summary_path = result_dir / "dynamics_summary.json"
    if not summary_path.exists():
        summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r") as f:
        return json.load(f)


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def check_summary_consistency(summary: dict[str, Any], trajectory: Any) -> tuple[bool, list[str]]:
    """Check lightweight consistency between summary and trajectory.

    This function only checks fields that current scripts already write. It is
    intentionally conservative and does not enforce future metadata fields.
    """
    messages: list[str] = []
    ok = True

    steps = _as_list(trajectory["snapshot_steps"])
    summary_steps = _as_list(summary.get("snapshot_steps"))
    if steps is not None:
        n_steps = len(steps)
        if "n_snapshots" in summary and int(summary["n_snapshots"]) != n_steps:
            ok = False
            messages.append(
                f"summary consistency: FAIL n_snapshots={summary['n_snapshots']} "
                f"!= len(snapshot_steps)={n_steps}"
            )
        if summary_steps is not None and list(summary_steps) != list(steps):
            ok = False
            messages.append("summary consistency: FAIL summary snapshot_steps != trajectory snapshot_steps")
        if n_steps < 3:
            messages.append(
                "analysis readiness: WARN fewer than 3 snapshots; "
                "schema is valid but PCA/update-coherence analysis is not meaningful"
            )
        else:
            messages.append("analysis readiness: OK")

    psnrs = _as_list(trajectory.get("psnrs"))
    if psnrs:
        final_psnr = float(psnrs[-1])
        # Accept small summary rounding differences from historical outputs.
        if "final_psnr" in summary and abs(float(summary["final_psnr"]) - final_psnr) > 5e-3:
            ok = False
            messages.append(
                f"summary consistency: FAIL final_psnr={summary['final_psnr']} "
                f"!= last trajectory psnr={final_psnr}"
            )

    losses = _as_list(trajectory.get("losses"))
    if losses:
        final_loss = float(losses[-1])
        if "final_loss" in summary and abs(float(summary["final_loss"]) - final_loss) > 1e-6:
            ok = False
            messages.append(
                f"summary consistency: FAIL final_loss={summary['final_loss']} "
                f"!= last trajectory loss={final_loss}"
            )

    if ok:
        messages.append("summary consistency: OK")
    return ok, messages


def validate_result_dir(
    result_dir: Path,
    model_family: str | None = None,
    require_summary: bool = True,
) -> tuple[bool, list[str]]:
    """Validate one fitting-dynamics result directory.

    Returns:
        (ok, messages). ok is False if any required check fails.
    """
    messages: list[str] = []
    trajectory_path = result_dir / "trajectory.npz"
    if not result_dir.exists():
        return False, [f"结果目录不存在: {result_dir}"]
    if not trajectory_path.exists():
        return False, [f"缺少 trajectory.npz: {trajectory_path}"]

    summary = load_summary(result_dir)
    if require_summary and not summary:
        return False, ["缺少 dynamics_summary.json 或 summary.json"]

    data = np.load(trajectory_path, allow_pickle=True)
    keys = set(data.files)
    family = model_family or infer_model_family(summary, keys)

    schema = validate_trajectory_schema(keys, model_family=family)
    if schema.ok:
        messages.append(f"trajectory schema: OK ({family})")
    else:
        messages.append(f"trajectory schema: FAIL missing={list(schema.missing_keys)}")

    shape_report = validate_trajectory_shapes(data, model_family=family)
    if shape_report.ok:
        messages.append("trajectory shapes: OK")
    else:
        messages.append(f"trajectory shapes: FAIL errors={list(shape_report.errors)}")

    summary_ok = True
    consistency_ok = True
    if summary:
        summary_report = validate_summary_schema(
            summary,
            conditional=(family != "siren"),
            allow_extra=True,
        )
        summary_ok = summary_report.ok
        if summary_report.ok:
            messages.append("summary schema: OK")
        else:
            messages.append(f"summary schema: FAIL missing={list(summary_report.missing_keys)}")

        consistency_ok, consistency_messages = check_summary_consistency(summary, data)
        messages.extend(consistency_messages)
    else:
        messages.append("summary schema: WARN dynamics_summary.json/summary.json not found")

    ok = schema.ok and shape_report.ok and summary_ok and consistency_ok
    return ok, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate fitting-dynamics output directory")
    parser.add_argument("result_dir", type=str, help="Directory containing trajectory.npz")
    parser.add_argument(
        "--model_family",
        type=str,
        default=None,
        choices=["siren", "conditional", "liif", "lte"],
        help="Override inferred schema family",
    )
    parser.add_argument(
        "--allow_missing_summary",
        action="store_true",
        help="Treat missing dynamics_summary.json/summary.json as a warning instead of failure",
    )
    args = parser.parse_args()

    ok, messages = validate_result_dir(
        Path(args.result_dir),
        model_family=args.model_family,
        require_summary=not args.allow_missing_summary,
    )
    print(f"Result directory: {args.result_dir}")
    for msg in messages:
        print(f"- {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
