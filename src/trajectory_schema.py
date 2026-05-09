"""
Trajectory output schema validation utilities.

This module is intentionally non-invasive: it documents and validates the
current trajectory.npz conventions used by experiments/Phase1_FittingDynamics
without changing existing experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


SIREN_REQUIRED_KEYS = {
    "full_snapshots",
    "snapshot_steps",
    "losses",
    "psnrs",
    "freq_ratios",
    "target_spectrum",
}

SIREN_OPTIONAL_KEYS = {
    "full_snapshots_aligned",
    "grad_norms",
}

CONDITIONAL_REQUIRED_KEYS = {
    "full_snapshots",
    "enc_snapshots",
    "dec_snapshots",
    "snapshot_steps",
    "losses",
    "psnrs",
    "freq_ratios",
    "target_spectrum",
}

CONDITIONAL_OPTIONAL_KEYS = {
    "dec_snapshots_aligned",
    "model_type",
}

SUMMARY_COMMON_KEYS = {
    "model_type",
    "image",
    "seed",
    "n_params",
    "total_steps",
    "n_snapshots",
    "final_psnr",
    "final_loss",
    "snapshot_steps",
}

SUMMARY_CONDITIONAL_KEYS = {
    "mode",
    "sr_scale",
    "lr_size",
    "hr_size",
    "n_encoder_params",
    "n_decoder_params",
}


@dataclass(frozen=True)
class SchemaValidationResult:
    """Result returned by schema validation helpers."""

    ok: bool
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def raise_for_error(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.ok:
            raise ValueError(
                "Schema validation failed: "
                f"missing={list(self.missing_keys)}, "
                f"unexpected={list(self.unexpected_keys)}"
            )


@dataclass(frozen=True)
class TrajectoryShapeReport:
    """Lightweight consistency report for trajectory arrays.

    This does not validate scientific correctness. It only checks structural
    assumptions shared by existing fitting-dynamics outputs.
    """

    ok: bool
    errors: tuple[str, ...]

    def raise_for_error(self) -> None:
        if not self.ok:
            raise ValueError("Trajectory shape validation failed: " + "; ".join(self.errors))


def _key_set(obj: Mapping[str, object] | Iterable[str]) -> set[str]:
    if isinstance(obj, Mapping):
        return set(obj.keys())
    return set(obj)


def _shape_of(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except TypeError:
        return None


def _length_of(value: object) -> int | None:
    shape = _shape_of(value)
    if shape is not None and len(shape) > 0:
        return shape[0]
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def validate_keys(
    keys_or_mapping: Mapping[str, object] | Iterable[str],
    required: Iterable[str],
    optional: Iterable[str] = (),
    allow_extra: bool = True,
) -> SchemaValidationResult:
    """Validate that required keys are present.

    Args:
        keys_or_mapping: Mapping or iterable of keys to validate.
        required: Required keys.
        optional: Optional known keys.
        allow_extra: If False, keys outside required/optional are reported.

    Returns:
        SchemaValidationResult with missing and unexpected keys.
    """
    keys = _key_set(keys_or_mapping)
    required_set = set(required)
    optional_set = set(optional)
    missing = tuple(sorted(required_set - keys))
    if allow_extra:
        unexpected: tuple[str, ...] = ()
    else:
        unexpected = tuple(sorted(keys - required_set - optional_set))
    return SchemaValidationResult(
        ok=(len(missing) == 0 and len(unexpected) == 0),
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def validate_trajectory_schema(
    keys_or_mapping: Mapping[str, object] | Iterable[str],
    model_family: str,
    allow_extra: bool = True,
) -> SchemaValidationResult:
    """Validate a trajectory.npz schema by model family.

    Args:
        keys_or_mapping: Mapping or iterable of NPZ keys.
        model_family: 'siren' or 'conditional'. Conditional covers LIIF/LTE/
            pretrained LIIF/LIIF-EQ trajectory files.
        allow_extra: Whether to allow extra keys.
    """
    family = model_family.lower()
    if family == "siren":
        return validate_keys(
            keys_or_mapping,
            required=SIREN_REQUIRED_KEYS,
            optional=SIREN_OPTIONAL_KEYS,
            allow_extra=allow_extra,
        )
    if family in {"conditional", "liif", "lte", "pretrained_liif", "liif_eq"}:
        return validate_keys(
            keys_or_mapping,
            required=CONDITIONAL_REQUIRED_KEYS,
            optional=CONDITIONAL_OPTIONAL_KEYS,
            allow_extra=allow_extra,
        )
    raise ValueError(f"Unknown model_family: {model_family}")


def validate_summary_schema(
    keys_or_mapping: Mapping[str, object] | Iterable[str],
    conditional: bool = False,
    allow_extra: bool = True,
) -> SchemaValidationResult:
    """Validate dynamics_summary.json keys."""
    required = set(SUMMARY_COMMON_KEYS)
    if conditional:
        required |= SUMMARY_CONDITIONAL_KEYS
    return validate_keys(keys_or_mapping, required=required, allow_extra=allow_extra)


def validate_trajectory_shapes(
    trajectory: Mapping[str, object],
    model_family: str,
) -> TrajectoryShapeReport:
    """Validate lightweight shape consistency for a trajectory mapping.

    Checks:
        - required keys exist;
        - snapshot arrays agree with snapshot_steps length;
        - aligned snapshots match their raw snapshot shapes when present;
        - psnrs/freq_ratios lengths match either snapshot count or snapshot
          count minus one, because current scripts include step 0 in
          snapshot_steps but usually append metrics only after training begins.

    This helper intentionally accepts both existing conventions rather than
    forcing a new one.
    """
    schema = validate_trajectory_schema(trajectory, model_family=model_family)
    errors = list(schema.missing_keys)
    if errors:
        return TrajectoryShapeReport(ok=False, errors=tuple(f"missing key: {k}" for k in errors))

    snapshot_len = _length_of(trajectory["snapshot_steps"])
    if snapshot_len is None:
        errors.append("snapshot_steps has no length")
        return TrajectoryShapeReport(ok=False, errors=tuple(errors))

    snapshot_keys = ["full_snapshots"]
    family = model_family.lower()
    if family != "siren":
        snapshot_keys.extend(["enc_snapshots", "dec_snapshots"])

    for key in snapshot_keys:
        shape = _shape_of(trajectory[key])
        if shape is None or len(shape) < 2:
            errors.append(f"{key} must be at least 2D")
            continue
        if shape[0] != snapshot_len:
            errors.append(
                f"{key} first dimension {shape[0]} != snapshot_steps length {snapshot_len}"
            )

    aligned_pairs = [
        ("full_snapshots", "full_snapshots_aligned"),
        ("dec_snapshots", "dec_snapshots_aligned"),
    ]
    for raw_key, aligned_key in aligned_pairs:
        if raw_key in trajectory and aligned_key in trajectory:
            raw_shape = _shape_of(trajectory[raw_key])
            aligned_shape = _shape_of(trajectory[aligned_key])
            if raw_shape != aligned_shape:
                errors.append(f"{aligned_key} shape {aligned_shape} != {raw_key} shape {raw_shape}")

    for metric_key in ["psnrs", "freq_ratios"]:
        metric_len = _length_of(trajectory[metric_key])
        if metric_len not in {snapshot_len, max(snapshot_len - 1, 0)}:
            errors.append(
                f"{metric_key} length {metric_len} is not compatible with "
                f"snapshot_steps length {snapshot_len}"
            )

    loss_len = _length_of(trajectory["losses"])
    if loss_len is None:
        errors.append("losses has no length")

    return TrajectoryShapeReport(ok=(len(errors) == 0), errors=tuple(errors))
