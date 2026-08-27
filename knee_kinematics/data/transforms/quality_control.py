"""Training-fitted quality control for segmented kinematic signals.

The adaptive segmenter already removes implausible intervals and
within-subject cycle-shape outliers.  This module adds population-level
quality control: it learns conservative per-DOF amplitude limits from the
training partition and rejects finite but clearly corrupted subjects before
normalization statistics are fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .gait_cycle import (
    AdaptiveGaitCycleSegmenter,
    GaitCycleSegmentationResult,
)


RAW_DOF_NAMES = (
    "varus_valgus",
    "internal_external_rotation",
    "flexion_extension",
    "anterior_posterior_translation",
    "superior_inferior_translation",
    "medial_lateral_translation",
)

BilateralSegmentation = tuple[
    GaitCycleSegmentationResult,
    GaitCycleSegmentationResult,
]


def segment_bilateral_signals(
    raw_data: np.ndarray | Any,
    segmentation_config: Mapping[str, Any],
) -> list[BilateralSegmentation]:
    """Segment every ``[12, T]`` subject without using labels or test data."""
    data = np.asarray(raw_data, dtype=np.float32)
    if data.ndim != 3 or data.shape[1] != 12:
        raise ValueError("raw_data must have shape [N, 12, T]")
    segmenter = AdaptiveGaitCycleSegmenter.from_config(segmentation_config)
    return [
        (
            segmenter.segment(subject[:6]),
            segmenter.segment(subject[6:]),
        )
        for subject in data
    ]


def subject_dof_scales(result: BilateralSegmentation) -> np.ndarray:
    """Return pooled dispersion for each raw-order DOF.

    Dispersion is computed across sides, cycles, and time.  In addition to
    within-cycle amplitude, this intentionally captures impossible baseline
    offsets between left and right signals.  Those offsets enter the global
    standardizer and therefore must not be hidden by a median-over-cycles
    statistic.
    """
    available = [
        side.cycles
        for side in result
        if side.cycle_count > 0
    ]
    if not available:
        return np.full(6, np.nan, dtype=np.float64)
    cycles = np.concatenate(available, axis=0).astype(np.float64, copy=False)
    return cycles.std(axis=(0, 2))


@dataclass(frozen=True)
class QualityControlDecision:
    """One subject-level quality-control decision."""

    accepted: bool
    reasons: tuple[str, ...]
    dof_scales: np.ndarray
    outlier_dof_indices: tuple[int, ...]


@dataclass(frozen=True)
class AdaptiveKinematicQualityControl:
    """Training-fitted conservative quality limits for adaptive cycles."""

    min_cycles_per_side: int
    robust_z_threshold: float
    min_upper_scale_factor: float
    scale_center: np.ndarray
    scale_sigma: np.ndarray
    upper_scale_limit: np.ndarray
    fitted_subjects: int

    @classmethod
    def fit(
        cls,
        results: Sequence[BilateralSegmentation],
        config: Mapping[str, Any] | None = None,
    ) -> "AdaptiveKinematicQualityControl":
        """Fit amplitude limits using segmented training subjects only."""
        values = dict(config or {})
        supported = {
            "enabled",
            "min_cycles_per_side",
            "robust_z_threshold",
            "min_upper_scale_factor",
            "min_reference_subjects",
        }
        unknown = set(values) - supported
        if unknown:
            raise KeyError(
                f"unsupported quality-control keys: {sorted(unknown)}"
            )
        min_cycles = int(values.get("min_cycles_per_side", 2))
        robust_z = float(values.get("robust_z_threshold", 6.0))
        minimum_factor = float(values.get("min_upper_scale_factor", 3.0))
        minimum_subjects = int(values.get("min_reference_subjects", 20))
        if min_cycles < 1:
            raise ValueError("min_cycles_per_side must be positive")
        if robust_z <= 0:
            raise ValueError("robust_z_threshold must be positive")
        if minimum_factor <= 1:
            raise ValueError("min_upper_scale_factor must exceed 1")
        if minimum_subjects < 2:
            raise ValueError("min_reference_subjects must be at least 2")

        candidate_scales = [
            subject_dof_scales(result)
            for result in results
            if all(side.cycle_count >= min_cycles for side in result)
        ]
        if len(candidate_scales) < minimum_subjects:
            raise ValueError(
                "too few training subjects passed cycle-count quality "
                f"control: {len(candidate_scales)} < {minimum_subjects}"
            )
        scales = np.stack(candidate_scales).astype(np.float64, copy=False)
        center = np.median(scales, axis=0)
        mad_sigma = 1.4826 * np.median(
            np.abs(scales - center[None, :]), axis=0
        )
        q25, q75 = np.quantile(scales, [0.25, 0.75], axis=0)
        iqr_sigma = (q75 - q25) / 1.349
        scale_sigma = np.maximum(mad_sigma, iqr_sigma)
        numerical_floor = np.maximum(np.abs(center) * 1e-6, 1e-8)
        scale_sigma = np.maximum(scale_sigma, numerical_floor)
        upper_limit = np.maximum(
            center + robust_z * scale_sigma,
            center * minimum_factor,
        )
        return cls(
            min_cycles_per_side=min_cycles,
            robust_z_threshold=robust_z,
            min_upper_scale_factor=minimum_factor,
            scale_center=center,
            scale_sigma=scale_sigma,
            upper_scale_limit=upper_limit,
            fitted_subjects=len(candidate_scales),
        )

    def evaluate(
        self,
        result: BilateralSegmentation,
    ) -> QualityControlDecision:
        """Reject insufficient-cycle or population-amplitude outliers."""
        reasons: list[str] = []
        side_names = ("left", "right")
        for side_name, side in zip(side_names, result):
            if side.cycle_count < self.min_cycles_per_side:
                reasons.append(
                    f"{side_name} retained {side.cycle_count} cycles; "
                    f"requires {self.min_cycles_per_side}"
                )
        scales = subject_dof_scales(result)
        nonfinite = np.flatnonzero(~np.isfinite(scales))
        if nonfinite.size:
            reasons.append(
                "non-finite segmented DOF scales: "
                + ", ".join(RAW_DOF_NAMES[index] for index in nonfinite)
            )
        outliers = np.flatnonzero(scales > self.upper_scale_limit)
        if outliers.size:
            reasons.append(
                "extreme segmented DOF dispersion: "
                + ", ".join(
                    f"{RAW_DOF_NAMES[index]}={scales[index]:.6g}"
                    f">{self.upper_scale_limit[index]:.6g}"
                    for index in outliers
                )
            )
        return QualityControlDecision(
            accepted=not reasons,
            reasons=tuple(reasons),
            dof_scales=scales,
            outlier_dof_indices=tuple(int(index) for index in outliers),
        )

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable fitted thresholds and provenance."""
        return {
            "enabled": True,
            "fit_partition": "training_only",
            "fit_stage": "after_adaptive_segmentation",
            "min_cycles_per_side": self.min_cycles_per_side,
            "robust_z_threshold": self.robust_z_threshold,
            "min_upper_scale_factor": self.min_upper_scale_factor,
            "fitted_subjects": self.fitted_subjects,
            "raw_dof_names": list(RAW_DOF_NAMES),
            "scale_center": self.scale_center.tolist(),
            "scale_sigma": self.scale_sigma.tolist(),
            "upper_scale_limit": self.upper_scale_limit.tolist(),
        }


__all__ = [
    "AdaptiveKinematicQualityControl",
    "BilateralSegmentation",
    "QualityControlDecision",
    "RAW_DOF_NAMES",
    "segment_bilateral_signals",
    "subject_dof_scales",
]
