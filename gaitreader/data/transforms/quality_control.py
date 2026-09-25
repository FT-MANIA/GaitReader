"""Subject quality control using a fixed minimum cycle count on each side."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .gait_cycle import (
    GaitParser,
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
    segmenter = GaitParser.from_config(segmentation_config)
    return [
        (
            segmenter.segment(subject[:6]),
            segmenter.segment(subject[6:]),
        )
        for subject in data
    ]





@dataclass(frozen=True)
class QualityControlDecision:
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AdaptiveKinematicQualityControl:
    """Fixed bilateral cycle-count rule, independent of training subjects."""

    min_cycles_per_side: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AdaptiveKinematicQualityControl":
        return cls(min_cycles_per_side=int(config.get("min_cycles_per_side", 2)))

    def evaluate(self, result: BilateralSegmentation) -> QualityControlDecision:
        reasons = tuple(
            f"{name} retained {side.cycle_count} cycles; requires {self.min_cycles_per_side}"
            for name, side in zip(("left", "right"), result)
            if side.cycle_count < self.min_cycles_per_side
        )
        return QualityControlDecision(accepted=not reasons, reasons=reasons)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "method": "bilateral_minimum_cycle_count",
            "min_cycles_per_side": self.min_cycles_per_side,
        }


__all__ = [
    "AdaptiveKinematicQualityControl",
    "BilateralSegmentation",
    "QualityControlDecision",
    "RAW_DOF_NAMES",
    "segment_bilateral_signals",
]
