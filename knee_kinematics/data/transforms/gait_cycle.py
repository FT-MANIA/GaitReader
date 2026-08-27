"""Adaptive gait-cycle detection and fixed-length time normalization.

The repository CSV stores one continuous 600-sample recording per side in the
raw DOF order ``[VV, IE, FE, AP, PD/SI, ML]``.  This module estimates cycle
boundaries from the flexion/extension trace, rejects implausible intervals, and
linearly normalizes every retained cycle to a common number of samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.signal import find_peaks, savgol_filter


@dataclass(frozen=True)
class GaitCycleSegmentationResult:
    """Cycles plus diagnostics for one unilateral continuous recording."""

    cycles: np.ndarray
    boundaries: np.ndarray
    period_samples: float | None
    quality_scores: np.ndarray
    rejected_intervals: int
    failure_reason: str | None = None

    @property
    def cycle_count(self) -> int:
        """Return the number of retained cycles."""
        return int(self.cycles.shape[0])


class AdaptiveGaitCycleSegmenter:
    """Detect flexion-anchored gait cycles and normalize them in time.

    Detection operates on raw repository-order unilateral data ``[6, T]``.
    Flexion minima are used as reproducible kinematic cycle anchors.  These are
    kinematic proxy events rather than force-plate heel-strike annotations.
    """

    def __init__(
        self,
        *,
        sampling_rate_hz: float = 60.0,
        target_length: int = 100,
        reference_dof_index: int = 2,
        min_cycle_seconds: float = 0.4,
        max_cycle_seconds: float = 4.0,
        smoothing_window_seconds: float = 0.15,
        smoothing_polyorder: int = 3,
        peak_prominence_fraction: float = 0.15,
        peak_distance_fraction: float = 0.55,
        period_min_correlation: float = 0.05,
        period_relative_min: float = 0.55,
        period_relative_max: float = 1.60,
        min_cycles: int = 1,
        similarity_filter: bool = True,
        min_cycle_similarity: float = -1.0,
        similarity_mad_scale: float = 3.0,
    ) -> None:
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if target_length < 2:
            raise ValueError("target_length must be at least 2")
        if not 0 <= reference_dof_index < 6:
            raise ValueError("reference_dof_index must lie in [0, 5]")
        if not 0 < min_cycle_seconds < max_cycle_seconds:
            raise ValueError(
                "cycle duration must satisfy 0 < min_cycle_seconds < "
                "max_cycle_seconds"
            )
        if smoothing_window_seconds <= 0 or smoothing_polyorder < 1:
            raise ValueError("invalid smoothing configuration")
        if not 0 < peak_prominence_fraction < 1:
            raise ValueError("peak_prominence_fraction must lie in (0, 1)")
        if not 0 < peak_distance_fraction <= 1:
            raise ValueError("peak_distance_fraction must lie in (0, 1]")
        if not -1 <= period_min_correlation <= 1:
            raise ValueError("period_min_correlation must lie in [-1, 1]")
        if not 0 < period_relative_min < period_relative_max:
            raise ValueError("invalid period-relative duration limits")
        if min_cycles < 1:
            raise ValueError("min_cycles must be positive")
        if not -1 <= min_cycle_similarity <= 1:
            raise ValueError("min_cycle_similarity must lie in [-1, 1]")
        if similarity_mad_scale < 0:
            raise ValueError("similarity_mad_scale must be non-negative")

        self.sampling_rate_hz = float(sampling_rate_hz)
        self.target_length = int(target_length)
        self.reference_dof_index = int(reference_dof_index)
        self.min_cycle_seconds = float(min_cycle_seconds)
        self.max_cycle_seconds = float(max_cycle_seconds)
        self.smoothing_window_seconds = float(smoothing_window_seconds)
        self.smoothing_polyorder = int(smoothing_polyorder)
        self.peak_prominence_fraction = float(peak_prominence_fraction)
        self.peak_distance_fraction = float(peak_distance_fraction)
        self.period_min_correlation = float(period_min_correlation)
        self.period_relative_min = float(period_relative_min)
        self.period_relative_max = float(period_relative_max)
        self.min_cycles = int(min_cycles)
        self.similarity_filter = bool(similarity_filter)
        self.min_cycle_similarity = float(min_cycle_similarity)
        self.similarity_mad_scale = float(similarity_mad_scale)

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any]
    ) -> "AdaptiveGaitCycleSegmenter":
        """Construct a segmenter from a ``dataset.segmentation`` mapping."""
        supported = {
            "sampling_rate_hz",
            "target_length",
            "reference_dof_index",
            "min_cycle_seconds",
            "max_cycle_seconds",
            "smoothing_window_seconds",
            "smoothing_polyorder",
            "peak_prominence_fraction",
            "peak_distance_fraction",
            "period_min_correlation",
            "period_relative_min",
            "period_relative_max",
            "min_cycles",
            "similarity_filter",
            "min_cycle_similarity",
            "similarity_mad_scale",
        }
        unknown = set(config) - supported - {"method"}
        if unknown:
            raise KeyError(
                "unsupported adaptive segmentation keys: "
                f"{sorted(unknown)}"
            )
        return cls(
            **{key: config[key] for key in supported if key in config}
        )

    def _empty(self, reason: str) -> GaitCycleSegmentationResult:
        """Return a shape-stable failed result."""
        return GaitCycleSegmentationResult(
            cycles=np.empty(
                (0, 6, self.target_length), dtype=np.float32
            ),
            boundaries=np.empty((0, 2), dtype=np.int64),
            period_samples=None,
            quality_scores=np.empty((0,), dtype=np.float32),
            rejected_intervals=0,
            failure_reason=reason,
        )

    def _smooth(self, signal: np.ndarray) -> np.ndarray:
        """Apply a length-safe Savitzky–Golay smoother."""
        desired = max(
            self.smoothing_polyorder + 2,
            int(round(self.smoothing_window_seconds * self.sampling_rate_hz)),
        )
        if desired % 2 == 0:
            desired += 1
        maximum = signal.size if signal.size % 2 == 1 else signal.size - 1
        window = min(desired, maximum)
        minimum = self.smoothing_polyorder + 2
        if minimum % 2 == 0:
            minimum += 1
        if window < minimum:
            return signal.copy()
        return savgol_filter(
            signal,
            window_length=window,
            polyorder=self.smoothing_polyorder,
            mode="interp",
        )

    def _estimate_period(self, signal: np.ndarray) -> float | None:
        """Estimate period from autocorrelation, corrected by swing spacing."""
        centered = signal - signal.mean()
        energy = float(np.dot(centered, centered))
        if not np.isfinite(energy) or energy <= np.finfo(float).eps:
            return None
        correlation = np.correlate(centered, centered, mode="full")[
            signal.size - 1 :
        ]
        correlation = correlation / max(float(correlation[0]), 1e-12)
        min_lag = max(
            2, int(round(self.sampling_rate_hz * self.min_cycle_seconds))
        )
        max_lag = min(
            signal.size - 2,
            int(round(self.sampling_rate_hz * self.max_cycle_seconds)),
        )
        if min_lag >= max_lag:
            return None
        region = correlation[min_lag : max_lag + 1]
        candidates, _ = find_peaks(
            region,
            prominence=max(0.01, self.period_min_correlation / 2.0),
        )
        autocorrelation_period: float | None
        if candidates.size:
            lags = candidates + min_lag
            values = correlation[lags]
            acceptable = values >= max(
                self.period_min_correlation, float(values.max()) * 0.85
            )
            if acceptable.any():
                autocorrelation_period = float(lags[acceptable][0])
            else:
                autocorrelation_period = float(
                    lags[int(np.argmax(values))]
                )
        else:
            best = int(np.argmax(region)) + min_lag
            autocorrelation_period = (
                float(best)
                if correlation[best] >= self.period_min_correlation
                else None
            )

        dynamic_range = float(np.ptp(signal))
        swing_peaks, _ = find_peaks(
            signal,
            distance=max(2, int(round(min_lag * 0.8))),
            prominence=max(
                np.finfo(float).eps,
                dynamic_range * self.peak_prominence_fraction,
            ),
        )
        intervals = np.diff(swing_peaks)
        intervals = intervals[
            (intervals >= min_lag) & (intervals <= max_lag)
        ]
        peak_period = (
            float(np.median(intervals)) if intervals.size else None
        )
        if peak_period is None:
            return autocorrelation_period
        if autocorrelation_period is None:
            return peak_period
        at_search_edge = (
            autocorrelation_period <= min_lag * 1.10
            or autocorrelation_period >= max_lag * 0.95
        )
        ratio = peak_period / autocorrelation_period
        if at_search_edge or 0.75 <= ratio <= 1.33:
            return peak_period
        return autocorrelation_period

    def _flexion_boundaries(
        self, signal: np.ndarray, period: float
    ) -> np.ndarray:
        """Locate extension minima surrounding consecutive swing peaks."""
        dynamic_range = float(np.ptp(signal))
        prominence = max(
            np.finfo(float).eps,
            dynamic_range * self.peak_prominence_fraction,
        )
        distance = max(2, int(round(period * self.peak_distance_fraction)))
        swing_peaks, _ = find_peaks(
            signal, distance=distance, prominence=prominence
        )

        boundaries: list[int] = []
        if swing_peaks.size >= 2:
            leading_start = max(0, int(round(swing_peaks[0] - period)))
            if swing_peaks[0] - leading_start >= 2:
                leading = signal[leading_start : swing_peaks[0] + 1]
                leading_minimum = int(np.argmin(leading))
                if 0 < leading_minimum < leading.size - 1:
                    boundaries.append(leading_start + leading_minimum)
            for left, right in zip(swing_peaks[:-1], swing_peaks[1:]):
                if right - left >= 2:
                    interval = signal[left : right + 1]
                    boundaries.append(int(left + np.argmin(interval)))
            trailing_end = min(
                signal.size - 1, int(round(swing_peaks[-1] + period))
            )
            if trailing_end - swing_peaks[-1] >= 2:
                trailing = signal[swing_peaks[-1] : trailing_end + 1]
                trailing_minimum = int(np.argmin(trailing))
                if 0 < trailing_minimum < trailing.size - 1:
                    boundaries.append(
                        int(swing_peaks[-1] + trailing_minimum)
                    )

        direct_minima, _ = find_peaks(
            -signal,
            distance=distance,
            prominence=max(
                np.finfo(float).eps,
                prominence * 0.5,
            ),
        )
        if len(boundaries) < 2:
            boundaries.extend(int(value) for value in direct_minima)
        elif direct_minima.size:
            tolerance = max(2, int(round(period * 0.20)))
            for minimum in direct_minima:
                if all(
                    abs(int(minimum) - boundary) > tolerance
                    for boundary in boundaries
                ):
                    boundaries.append(int(minimum))

        if not boundaries:
            return np.empty((0,), dtype=np.int64)
        return np.asarray(sorted(set(boundaries)), dtype=np.int64)

    def _intervals(
        self, boundaries: np.ndarray, period: float
    ) -> tuple[np.ndarray, int]:
        """Create physiologically and period-consistent intervals."""
        if boundaries.size < 2:
            return np.empty((0, 2), dtype=np.int64), 0
        absolute_min = self.sampling_rate_hz * self.min_cycle_seconds
        absolute_max = self.sampling_rate_hz * self.max_cycle_seconds
        minimum = max(absolute_min, period * self.period_relative_min)
        maximum = min(absolute_max, period * self.period_relative_max)
        intervals: list[tuple[int, int]] = []
        rejected = 0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            duration = int(end - start)
            if minimum <= duration <= maximum:
                intervals.append((int(start), int(end)))
            else:
                rejected += 1
        return np.asarray(intervals, dtype=np.int64).reshape(-1, 2), rejected

    def _interpolate(self, cycle: np.ndarray) -> np.ndarray:
        """Linearly normalize ``[6, L]`` to ``[6, target_length]``."""
        source = np.linspace(0.0, 1.0, cycle.shape[1], dtype=np.float64)
        target = np.linspace(
            0.0, 1.0, self.target_length, dtype=np.float64
        )
        normalized = np.stack(
            [np.interp(target, source, channel) for channel in cycle],
            axis=0,
        )
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def _similarity_scores(cycles: np.ndarray) -> np.ndarray:
        """Return robust median pairwise shape correlation per cycle."""
        centered = cycles - cycles.mean(axis=-1, keepdims=True)
        scale = centered.std(axis=-1, keepdims=True)
        normalized = centered / np.maximum(scale, 1e-6)
        flat = normalized.reshape(normalized.shape[0], -1)
        correlation = np.corrcoef(flat)
        correlation = np.nan_to_num(
            correlation, nan=-1.0, posinf=-1.0, neginf=-1.0
        )
        np.fill_diagonal(correlation, np.nan)
        return np.nanmedian(correlation, axis=1).astype(np.float32)

    def _filter_by_similarity(
        self,
        cycles: np.ndarray,
        boundaries: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Remove correlation outliers without forcing a fixed drop ratio."""
        count = cycles.shape[0]
        if count <= 2 or not self.similarity_filter:
            scores = (
                np.ones(count, dtype=np.float32)
                if count <= 1
                else self._similarity_scores(cycles)
            )
            return cycles, boundaries, scores, 0
        scores = self._similarity_scores(cycles)
        center = float(np.median(scores))
        mad = float(np.median(np.abs(scores - center)))
        robust_floor = center - (
            self.similarity_mad_scale * 1.4826 * mad
        )
        threshold = max(self.min_cycle_similarity, robust_floor)
        keep = np.flatnonzero(scores >= threshold)
        return (
            cycles[keep],
            boundaries[keep],
            scores[keep],
            int(count - keep.size),
        )

    def segment(
        self, leg_data: np.ndarray | Any
    ) -> GaitCycleSegmentationResult:
        """Segment raw-order unilateral data shaped ``[6, T]``."""
        data = np.asarray(leg_data, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] != 6:
            raise ValueError("leg_data must have shape [6, T]")
        if data.shape[1] < 3:
            return self._empty("recording is too short")
        if not np.isfinite(data).all():
            return self._empty("recording contains NaN or Inf")
        flexion = data[self.reference_dof_index]
        dynamic_range = float(np.ptp(flexion))
        if dynamic_range <= max(1e-8, abs(float(flexion.mean())) * 1e-8):
            return self._empty("reference DOF has negligible dynamic range")
        smoothed = self._smooth(flexion)
        period = self._estimate_period(smoothed)
        if period is None:
            return self._empty("no reliable gait period was found")
        boundaries = self._flexion_boundaries(smoothed, period)
        intervals, rejected = self._intervals(boundaries, period)
        if intervals.shape[0] < self.min_cycles:
            result = self._empty(
                f"fewer than {self.min_cycles} valid cycles were found"
            )
            return GaitCycleSegmentationResult(
                **{
                    **result.__dict__,
                    "period_samples": period,
                    "rejected_intervals": rejected,
                }
            )
        cycles = np.stack(
            [
                self._interpolate(data[:, start:end])
                for start, end in intervals
            ],
            axis=0,
        )
        cycles, intervals, scores, filtered = self._filter_by_similarity(
            cycles, intervals
        )
        rejected += filtered
        if cycles.shape[0] < self.min_cycles:
            result = self._empty(
                f"fewer than {self.min_cycles} cycles passed quality control"
            )
            return GaitCycleSegmentationResult(
                **{
                    **result.__dict__,
                    "period_samples": period,
                    "rejected_intervals": rejected,
                }
            )
        return GaitCycleSegmentationResult(
            cycles=cycles.astype(np.float32, copy=False),
            boundaries=intervals,
            period_samples=period,
            quality_scores=scores,
            rejected_intervals=rejected,
            failure_reason=None,
        )


def effective_cycle_length(dataset_config: Mapping[str, Any]) -> int:
    """Return the tensor length produced by the configured segmentation."""
    segmentation = dataset_config.get("segmentation", {})
    method = segmentation.get("method", "fixed_window")
    if method == "adaptive":
        return int(segmentation.get("target_length", 100))
    if method == "fixed_window":
        return int(dataset_config["cycle_length"])
    raise ValueError(
        "dataset.segmentation.method must be 'adaptive' or 'fixed_window'"
    )


__all__ = [
    "AdaptiveGaitCycleSegmenter",
    "GaitCycleSegmentationResult",
    "effective_cycle_length",
]
