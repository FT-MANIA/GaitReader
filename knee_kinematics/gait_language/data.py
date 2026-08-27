"""Convert repository cycle batches into bilateral gait sentences."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _side_timing(
    metadata: list[dict[str, Any]],
    side: str,
    cycle_count: int,
    *,
    sampling_rate_hz: float,
    recording_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return duration, center, preceding interval, and quality per cycle."""
    timing = torch.zeros(
        len(metadata), cycle_count, 4, dtype=dtype, device=device
    )
    for subject_index, subject_metadata in enumerate(metadata):
        segmentation = subject_metadata.get("segmentation", {})
        side_metadata = segmentation.get(side, {})
        boundaries = side_metadata.get("boundaries", [])
        quality = side_metadata.get("quality_scores", [])
        usable = min(cycle_count, len(boundaries))
        if usable == 0:
            continue
        boundary_tensor = torch.as_tensor(
            boundaries[:usable], dtype=dtype, device=device
        )
        starts = boundary_tensor[:, 0]
        ends = boundary_tensor[:, 1]
        centers = (starts + ends) * 0.5
        intervals = torch.zeros_like(centers)
        if usable > 1:
            intervals[1:] = centers[1:] - centers[:-1]
            intervals[0] = intervals[1]
        else:
            intervals[0] = ends[0] - starts[0]
        timing[subject_index, :usable, 0] = (
            ends - starts
        ) / sampling_rate_hz
        timing[subject_index, :usable, 1] = centers / recording_length
        timing[subject_index, :usable, 2] = intervals / sampling_rate_hz
        if quality:
            quality_tensor = torch.as_tensor(
                quality[:usable], dtype=dtype, device=device
            )
            timing[subject_index, : quality_tensor.numel(), 3] = quality_tensor
    return timing


def build_language_batch(
    batch: dict[str, Any],
    *,
    sampling_rate_hz: float,
    recording_length: int,
) -> dict[str, Any]:
    """Create words ``[B,2,W,6,T]`` plus masks and timing metadata."""
    left = batch["left_cycles"]
    right = batch["right_cycles"]
    left_mask = batch["left_cycle_mask"]
    right_mask = batch["right_cycle_mask"]
    batch_size = left.shape[0]
    word_count = max(left.shape[1], right.shape[1])
    time_steps = left.shape[-1] if left.shape[1] else right.shape[-1]
    words = left.new_zeros(batch_size, 2, word_count, 6, time_steps)
    word_mask = torch.zeros(
        batch_size, 2, word_count, dtype=torch.bool, device=left.device
    )
    words[:, 0, : left.shape[1]] = left
    words[:, 1, : right.shape[1]] = right
    word_mask[:, 0, : left_mask.shape[1]] = left_mask
    word_mask[:, 1, : right_mask.shape[1]] = right_mask

    timing = left.new_zeros(batch_size, 2, word_count, 4)
    timing[:, 0] = _side_timing(
        batch["metadata"],
        "left",
        word_count,
        sampling_rate_hz=sampling_rate_hz,
        recording_length=recording_length,
        dtype=left.dtype,
        device=left.device,
    )
    timing[:, 1] = _side_timing(
        batch["metadata"],
        "right",
        word_count,
        sampling_rate_hz=sampling_rate_hz,
        recording_length=recording_length,
        dtype=left.dtype,
        device=left.device,
    )
    timing = timing * word_mask[..., None]
    return {
        "words": words,
        "word_mask": word_mask,
        "timing": timing,
        "subject_id": batch["subject_id"],
        "disease_label": batch["disease_label"],
        "affected_side_label": batch["affected_side_label"],
        "affected_side_valid_mask": batch["affected_side_valid_mask"],
    }


def move_language_batch(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Move tensor values in a language batch to one device."""
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def summarize_language_dataset(
    dataset: Any, *, sampling_rate_hz: float
) -> dict[str, Any]:
    """Summarize word counts, durations, quality, and bilateral timing."""
    counts = {"left": [], "right": []}
    durations = {"left": [], "right": []}
    quality = {"left": [], "right": []}
    nearest_bilateral_offsets = []
    for index in range(len(dataset)):
        sample = dataset[index]
        segmentation = sample["metadata"]["segmentation"]
        centers: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            side_metadata = segmentation[side]
            boundaries = np.asarray(side_metadata["boundaries"], dtype=float)
            counts[side].append(int(boundaries.shape[0]))
            if boundaries.size:
                durations[side].extend(
                    ((boundaries[:, 1] - boundaries[:, 0]) / sampling_rate_hz).tolist()
                )
                centers[side] = boundaries.mean(axis=1) / sampling_rate_hz
            else:
                centers[side] = np.empty(0)
            quality[side].extend(side_metadata["quality_scores"])
        if centers["left"].size and centers["right"].size:
            offsets = np.abs(
                centers["left"][:, None] - centers["right"][None, :]
            )
            nearest_bilateral_offsets.extend(offsets.min(axis=1).tolist())

    def distribution(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {
            "min": float(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "max": float(array.max()),
        }

    return {
        "subjects": len(dataset),
        "cycle_count": {
            side: distribution(values) for side, values in counts.items()
        },
        "duration_seconds": {
            side: distribution(values) for side, values in durations.items()
        },
        "quality_score": {
            side: distribution(values) for side, values in quality.items()
        },
        "nearest_bilateral_center_offset_seconds": distribution(
            nearest_bilateral_offsets
        ),
    }


__all__ = [
    "build_language_batch",
    "move_language_batch",
    "summarize_language_dataset",
]
