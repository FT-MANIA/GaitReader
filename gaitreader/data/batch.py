"""Convert repository cycle batches into bilateral gait sentences."""

from __future__ import annotations

from typing import Any

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
