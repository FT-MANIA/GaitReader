"""Subject adapters, adaptive segmentation, split construction, and collation."""

from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms.gait_cycle import (
    AdaptiveGaitCycleSegmenter,
    GaitCycleSegmentationResult,
)
from .transforms.quality_control import (
    AdaptiveKinematicQualityControl,
    BilateralSegmentation,
    segment_bilateral_signals,
)


logger = logging.getLogger(__name__)


class KinematicSubjectSample(TypedDict, total=False):
    """One subject with variable-size, non-synchronized left/right cycle sets."""

    subject_id: str | int
    left_cycles: torch.Tensor
    right_cycles: torch.Tensor
    left_cycle_mask: torch.Tensor
    right_cycle_mask: torch.Tensor
    is_healthy: bool
    affected_side: str | None
    center_id: str | int | None
    split: str | None
    metadata: dict[str, Any]
    disease_label: int


class KinematicDOFStandardizer:
    """Per-DOF standardizer fitted on a training partition only."""

    MODEL_DOF_ORDER = torch.tensor([2, 0, 1, 3, 5, 4], dtype=torch.long)

    def __init__(
        self,
        mean: torch.Tensor,
        standard_deviation: torch.Tensor,
        *,
        fit_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if mean.shape != (6,) or standard_deviation.shape != (6,):
            raise ValueError("mean and standard_deviation must have shape [6]")
        self.mean = mean.detach().float().cpu()
        self.standard_deviation = standard_deviation.detach().float().cpu().clamp_min(1e-6)
        self.fit_metadata = dict(fit_metadata or {})

    @classmethod
    def fit(cls, bilateral_data: torch.Tensor | Any) -> "KinematicDOFStandardizer":
        """Fit from repository-order ``[N,12,T]`` training data."""
        data = torch.as_tensor(bilateral_data, dtype=torch.float32)
        if data.ndim != 3 or data.shape[1] != 12:
            raise ValueError("bilateral_data must have shape [N, 12, T]")
        sides = torch.cat([data[:, :6], data[:, 6:]], dim=0)
        sides = sides.index_select(1, cls.MODEL_DOF_ORDER)
        mean = sides.mean(dim=(0, 2))
        standard_deviation = sides.std(dim=(0, 2), unbiased=False)
        if not torch.isfinite(mean).all() or not torch.isfinite(standard_deviation).all():
            raise ValueError("training data contain non-finite standardization statistics")
        return cls(
            mean,
            standard_deviation,
            fit_metadata={
                "fit_stage": "raw_recordings",
                "quality_controlled": False,
            },
        )

    @classmethod
    def fit_segmented(
        cls,
        results: list[BilateralSegmentation],
        quality_control: AdaptiveKinematicQualityControl,
    ) -> "KinematicDOFStandardizer":
        """Fit mean/std from accepted, time-normalized training cycles."""
        retained: list[np.ndarray] = []
        accepted_subjects = 0
        for result in results:
            if not quality_control.evaluate(result).accepted:
                continue
            subject_cycles = [
                side.cycles
                for side in result
                if side.cycle_count > 0
            ]
            if not subject_cycles:
                continue
            retained.extend(subject_cycles)
            accepted_subjects += 1
        if not retained:
            raise ValueError(
                "quality control retained no segmented training cycles"
            )
        cycles = torch.as_tensor(
            np.concatenate(retained, axis=0),
            dtype=torch.float32,
        )
        cycles = cycles.index_select(1, cls.MODEL_DOF_ORDER)
        mean = cycles.mean(dim=(0, 2))
        standard_deviation = cycles.std(dim=(0, 2), unbiased=False)
        if not torch.isfinite(mean).all() or not torch.isfinite(
            standard_deviation
        ).all():
            raise ValueError(
                "quality-controlled cycles produced non-finite statistics"
            )
        return cls(
            mean,
            standard_deviation,
            fit_metadata={
                "fit_stage": "after_adaptive_segmentation_and_quality_control",
                "quality_controlled": True,
                "accepted_subjects": accepted_subjects,
                "retained_cycles": int(cycles.shape[0]),
            },
        )

    def transform(self, cycles: torch.Tensor) -> torch.Tensor:
        """Standardize model-order cycles ``[...,6,T]`` without changing shape."""
        mean = self.mean.to(device=cycles.device, dtype=cycles.dtype)
        std = self.standard_deviation.to(device=cycles.device, dtype=cycles.dtype)
        return (cycles - mean.reshape(*([1] * (cycles.ndim - 2)), 6, 1)) / std.reshape(
            *([1] * (cycles.ndim - 2)), 6, 1
        )

    def summary(self) -> dict[str, Any]:
        """Return serializable normalization statistics and fit provenance."""
        return {
            "dof_order": [
                "flexion_extension",
                "adduction_abduction",
                "internal_external_rotation",
                "anterior_posterior_translation",
                "medial_lateral_translation",
                "superior_inferior_translation",
            ],
            "mean": self.mean.tolist(),
            "standard_deviation": self.standard_deviation.tolist(),
            **self.fit_metadata,
        }


class RepositoryKinematicDataset(Dataset[KinematicSubjectSample]):
    """Adapt repository ``[N,12,T]`` arrays to subject-level gait-language cycles.

    Repository DOFs are explicitly reordered from ``[VV, IE, FE, AP, PD, ML]``
    to the model order ``[FE, VV, IE, AP, ML, SI]``. Adaptive mode detects
    cycles before this reorder and normalizes each retained cycle in time.
    Fixed-window mode remains available for old-checkpoint compatibility.
    Disease labels are retained only as downstream metadata and are never used
    by the gait-language tokenizer and sentence models.
    """

    def __init__(
        self,
        raw_data: torch.Tensor | Any,
        *,
        labels: torch.Tensor | Any | None,
        trace_info: Any | None,
        standardizer: KinematicDOFStandardizer,
        cycle_length: int = 200,
        segmentation_config: Mapping[str, Any] | None = None,
        quality_control: AdaptiveKinematicQualityControl | None = None,
        adaptive_results: list[BilateralSegmentation] | None = None,
        split: str | None = None,
    ) -> None:
        data = torch.as_tensor(raw_data, dtype=torch.float32)
        if data.ndim != 3 or data.shape[1] != 12:
            raise ValueError("raw_data must have shape [N, 12, T]")
        labels_tensor = (
            torch.as_tensor(labels, dtype=torch.long)
            if labels is not None
            else torch.zeros(data.shape[0], dtype=torch.long)
        )
        if labels_tensor.shape != (data.shape[0],):
            raise ValueError("labels must have shape [N]")

        segmentation = dict(segmentation_config or {})
        method = str(segmentation.get("method", "fixed_window"))
        if method not in {"fixed_window", "adaptive"}:
            raise ValueError(
                "segmentation method must be 'fixed_window' or 'adaptive'"
            )
        self.segmentation_method = method
        self._adaptive_cycles: list[
            tuple[
                GaitCycleSegmentationResult,
                GaitCycleSegmentationResult,
            ]
        ] | None = None
        self.excluded_subjects: list[dict[str, Any]] = []
        adaptive_summary: dict[str, Any] = {}
        selected_indices = list(range(data.shape[0]))
        if method == "fixed_window":
            if cycle_length <= 0 or data.shape[-1] < cycle_length:
                raise ValueError(
                    "cycle_length must be positive and no longer than the signal"
                )
            self.cycle_length = int(cycle_length)
            self.num_cycles: int | None = (
                data.shape[-1] // self.cycle_length
            )
        else:
            segmenter = AdaptiveGaitCycleSegmenter.from_config(segmentation)
            if adaptive_results is not None and len(adaptive_results) != data.shape[0]:
                raise ValueError(
                    "adaptive_results must contain one result per subject"
                )
            adaptive_cycles = []
            selected_indices = []
            left_failures = 0
            right_failures = 0
            total_rejected = 0
            left_cycle_counts: list[int] = []
            right_cycle_counts: list[int] = []
            for source_index in range(data.shape[0]):
                if adaptive_results is None:
                    subject = data[source_index].cpu().numpy()
                    left = segmenter.segment(subject[:6])
                    right = segmenter.segment(subject[6:])
                else:
                    left, right = adaptive_results[source_index]
                left_failures += int(left.cycle_count == 0)
                right_failures += int(right.cycle_count == 0)
                total_rejected += (
                    left.rejected_intervals + right.rejected_intervals
                )
                if left.cycle_count == 0 and right.cycle_count == 0:
                    trace = (
                        trace_info[source_index]
                        if trace_info is not None
                        else {}
                    )
                    subject_id = (
                        trace.get(
                            "subject_id",
                            trace.get("person_id", source_index),
                        )
                        if isinstance(trace, dict)
                        else source_index
                    )
                    self.excluded_subjects.append(
                        {
                            "source_index": source_index,
                            "subject_id": subject_id,
                            "disease_label": int(
                                labels_tensor[source_index].item()
                            ),
                            "left_failure_reason": left.failure_reason,
                            "right_failure_reason": right.failure_reason,
                        }
                    )
                    continue
                if quality_control is not None:
                    decision = quality_control.evaluate((left, right))
                    if not decision.accepted:
                        trace = (
                            trace_info[source_index]
                            if trace_info is not None
                            else {}
                        )
                        subject_id = (
                            trace.get(
                                "subject_id",
                                trace.get("person_id", source_index),
                            )
                            if isinstance(trace, dict)
                            else source_index
                        )
                        self.excluded_subjects.append(
                            {
                                "source_index": source_index,
                                "subject_id": subject_id,
                                "disease_label": int(
                                    labels_tensor[source_index].item()
                                ),
                                "quality_control_reasons": list(
                                    decision.reasons
                                ),
                                "raw_order_dof_scales": (
                                    decision.dof_scales.tolist()
                                ),
                                "outlier_dof_indices": list(
                                    decision.outlier_dof_indices
                                ),
                                "left_cycle_count": left.cycle_count,
                                "right_cycle_count": right.cycle_count,
                            }
                        )
                        continue
                selected_indices.append(source_index)
                adaptive_cycles.append((left, right))
                left_cycle_counts.append(left.cycle_count)
                right_cycle_counts.append(right.cycle_count)
            if not selected_indices:
                raise ValueError(
                    f"adaptive segmentation retained no subjects in {split}"
                )
            self._adaptive_cycles = adaptive_cycles
            self.cycle_length = segmenter.target_length
            self.num_cycles = None
            source_class_counts = torch.bincount(
                labels_tensor, minlength=3
            ).tolist()
            retained_class_counts = torch.bincount(
                labels_tensor[
                    torch.as_tensor(selected_indices, dtype=torch.long)
                ],
                minlength=3,
            ).tolist()
            adaptive_summary = {
                "left_side_failures": left_failures,
                "right_side_failures": right_failures,
                "rejected_intervals": total_rejected,
                "left_cycle_count": {
                    "min": min(left_cycle_counts),
                    "median": float(np.median(left_cycle_counts)),
                    "max": max(left_cycle_counts),
                },
                "right_cycle_count": {
                    "min": min(right_cycle_counts),
                    "median": float(np.median(right_cycle_counts)),
                    "max": max(right_cycle_counts),
                },
                "source_class_counts": source_class_counts,
                "retained_class_counts": retained_class_counts,
                "excluded_class_counts": [
                    source - retained
                    for source, retained in zip(
                        source_class_counts, retained_class_counts
                    )
                ],
                "quality_control": (
                    quality_control.summary()
                    if quality_control is not None
                    else {"enabled": False}
                ),
            }
            logger.info(
                "adaptive segmentation split=%s subjects=%d retained=%d "
                "excluded=%d left_failures=%d right_failures=%d "
                "rejected_cycles=%d",
                split,
                data.shape[0],
                len(selected_indices),
                data.shape[0] - len(selected_indices),
                left_failures,
                right_failures,
                total_rejected,
            )

        index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
        self.raw_data = data.index_select(0, index_tensor)
        self.labels = labels_tensor.index_select(0, index_tensor)
        if trace_info is None:
            self.trace_info = None
        elif isinstance(trace_info, np.ndarray):
            self.trace_info = trace_info[np.asarray(selected_indices)]
        else:
            self.trace_info = [trace_info[index] for index in selected_indices]
        self.standardizer = standardizer
        self.split = split
        self.segmentation_summary = {
            "method": method,
            "source_subjects": int(data.shape[0]),
            "retained_subjects": len(selected_indices),
            "excluded_subjects": int(data.shape[0] - len(selected_indices)),
            "excluded_subject_details": self.excluded_subjects,
            "target_length": self.cycle_length,
            "standardization": self.standardizer.summary(),
            **adaptive_summary,
        }

    def __len__(self) -> int:
        """Return the number of repository subjects."""
        return self.raw_data.shape[0]

    def _side_cycles(self, side: torch.Tensor) -> torch.Tensor:
        """Reorder and fixed-window one ``[6,T]`` side."""
        if self.num_cycles is None:
            raise RuntimeError(
                "_side_cycles is only available in fixed-window mode"
            )
        order = KinematicDOFStandardizer.MODEL_DOF_ORDER.to(side.device)
        side = side.index_select(0, order)
        usable = self.num_cycles * self.cycle_length
        cycles = side[:, :usable].reshape(6, self.num_cycles, self.cycle_length)
        cycles = cycles.permute(1, 0, 2).contiguous()
        return self.standardizer.transform(cycles)

    def _adaptive_side_cycles(self, cycles: np.ndarray) -> torch.Tensor:
        """Reorder normalized raw-order cycles and standardize them."""
        tensor = torch.as_tensor(cycles, dtype=torch.float32)
        order = KinematicDOFStandardizer.MODEL_DOF_ORDER
        tensor = tensor.index_select(1, order)
        return self.standardizer.transform(tensor)

    @staticmethod
    def _segmentation_metadata(
        result: GaitCycleSegmentationResult,
    ) -> dict[str, Any]:
        """Return serializable per-side segmentation provenance."""
        return {
            "cycle_count": result.cycle_count,
            "boundaries": result.boundaries.tolist(),
            "period_samples": result.period_samples,
            "quality_scores": result.quality_scores.tolist(),
            "rejected_intervals": result.rejected_intervals,
            "failure_reason": result.failure_reason,
        }

    def __getitem__(self, index: int) -> KinematicSubjectSample:
        """Return paired, independently segmented left/right cycle sets."""
        signal = self.raw_data[index]
        label = int(self.labels[index].item())
        trace = self.trace_info[index] if self.trace_info is not None else {}
        subject_id = (
            trace.get("subject_id", trace.get("person_id", index))
            if isinstance(trace, dict)
            else index
        )
        affected_side = trace.get("affected_side") if isinstance(trace, dict) else None
        if self._adaptive_cycles is None:
            left_cycles = self._side_cycles(signal[:6])
            right_cycles = self._side_cycles(signal[6:])
            segmentation_metadata = {
                "method": "fixed_window",
                "cycle_length": self.cycle_length,
            }
        else:
            left_result, right_result = self._adaptive_cycles[index]
            left_cycles = self._adaptive_side_cycles(left_result.cycles)
            right_cycles = self._adaptive_side_cycles(right_result.cycles)
            segmentation_metadata = {
                "method": "adaptive",
                "target_length": self.cycle_length,
                "left": self._segmentation_metadata(left_result),
                "right": self._segmentation_metadata(right_result),
            }
        return {
            "subject_id": subject_id,
            "left_cycles": left_cycles,
            "right_cycles": right_cycles,
            "is_healthy": label == 0,
            "affected_side": affected_side,
            "center_id": trace.get("source_file") if isinstance(trace, dict) else None,
            "disease_label": label,
            "metadata": {
                "trace_info": trace,
                "disease_label": label,
                "split": self.split,
                "segmentation": segmentation_metadata,
            },
        }


def build_repository_datasets(
    data: dict[str, Any],
    *,
    cycle_length: int = 200,
    segmentation_config: Mapping[str, Any] | None = None,
    quality_control_config: Mapping[str, Any] | None = None,
    ssl_validation_fraction: float = 0.1,
    downstream_validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, RepositoryKinematicDataset]:
    """Build leakage-safe SSL, fine-tuning, and two evaluation datasets.

    Normalization is fitted only on quality-controlled SSL healthy training
    cycles and then frozen for SSL, downstream, and both evaluation splits.
    Downstream quality-control limits may still be fitted on downstream train,
    but they never change the healthy normalization coordinate system.
    """
    required = {
        "ssl_data",
        "ssl_label",
        "dev_data",
        "dev_label",
        "dev_test_data",
        "dev_test_label",
        "ext_test_data",
        "ext_test_label",
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"Data is missing required partitions: {sorted(missing)}")
    if not 0.0 < ssl_validation_fraction < 1.0:
        raise ValueError("ssl_validation_fraction must lie in (0, 1)")
    if not 0.0 < downstream_validation_fraction < 1.0:
        raise ValueError("downstream_validation_fraction must lie in (0, 1)")
    ssl_size = len(data["ssl_data"])
    validation_size = max(2, round(ssl_size * ssl_validation_fraction))
    if ssl_size - validation_size < 2:
        raise ValueError("ssl_data must contain at least four subjects")
    permutation = torch.randperm(ssl_size, generator=torch.Generator().manual_seed(seed))
    validation_indices = permutation[:validation_size].numpy()
    training_indices = permutation[validation_size:].numpy()
    dev_labels = torch.as_tensor(data["dev_label"])
    dev_train_indices: list[int] = []
    dev_validation_indices: list[int] = []
    split_generator = torch.Generator().manual_seed(seed + 1)
    for label in dev_labels.unique(sorted=True):
        class_indices = (dev_labels == label).nonzero(as_tuple=False).flatten()
        class_indices = class_indices[
            torch.randperm(class_indices.numel(), generator=split_generator)
        ]
        validation_count = max(1, round(class_indices.numel() * downstream_validation_fraction))
        dev_validation_indices.extend(class_indices[:validation_count].tolist())
        dev_train_indices.extend(class_indices[validation_count:].tolist())
    dev_train_indices_array = np.asarray(sorted(dev_train_indices))
    dev_validation_indices_array = np.asarray(sorted(dev_validation_indices))
    segmentation = dict(segmentation_config or {})
    quality_values = dict(quality_control_config or {})
    use_adaptive_quality_control = (
        segmentation.get("method", "fixed_window") == "adaptive"
        and bool(quality_values.get("enabled", False))
    )
    ssl_quality_control: AdaptiveKinematicQualityControl | None = None
    downstream_quality_control: AdaptiveKinematicQualityControl | None = None
    ssl_training_results: list[BilateralSegmentation] | None = None
    downstream_training_results: list[BilateralSegmentation] | None = None
    if use_adaptive_quality_control:
        ssl_training_results = segment_bilateral_signals(
            data["ssl_data"][training_indices],
            segmentation,
        )
        ssl_quality_control = AdaptiveKinematicQualityControl.fit(
            ssl_training_results,
            quality_values,
        )
        ssl_standardizer = KinematicDOFStandardizer.fit_segmented(
            ssl_training_results,
            ssl_quality_control,
        )
        downstream_training_results = segment_bilateral_signals(
            data["dev_data"][dev_train_indices_array],
            segmentation,
        )
        downstream_quality_control = AdaptiveKinematicQualityControl.fit(
            downstream_training_results,
            quality_values,
        )
    else:
        ssl_standardizer = KinematicDOFStandardizer.fit(
            data["ssl_data"][training_indices]
        )
    downstream_standardizer = ssl_standardizer

    def dataset(
        partition: str,
        standardizer: KinematicDOFStandardizer,
        quality_control: AdaptiveKinematicQualityControl | None,
    ) -> RepositoryKinematicDataset:
        return RepositoryKinematicDataset(
            data[f"{partition}_data"],
            labels=data[f"{partition}_label"],
            trace_info=data.get(f"{partition}_trace_info"),
            standardizer=standardizer,
            cycle_length=cycle_length,
            segmentation_config=segmentation_config,
            quality_control=quality_control,
            split=partition,
        )

    ssl_train = RepositoryKinematicDataset(
        data["ssl_data"][training_indices],
        labels=data["ssl_label"][training_indices],
        trace_info=data.get("ssl_trace_info")[training_indices]
        if data.get("ssl_trace_info") is not None
        else None,
        standardizer=ssl_standardizer,
        cycle_length=cycle_length,
        segmentation_config=segmentation_config,
        quality_control=ssl_quality_control,
        adaptive_results=ssl_training_results,
        split="ssl_train",
    )
    ssl_validation = RepositoryKinematicDataset(
        data["ssl_data"][validation_indices],
        labels=data["ssl_label"][validation_indices],
        trace_info=data.get("ssl_trace_info")[validation_indices]
        if data.get("ssl_trace_info") is not None
        else None,
        standardizer=ssl_standardizer,
        cycle_length=cycle_length,
        segmentation_config=segmentation_config,
        quality_control=ssl_quality_control,
        split="ssl_validation",
    )
    dev_train = RepositoryKinematicDataset(
        data["dev_data"][dev_train_indices_array],
        labels=data["dev_label"][dev_train_indices_array],
        trace_info=data.get("dev_trace_info")[dev_train_indices_array]
        if data.get("dev_trace_info") is not None
        else None,
        standardizer=downstream_standardizer,
        cycle_length=cycle_length,
        segmentation_config=segmentation_config,
        quality_control=downstream_quality_control,
        adaptive_results=downstream_training_results,
        split="dev_train",
    )
    dev_validation = RepositoryKinematicDataset(
        data["dev_data"][dev_validation_indices_array],
        labels=data["dev_label"][dev_validation_indices_array],
        trace_info=data.get("dev_trace_info")[dev_validation_indices_array]
        if data.get("dev_trace_info") is not None
        else None,
        standardizer=downstream_standardizer,
        cycle_length=cycle_length,
        segmentation_config=segmentation_config,
        quality_control=downstream_quality_control,
        split="dev_validation",
    )
    return {
        "ssl_data": ssl_train,
        "ssl_validation_data": ssl_validation,
        "dev_data": dev_train,
        "dev_validation_data": dev_validation,
        "dev_test_data": dataset(
            "dev_test",
            downstream_standardizer,
            downstream_quality_control,
        ),
        "ext_test_data": dataset(
            "ext_test",
            downstream_standardizer,
            downstream_quality_control,
        ),
    }


def _validate_cycle_tensor(cycles: torch.Tensor, name: str) -> None:
    """Validate an unbatched cycle set ``[C, 6, T]``."""
    if not isinstance(cycles, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if cycles.ndim != 3 or cycles.shape[1] != 6:
        raise ValueError(f"{name} must have shape [C, 6, T]")
    if not torch.is_floating_point(cycles):
        raise TypeError(f"{name} must be floating point")


def collate_kinematic_subjects(
    samples: list[KinematicSubjectSample],
) -> dict[str, Any]:
    """Pad variable cycle counts without manufacturing missing-side observations.

    Cycle masks use ``True`` for a real cycle and ``False`` for cycle padding.
    Empty sides are represented by an all-false mask, never by copying the other
    side. All non-empty tensors in one batch must share ``[6, T]`` and dtype.
    """
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    empty_reference: torch.Tensor | None = None
    for sample in samples:
        for side in ("left_cycles", "right_cycles"):
            cycles = sample.get(side)
            if cycles is not None:
                _validate_cycle_tensor(cycles, side)
                if cycles.shape[0] > 0 and empty_reference is None:
                    empty_reference = cycles
    if empty_reference is None:
        raise ValueError("at least one real cycle is required to infer T and dtype")

    batch_size = len(samples)
    time_steps = empty_reference.shape[-1]
    dtype = empty_reference.dtype
    device = empty_reference.device
    left_counts = [int(sample.get("left_cycles", empty_reference[:0]).shape[0]) for sample in samples]
    right_counts = [int(sample.get("right_cycles", empty_reference[:0]).shape[0]) for sample in samples]
    left_max = max(left_counts, default=0)
    right_max = max(right_counts, default=0)

    left = torch.zeros(batch_size, left_max, 6, time_steps, dtype=dtype, device=device)
    right = torch.zeros(batch_size, right_max, 6, time_steps, dtype=dtype, device=device)
    left_mask = torch.zeros(batch_size, left_max, dtype=torch.bool, device=device)
    right_mask = torch.zeros(batch_size, right_max, dtype=torch.bool, device=device)

    for index, sample in enumerate(samples):
        for side_name, target, target_mask, count in (
            ("left_cycles", left, left_mask, left_counts[index]),
            ("right_cycles", right, right_mask, right_counts[index]),
        ):
            cycles = sample.get(side_name)
            if cycles is None or count == 0:
                continue
            if cycles.shape[1:] != (6, time_steps):
                raise ValueError(f"all {side_name} tensors must share shape [C, 6, {time_steps}]")
            if cycles.dtype != dtype or cycles.device != device:
                raise ValueError("all cycle tensors must share dtype and device")
            target[index, :count].copy_(cycles)
            source_mask = sample.get(side_name.replace("cycles", "cycle_mask"))
            if source_mask is None:
                target_mask[index, :count] = True
            else:
                if source_mask.dtype != torch.bool or source_mask.shape != (count,):
                    raise ValueError(f"{side_name.replace('cycles', 'cycle_mask')} must be BoolTensor[C]")
                target_mask[index, :count] = source_mask.to(device=device)

    return {
        "subject_id": [sample.get("subject_id", index) for index, sample in enumerate(samples)],
        "left_cycles": left,
        "right_cycles": right,
        "left_cycle_mask": left_mask,
        "right_cycle_mask": right_mask,
        "left_available": left_mask.any(dim=1),
        "right_available": right_mask.any(dim=1),
        "is_healthy": torch.tensor(
            [bool(sample.get("is_healthy", False)) for sample in samples],
            dtype=torch.bool,
            device=device,
        ),
        "affected_side": [sample.get("affected_side") for sample in samples],
        "affected_side_label": torch.tensor(
            [0 if sample.get("affected_side") == "left" else 1 for sample in samples],
            dtype=torch.long,
            device=device,
        ),
        "affected_side_valid_mask": torch.tensor(
            [sample.get("affected_side") in {"left", "right"} for sample in samples],
            dtype=torch.bool,
            device=device,
        ),
        "center_id": [sample.get("center_id") for sample in samples],
        "split": [sample.get("metadata", {}).get("split") for sample in samples],
        "metadata": [sample.get("metadata", {}) for sample in samples],
        "disease_label": torch.tensor(
            [int(sample.get("disease_label", 0)) for sample in samples],
            dtype=torch.long,
            device=device,
        ),
    }
