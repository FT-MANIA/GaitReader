"""Data loading, adaptive gait segmentation, and collation APIs."""

from .builders import (
    build_data_loaders,
    build_repository_datasets,
    build_segmentation_report,
)
from .collate import collate_kinematic_subjects
from .loading import (
    GaitFeatureParseError,
    extract_subject_arrays,
    load_gait_data,
    parse_features_to_array,
    remove_nonfinite_subjects,
)
from .repository import (
    KinematicDOFStandardizer,
    KinematicSubjectSample,
    RepositoryKinematicDataset,
)
from .transforms import (
    AdaptiveGaitCycleSegmenter,
    AdaptiveKinematicQualityControl,
    BilateralSegmentation,
    GaitCycleSegmentationResult,
    QualityControlDecision,
    effective_cycle_length,
    segment_bilateral_signals,
    subject_dof_scales,
)

__all__ = [
    "AdaptiveGaitCycleSegmenter",
    "AdaptiveKinematicQualityControl",
    "BilateralSegmentation",
    "GaitCycleSegmentationResult",
    "GaitFeatureParseError",
    "KinematicDOFStandardizer",
    "KinematicSubjectSample",
    "QualityControlDecision",
    "RepositoryKinematicDataset",
    "build_data_loaders",
    "build_repository_datasets",
    "build_segmentation_report",
    "collate_kinematic_subjects",
    "extract_subject_arrays",
    "effective_cycle_length",
    "load_gait_data",
    "parse_features_to_array",
    "remove_nonfinite_subjects",
    "segment_bilateral_signals",
    "subject_dof_scales",
]
