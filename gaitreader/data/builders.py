"""Dataset/DataLoader construction and segmentation provenance reports."""

from __future__ import annotations

import random
from typing import Any, Mapping

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from .collate import collate_kinematic_subjects
from .loading import load_gait_data
from .repository import build_repository_datasets


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed each DataLoader worker from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class LabeledSubjectSubset(Subset):
    """Selected subjects with labels exposed for downstream class weights."""

    @property
    def labels(self):
        return self.dataset.labels[self.indices]


def build_data_loaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Map repository partitions to leakage-safe workflow stages.

    ``ssl_data`` drives SSL; ``dev_data`` drives fine-tuning; and
    ``dev_test_data`` plus ``ext_test_data`` are evaluation-only.
    """
    data = load_gait_data(config)
    dataset_config = config["dataset"]
    segmentation = dict(dataset_config.get("segmentation") or {})
    # Window size belongs to the downstream word conversion, not the adaptive
    # reference segmenter. Remove it in BOTH arms before constructing datasets.
    fixed_window_length = segmentation.pop("fixed_window_length", dataset_config["cycle_length"])
    fixed_windows = segmentation.get("method") == "fixed_window"
    if fixed_windows:
        # Hold cohort eligibility and healthy-train normalization fixed across
        # the segmentation ablation; only the resulting word boundaries differ.
        segmentation["method"] = "adaptive"
    datasets = build_repository_datasets(
        data,
        cycle_length=dataset_config["cycle_length"],
        segmentation_config=segmentation,
        quality_control_config=dataset_config.get("quality_control"),
        ssl_validation_fraction=dataset_config["ssl_validation_fraction"],
        downstream_validation_fraction=dataset_config.get(
            "downstream_validation_fraction", 0.15
        ),
        seed=config["training"]["seed"],
    )
    if fixed_windows:
        for dataset in datasets.values():
            dataset.use_fixed_windows(
                fixed_window_length, dataset_config["cycle_length"])
    training_config = config["training"]
    # Subsample only after fitting preprocessing on the full training partition.
    # A fixed class permutation makes the fractional subsets nested.
    fraction = dataset_config.get("downstream_label_fraction", 1.0)
    subset_seed = dataset_config.get("downstream_label_seed")
    if subset_seed is None:
        subset_seed = training_config["seed"]
    full_train = datasets["dev_data"]
    subset_generator = torch.Generator().manual_seed(subset_seed)
    selected = []
    for label in full_train.labels.unique(sorted=True):
        indices = (full_train.labels == label).nonzero(as_tuple=False).flatten()
        indices = indices[torch.randperm(len(indices), generator=subset_generator)]
        count = max(1, round(len(indices) * fraction))
        selected.extend(indices[:count].tolist())
    selected.sort()
    train_subset = LabeledSubjectSubset(full_train, selected)
    train_subset.label_subset_summary = {
        "requested_fraction": fraction,
        "subset_seed": subset_seed,
        "full_training_subjects": len(full_train),
        "selected_subjects": len(selected),
        "class_counts": torch.bincount(train_subset.labels, minlength=3).tolist(),
        "indices_in_full_training_dataset": selected,
        "subjects": [
            {"subject_id": str(full_train[index]["subject_id"]),
             "disease_label": int(full_train.labels[index])}
            for index in selected
        ],
    }
    train_subset.segmentation_summary = {
        **full_train.segmentation_summary,
        "label_subset": train_subset.label_subset_summary,
    }
    datasets["dev_data"] = train_subset
    common = {
        "batch_size": training_config["batch_size"],
        "num_workers": training_config["num_workers"],
        "collate_fn": collate_kinematic_subjects,
        "worker_init_fn": seed_dataloader_worker,
    }
    return {
        "ssl_data": DataLoader(
            datasets["ssl_data"], shuffle=True, generator=torch.Generator().manual_seed(training_config["seed"]), **common
        ),
        "ssl_validation_data": DataLoader(
            datasets["ssl_validation_data"], shuffle=False, generator=torch.Generator().manual_seed(training_config["seed"]), **common
        ),
        "dev_data": DataLoader(
            datasets["dev_data"], shuffle=True, generator=torch.Generator().manual_seed(training_config["seed"]), **common
        ),
        "dev_validation_data": DataLoader(
            datasets["dev_validation_data"], shuffle=False, generator=torch.Generator().manual_seed(training_config["seed"]), **common
        ),
        "dev_test_data": DataLoader(
            datasets["dev_test_data"], shuffle=False, generator=torch.Generator().manual_seed(training_config["seed"]), **common
        ),
        "ext_test_data": DataLoader(
            datasets["ext_test_data"], shuffle=False, generator=torch.Generator().manual_seed(training_config["seed"]), **common
        ),
    }


def build_segmentation_report(
    loaders: Mapping[str, DataLoader],
) -> dict[str, Any]:
    """Collect serializable partition-level segmentation summaries."""
    return {
        name: dict(
            getattr(
                loader.dataset,
                "segmentation_summary",
                {"method": "unknown"},
            )
        )
        for name, loader in loaders.items()
    }


__all__ = [
    "build_data_loaders",
    "build_repository_datasets",
    "build_segmentation_report",
]
