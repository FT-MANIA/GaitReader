"""Dataset/DataLoader construction and segmentation provenance reports."""

from __future__ import annotations

import random
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from .collate import collate_kinematic_subjects
from .loading import load_gait_data
from .repository import build_repository_datasets


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed each DataLoader worker from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_data_loaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Map repository partitions to leakage-safe workflow stages.

    ``ssl_data`` drives SSL; ``dev_data`` drives fine-tuning; and
    ``dev_test_data`` plus ``ext_test_data`` are evaluation-only.
    """
    data = load_gait_data(config)
    dataset_config = config["dataset"]
    datasets = build_repository_datasets(
        data,
        cycle_length=dataset_config["cycle_length"],
        segmentation_config=dataset_config.get("segmentation"),
        quality_control_config=dataset_config.get("quality_control"),
        ssl_validation_fraction=dataset_config["ssl_validation_fraction"],
        downstream_validation_fraction=dataset_config.get(
            "downstream_validation_fraction", 0.15
        ),
        seed=config["training"]["seed"],
    )
    training_config = config["training"]
    generator = torch.Generator().manual_seed(training_config["seed"])
    common = {
        "batch_size": training_config["batch_size"],
        "num_workers": training_config["num_workers"],
        "collate_fn": collate_kinematic_subjects,
        "worker_init_fn": seed_dataloader_worker,
        "generator": generator,
    }
    return {
        "ssl_data": DataLoader(
            datasets["ssl_data"], shuffle=True, **common
        ),
        "ssl_validation_data": DataLoader(
            datasets["ssl_validation_data"], shuffle=False, **common
        ),
        "dev_data": DataLoader(
            datasets["dev_data"], shuffle=True, **common
        ),
        "dev_validation_data": DataLoader(
            datasets["dev_validation_data"], shuffle=False, **common
        ),
        "dev_test_data": DataLoader(
            datasets["dev_test_data"], shuffle=False, **common
        ),
        "ext_test_data": DataLoader(
            datasets["ext_test_data"], shuffle=False, **common
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
