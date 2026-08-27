"""Repository CSV parsing, quality control, and fixed data partitioning."""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)
LABEL_TO_INDEX = {"Healthy": 0, "ACLD": 1, "KOA": 2}


class GaitFeatureParseError(ValueError):
    """A serialized gait feature field could not be parsed safely."""


def parse_features_to_array(value: str | Any) -> np.ndarray:
    """Parse a serialized gait matrix into a floating NumPy array."""
    if not isinstance(value, str):
        return np.asarray(value, dtype=float)
    text = value.strip()
    json_text = re.sub(r"\b(?:nan|NaN|NAN)\b", "null", text)
    try:
        return np.asarray(json.loads(json_text), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError) as json_error:
        literal_text = re.sub(r"\b(?:nan|NaN|NAN)\b", "None", text)
        try:
            return np.asarray(ast.literal_eval(literal_text), dtype=float)
        except (SyntaxError, ValueError, TypeError) as error:
            json_detail = (
                f"{json_error.msg} at character {json_error.pos}"
                if isinstance(json_error, json.JSONDecodeError)
                else str(json_error)
            )
            raise GaitFeatureParseError(
                "could not parse gait features; "
                f"JSON parser: {json_detail}; "
                f"literal parser: {error}"
            ) from error


def extract_subject_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert paired leg rows into subject arrays shaped ``[N, 12, T]``."""
    required = {
        "person_id",
        "leg",
        "label",
        "features",
        "gender",
        "age",
        "bmi",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset CSV is missing columns: {sorted(missing)}")
    signals: list[np.ndarray] = []
    labels: list[int] = []
    demographics: list[list[float]] = []
    trace_info: list[dict[str, Any]] = []
    for person_id, subject_rows in frame.groupby("person_id", sort=False):
        side_rows = {
            str(row["leg"]).strip().lower(): row
            for _, row in subject_rows.iterrows()
        }
        if "left" not in side_rows or "right" not in side_rows:
            logger.warning(
                "Skipping subject %s because one side is missing", person_id
            )
            continue
        parsed_sides: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            row = side_rows[side]
            try:
                parsed_sides[side] = parse_features_to_array(
                    row["features"]
                )
            except GaitFeatureParseError as error:
                raise GaitFeatureParseError(
                    "invalid gait feature field for "
                    f"person_id={person_id!r}, "
                    f"source_file={row.get('source_file')!r}, "
                    f"original_id={row.get('original_id')!r}, "
                    f"leg={side!r}: {error}"
                ) from error
        left = parsed_sides["left"]
        right = parsed_sides["right"]
        if (
            left.ndim != 2
            or right.ndim != 2
            or left.shape[1] != 6
            or right.shape[1] != 6
        ):
            logger.warning(
                "Skipping subject %s because feature shape is invalid",
                person_id,
            )
            continue
        if left.shape[0] != right.shape[0]:
            logger.warning(
                "Skipping subject %s because side lengths differ", person_id
            )
            continue
        subject_labels = {str(value) for value in subject_rows["label"]}
        if "KOA" in subject_labels:
            label = LABEL_TO_INDEX["KOA"]
        elif "ACLD" in subject_labels:
            label = LABEL_TO_INDEX["ACLD"]
        else:
            label = LABEL_TO_INDEX["Healthy"]
        first = subject_rows.iloc[0]
        signals.append(
            np.concatenate([left.T, right.T], axis=0).astype(np.float32)
        )
        labels.append(label)
        demographics.append(
            [
                float(first.get("gender", 0)),
                float(first.get("age", 0)),
                float(first.get("bmi", 0)),
            ]
        )
        affected_side = (
            next(
                (
                    side
                    for side in ("left", "right")
                    if str(side_rows[side]["label"]) == "ACLD"
                ),
                None,
            )
            if "ACLD" in subject_labels
            else None
        )
        trace_info.append(
            {
                "person_id": person_id,
                "subject_id": (
                    f"{first.get('source_file')}:{first.get('original_id')}"
                ),
                "source_file": first.get("source_file"),
                "original_id": first.get("original_id"),
                "affected_side": affected_side,
            }
        )
    if not signals:
        raise ValueError(
            "no complete bilateral subjects were extracted from the CSV"
        )
    return (
        np.stack(signals),
        np.asarray(labels, dtype=np.int64),
        np.asarray(demographics, dtype=np.float32),
        np.asarray(trace_info, dtype=object),
    )


def remove_nonfinite_subjects(
    data: np.ndarray,
    labels: np.ndarray,
    demographics: np.ndarray,
    trace_info: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove subjects containing NaN/Inf while keeping arrays aligned."""
    valid = np.isfinite(data).all(axis=(1, 2))
    dropped = int((~valid).sum())
    if dropped:
        logger.warning(
            "Quality control removed %d non-finite subjects", dropped
        )
    return data[valid], labels[valid], demographics[valid], trace_info[valid]


def _dataset_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve explicit CSV paths with old-checkpoint fallback support."""
    dataset_config = config.get("dataset", config)
    required = ("ssl_csv", "dev_csv", "ext_test_csv")
    if all(key in dataset_config for key in required):
        return {
            "ssl": Path(dataset_config["ssl_csv"]),
            "dev": Path(dataset_config["dev_csv"]),
            "test": Path(dataset_config["ext_test_csv"]),
        }
    root = Path(config["dataset_path"])
    return {
        "ssl": root / "ssl_healthy_dataset.csv",
        "dev": root / "dev_dataset.csv",
        "test": root / "test_dataset.csv",
    }


def load_gait_data(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load SSL, fine-tuning, internal-test, and external-test partitions."""
    logger.info("Loading Knee Gait Kinematic Dataset")
    paths = _dataset_paths(config)
    missing_paths = [
        str(path) for path in paths.values() if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"dataset CSV files not found: {missing_paths}")
    extracted: dict[str, tuple[np.ndarray, ...]] = {}
    for name, path in paths.items():
        parse_error: GaitFeatureParseError | None = None
        for attempt in range(2):
            before = (path.stat().st_size, path.stat().st_mtime_ns)
            frame = pd.read_csv(path)
            after = (path.stat().st_size, path.stat().st_mtime_ns)
            if before != after:
                logger.warning(
                    "Dataset CSV changed while being read; retrying: %s",
                    path,
                )
                if attempt == 0:
                    continue
                raise RuntimeError(
                    f"dataset CSV changed repeatedly while being read: {path}"
                )
            try:
                extracted[name] = remove_nonfinite_subjects(
                    *extract_subject_arrays(frame)
                )
                parse_error = None
                break
            except GaitFeatureParseError as error:
                parse_error = error
                if attempt == 0:
                    logger.warning(
                        "Feature parsing failed; rereading CSV once in case "
                        "the first read observed incomplete content: %s",
                        path,
                    )
        if parse_error is not None:
            raise GaitFeatureParseError(
                f"failed to parse stable dataset CSV {path}: {parse_error}"
            ) from parse_error
    x_ssl, y_ssl, demo_ssl, trace_ssl = extracted["ssl"]
    x_dev, y_dev, demo_dev, trace_dev = extracted["dev"]
    x_test, y_test, demo_test, trace_test = extracted["test"]

    dataset_config = config.get("dataset", config)
    test_size = dataset_config["internal_test_size"]
    training_config = config.get("training", config)
    seed = training_config.get("seed", config.get("seed", 42))
    dev_indices, test_indices = train_test_split(
        np.arange(len(y_dev)),
        test_size=test_size,
        random_state=seed,
        stratify=y_dev,
    )
    return {
        "ssl_data": x_ssl,
        "ssl_label": y_ssl,
        "ssl_demo": demo_ssl,
        "ssl_trace_info": trace_ssl,
        "dev_data": x_dev[dev_indices],
        "dev_label": y_dev[dev_indices],
        "dev_demo": demo_dev[dev_indices],
        "dev_trace_info": trace_dev[dev_indices],
        "dev_test_data": x_dev[test_indices],
        "dev_test_label": y_dev[test_indices],
        "dev_test_demo": demo_dev[test_indices],
        "dev_test_trace_info": trace_dev[test_indices],
        "ext_test_data": x_test,
        "ext_test_label": y_test,
        "ext_test_demo": demo_test,
        "ext_test_trace_info": trace_test,
    }


__all__ = [
    "GaitFeatureParseError",
    "LABEL_TO_INDEX",
    "extract_subject_arrays",
    "load_gait_data",
    "parse_features_to_array",
    "remove_nonfinite_subjects",
]
