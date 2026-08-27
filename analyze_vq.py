"""Offline diagnostics for a trained gait-language VQ tokenizer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import run
from knee_kinematics.data.builders import build_data_loaders
from knee_kinematics.gait_language.data import (
    build_language_batch,
    move_language_batch,
)


DOF_NAMES = (
    "flexion_extension",
    "adduction_abduction",
    "internal_external_rotation",
    "anterior_posterior_translation",
    "medial_lateral_translation",
    "superior_inferior_translation",
)


def get_args() -> argparse.Namespace:
    """Define checkpoint, data split, similarity, and output parameters."""
    parser = argparse.ArgumentParser("Offline VQ diagnostics")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Analyze only this run; otherwise analyze every VQ run",
    )
    parser.add_argument(
        "--results-dir",
        default="Results/gait_language",
        help="Directory recursively searched for VQ experiment runs",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--args-json", default=None)
    parser.add_argument(
        "--split",
        choices=(
            "ssl_data",
            "ssl_validation_data",
            "dev_data",
            "dev_validation_data",
            "dev_test_data",
            "ext_test_data",
        ),
        default="ssl_validation_data",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--waveform-similarity-threshold",
        type=float,
        default=0.90,
        help="Pearson correlation defining two cycles as similar",
    )
    parser.add_argument(
        "--near-code-similarity-threshold",
        type=float,
        default=0.90,
        help="Prototype cosine similarity defining two codes as near",
    )
    parser.add_argument(
        "--prototype-duplicate-threshold",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--decoded-waveform-duplicate-threshold",
        type=float,
        default=0.98,
    )
    return parser.parse_args()


def _vq_run_directories(results_dir: Path) -> list[Path]:
    """Return every experiment directory containing a VQ checkpoint."""
    return sorted(
        checkpoint.parent
        for checkpoint in results_dir.rglob("best_vq.pt")
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _empty_pair_statistics() -> dict[str, float]:
    return {
        "candidate_pair_count": 0.0,
        "similar_pair_count": 0.0,
        "same_code_count": 0.0,
        "near_code_count": 0.0,
        "waveform_correlation_sum": 0.0,
        "prototype_similarity_sum": 0.0,
    }


def _merge_pair_statistics(
    target: dict[str, float], source: dict[str, float]
) -> None:
    for key, value in source.items():
        target[key] += value


def _cycle_pair_statistics(
    first_waveforms: torch.Tensor,
    first_codes: torch.Tensor,
    second_waveforms: torch.Tensor,
    second_codes: torch.Tensor,
    prototype_similarity: torch.Tensor,
    *,
    waveform_threshold: float,
    near_code_threshold: float,
    same_collection: bool,
) -> dict[str, float]:
    """Compare waveform similarity with assigned-prototype similarity."""
    first_normalized = F.normalize(
        first_waveforms - first_waveforms.mean(dim=-1, keepdim=True),
        dim=-1,
    )
    second_normalized = F.normalize(
        second_waveforms - second_waveforms.mean(dim=-1, keepdim=True),
        dim=-1,
    )
    correlations = first_normalized @ second_normalized.transpose(0, 1)
    if same_collection:
        pair_indices = torch.triu_indices(
            first_waveforms.shape[0],
            second_waveforms.shape[0],
            offset=1,
        )
        first_indices, second_indices = pair_indices
    else:
        first_indices, second_indices = torch.meshgrid(
            torch.arange(first_waveforms.shape[0]),
            torch.arange(second_waveforms.shape[0]),
            indexing="ij",
        )
        first_indices = first_indices.reshape(-1)
        second_indices = second_indices.reshape(-1)
    pair_correlations = correlations[first_indices, second_indices]
    similar = pair_correlations >= waveform_threshold
    selected_first_codes = first_codes[first_indices[similar]]
    selected_second_codes = second_codes[second_indices[similar]]
    selected_prototype_similarity = prototype_similarity[
        selected_first_codes, selected_second_codes
    ]
    return {
        "candidate_pair_count": float(pair_correlations.numel()),
        "similar_pair_count": float(similar.sum()),
        "same_code_count": float(
            (selected_first_codes == selected_second_codes).sum()
        ),
        "near_code_count": float(
            (
                selected_prototype_similarity >= near_code_threshold
            ).sum()
        ),
        "waveform_correlation_sum": float(pair_correlations[similar].sum()),
        "prototype_similarity_sum": float(
            selected_prototype_similarity.sum()
        ),
    }


def _pair_statistics_row(
    statistics: dict[str, float],
    *,
    subject_id: str,
    dof_index: int,
    scope: str,
) -> dict[str, Any]:
    similar_count = statistics["similar_pair_count"]
    return {
        "subject_id": subject_id,
        "dof_index": dof_index,
        "dof_name": DOF_NAMES[dof_index],
        "scope": scope,
        "candidate_pair_count": int(statistics["candidate_pair_count"]),
        "similar_pair_count": int(similar_count),
        "same_code_rate": (
            statistics["same_code_count"] / similar_count
            if similar_count
            else np.nan
        ),
        "near_code_rate": (
            statistics["near_code_count"] / similar_count
            if similar_count
            else np.nan
        ),
        "mean_similar_waveform_correlation": (
            statistics["waveform_correlation_sum"] / similar_count
            if similar_count
            else np.nan
        ),
        "mean_assigned_prototype_similarity": (
            statistics["prototype_similarity_sum"] / similar_count
            if similar_count
            else np.nan
        ),
    }


def _correlation_from_sums(
    count: float,
    x_sum: float,
    y_sum: float,
    x_square_sum: float,
    y_square_sum: float,
    product_sum: float,
) -> float:
    numerator = count * product_sum - x_sum * y_sum
    denominator = math.sqrt(
        max(count * x_square_sum - x_sum**2, 0.0)
        * max(count * y_square_sum - y_sum**2, 0.0)
    )
    return numerator / max(denominator, 1e-12)


def _prototype_diagnostics(
    tokenizer: torch.nn.Module,
    *,
    embedding_threshold: float,
    waveform_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], np.ndarray]:
    embeddings = F.normalize(
        tokenizer.codebook.embedding.detach().float().cpu(), dim=-1
    )
    decoded = tokenizer.decode_codebook().detach().float().cpu()
    pair_rows = []
    summary_rows = []
    for dof_index, dof_name in enumerate(DOF_NAMES):
        embedding_similarity = embeddings[dof_index] @ embeddings[
            dof_index
        ].transpose(0, 1)
        centered = decoded[dof_index] - decoded[dof_index].mean(
            dim=-1, keepdim=True
        )
        normalized_waveforms = F.normalize(centered, dim=-1)
        waveform_similarity = (
            normalized_waveforms @ normalized_waveforms.transpose(0, 1)
        )
        pairs = torch.triu_indices(
            embeddings.shape[1], embeddings.shape[1], offset=1
        )
        first_codes, second_codes = pairs
        code_similarity = embedding_similarity[first_codes, second_codes]
        decoded_similarity = waveform_similarity[first_codes, second_codes]
        waveform_rmse = (
            decoded[dof_index, first_codes]
            - decoded[dof_index, second_codes]
        ).square().mean(dim=-1).sqrt()
        embedding_near = code_similarity >= embedding_threshold
        waveform_near = decoded_similarity >= waveform_threshold
        joint_duplicate = embedding_near & waveform_near
        duplicate_codes = torch.unique(
            torch.cat(
                [
                    first_codes[joint_duplicate],
                    second_codes[joint_duplicate],
                ]
            )
        )
        nearest_similarity = embedding_similarity.masked_fill(
            torch.eye(embeddings.shape[1], dtype=torch.bool), -1.0
        ).max(dim=1).values
        for pair_index in range(first_codes.numel()):
            pair_rows.append(
                {
                    "dof_index": dof_index,
                    "dof_name": dof_name,
                    "first_code": int(first_codes[pair_index]),
                    "second_code": int(second_codes[pair_index]),
                    "embedding_cosine_similarity": float(
                        code_similarity[pair_index]
                    ),
                    "embedding_euclidean_distance": math.sqrt(
                        max(
                            2.0
                            - 2.0 * float(code_similarity[pair_index]),
                            0.0,
                        )
                    ),
                    "decoded_waveform_correlation": float(
                        decoded_similarity[pair_index]
                    ),
                    "decoded_waveform_rmse": float(
                        waveform_rmse[pair_index]
                    ),
                    "embedding_near": bool(embedding_near[pair_index]),
                    "decoded_waveform_near": bool(
                        waveform_near[pair_index]
                    ),
                    "joint_duplicate": bool(joint_duplicate[pair_index]),
                }
            )
        total_pairs = first_codes.numel()
        summary_rows.append(
            {
                "dof_index": dof_index,
                "dof_name": dof_name,
                "prototype_count": embeddings.shape[1],
                "prototype_pair_count": total_pairs,
                "embedding_near_pair_count": int(embedding_near.sum()),
                "embedding_near_pair_ratio": float(
                    embedding_near.float().mean()
                ),
                "decoded_near_pair_count": int(waveform_near.sum()),
                "decoded_near_pair_ratio": float(
                    waveform_near.float().mean()
                ),
                "joint_duplicate_pair_count": int(
                    joint_duplicate.sum()
                ),
                "joint_duplicate_pair_ratio": float(
                    joint_duplicate.float().mean()
                ),
                "prototype_in_joint_duplicate_count": int(
                    duplicate_codes.numel()
                ),
                "prototype_in_joint_duplicate_ratio": float(
                    duplicate_codes.numel() / embeddings.shape[1]
                ),
                "maximum_off_diagonal_similarity": float(
                    nearest_similarity.max()
                ),
                "mean_nearest_prototype_similarity": float(
                    nearest_similarity.mean()
                ),
            }
        )
    return pd.DataFrame(pair_rows), summary_rows, decoded.numpy()


def analyze_vq(
    tokenizer: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    sampling_rate_hz: float,
    recording_length: int,
    waveform_similarity_threshold: float,
    near_code_similarity_threshold: float,
) -> dict[str, Any]:
    """Collect usage, waveform, reconstruction, and consistency statistics."""
    codebook_size = tokenizer.codebook.codebook_size
    word_length = tokenizer.word_decoder.word_length
    code_counts = torch.zeros(6, codebook_size, dtype=torch.int64)
    waveform_sum = torch.zeros(
        6, codebook_size, word_length, dtype=torch.float64
    )
    waveform_square_sum = torch.zeros_like(waveform_sum)
    reconstruction_sse = torch.zeros(6, dtype=torch.float64)
    velocity_sse = torch.zeros(6, dtype=torch.float64)
    point_count = torch.zeros(6, dtype=torch.float64)
    velocity_point_count = torch.zeros(6, dtype=torch.float64)
    original_sum = torch.zeros(6, dtype=torch.float64)
    reconstructed_sum = torch.zeros(6, dtype=torch.float64)
    original_square_sum = torch.zeros(6, dtype=torch.float64)
    reconstructed_square_sum = torch.zeros(6, dtype=torch.float64)
    product_sum = torch.zeros(6, dtype=torch.float64)
    cycle_correlation_sum = torch.zeros(6, dtype=torch.float64)
    cycle_correlation_square_sum = torch.zeros(6, dtype=torch.float64)
    cycle_count = torch.zeros(6, dtype=torch.float64)
    prototype_similarity = (
        F.normalize(tokenizer.codebook.embedding.detach().float().cpu(), dim=-1)
        @ F.normalize(
            tokenizer.codebook.embedding.detach().float().cpu(), dim=-1
        ).transpose(1, 2)
    )
    subject_rows = []
    aggregate_pairs = {
        scope: [_empty_pair_statistics() for _ in range(6)]
        for scope in ("within_side", "across_sides")
    }

    tokenizer.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            language_batch = build_language_batch(
                raw_batch,
                sampling_rate_hz=sampling_rate_hz,
                recording_length=recording_length,
            )
            device_batch = move_language_batch(language_batch, device)
            output = tokenizer(
                device_batch["words"],
                device_batch["word_mask"],
                device_batch["timing"],
            )
            words = language_batch["words"].float().cpu()
            word_mask = language_batch["word_mask"].cpu()
            indices = output["indices"].cpu()
            reconstructed = output["reconstructed"].float().cpu()

            for dof_index in range(6):
                original = words[..., dof_index, :][word_mask]
                prediction = reconstructed[..., dof_index, :][word_mask]
                assignments = indices[..., dof_index][word_mask]
                counts = torch.bincount(
                    assignments, minlength=codebook_size
                )
                code_counts[dof_index] += counts
                waveform_sum[dof_index].index_add_(
                    0, assignments, original.double()
                )
                waveform_square_sum[dof_index].index_add_(
                    0, assignments, original.double().square()
                )
                error = prediction.double() - original.double()
                velocity_error = prediction.double().diff(dim=-1) - (
                    original.double().diff(dim=-1)
                )
                reconstruction_sse[dof_index] += error.square().sum()
                velocity_sse[dof_index] += velocity_error.square().sum()
                point_count[dof_index] += original.numel()
                velocity_point_count[dof_index] += velocity_error.numel()
                original_sum[dof_index] += original.double().sum()
                reconstructed_sum[dof_index] += prediction.double().sum()
                original_square_sum[dof_index] += (
                    original.double().square().sum()
                )
                reconstructed_square_sum[dof_index] += (
                    prediction.double().square().sum()
                )
                product_sum[dof_index] += (
                    original.double() * prediction.double()
                ).sum()
                original_centered = original - original.mean(
                    dim=-1, keepdim=True
                )
                prediction_centered = prediction - prediction.mean(
                    dim=-1, keepdim=True
                )
                correlations = (
                    F.normalize(original_centered, dim=-1)
                    * F.normalize(prediction_centered, dim=-1)
                ).sum(dim=-1)
                cycle_correlation_sum[dof_index] += correlations.double().sum()
                cycle_correlation_square_sum[dof_index] += (
                    correlations.double().square().sum()
                )
                cycle_count[dof_index] += correlations.numel()

            for subject_index, subject_id in enumerate(
                language_batch["subject_id"]
            ):
                for dof_index in range(6):
                    within_statistics = _empty_pair_statistics()
                    for side_index in range(2):
                        valid = word_mask[subject_index, side_index]
                        side_statistics = _cycle_pair_statistics(
                            words[
                                subject_index,
                                side_index,
                                valid,
                                dof_index,
                            ],
                            indices[
                                subject_index,
                                side_index,
                                valid,
                                dof_index,
                            ],
                            words[
                                subject_index,
                                side_index,
                                valid,
                                dof_index,
                            ],
                            indices[
                                subject_index,
                                side_index,
                                valid,
                                dof_index,
                            ],
                            prototype_similarity[dof_index],
                            waveform_threshold=waveform_similarity_threshold,
                            near_code_threshold=near_code_similarity_threshold,
                            same_collection=True,
                        )
                        _merge_pair_statistics(
                            within_statistics, side_statistics
                        )
                    left_valid = word_mask[subject_index, 0]
                    right_valid = word_mask[subject_index, 1]
                    across_statistics = _cycle_pair_statistics(
                        words[subject_index, 0, left_valid, dof_index],
                        indices[subject_index, 0, left_valid, dof_index],
                        words[subject_index, 1, right_valid, dof_index],
                        indices[subject_index, 1, right_valid, dof_index],
                        prototype_similarity[dof_index],
                        waveform_threshold=waveform_similarity_threshold,
                        near_code_threshold=near_code_similarity_threshold,
                        same_collection=False,
                    )
                    for scope, statistics in (
                        ("within_side", within_statistics),
                        ("across_sides", across_statistics),
                    ):
                        _merge_pair_statistics(
                            aggregate_pairs[scope][dof_index], statistics
                        )
                        subject_rows.append(
                            _pair_statistics_row(
                                statistics,
                                subject_id=str(subject_id),
                                dof_index=dof_index,
                                scope=scope,
                            )
                        )

    count_denominator = code_counts.clamp_min(1)[..., None].double()
    waveform_mean = waveform_sum / count_denominator
    waveform_variance = (
        waveform_square_sum / count_denominator - waveform_mean.square()
    ).clamp_min(0.0)
    unused = code_counts == 0
    waveform_mean[unused] = torch.nan
    waveform_variance[unused] = torch.nan

    usage_rows = []
    reconstruction_rows = []
    dof_summary = []
    consistency_summary = []
    for dof_index, dof_name in enumerate(DOF_NAMES):
        counts = code_counts[dof_index]
        probabilities = counts.double() / counts.sum()
        nonzero = probabilities > 0
        perplexity = float(
            (-(probabilities[nonzero] * probabilities[nonzero].log()).sum()).exp()
        )
        active_codes = int(nonzero.sum())
        for code_index in range(codebook_size):
            usage_rows.append(
                {
                    "dof_index": dof_index,
                    "dof_name": dof_name,
                    "code_index": code_index,
                    "assignment_count": int(counts[code_index]),
                    "assignment_probability": float(
                        probabilities[code_index]
                    ),
                    "active": bool(nonzero[code_index]),
                    "mean_pointwise_waveform_variance": float(
                        waveform_variance[dof_index, code_index].mean()
                    ),
                    "maximum_pointwise_waveform_variance": float(
                        waveform_variance[dof_index, code_index].max()
                    ),
                }
            )
        point_total = float(point_count[dof_index])
        cycles = float(cycle_count[dof_index])
        pooled_correlation = _correlation_from_sums(
            point_total,
            float(original_sum[dof_index]),
            float(reconstructed_sum[dof_index]),
            float(original_square_sum[dof_index]),
            float(reconstructed_square_sum[dof_index]),
            float(product_sum[dof_index]),
        )
        mean_cycle_correlation = float(
            cycle_correlation_sum[dof_index] / cycles
        )
        cycle_correlation_variance = max(
            float(cycle_correlation_square_sum[dof_index] / cycles)
            - mean_cycle_correlation**2,
            0.0,
        )
        reconstruction_row = {
            "dof_index": dof_index,
            "dof_name": dof_name,
            "cycle_count": int(cycles),
            "rmse": math.sqrt(
                float(reconstruction_sse[dof_index] / point_count[dof_index])
            ),
            "pooled_pearson_correlation": pooled_correlation,
            "mean_cycle_pearson_correlation": mean_cycle_correlation,
            "std_cycle_pearson_correlation": math.sqrt(
                cycle_correlation_variance
            ),
            "velocity_rmse": math.sqrt(
                float(
                    velocity_sse[dof_index]
                    / velocity_point_count[dof_index]
                )
            ),
        }
        reconstruction_rows.append(reconstruction_row)
        dof_summary.append(
            {
                "dof_index": dof_index,
                "dof_name": dof_name,
                "assignment_count": int(counts.sum()),
                "active_code_count": active_codes,
                "active_code_ratio": active_codes / codebook_size,
                "perplexity": perplexity,
                **{
                    key: value
                    for key, value in reconstruction_row.items()
                    if key not in {"dof_index", "dof_name"}
                },
            }
        )
        for scope in ("within_side", "across_sides"):
            consistency_summary.append(
                _pair_statistics_row(
                    aggregate_pairs[scope][dof_index],
                    subject_id="ALL",
                    dof_index=dof_index,
                    scope=scope,
                )
            )

    return {
        "dof_summary": dof_summary,
        "consistency_summary": consistency_summary,
        "code_usage": pd.DataFrame(usage_rows),
        "reconstruction": pd.DataFrame(reconstruction_rows),
        "subject_consistency": pd.DataFrame(subject_rows),
        "waveform_mean": waveform_mean.numpy(),
        "waveform_variance": waveform_variance.numpy(),
        "code_counts": code_counts.numpy(),
    }


def _analyze_run(
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    *,
    use_path_overrides: bool,
) -> bool:
    """Analyze one experiment and return whether new results were written."""
    checkpoint_path = (
        Path(args.checkpoint)
        if use_path_overrides and args.checkpoint
        else run_dir / "best_vq.pt"
    )
    args_path = (
        Path(args.args_json)
        if use_path_overrides and args.args_json
        else run_dir / "args.json"
    )
    output_dir = (
        Path(args.output_dir)
        if use_path_overrides and args.output_dir
        else run_dir / f"vq_analysis_{args.split}"
    )
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(f"Skipping analyzed run: {run_dir}")
        return False
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_args = argparse.Namespace(
        **json.loads(args_path.read_text(encoding="utf-8"))
    )
    if args.batch_size is not None:
        experiment_args.batch_size = args.batch_size
    if args.num_workers is not None:
        experiment_args.num_workers = args.num_workers
    loaders = build_data_loaders(run._data_config(experiment_args))
    tokenizer = run._build_vq(experiment_args).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    tokenizer.load_state_dict(checkpoint["model"])

    analysis = analyze_vq(
        tokenizer,
        loaders[args.split],
        device,
        sampling_rate_hz=experiment_args.sampling_rate_hz,
        recording_length=experiment_args.recording_length,
        waveform_similarity_threshold=args.waveform_similarity_threshold,
        near_code_similarity_threshold=args.near_code_similarity_threshold,
    )
    prototype_pairs, prototype_summary, decoded_prototypes = (
        _prototype_diagnostics(
            tokenizer,
            embedding_threshold=args.prototype_duplicate_threshold,
            waveform_threshold=args.decoded_waveform_duplicate_threshold,
        )
    )

    analysis["code_usage"].to_csv(
        output_dir / "code_usage_and_variance.csv", index=False
    )
    analysis["reconstruction"].to_csv(
        output_dir / "dof_reconstruction_metrics.csv", index=False
    )
    analysis["subject_consistency"].to_csv(
        output_dir / "subject_cycle_consistency.csv", index=False
    )
    pd.DataFrame(analysis["consistency_summary"]).to_csv(
        output_dir / "dof_cycle_consistency_summary.csv", index=False
    )
    prototype_pairs.to_csv(
        output_dir / "prototype_pair_similarity.csv", index=False
    )
    pd.DataFrame(prototype_summary).to_csv(
        output_dir / "prototype_duplicate_summary.csv", index=False
    )
    np.savez_compressed(
        output_dir / "code_waveform_statistics.npz",
        counts=analysis["code_counts"],
        mean_waveforms=analysis["waveform_mean"],
        waveform_variances=analysis["waveform_variance"],
        decoded_prototypes=decoded_prototypes,
        dof_names=np.asarray(DOF_NAMES),
    )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "data_split": args.split,
        "waveform_similarity_threshold": (
            args.waveform_similarity_threshold
        ),
        "near_code_similarity_threshold": (
            args.near_code_similarity_threshold
        ),
        "prototype_duplicate_threshold": (
            args.prototype_duplicate_threshold
        ),
        "decoded_waveform_duplicate_threshold": (
            args.decoded_waveform_duplicate_threshold
        ),
        "dof_summary": analysis["dof_summary"],
        "cycle_consistency_summary": analysis[
            "consistency_summary"
        ],
        "prototype_duplicate_summary": prototype_summary,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"VQ checkpoint: {checkpoint_path}")
    print(f"Analysis split: {args.split}")
    print(f"Analysis output: {output_dir}")
    for dof in analysis["dof_summary"]:
        print(
            f"DOF {dof['dof_index']} {dof['dof_name']}: "
            f"active={dof['active_code_count']}/{experiment_args.codebook_size} "
            f"perplexity={dof['perplexity']:.2f} "
            f"rmse={dof['rmse']:.4f} "
            f"corr={dof['mean_cycle_pearson_correlation']:.4f} "
            f"velocity_rmse={dof['velocity_rmse']:.4f}"
        )
    return True


def main() -> None:
    """Analyze all unfinished VQ experiments and export their artifacts."""
    args = get_args()
    device = _device(args.device)
    run_directories = (
        [Path(args.run_dir)]
        if args.run_dir
        else _vq_run_directories(Path(args.results_dir))
    )
    analyzed_count = 0
    skipped_count = 0
    for run_dir in run_directories:
        print(f"Processing VQ run: {run_dir}")
        analyzed = _analyze_run(
            run_dir,
            args,
            device,
            use_path_overrides=args.run_dir is not None,
        )
        analyzed_count += int(analyzed)
        skipped_count += int(not analyzed)
    print(
        f"VQ analysis complete: discovered={len(run_directories)}, "
        f"analyzed={analyzed_count}, skipped={skipped_count}"
    )


if __name__ == "__main__":
    main()
