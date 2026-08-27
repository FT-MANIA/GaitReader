"""One-click entry point for the complete gait-language experiment."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from knee_kinematics.data.builders import build_data_loaders
from knee_kinematics.gait_language.data import summarize_language_dataset
from knee_kinematics.gait_language.models import (
    GaitLanguageDownstreamModel,
    GaitLanguageSSLModel,
)
from knee_kinematics.gait_language.trainer import (
    evaluate_downstream,
    fit_downstream,
    fit_ssl,
    fit_vq,
)
from knee_kinematics.gait_language.vq import GaitVQTokenizer


def build_parser() -> argparse.ArgumentParser:
    """Build the shared data, model, SSL, and optimization parser."""
    parser = argparse.ArgumentParser(
        "Gait-language pretraining and downstream experiment"
    )

    # Experiment
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "vq",
            "ssl",
            "ssl_downstream",
            "downstream",
            "evaluate",
        ),
        default="all",
    )
    parser.add_argument(
        "--output-dir", default="Results/gait_language/dev_exp"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Unique child directory name; defaults to dev_exp_MMDD_HHMM",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Dataset
    parser.add_argument(
        "--ssl-csv",
        default="Dataset/SSL_Healthy/ssl_healthy_dataset.csv",
    )
    parser.add_argument(
        "--dev-csv", default="Dataset/KGKD/dev_dataset.csv"
    )
    parser.add_argument(
        "--ext-test-csv", default="Dataset/KGKD/test_dataset.csv"
    )
    parser.add_argument("--recording-length", type=int, default=600)
    parser.add_argument("--sampling-rate-hz", type=float, default=60.0)
    parser.add_argument("--word-length", type=int, default=100)
    parser.add_argument("--ssl-validation-fraction", type=float, default=0.10)
    parser.add_argument(
        "--downstream-validation-fraction", type=float, default=0.15
    )
    parser.add_argument("--internal-test-size", type=float, default=0.20)

    # Adaptive gait-cycle tokenizer
    parser.add_argument("--reference-dof-index", type=int, default=2)
    parser.add_argument("--min-cycle-seconds", type=float, default=0.4)
    parser.add_argument("--max-cycle-seconds", type=float, default=4.0)
    parser.add_argument(
        "--smoothing-window-seconds", type=float, default=0.15
    )
    parser.add_argument("--smoothing-polyorder", type=int, default=3)
    parser.add_argument(
        "--peak-prominence-fraction", type=float, default=0.15
    )
    parser.add_argument(
        "--peak-distance-fraction", type=float, default=0.55
    )
    parser.add_argument(
        "--period-min-correlation", type=float, default=0.05
    )
    parser.add_argument("--period-relative-min", type=float, default=0.55)
    parser.add_argument("--period-relative-max", type=float, default=1.60)
    parser.add_argument("--min-cycles", type=int, default=1)
    parser.add_argument(
        "--similarity-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-cycle-similarity", type=float, default=-1.0)
    parser.add_argument("--similarity-mad-scale", type=float, default=3.0)

    # Subject quality control
    parser.add_argument(
        "--quality-control",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-cycles-per-side", type=int, default=2)
    parser.add_argument("--robust-z-threshold", type=float, default=6.0)
    parser.add_argument("--min-upper-scale-factor", type=float, default=3.0)
    parser.add_argument("--min-reference-subjects", type=int, default=20)

    # Word tokenizer and codebook
    parser.add_argument("--word-dim", type=int, default=128)
    parser.add_argument("--word-hidden-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--codebook-decay", type=float, default=0.99)
    parser.add_argument("--dead-code-threshold", type=float, default=1.0)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--velocity-weight", type=float, default=0.20)
    parser.add_argument(
        "--vq-decoder",
        choices=(
            "mlp",
            "temporal_transformer",
            "sentence_transformer",
            "local_context_sentence",
        ),
        default="local_context_sentence",
    )
    parser.add_argument("--vq-decoder-depth", type=int, default=2)
    parser.add_argument("--vq-decoder-heads", type=int, default=4)
    parser.add_argument("--vq-decoder-ff-dim", type=int, default=512)
    parser.add_argument("--vq-decoder-dropout", type=float, default=0.10)
    parser.add_argument("--vq-decoder-phase-tokens", type=int, default=20)
    parser.add_argument(
        "--vq-context-residual-scale", type=float, default=0.5
    )
    parser.add_argument(
        "--vq-local-reconstruction-weight", type=float, default=2.0
    )
    parser.add_argument(
        "--vq-residual-energy-weight",
        type=float,
        default=0.01,
        help=(
            "Weight on the valid-word mean squared scaled contextual "
            "residual; only affects local_context_sentence"
        ),
    )

    # Sentence encoder and masked tasks
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--sentence-depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--ssl-within-task",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ssl-cross-dof-task",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ssl-rhythm-task",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ssl-bilateral-context-task",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ssl-contralateral-task",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ssl-bilateral-pair-task",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ssl-swap-task",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--word-mask-ratio", type=float, default=0.30)
    parser.add_argument("--bilateral-mask-ratio", type=float, default=0.30)
    parser.add_argument(
        "--contralateral-mask-ratio", type=float, default=0.30
    )
    parser.add_argument("--rhythm-mask-ratio", type=float, default=0.30)
    parser.add_argument("--span-length", type=int, default=2)
    parser.add_argument("--within-weight", type=float, default=1.0)
    parser.add_argument("--cross-dof-weight", type=float, default=1.0)
    parser.add_argument("--rhythm-weight", type=float, default=0.50)
    parser.add_argument(
        "--duration-prediction-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--interval-prediction-weight", type=float, default=1.0
    )
    parser.add_argument("--bilateral-weight", type=float, default=1.0)
    parser.add_argument("--contralateral-weight", type=float, default=0.50)
    parser.add_argument("--bilateral-pair-weight", type=float, default=1.0)
    parser.add_argument("--swap-weight", type=float, default=0.10)
    parser.add_argument("--conditional-code-top-k", type=int, default=5)
    parser.add_argument(
        "--conditional-soft-target-temperature",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--cross-dof-hard-code-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--cross-dof-soft-code-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--cross-dof-prototype-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--contralateral-hard-code-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--contralateral-soft-code-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--contralateral-prototype-weight", type=float, default=1.0
    )

    # VQ training
    parser.add_argument("--vq-epochs", type=int, default=100)
    parser.add_argument("--vq-learning-rate", type=float, default=3e-4)
    parser.add_argument("--vq-weight-decay", type=float, default=1e-4)
    parser.add_argument("--vq-patience", type=int, default=10)
    parser.add_argument("--vq-checkpoint", default=None)

    # Masked sentence pretraining
    parser.add_argument("--ssl-epochs", type=int, default=100)
    parser.add_argument("--ssl-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ssl-weight-decay", type=float, default=1e-2)
    parser.add_argument("--ssl-patience", type=int, default=10)
    parser.add_argument(
        "--validation-mask-seed",
        type=int,
        default=None,
        help="Fixed SSL validation mask seed; defaults to seed + 10000",
    )
    parser.add_argument("--ssl-checkpoint", default=None)

    # Downstream training
    parser.add_argument("--downstream-epochs", type=int, default=50)
    parser.add_argument(
        "--downstream-learning-rate", type=float, default=3e-4
    )
    parser.add_argument(
        "--downstream-weight-decay", type=float, default=1e-2
    )
    parser.add_argument("--downstream-patience", type=int, default=10)
    parser.add_argument("--affected-side-weight", type=float, default=0.20)
    parser.add_argument("--classifier-dropout", type=float, default=0.20)
    parser.add_argument("--deviation-dof-dim", type=int, default=64)
    parser.add_argument("--deviation-std-floor", type=float, default=0.05)
    parser.add_argument(
        "--freeze-sentence-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--downstream-checkpoint", default=None)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    return parser


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a development experiment."""
    return build_parser().parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _create_run_directory(args: argparse.Namespace) -> Path:
    """Create one exclusive output directory for this invocation."""
    run_name = args.run_name or (
        f"dev_exp_{datetime.now().strftime('%m%d_%H%M')}"
    )
    if Path(run_name).name != run_name:
        raise ValueError("run_name must be one directory name")
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _data_config(args: argparse.Namespace) -> dict:
    return {
        "dataset": {
            "ssl_csv": args.ssl_csv,
            "dev_csv": args.dev_csv,
            "ext_test_csv": args.ext_test_csv,
            "cycle_length": args.word_length,
            "segmentation": {
                "method": "adaptive",
                "sampling_rate_hz": args.sampling_rate_hz,
                "target_length": args.word_length,
                "reference_dof_index": args.reference_dof_index,
                "min_cycle_seconds": args.min_cycle_seconds,
                "max_cycle_seconds": args.max_cycle_seconds,
                "smoothing_window_seconds": args.smoothing_window_seconds,
                "smoothing_polyorder": args.smoothing_polyorder,
                "peak_prominence_fraction": args.peak_prominence_fraction,
                "peak_distance_fraction": args.peak_distance_fraction,
                "period_min_correlation": args.period_min_correlation,
                "period_relative_min": args.period_relative_min,
                "period_relative_max": args.period_relative_max,
                "min_cycles": args.min_cycles,
                "similarity_filter": args.similarity_filter,
                "min_cycle_similarity": args.min_cycle_similarity,
                "similarity_mad_scale": args.similarity_mad_scale,
            },
            "quality_control": {
                "enabled": args.quality_control,
                "min_cycles_per_side": args.min_cycles_per_side,
                "robust_z_threshold": args.robust_z_threshold,
                "min_upper_scale_factor": args.min_upper_scale_factor,
                "min_reference_subjects": args.min_reference_subjects,
            },
            "ssl_validation_fraction": args.ssl_validation_fraction,
            "downstream_validation_fraction": (
                args.downstream_validation_fraction
            ),
            "internal_test_size": args.internal_test_size,
        },
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    }


def _build_vq(args: argparse.Namespace) -> GaitVQTokenizer:
    return GaitVQTokenizer(
        word_length=args.word_length,
        word_dim=args.word_dim,
        hidden_dim=args.word_hidden_dim,
        codebook_size=args.codebook_size,
        codebook_decay=args.codebook_decay,
        dead_code_threshold=args.dead_code_threshold,
        commitment_weight=args.commitment_weight,
        velocity_weight=args.velocity_weight,
        decoder_type=getattr(args, "vq_decoder", "mlp"),
        decoder_depth=getattr(args, "vq_decoder_depth", 2),
        decoder_num_heads=getattr(args, "vq_decoder_heads", 4),
        decoder_feedforward_dim=getattr(args, "vq_decoder_ff_dim", 512),
        decoder_dropout=getattr(args, "vq_decoder_dropout", 0.10),
        decoder_phase_tokens=getattr(
            args, "vq_decoder_phase_tokens", 20
        ),
        context_residual_scale=getattr(
            args, "vq_context_residual_scale", 0.5
        ),
        local_reconstruction_weight=getattr(
            args, "vq_local_reconstruction_weight", 1.0
        ),
        residual_energy_weight=getattr(
            args, "vq_residual_energy_weight", 0.0
        ),
        max_words=args.max_words,
    )


def _build_ssl(
    args: argparse.Namespace, tokenizer: GaitVQTokenizer
) -> GaitLanguageSSLModel:
    return GaitLanguageSSLModel(
        tokenizer,
        word_dim=args.word_dim,
        codebook_size=args.codebook_size,
        max_words=args.max_words,
        depth=args.sentence_depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
        within_task=args.ssl_within_task,
        cross_dof_task=args.ssl_cross_dof_task,
        rhythm_task=args.ssl_rhythm_task,
        bilateral_context_task=args.ssl_bilateral_context_task,
        contralateral_task=args.ssl_contralateral_task,
        bilateral_pair_task=args.ssl_bilateral_pair_task,
        swap_task=args.ssl_swap_task,
        word_mask_ratio=args.word_mask_ratio,
        bilateral_mask_ratio=args.bilateral_mask_ratio,
        contralateral_mask_ratio=args.contralateral_mask_ratio,
        rhythm_mask_ratio=args.rhythm_mask_ratio,
        span_length=args.span_length,
        within_weight=args.within_weight,
        cross_dof_weight=args.cross_dof_weight,
        rhythm_weight=args.rhythm_weight,
        duration_prediction_weight=args.duration_prediction_weight,
        interval_prediction_weight=args.interval_prediction_weight,
        bilateral_weight=args.bilateral_weight,
        contralateral_weight=args.contralateral_weight,
        bilateral_pair_weight=args.bilateral_pair_weight,
        swap_weight=args.swap_weight,
        conditional_code_top_k=args.conditional_code_top_k,
        conditional_soft_target_temperature=(
            args.conditional_soft_target_temperature
        ),
        cross_dof_hard_code_weight=args.cross_dof_hard_code_weight,
        cross_dof_soft_code_weight=args.cross_dof_soft_code_weight,
        cross_dof_prototype_weight=args.cross_dof_prototype_weight,
        contralateral_hard_code_weight=(
            args.contralateral_hard_code_weight
        ),
        contralateral_soft_code_weight=(
            args.contralateral_soft_code_weight
        ),
        contralateral_prototype_weight=(
            args.contralateral_prototype_weight
        ),
    )


def _load_model(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])


def run_experiment(args: argparse.Namespace) -> dict | None:
    """Run one configured vocabulary, SSL, and downstream experiment."""
    if args.validation_mask_seed is None:
        args.validation_mask_seed = args.seed + 10_000
    _set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = _device(args.device)
    mixed_precision = args.mixed_precision and device.type == "cuda"
    output_dir = _create_run_directory(args)
    args.run_dir = str(output_dir)
    with (output_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    print(f"device={device}")
    print(f"run_dir={output_dir}")
    print("Loading and segmenting gait recordings")
    loaders = build_data_loaders(_data_config(args))
    print(
        "subjects: "
        + ", ".join(
            f"{name}={len(loader.dataset)}"
            for name, loader in loaders.items()
        )
    )
    word_statistics = {
        "ssl_data": summarize_language_dataset(
            loaders["ssl_data"].dataset,
            sampling_rate_hz=args.sampling_rate_hz,
        ),
        "dev_data": summarize_language_dataset(
            loaders["dev_data"].dataset,
            sampling_rate_hz=args.sampling_rate_hz,
        ),
    }
    with (output_dir / "word_statistics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(word_statistics, handle, ensure_ascii=False, indent=2)

    tokenizer = _build_vq(args).to(device)
    vq_checkpoint = Path(
        args.vq_checkpoint or output_dir / "best_vq.pt"
    )
    if args.stage in {"all", "vq"}:
        vq_checkpoint = fit_vq(
            tokenizer,
            loaders["ssl_data"],
            loaders["ssl_validation_data"],
            device,
            output_dir=output_dir,
            epochs=args.vq_epochs,
            learning_rate=args.vq_learning_rate,
            weight_decay=args.vq_weight_decay,
            patience=args.vq_patience,
            mixed_precision=mixed_precision,
            gradient_clip=args.gradient_clip,
            sampling_rate_hz=args.sampling_rate_hz,
            recording_length=args.recording_length,
        )
    else:
        _load_model(tokenizer, vq_checkpoint)
    if args.stage == "vq":
        print(f"VQ checkpoint: {vq_checkpoint}")
        return None

    ssl_model = _build_ssl(args, tokenizer).to(device)
    ssl_checkpoint = Path(
        args.ssl_checkpoint or output_dir / "best_ssl.pt"
    )
    if args.stage in {"all", "ssl", "ssl_downstream"}:
        ssl_checkpoint = fit_ssl(
            ssl_model,
            loaders["ssl_data"],
            loaders["ssl_validation_data"],
            device,
            output_dir=output_dir,
            epochs=args.ssl_epochs,
            learning_rate=args.ssl_learning_rate,
            weight_decay=args.ssl_weight_decay,
            patience=args.ssl_patience,
            mixed_precision=mixed_precision,
            gradient_clip=args.gradient_clip,
            sampling_rate_hz=args.sampling_rate_hz,
            recording_length=args.recording_length,
            validation_mask_seed=args.validation_mask_seed,
        )
    else:
        _load_model(ssl_model, ssl_checkpoint)
    if args.stage == "ssl":
        print(f"SSL checkpoint: {ssl_checkpoint}")
        return None

    downstream = GaitLanguageDownstreamModel(
        ssl_model.sentence_encoder,
        word_dim=args.word_dim,
        dropout=args.classifier_dropout,
        deviation_dof_dim=args.deviation_dof_dim,
        deviation_std_floor=args.deviation_std_floor,
    ).to(device)
    if args.freeze_sentence_encoder:
        for parameter in downstream.sentence_encoder.parameters():
            parameter.requires_grad = False
    downstream_checkpoint = Path(
        args.downstream_checkpoint or output_dir / "best_downstream.pt"
    )
    if args.stage == "evaluate":
        _load_model(downstream, downstream_checkpoint)
    else:
        downstream_checkpoint = fit_downstream(
            downstream,
            loaders["dev_data"],
            loaders["dev_validation_data"],
            device,
            output_dir=output_dir,
            epochs=args.downstream_epochs,
            learning_rate=args.downstream_learning_rate,
            weight_decay=args.downstream_weight_decay,
            patience=args.downstream_patience,
            affected_side_weight=args.affected_side_weight,
            mixed_precision=mixed_precision,
            gradient_clip=args.gradient_clip,
            sampling_rate_hz=args.sampling_rate_hz,
            recording_length=args.recording_length,
        )
    internal = evaluate_downstream(
        downstream,
        loaders["dev_test_data"],
        device,
        sampling_rate_hz=args.sampling_rate_hz,
        recording_length=args.recording_length,
        mixed_precision=mixed_precision,
    )
    external = evaluate_downstream(
        downstream,
        loaders["ext_test_data"],
        device,
        sampling_rate_hz=args.sampling_rate_hz,
        recording_length=args.recording_length,
        mixed_precision=mixed_precision,
    )
    results = {
        "internal_test": internal,
        "external_test": external,
        "vq_checkpoint": str(vq_checkpoint),
        "ssl_checkpoint": str(ssl_checkpoint),
        "downstream_checkpoint": str(downstream_checkpoint),
    }
    with (output_dir / "evaluation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def main() -> None:
    """Run one development experiment from command-line arguments."""
    run_experiment(get_args())


if __name__ == "__main__":
    main()
