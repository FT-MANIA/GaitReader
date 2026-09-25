"""Published settings and the small set of experiments reported in the paper."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from .utils import read_json

ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = ROOT / "configs" / "paper.json"
ABLATIONS = ("full", "without_vocabulary", "without_waveform",
             "without_attributes_waveform", "without_gaitparser")
MODULE_OVERRIDES = {
    "without_vq_attribute_separation": {"vq_v2_separate_shape": False},
    "without_vq_geometry": {"vq_geometry_weight": 0.0},
    "without_vq_shape_reconstruction": {"vq_v2_shape_loss_weight": 0.0},
    "without_vq_waveform_reconstruction": {"vq_v2_waveform_loss_weight": 0.0},
    "without_ssl_attribute_loss": {"ssl_attribute_weight": 0.0},
    "without_ssl_code_loss": {"masked_top1_weight": 0.0},
    "without_bilateral_context": {"bilateral_depth": 0},
}
EMBEDDING_DEFAULTS = {
    f"ssl_use_{name}_embedding": True
    for name in ("shape", "mean", "scale", "dof", "cycle", "duration", "interval")
}
EMBEDDING_OVERRIDES = {
    "without_shape_embedding": {"ssl_use_shape_embedding": False},
    "without_attribute_embedding": {"ssl_use_mean_embedding": False, "ssl_use_scale_embedding": False},
    "without_mean_embedding": {"ssl_use_mean_embedding": False},
    "without_scale_embedding": {"ssl_use_scale_embedding": False},
    "without_dof_embedding": {"ssl_use_dof_embedding": False},
    "without_cycle_embedding": {"ssl_use_cycle_embedding": False},
    "without_timing_embedding": {"ssl_use_duration_embedding": False, "ssl_use_interval_embedding": False},
    "without_duration_embedding": {"ssl_use_duration_embedding": False},
    "without_interval_embedding": {"ssl_use_interval_embedding": False},
}
for _name, _included in (
    ("embedding_shape_only", ("shape",)),
    ("embedding_shape_attributes", ("shape", "mean", "scale")),
    ("embedding_shape_attributes_dof", ("shape", "mean", "scale", "dof")),
):
    EMBEDDING_OVERRIDES[_name] = {
        key: key in {f"ssl_use_{name}_embedding" for name in _included}
        for key in EMBEDDING_DEFAULTS
    }
# Legacy one-factor sweeps, not a Cartesian grid. Keep encoder/CNN widths fixed.
CODEBOOK_CONFIGURATIONS = {
    "codebook_k128_d128": (128, 128),
    "codebook_k64_d128": (64, 128),
    "codebook_k256_d128": (256, 128),
    "codebook_k128_d32": (128, 32),
    "codebook_k128_d64": (128, 64),
}
MASK_RATIO_CONFIGURATIONS = {
    f"mask_ratio_{percent:03d}": percent / 100
    for percent in (10, 15, 30, 50, 75)
}
COMPARISONS = ("timesnet", "patchtst", "itransformer", "ts_tcc", "ts2vec",
               "trep", "vqshape", "heartlang")

def build_parser():
    parser = argparse.ArgumentParser(description="GaitReader: pretraining followed directly by fixed-test five-fold evaluation.")
    parser.add_argument("--config", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--suite", choices=("full", "pretraining", "ablations", "full_ablations", "modules", "embeddings",
                        "codebook", "codebook_k", "codebook_d", "mask_ratio", "comparison", "all"), default="full")
    parser.add_argument("--methods", nargs="+", help="Select names from the chosen suite.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--resume-dir", type=Path, help="Resume this integrated run; saved settings take precedence.")
    parser.add_argument("--seed", type=int, help="Set every stage, split, fold and label-subset seed.")
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--ssl-csv", type=Path)
    parser.add_argument("--dev-csv", type=Path)
    parser.add_argument("--ext-test-csv", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                        help="Override an existing config field, e.g. --set downstream_epochs=200.")
    parser.add_argument("--checkpoint-map", type=Path,
                        help="JSON: method -> {vq: path, ssl: path}; load pretrained checkpoints, then run all five folds.")
    parser.add_argument("--subject-groups-csv", type=Path,
                        help="Optional subject_id,group_id mapping for repeated-record aliases.")
    parser.add_argument("--fold-manifest", type=Path, help="Reuse and validate a prior subject fold manifest.")
    parser.add_argument("--fetch-sources", action="store_true", help="Fetch pinned comparison sources, then exit.")
    return parser

def experiment_specs(suite, selected=None):
    full = [{"name": "full", "variant": "full", "mode": "full_finetune", "fraction": 1.0}]
    labels = [{"name": f"{mode}_labels{round(fraction * 100):03d}",
               "variant": "scratch" if mode == "scratch" else "full",
               "mode": "linear_probe" if mode == "linear_probe" else "full_finetune",
               "fraction": fraction}
              for fraction in (1.0, 0.1, 0.25, 0.5)
              for mode in ("finetune", "scratch", "linear_probe")]
    ablations = [{"name": name, "variant": name, "mode": "full_finetune", "fraction": 1.0}
                 for name in ABLATIONS]
    modules = [{"name": name, "variant": name, "mode": "full_finetune", "fraction": 1.0,
                "overrides": dict(overrides)} for name, overrides in MODULE_OVERRIDES.items()]
    embeddings = [{"name": name, "variant": name, "mode": "full_finetune", "fraction": 1.0,
                   "overrides": dict(overrides)} for name, overrides in EMBEDDING_OVERRIDES.items()]
    codebooks = [{"name": name, "variant": name, "mode": "full_finetune", "fraction": 1.0,
                  "overrides": {"codebook_size": size, "code_dim": dimension}}
                 for name, (size, dimension) in CODEBOOK_CONFIGURATIONS.items()]
    mask_ratios = [{"name": name, "variant": name, "mode": "full_finetune", "fraction": 1.0,
                    "overrides": {"random_mask_ratio": ratio}}
                   for name, ratio in MASK_RATIO_CONFIGURATIONS.items()]
    comparisons = [{"name": name, "variant": name, "mode": "full_finetune", "fraction": 1.0,
                    "adapter": "comparison"} for name in COMPARISONS]
    full_ablations = ablations + modules + embeddings
    suites = {"full": full, "pretraining": labels, "ablations": ablations,
              "full_ablations": full_ablations,
              "modules": full + modules, "embeddings": full + embeddings, "codebook": codebooks,
              "codebook_k": codebooks[:3], "codebook_d": [codebooks[0], *codebooks[3:]],
              "mask_ratio": mask_ratios,
              "comparison": comparisons, "all": full_ablations + codebooks + mask_ratios + labels + comparisons}
    specs = suites[suite]
    if selected:
        available = {s["name"] for s in specs}
        if set(selected) - available:
            raise ValueError(f"Unknown methods: {set(selected) - available}; available: {sorted(available)}")
        specs = [s for s in specs if s["name"] in selected]
    return specs

def method_args(config, spec):
    # Older saved plans implicitly used shape/attribute separation.
    values = {"vq_v2_separate_shape": True, **EMBEDDING_DEFAULTS, **deepcopy(config)}
    values.update(spec.get("overrides", {}))
    values.update(downstream_label_fraction=spec["fraction"], downstream_encoder_mode=spec["mode"])
    values["downstream_from_scratch"] = spec["variant"] == "scratch"
    values["ssl_objective"] = "waveform" if spec["variant"] == "without_vocabulary" else "code"
    if spec["variant"] == "without_gaitparser":
        values["segmentation_method"] = "fixed_window"
    if spec["variant"] in ("without_vocabulary", "without_attributes_waveform", "scratch"):
        values["ssl_attribute_weight"] = values["ssl_code_waveform_weight"] = 0.0
    elif spec["variant"] == "without_waveform":
        values["ssl_code_waveform_weight"] = 0.0
    for key in ("split_seed", "vq_training_seed", "ssl_training_seed",
                "downstream_training_seed", "validation_mask_seed", "downstream_label_seed"):
        values[key] = values["seed"]
    if spec.get("adapter") == "comparison":
        values = {**read_json(ROOT / "configs" / "comparison.json"), **values}
        values["comparison_model"] = spec["variant"]
        values["comparison_encoder_mode"] = spec["mode"]
        values["mixed_precision"] = False
    return argparse.Namespace(**values)

def data_config(args: argparse.Namespace) -> dict:
    return {
        "dataset": {
            "ssl_csv": args.ssl_csv,
            "dev_csv": args.dev_csv,
            "ext_test_csv": args.ext_test_csv,
            "cycle_length": args.word_length,
            "downstream_label_fraction": vars(args).get("downstream_label_fraction", 1.0),
            "downstream_label_seed": vars(args).get("downstream_label_seed"),
            "segmentation": {
                "method": vars(args).get("segmentation_method", "adaptive"),
                "fixed_window_length": vars(args).get("fixed_window_length", 100),
                "sampling_rate_hz": args.sampling_rate_hz,
                "target_length": args.word_length,
                "reference_dof_index": args.reference_dof_index,
                "min_cycle_seconds": args.min_cycle_seconds,
                "max_cycle_seconds": args.max_cycle_seconds,
                "smoothing_window_seconds": args.smoothing_window_seconds,
                "smoothing_polyorder": args.smoothing_polyorder,
                "peak_prominence_fraction": args.peak_prominence_fraction,
                "peak_distance_fraction": args.peak_distance_fraction,
                "boundary_post_peak_min_fraction": args.boundary_post_peak_min_fraction,
                "boundary_post_peak_max_fraction": args.boundary_post_peak_max_fraction,
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
            },
            "ssl_validation_fraction": args.ssl_validation_fraction,
            "downstream_validation_fraction": (
                args.downstream_validation_fraction
            ),
            "internal_test_size": args.internal_test_size,
        },
        "training": {
            "seed": args.split_seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    }
