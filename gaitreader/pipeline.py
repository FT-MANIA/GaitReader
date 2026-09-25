"""One workflow: healthy pretraining -> five independent downstream fits."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import torch

from .config import EMBEDDING_DEFAULTS, PAPER_CONFIG, build_parser, data_config, experiment_specs, method_args
from .data.builders import build_data_loaders, build_segmentation_report
from .evaluation import prepare_folds, fold_loaders, summarize
from .factory import build_encoder, build_ssl
from .models import VQGait, GaitClassifier
from .training import fit_vq, fit_ssl, fit_downstream, evaluate_downstream
from .utils import device_for, environment, file_hash, load_weights, read_json, save_json, seed_stage


def training_options(args, stage, output, resume):
    return dict(output_dir=output, epochs=getattr(args, f"{stage}_epochs"),
                learning_rate=getattr(args, f"{stage}_learning_rate"),
                weight_decay=getattr(args, f"{stage}_weight_decay"),
                patience=getattr(args, f"{stage}_patience"),
                mixed_precision=args.mixed_precision and device_for(args.device).type == "cuda",
                gradient_clip=args.gradient_clip, sampling_rate_hz=args.sampling_rate_hz,
                recording_length=args.recording_length, resume=resume)


def checkpoint_record(path):
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path)}


def pretraining_key(args, stage, plan, parent=None):
    # Only relevant settings identify dependencies. Full/probe/label fractions
    # share pretraining; SSL ablations share VQ; fixed windows train a new VQ.
    values = vars(args)
    shared = {key: values[key] for key in (
        "seed", "batch_size", "num_workers", "deterministic", "mixed_precision",
        "gradient_clip", "recording_length", "word_length", "word_dim", "code_dim")}
    if stage == "vq":
        keys = [k for k in values if k.startswith(("vq_", "codebook_"))]
        # True was implicit in earlier release plans; retain their baseline key.
        if values["vq_v2_separate_shape"]:
            keys.remove("vq_v2_separate_shape")
        keys += ["commitment_weight", "dead_code_threshold", "decoder_ff_dim"]
    else:
        keys = [k for k in values if k.startswith(("ssl_", "masked_", "encoder_", "patch_"))]
        # All-on embeddings were implicit in old plans. Only disabled inputs
        # change SSL's signature, keeping the existing Full cache key intact.
        keys = [k for k in keys if k not in EMBEDDING_DEFAULTS or not values[k]]
        keys += ["random_mask_ratio", "bilateral_depth", "dropout", "max_words"]
    settings = {**shared, **{k: values[k] for k in keys}}
    dataset = data_config(args)["dataset"]
    dataset.pop("downstream_label_fraction")
    dataset.pop("downstream_label_seed")
    signature = {"stage": stage, "settings": settings, "dataset": dataset,
                 "csv_hashes": plan["data_hashes"], "parent": parent}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    return digest, signature


def pretrain(args, spec, loaders, plan, directory, resume):
    if args.downstream_from_scratch:
        return {"vq": None, "ssl": None}
    supplied = plan["checkpoints"].get(spec["name"], plan["checkpoints"].get(spec["variant"], {}))
    if args.ssl_objective == "code" and "ssl" in supplied and "vq" not in supplied:
        raise ValueError("Code-based SSL imports require both the matching VQ and SSL checkpoints.")
    paths = {"vq": None, "ssl": None}
    device = device_for(args.device)
    tokenizer = None
    if args.ssl_objective == "code":
        seed_stage(args.seed, loaders)
        tokenizer = VQGait(args).to(device)
        key, signature = pretraining_key(args, "vq", plan)
        output = directory / "pretraining" / "vq" / key
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / "signature.json", signature)
        save_json(output / "args.json", vars(args))
        if "vq" in supplied:
            path = Path(supplied["vq"]["path"])
            load_weights(tokenizer, path)
        elif (output / "complete.json").exists():
            path = output / "best_vq.pt"
            load_weights(tokenizer, path)
        else:
            print(f"VQ-Gait pretraining: {output}", flush=True)
            path = fit_vq(tokenizer, loaders["ssl_data"], loaders["ssl_validation_data"], device,
                          **training_options(args, "vq", output, resume))
            save_json(output / "complete.json", checkpoint_record(path))
        paths["vq"] = checkpoint_record(path)
    if "ssl" in supplied:
        paths["ssl"] = supplied["ssl"]
        return paths
    seed_stage(args.seed, loaders)
    model = build_ssl(args, tokenizer).to(device)
    key, signature = pretraining_key(args, "ssl", plan, paths["vq"]["sha256"] if paths["vq"] else None)
    output = directory / "pretraining" / "ssl" / key
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "signature.json", signature)
    save_json(output / "args.json", vars(args))
    path = output / "best_ssl.pt"
    if not (output / "complete.json").exists():
        print(f"GaitFormer pretraining: {output}", flush=True)
        path = fit_ssl(model, loaders["ssl_data"], loaders["ssl_validation_data"], device,
                       validation_mask_seed=args.seed, training_mask_seed=args.seed,
                       **training_options(args, "ssl", output, resume))
        save_json(output / "complete.json", checkpoint_record(path))
    paths["ssl"] = checkpoint_record(path)
    return paths


def train_fold(args, spec, checkpoints, fold, pool, original, directory, resume):
    output = directory / spec["name"] / f"fold_{fold['fold']:02d}"
    if (output / "evaluation.json").exists():
        print(f"Skip completed fold: {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    loaders = fold_loaders(pool, fold, original, args)
    device = device_for(args.device)
    save_json(output / "args.json", {**vars(args), "cv_fold": fold["fold"], "checkpoints": checkpoints})
    save_json(output / "downstream_label_subset.json", fold["labeled_subsets"][str(args.downstream_label_fraction)])
    save_json(output / "split.json", {k: v for k, v in fold.items() if k != "labeled_subsets"})
    if spec.get("adapter") == "comparison":
        from .comparisons.workflow import train_comparison_fold
        train_comparison_fold(args, checkpoints, loaders, device, output, resume)
        return
    seed_stage(args.seed, loaders)
    if args.downstream_from_scratch:
        encoder = build_encoder(args)
    else:
        # Construction order matches the archived CV script. Every fold reloads
        # the SAME pretrained model, never a previous fold's supervised weights.
        tokenizer = VQGait(args) if checkpoints["vq"] else None
        if tokenizer is not None:
            load_weights(tokenizer, checkpoints["vq"]["path"])
        ssl = build_ssl(args, tokenizer)
        load_weights(ssl, checkpoints["ssl"]["path"])
        encoder = ssl.encoder
        del ssl, tokenizer
    seed_stage(args.seed, loaders)
    model = GaitClassifier(encoder, fine_tune_encoder=args.downstream_encoder_mode == "full_finetune").to(device)
    save_json(output / "parameters.json", {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoder_trainable": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)})
    print(f"{spec['name']} fold={fold['fold']}/5 mode={args.downstream_encoder_mode} labels={args.downstream_label_fraction}", flush=True)
    path = fit_downstream(model, loaders["dev_data"], loaders["dev_validation_data"], device,
                          **training_options(args, "downstream", output, resume))
    result = {name: evaluate_downstream(model, loaders[key], device,
              sampling_rate_hz=args.sampling_rate_hz, recording_length=args.recording_length,
              mixed_precision=args.mixed_precision and device.type == "cuda")
              for name, key in (("internal_test", "dev_test_data"), ("external_test", "ext_test_data"))}
    result.update(fold=fold["fold"], downstream_checkpoint=str(path), checkpoints=checkpoints)
    save_json(output / "evaluation.json", result)


def create_plan(options):
    config = {**read_json(PAPER_CONFIG), **read_json(options.config)}
    for override in options.set:
        key, value = override.split("=", 1)
        if key not in config:
            raise ValueError(f"Unknown configuration field: {key}")
        config[key] = json.loads(value)
    if options.seed is not None:
        config["seed"] = config["fold_seed"] = options.seed
    if options.device:
        config["device"] = options.device
    for key in ("ssl_csv", "dev_csv", "ext_test_csv"):
        config[key] = str(Path(getattr(options, key) or config[key]).resolve())
    mapping = {}
    if options.subject_groups_csv:
        with options.subject_groups_csv.open(encoding="utf-8-sig", newline="") as handle:
            mapping = {row["subject_id"]: row["group_id"] for row in csv.DictReader(handle)}
    checkpoints = {}
    if options.checkpoint_map:
        for name, stages in read_json(options.checkpoint_map).items():
            checkpoints[name] = {stage: checkpoint_record(path) for stage, path in stages.items()}
    specs = experiment_specs(options.suite, options.methods)
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)): file_hash(p)
               for p in sorted((root / "gaitreader").rglob("*.py"))}
    return {"protocol": "gaitreader_paper_fixed_tests_cv_v1", "config": config,
            "methods": specs, "fold_seed": config["fold_seed"],
            "subject_group_mapping": mapping, "label_subset_seed": config["seed"],
            "label_fractions": sorted({s["fraction"] for s in specs}),
            "checkpoints": checkpoints, "environment": environment(), "source_hashes": sources,
            "data_hashes": {k: file_hash(config[k]) for k in ("ssl_csv", "dev_csv", "ext_test_csv")}}


def main():
    options = build_parser().parse_args()
    if options.fetch_sources:
        from .comparisons.sources import fetch_sources
        fetch_sources()
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    if options.resume_dir:
        directory = options.resume_dir.resolve()
        plan = read_json(directory / "plan.json")
        for key, digest in plan["data_hashes"].items():
            if file_hash(plan["config"][key]) != digest:
                raise ValueError(f"Data changed since this run started: {key}")
        for stages in plan["checkpoints"].values():
            for record in stages.values():
                if file_hash(record["path"]) != record["sha256"]:
                    raise ValueError(f"Checkpoint changed: {record['path']}")
    else:
        plan = create_plan(options)
        directory = (options.output_dir / f"gaitreader_{datetime.now():%Y%m%d_%H%M%S}").resolve()
        directory.mkdir(parents=True, exist_ok=False)
        save_json(directory / "plan.json", plan)
        if options.fold_manifest:
            save_json(directory / "fold_manifest.json", read_json(options.fold_manifest))
    torch.use_deterministic_algorithms(plan["config"]["deterministic"])
    torch.backends.cudnn.deterministic = plan["config"]["deterministic"]
    print(f"Integrated pretraining + five-fold output: {directory}", flush=True)
    current_input = None
    for spec in plan["methods"]:
        args = method_args(plan["config"], spec)
        print(f"Method: {spec['name']} overrides={spec.get('overrides', {})}", flush=True)
        config = data_config(args)
        config["dataset"]["downstream_label_fraction"] = 1.0
        if config != current_input:
            print(f"Loading subjects: {args.segmentation_method}", flush=True)
            original = build_data_loaders(config)
            pool, manifest = prepare_folds(plan, original, directory)
            current_input = config
        method_directory = directory / spec["name"]
        method_directory.mkdir(parents=True, exist_ok=True)
        save_json(method_directory / "preprocessing.json", build_segmentation_report(original))
        reference_file = method_directory / "checkpoints.json"
        if reference_file.exists():
            checkpoints = read_json(reference_file)
            for record in checkpoints.values():
                if record and file_hash(record["path"]) != record["sha256"]:
                    raise ValueError(f"Pretrained checkpoint changed: {record['path']}")
        else:
            if spec.get("adapter") == "comparison":
                from .comparisons.workflow import pretrain_comparison
                checkpoints = pretrain_comparison(args, spec, original, plan, directory, bool(options.resume_dir))
            else:
                checkpoints = pretrain(args, spec, original, plan, directory, bool(options.resume_dir))
            save_json(reference_file, checkpoints)
        for fold in manifest["folds"]:
            train_fold(args, spec, checkpoints, fold, pool, original, directory, bool(options.resume_dir))
            summarize(plan, directory)


if __name__ == "__main__":
    main()
