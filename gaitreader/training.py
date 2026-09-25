"""Paper training loops: same reductions, optimizer, precision and stopping."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from .data.batch import build_language_batch, move_language_batch
from .utils import _atomic_save, _resume_training, _save_last, _append_metrics

CLASS_NAMES = ("Healthy", "ACLD", "KOA")

def _language_batch(
    batch: dict[str, Any],
    device: torch.device,
    *,
    sampling_rate_hz: float,
    recording_length: int,
) -> dict[str, Any]:
    return move_language_batch(
        build_language_batch(
            batch,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        ),
        device,
    )


def _mean_metrics(
    sums: dict[str, float], sample_count: int
) -> dict[str, float]:
    return {key: value / sample_count for key, value in sums.items()}


def _optimizer_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    gradient_clip: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # Clip only optimized, trainable parameters; frozen teachers are excluded.
    parameters = (
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    )
    nn.utils.clip_grad_norm_(parameters, gradient_clip)
    scaler.step(optimizer)
    scaler.update()


def run_vq_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    gradient_clip: float,
    sampling_rate_hz: float,
    recording_length: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    keys = (
        "loss",
        "reconstruction_loss",
        "commitment_loss",
        "geometry_loss",
        "active_code_ratio",
        "perplexity",
    )
    keys += model.extra_loss_keys
    sums = {key: 0.0 for key in keys}
    sample_count = 0
    for raw_batch in loader:
        batch = _language_batch(
            raw_batch,
            device,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        )
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=mixed_precision
        ):
            output = model(
                batch["words"], batch["word_mask"], batch["timing"]
            )
        if training:
            _optimizer_step(
                output["loss"], model, optimizer, scaler, gradient_clip
            )
        batch_size = batch["words"].shape[0]
        sample_count += batch_size
        for key in keys:
            sums[key] += float(output[key].detach()) * batch_size
    return _mean_metrics(sums, sample_count)


def fit_vq(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    mixed_precision: bool,
    gradient_clip: float,
    sampling_rate_hz: float,
    recording_length: int,
    resume: bool = False,
) -> Path:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = None
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)
    best_loss = float("inf")
    stale_epochs = 0
    checkpoint = output_dir / "best_vq.pt"
    metrics_path = output_dir / "metrics.jsonl"
    start, best_loss, stale_epochs = _resume_training(
        model, optimizer, scaler, (train_loader, validation_loader), output_dir,
        "vq", epochs, best_loss, resume, scheduler,
    )
    for epoch in range(start, epochs):
        epoch_start_lr = optimizer.param_groups[0]["lr"]
        train = run_vq_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        )
        validation = run_vq_epoch(
            model,
            validation_loader,
            device,
            optimizer=None,
            scaler=scaler,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        )
        metrics = {
            "train/lr_start": epoch_start_lr,
            "train/lr_end": optimizer.param_groups[0]["lr"],
            "train/global_step": (epoch + 1) * len(train_loader),
            **{f"train/{key}": value for key, value in train.items()},
            **{
                f"validation/{key}": value
                for key, value in validation.items()
            },
        }
        _append_metrics(metrics_path, "vq", epoch, metrics)
        print(
            f"VQ epoch={epoch:03d} "
            f"train={train['loss']:.5f} "
            f"validation={validation['loss']:.5f} "
            f"geometry={validation['geometry_loss']:.5f} "
            f"active={validation['active_code_ratio']:.3f} "
            f"perplexity={validation['perplexity']:.1f}"
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            _atomic_save(
                {
                    "stage": "vq",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": None,
                    "global_step": (epoch + 1) * len(train_loader),
                    "validation_loss": best_loss,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
        _save_last(model, optimizer, scaler, (train_loader, validation_loader),
                   output_dir, "vq", epoch, best_loss, stale_epochs,
                   stale_epochs >= patience or epoch + 1 >= epochs, scheduler)
        if stale_epochs >= patience:
            break
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)[
            "model"
        ]
    )
    return checkpoint

def run_ssl_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    gradient_clip: float,
    sampling_rate_hz: float,
    recording_length: int,
    mask_seed: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    generator = None
    if mask_seed is not None:
        # Independent streams: disabling a task cannot shift another task's masks.
        generator = torch.Generator(device=device).manual_seed(mask_seed)
    keys = model.metric_keys + getattr(model, "extra_metric_keys", ())
    sums = {key: 0.0 for key in keys}
    sample_count = 0
    for raw_batch in loader:
        batch = _language_batch(
            raw_batch,
            device,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        )
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=mixed_precision
        ):
            output = model(
                batch["words"],
                batch["word_mask"],
                batch["timing"],
                mask_generator=generator,
            )
        if training:
            _optimizer_step(
                output["loss"], model, optimizer, scaler, gradient_clip
            )
        batch_size = batch["words"].shape[0]
        sample_count += batch_size
        for key in keys:
            sums[key] += float(output[key].detach()) * batch_size
    return _mean_metrics(sums, sample_count)

def fit_ssl(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    mixed_precision: bool,
    gradient_clip: float,
    sampling_rate_hz: float,
    recording_length: int,
    validation_mask_seed: int,
    training_mask_seed: int | None = None,
    resume: bool = False,
) -> Path:
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = None
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)
    best_loss = float("inf")
    stale_epochs = 0
    checkpoint = output_dir / "best_ssl.pt"
    metrics_path = output_dir / "metrics.jsonl"
    start, best_loss, stale_epochs = _resume_training(
        model, optimizer, scaler, (train_loader, validation_loader), output_dir,
        "ssl", epochs, best_loss, resume, scheduler,
    )
    for epoch in range(start, epochs):
        epoch_start_lr = optimizer.param_groups[0]["lr"]
        train = run_ssl_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
            mask_seed=None if training_mask_seed is None else training_mask_seed + epoch,
        )
        validation = run_ssl_epoch(
            model,
            validation_loader,
            device,
            optimizer=None,
            scaler=scaler,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
            mask_seed=validation_mask_seed,
        )
        metrics = {
            "train/lr_start": epoch_start_lr,
            "train/lr_end": optimizer.param_groups[0]["lr"],
            "train/global_step": (epoch + 1) * len(train_loader),
            **{f"train/{key}": value for key, value in train.items()},
            **{
                f"validation/{key}": value
                for key, value in validation.items()
            },
        }
        _append_metrics(metrics_path, "ssl", epoch, metrics)
        detail = f"waveform_mse={validation['waveform_loss']:.5f}" if "waveform_loss" in validation else f"code_acc={validation['masked_top1_accuracy']:.3f}"
        print(f"SSL epoch={epoch:03d} train={train['loss']:.5f} validation={validation['loss']:.5f} {detail}")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            _atomic_save(
                {
                    "stage": "ssl",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "encoder": model.encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": None,
                    "global_step": (epoch + 1) * len(train_loader),
                    "validation_loss": best_loss,
                    "validation_mask_seed": validation_mask_seed,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
        _save_last(model, optimizer, scaler, (train_loader, validation_loader),
                   output_dir, "ssl", epoch, best_loss, stale_epochs,
                   stale_epochs >= patience or epoch + 1 >= epochs, scheduler)
        if stale_epochs >= patience:
            break
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)[
            "model"
        ]
    )
    return checkpoint

def run_downstream_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    class_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    gradient_clip: float,
    sampling_rate_hz: float,
    recording_length: int,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0}
    sample_count = 0
    labels = []
    probabilities = []
    predictions = []
    for raw_batch in loader:
        batch = _language_batch(
            raw_batch,
            device,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        )
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=mixed_precision
        ):
            output = model(
                batch["words"], batch["word_mask"], batch["timing"]
            )
            loss = F.cross_entropy(
                output["disease_logits"],
                batch["disease_label"],
                weight=class_weights,
            )
        if training:
            _optimizer_step(
                loss, model, optimizer, scaler, gradient_clip
            )

        batch_size = batch["words"].shape[0]
        sample_count += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        probability = output["disease_logits"].float().softmax(dim=-1)
        labels.append(batch["disease_label"].detach().cpu())
        probabilities.append(probability.detach().cpu())
        predictions.append(probability.argmax(dim=-1).detach().cpu())

    y_true = torch.cat(labels).numpy()
    y_probability = torch.cat(probabilities).double().numpy()
    y_probability /= y_probability.sum(axis=1, keepdims=True)
    y_prediction = torch.cat(predictions).numpy()
    class_indices = np.arange(len(CLASS_NAMES))
    per_class_precision = precision_score(
        y_true,
        y_prediction,
        labels=class_indices,
        average=None,
        zero_division=0,
    )
    per_class_recall = recall_score(
        y_true,
        y_prediction,
        labels=class_indices,
        average=None,
        zero_division=0,
    )
    class_confusion_matrix = confusion_matrix(
        y_true,
        y_prediction,
        labels=class_indices,
    )
    return {
        **{key: value / sample_count for key, value in totals.items()},
        "accuracy": accuracy_score(y_true, y_prediction),
        "macro_f1": f1_score(
            y_true, y_prediction, average="macro", zero_division=0
        ),
        "macro_auroc": roc_auc_score(
            y_true,
            y_probability,
            average="macro",
            multi_class="ovr",
            labels=class_indices,
        ),
        "per_class_precision": {
            name: float(per_class_precision[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_recall": {
            name: float(per_class_recall[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": class_confusion_matrix.tolist(),
    }

def fit_downstream(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    mixed_precision: bool,
    gradient_clip: float,
    sampling_rate_hz: float,
    recording_length: int,
    resume: bool = False,
) -> Path:
    counts = torch.bincount(train_loader.dataset.labels, minlength=3).float()
    class_weights = (counts.sum() / counts).to(device)
    class_weights = class_weights / class_weights.mean()
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = None
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)
    best_score = -1.0
    stale_epochs = 0
    checkpoint = output_dir / "best_downstream.pt"
    metrics_path = output_dir / "metrics.jsonl"
    start, best_score, stale_epochs = _resume_training(
        model, optimizer, scaler, (train_loader, validation_loader), output_dir,
        "downstream", epochs, best_score, resume, scheduler,
    )
    for epoch in range(start, epochs):
        epoch_start_lr = optimizer.param_groups[0]["lr"]
        train = run_downstream_epoch(
            model,
            train_loader,
            device,
            class_weights=class_weights,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        )
        validation = run_downstream_epoch(
            model,
            validation_loader,
            device,
            class_weights=class_weights,
            optimizer=None,
            scaler=scaler,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        )
        metrics = {
            "train/lr_start": epoch_start_lr,
            "train/lr_end": optimizer.param_groups[0]["lr"],
            "train/global_step": (epoch + 1) * len(train_loader),
            **{f"train/{key}": value for key, value in train.items()},
            **{
                f"validation/{key}": value
                for key, value in validation.items()
            },
        }
        _append_metrics(metrics_path, "downstream", epoch, metrics)
        print(
            f"Downstream epoch={epoch:03d} "
            f"train_loss={train['loss']:.5f} "
            f"val_f1={validation['macro_f1']:.4f} "
            f"val_auc={validation['macro_auroc']:.4f}"
        )
        if validation["macro_f1"] > best_score:
            best_score = validation["macro_f1"]
            stale_epochs = 0
            _atomic_save(
                {
                    "stage": "downstream",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": None,
                    "global_step": (epoch + 1) * len(train_loader),
                    "validation_macro_f1": best_score,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
        _save_last(model, optimizer, scaler, (train_loader, validation_loader),
                   output_dir, "downstream", epoch, best_score, stale_epochs,
                   stale_epochs >= patience or epoch + 1 >= epochs, scheduler)
        if stale_epochs >= patience:
            break
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)[
            "model"
        ]
    )
    return checkpoint

def evaluate_downstream(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    sampling_rate_hz: float,
    recording_length: int,
    mixed_precision: bool,
) -> dict[str, Any]:
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)
    return run_downstream_epoch(
        model,
        loader,
        device,
        class_weights=torch.ones(3, device=device),
        optimizer=None,
        scaler=scaler,
        mixed_precision=mixed_precision,
        gradient_clip=1.0,
        sampling_rate_hz=sampling_rate_hz,
        recording_length=recording_length,
    )
