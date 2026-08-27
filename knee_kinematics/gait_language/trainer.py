"""Compact VQ, masked-language, and downstream training loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import build_language_batch, move_language_batch


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


def _append_metrics(
    path: Path, stage: str, epoch: int, metrics: dict[str, float]
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"stage": stage, "epoch": epoch, **metrics},
                ensure_ascii=False,
            )
            + "\n"
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
    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()


@torch.no_grad()
def fit_healthy_deviation_reference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    sampling_rate_hz: float,
    recording_length: int,
) -> None:
    """Fit side/DOF token moments from healthy KGKD training subjects."""
    model.eval()
    reference = model.deviation_encoder
    token_sum = torch.zeros_like(reference.reference_mean, dtype=torch.float32)
    token_square_sum = torch.zeros_like(token_sum)
    token_count = token_sum.new_zeros(2, 6, 1)
    for raw_batch in loader:
        batch = _language_batch(
            raw_batch,
            device,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        )
        tokens = model.sentence_encoder(
            batch["words"], batch["word_mask"], batch["timing"]
        )["tokens"].float()
        healthy = batch["disease_label"].eq(0)
        valid = (
            batch["word_mask"][..., None]
            & healthy[:, None, None, None]
        ).expand(-1, -1, -1, 6)
        weights = valid[..., None].to(tokens.dtype)
        token_sum += (tokens * weights).sum(dim=(0, 2))
        token_square_sum += (tokens.square() * weights).sum(dim=(0, 2))
        token_count += weights.sum(dim=(0, 2))
    mean = token_sum / token_count
    variance = token_square_sum / token_count - mean.square()
    reference.set_reference(mean, variance.clamp_min(0.0).sqrt())


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
    """Run one vocabulary construction or validation epoch."""
    training = optimizer is not None
    model.train(training)
    keys = (
        "loss",
        "reconstruction_loss",
        "local_reconstruction_loss",
        "residual_energy_loss",
        "raw_context_residual_rms",
        "scaled_context_residual_rms",
        "local_to_final_improvement",
        "context_residual_rms",
        "velocity_loss",
        "commitment_loss",
        "active_code_ratio",
        "perplexity",
    )
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
            device_type=device.type,
            enabled=mixed_precision,
        ):
            output = model(
                batch["words"], batch["word_mask"], batch["timing"]
            )
        if training:
            _optimizer_step(
                output["loss"],
                model,
                optimizer,
                scaler,
                gradient_clip,
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
) -> Path:
    """Fit the vocabulary tokenizer and save the best checkpoint."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=mixed_precision
    )
    best_loss = float("inf")
    stale_epochs = 0
    checkpoint = output_dir / "best_vq.pt"
    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(epochs):
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
            f"active={validation['active_code_ratio']:.3f} "
            f"perplexity={validation['perplexity']:.1f}"
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            torch.save(
                {
                    "stage": "vq",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "validation_loss": best_loss,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
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
    """Run one SSL epoch, optionally using a reproducible mask stream."""
    training = optimizer is not None
    model.train(training)
    model.target_tokenizer.eval()
    mask_generator = None
    if mask_seed is not None:
        mask_generator = torch.Generator(device=device).manual_seed(
            mask_seed
        )
    keys = (
        "loss",
        "within_loss",
        "cross_dof_loss",
        "cross_dof_hard_loss",
        "cross_dof_soft_loss",
        "cross_dof_prototype_loss",
        "rhythm_loss",
        "duration_loss",
        "interval_loss",
        "duration_mae",
        "interval_mae",
        "bilateral_loss",
        "contralateral_loss",
        "contralateral_hard_loss",
        "contralateral_soft_loss",
        "contralateral_prototype_loss",
        "bilateral_pair_loss",
        "swap_loss",
        "within_accuracy",
        "cross_dof_accuracy",
        "cross_dof_topk_accuracy",
        "bilateral_accuracy",
        "contralateral_accuracy",
        "contralateral_topk_accuracy",
        "bilateral_pair_accuracy",
    )
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
            device_type=device.type,
            enabled=mixed_precision,
        ):
            output = model(
                batch["words"],
                batch["word_mask"],
                batch["timing"],
                mask_generator=mask_generator,
            )
        if training:
            _optimizer_step(
                output["loss"],
                model,
                optimizer,
                scaler,
                gradient_clip,
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
) -> Path:
    """Fit masked gait-language tasks and save the best sentence encoder."""
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=mixed_precision
    )
    best_loss = float("inf")
    stale_epochs = 0
    checkpoint = output_dir / "best_ssl.pt"
    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(epochs):
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
            mask_seed=None,
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
            **{f"train/{key}": value for key, value in train.items()},
            **{
                f"validation/{key}": value
                for key, value in validation.items()
            },
        }
        _append_metrics(metrics_path, "ssl", epoch, metrics)
        print(
            f"SSL epoch={epoch:03d} "
            f"train={train['loss']:.5f} "
            f"validation={validation['loss']:.5f} "
            f"word_acc={validation['within_accuracy']:.3f} "
            f"dof_topk={validation['cross_dof_topk_accuracy']:.3f} "
            f"duration_mae={validation['duration_mae']:.3f} "
            f"pair_acc={validation['bilateral_pair_accuracy']:.3f}"
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            torch.save(
                {
                    "stage": "ssl",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "sentence_encoder": model.sentence_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "validation_loss": best_loss,
                    "validation_mask_seed": validation_mask_seed,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)[
            "model"
        ]
    )
    return checkpoint


def _downstream_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    class_weights: torch.Tensor,
    affected_side_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    disease = F.cross_entropy(
        output["disease_logits"],
        batch["disease_label"],
        weight=class_weights,
    )
    valid = batch["affected_side_valid_mask"]
    affected = output["affected_side_logits"].sum() * 0.0
    if valid.any():
        affected = F.cross_entropy(
            output["affected_side_logits"][valid],
            batch["affected_side_label"][valid],
        )
    return disease + affected_side_weight * affected, disease, affected


def run_downstream_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    class_weights: torch.Tensor,
    affected_side_weight: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    gradient_clip: float,
    sampling_rate_hz: float,
    recording_length: int,
) -> dict[str, float]:
    """Run downstream training or evaluation and calculate subject metrics."""
    training = optimizer is not None
    model.train(training)
    if training and not any(
        parameter.requires_grad
        for parameter in model.sentence_encoder.parameters()
    ):
        model.sentence_encoder.eval()
    total_loss = 0.0
    total_disease = 0.0
    total_affected = 0.0
    total_word_deviation = 0.0
    total_dof_deviation = 0.0
    total_side_deviation = 0.0
    total_subject_deviation = 0.0
    total_bilateral_magnitude_gap = 0.0
    sample_count = 0
    labels = []
    probabilities = []
    predictions = []
    affected_labels = []
    affected_predictions = []
    for raw_batch in loader:
        batch = _language_batch(
            raw_batch,
            device,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        )
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            enabled=mixed_precision,
        ):
            output = model(
                batch["words"], batch["word_mask"], batch["timing"]
            )
            loss, disease, affected = _downstream_loss(
                output,
                batch,
                class_weights,
                affected_side_weight,
            )
        if training:
            _optimizer_step(
                loss, model, optimizer, scaler, gradient_clip
            )
        batch_size = batch["words"].shape[0]
        sample_count += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_disease += float(disease.detach()) * batch_size
        total_affected += float(affected.detach()) * batch_size
        word_valid = batch["word_mask"][..., None].expand_as(
            output["word_deviation_magnitude"]
        )
        total_word_deviation += float(
            output["word_deviation_magnitude"][word_valid].mean().detach()
        ) * batch_size
        total_dof_deviation += float(
            output["dof_deviation_magnitude_mean"].mean().detach()
        ) * batch_size
        total_side_deviation += float(
            output["side_deviation_magnitude_mean"].mean().detach()
        ) * batch_size
        total_subject_deviation += float(
            output["subject_deviation_magnitude_mean"].mean().detach()
        ) * batch_size
        total_bilateral_magnitude_gap += float(
            output["bilateral_deviation_magnitude_gap"]
            .abs()
            .mean()
            .detach()
        ) * batch_size
        probability = output["disease_logits"].float().softmax(dim=-1)
        labels.append(batch["disease_label"].detach().cpu())
        probabilities.append(probability.detach().cpu())
        predictions.append(probability.argmax(dim=-1).detach().cpu())
        affected_valid = batch["affected_side_valid_mask"]
        if affected_valid.any():
            affected_labels.append(
                batch["affected_side_label"][affected_valid].detach().cpu()
            )
            affected_predictions.append(
                output["affected_side_logits"][affected_valid]
                .argmax(dim=-1)
                .detach()
                .cpu()
            )
    y_true = torch.cat(labels).numpy()
    y_probability = torch.cat(probabilities).double().numpy()
    y_probability /= y_probability.sum(axis=1, keepdims=True)
    y_prediction = torch.cat(predictions).numpy()
    affected_accuracy = 0.0
    if affected_labels:
        affected_accuracy = accuracy_score(
            torch.cat(affected_labels).numpy(),
            torch.cat(affected_predictions).numpy(),
        )
    return {
        "loss": total_loss / sample_count,
        "disease_loss": total_disease / sample_count,
        "affected_side_loss": total_affected / sample_count,
        "word_deviation_magnitude": (
            total_word_deviation / sample_count
        ),
        "dof_deviation_magnitude": total_dof_deviation / sample_count,
        "side_deviation_magnitude": total_side_deviation / sample_count,
        "subject_deviation_magnitude": (
            total_subject_deviation / sample_count
        ),
        "bilateral_magnitude_gap": (
            total_bilateral_magnitude_gap / sample_count
        ),
        "accuracy": accuracy_score(y_true, y_prediction),
        "macro_f1": f1_score(
            y_true, y_prediction, average="macro", zero_division=0
        ),
        "macro_auroc": roc_auc_score(
            y_true,
            y_probability,
            average="macro",
            multi_class="ovr",
            labels=np.arange(3),
        ),
        "affected_side_accuracy": affected_accuracy,
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
    affected_side_weight: float,
    mixed_precision: bool,
    gradient_clip: float,
    sampling_rate_hz: float,
    recording_length: int,
) -> Path:
    """Train deviation-aware disease heads by validation Macro-F1."""
    fit_healthy_deviation_reference(
        model,
        train_loader,
        device,
        sampling_rate_hz=sampling_rate_hz,
        recording_length=recording_length,
    )
    counts = torch.bincount(train_loader.dataset.labels, minlength=3).float()
    class_weights = (counts.sum() / counts.clamp_min(1.0)).to(device)
    class_weights = class_weights / class_weights.mean()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=mixed_precision
    )
    best_score = -1.0
    stale_epochs = 0
    checkpoint = output_dir / "best_downstream.pt"
    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(epochs):
        train = run_downstream_epoch(
            model,
            train_loader,
            device,
            class_weights=class_weights,
            affected_side_weight=affected_side_weight,
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
            affected_side_weight=affected_side_weight,
            optimizer=None,
            scaler=scaler,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
            sampling_rate_hz=sampling_rate_hz,
            recording_length=recording_length,
        )
        metrics = {
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
            torch.save(
                {
                    "stage": "downstream",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "validation_macro_f1": best_score,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
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
) -> dict[str, float]:
    """Evaluate a downstream checkpoint on an untouched subject split."""
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)
    class_weights = torch.ones(3, device=device)
    return run_downstream_epoch(
        model,
        loader,
        device,
        class_weights=class_weights,
        affected_side_weight=0.0,
        optimizer=None,
        scaler=scaler,
        mixed_precision=mixed_precision,
        gradient_clip=1.0,
        sampling_rate_hz=sampling_rate_hz,
        recording_length=recording_length,
    )


__all__ = [
    "evaluate_downstream",
    "fit_downstream",
    "fit_healthy_deviation_reference",
    "fit_ssl",
    "fit_vq",
]
