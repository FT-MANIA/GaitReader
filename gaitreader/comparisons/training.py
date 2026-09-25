"""Common training/evaluation protocol, with epoch-level exact RNG continuation."""
import json
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from .. import utils as run
from ..training import CLASS_NAMES, _optimizer_step
from ..utils import (_resume_training, _save_last,
                     _atomic_save, _append_metrics, _random_state, _restore_random)
from .data import prepare_batch


def classification_metrics(labels, probabilities):
    labels = torch.cat(labels).cpu().numpy()
    probabilities = torch.cat(probabilities).double().cpu().numpy()
    probabilities /= probabilities.sum(-1, keepdims=True)
    prediction = probabilities.argmax(-1)
    precision = precision_score(labels, prediction, labels=[0, 1, 2], average=None, zero_division=0)
    recall = recall_score(labels, prediction, labels=[0, 1, 2], average=None, zero_division=0)
    return dict(accuracy=float(accuracy_score(labels, prediction)),
                macro_f1=float(f1_score(labels, prediction, labels=[0, 1, 2], average='macro', zero_division=0)),
                macro_auroc=float(roc_auc_score(labels, probabilities, labels=[0, 1, 2], multi_class='ovr', average='macro')),
                per_class_precision=dict(zip(CLASS_NAMES, precision.tolist())),
                per_class_recall=dict(zip(CLASS_NAMES, recall.tolist())),
                confusion_matrix=confusion_matrix(labels, prediction, labels=[0, 1, 2]).tolist())


def epoch_pass(model, loader, device, args, stage, class_weights, *, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0., 0
    labels, probabilities, subjects = [], [], []
    for raw in loader:
        batch = prepare_batch(raw, device, args)
        # FP32 by default: FFT at length 600 and official quantizers are not all FP16-safe.
        with torch.set_grad_enabled(training):
            if stage == 'downstream':
                logits = model(batch)
                loss = F.cross_entropy(logits, batch['disease_label'], weight=class_weights)
                labels.append(batch['disease_label'].detach())
                probabilities.append(logits.detach().softmax(-1))
                subjects.extend(str(value) for value in batch['subject_id'])
            else:
                loss = model.pretrain_loss(batch, stage)
        if training:
            _optimizer_step(loss, model, optimizer, scaler, args.gradient_clip)
            model.after_update()
        b = len(batch['disease_label'])
        total += float(loss.detach()) * b
        count += b
    metrics = dict(loss=total / count)
    predictions = None
    if stage == 'downstream':
        metrics.update(classification_metrics(labels, probabilities))
        predictions = dict(subject_id=subjects, labels=torch.cat(labels).cpu(),
                           probabilities=torch.cat(probabilities).cpu())
    return metrics, predictions


def fit_stage(model, loaders, device, args, output_dir, stage, resume):
    model.set_stage(stage)
    run.seed_stage(args.seed, loaders)
    train = loaders['dev_data' if stage == 'downstream' else 'ssl_data']
    validation = loaders['dev_validation_data' if stage == 'downstream' else 'ssl_validation_data']
    epochs = getattr(args, f'{stage}_epochs')
    patience = getattr(args, f'{stage}_patience')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=getattr(args, f'{stage}_learning_rate'),
                                  weight_decay=getattr(args, f'{stage}_weight_decay'))
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    initial = -float('inf') if stage == 'downstream' else float('inf')
    start, best, stale = _resume_training(model, optimizer, scaler, (train, validation), output_dir,
                                         stage, epochs, initial, resume)
    weights = None
    if stage == 'downstream':
        counts = torch.bincount(train.dataset.labels, minlength=3).float().to(device)
        weights = counts.sum() / counts
        weights /= weights.mean()
    for epoch in range(start, epochs):
        train_metrics, _ = epoch_pass(model, train, device, args, stage, weights,
                                      optimizer=optimizer, scaler=scaler)
        state = _random_state((train, validation))
        run._set_seed(args.seed)
        validation.generator.manual_seed(args.seed)
        validation_metrics, _ = epoch_pass(model, validation, device, args, stage, weights)
        _restore_random(state, (train, validation))
        score = validation_metrics['macro_f1' if stage == 'downstream' else 'loss']
        improved = score > best if stage == 'downstream' else score < best
        if improved:
            best, stale = score, 0
            _atomic_save(dict(stage=stage, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                              validation_macro_f1=score if stage == 'downstream' else None,
                              validation_loss=validation_metrics['loss']), output_dir / f'best_{stage}.pt')
        else:
            stale += 1
        _append_metrics(output_dir / 'metrics.jsonl', stage, epoch,
                        {**{f'train/{key}': value for key, value in train_metrics.items()},
                         **{f'validation/{key}': value for key, value in validation_metrics.items()},
                         'train/lr': optimizer.param_groups[0]['lr']})
        finished = stale >= patience or epoch + 1 == epochs
        _save_last(model, optimizer, scaler, (train, validation), output_dir, stage, epoch, best, stale, finished)
        print(f'{args.comparison_model} {stage} epoch={epoch:03d} train_loss={train_metrics["loss"]:.6f} val_score={score:.6f}', flush=True)
        if finished:
            break
    model.load_state_dict(torch.load(output_dir / f'best_{stage}.pt', map_location=device, weights_only=False)['model'])


def evaluate(model, loaders, device, args, output_dir):
    result = {}
    for name, key in [('internal', 'dev_test_data'), ('external', 'ext_test_data')]:
        metrics, predictions = epoch_pass(model, loaders[key], device, args, 'downstream', None)
        result[f'{name}_test'] = metrics
        _atomic_save(predictions, output_dir / f'{name}_predictions.pt')
    (output_dir / 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
