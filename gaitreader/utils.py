"""Shared RNG, atomic checkpoints, epoch-boundary resume and serialization."""
from __future__ import annotations
import json
import os
import platform
import random
import shutil
import time
import hashlib
from pathlib import Path
import numpy as np
import torch

def _atomic_save(state: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    # Windows may briefly deny replacement while another process holds a file.
    # Retry only the rename; never remove the last committed checkpoint first.
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 5:
                raise
            delay = 0.1 * (2 ** attempt)
            print(f"Checkpoint replacement denied; retry {attempt + 1}/5 in {delay:.1f}s: {path}", flush=True)
            time.sleep(delay)


def _random_state(loaders) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": [loader.generator.get_state() if loader.generator is not None else None for loader in loaders],
    }


def _restore_random(state: dict, loaders) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    for loader, value in zip(loaders, state["loaders"]):
        if value is not None:
            loader.generator.set_state(value)


def _resume_training(model, optimizer, scaler, loaders, output_dir, stage,
                     epochs, initial_best, resume, scheduler=None):
    if not resume:
        return 0, initial_best, 0
    last = output_dir / f"last_{stage}.pt"
    best = output_dir / f"best_{stage}.pt"
    if not last.exists() and not best.exists():
        return 0, initial_best, 0
    state = torch.load(last if last.exists() else best, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler_state = state.get("scheduler")
    if scheduler_state is not None and scheduler_state["kind"] != "none":
        raise ValueError("The paper protocol uses constant learning rates; this checkpoint used a scheduler.")
    if "random_state" in state:
        # Keep best and last consistent even if a process died between saves.
        _atomic_save(state["best_checkpoint"], best)
        _restore_random(state["random_state"], loaders)
        if scaler.is_enabled() and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        best_value, stale = state["best_value"], state["stale_epochs"]
        finished = state["finished"]
    else:
        print(f"Resume {stage} from legacy best epoch {state['epoch']}: RNG/scaler/history unavailable; not exact continuation.")
        metric = "validation_macro_f1" if stage == "downstream" else "validation_loss"
        best_value, stale = state[metric], 0
        later = {"vq": ("ssl", "downstream"), "ssl": ("downstream",), "downstream": ()}[stage]
        finished = any((output_dir / f"best_{name}.pt").exists() for name in later)
    start = state["epoch"] + 1
    if not finished:
        # Preserve the original log, then remove uncommitted/replayed epochs.
        path = output_dir / "metrics.jsonl"
        if path.exists():
            shutil.copy2(path, output_dir / f"metrics_before_resume_{stage}_{start}.jsonl")
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            rows = [row for row in rows if row["stage"] != stage or row["epoch"] < start]
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            temporary.replace(path)
    print(f"Resume {stage}: {'completed; loading best' if finished else f'next epoch={start}'}")
    return epochs if finished else start, best_value, stale


def _save_last(model, optimizer, scaler, loaders, output_dir, stage,
               epoch, best_value, stale_epochs, finished, scheduler=None):
    _atomic_save({
        "stage": stage, "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "scheduler": None,
        "global_step": (epoch + 1) * len(loaders[0]),
        "random_state": _random_state(loaders), "best_value": best_value,
        "best_checkpoint": torch.load(output_dir / f"best_{stage}.pt", map_location="cpu", weights_only=False),
        "stale_epochs": stale_epochs, "finished": finished,
    }, output_dir / f"last_{stage}.pt")


def _append_metrics(
    path: Path, stage: str, epoch: int, metrics: dict[str, Any]
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"stage": stage, "epoch": epoch, **metrics},
                ensure_ascii=False,
            )
            + "\n"
        )

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_stage(seed, loaders):
    _set_seed(seed)
    for loader in loaders.values():
        loader.generator.manual_seed(seed)

def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)

def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_weights(model, path):
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    # Historical checkpoints contain a disabled rhythm head and timing-mask token.
    # Those parameters never participate in the paper's forward pass.
    state = {k: v for k, v in state.items()
             if not k.startswith("rhythm_head.") and not k.endswith("timing_mask_token")}
    model.load_state_dict(state, strict=True)

def environment():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "platform": platform.platform()}

def device_for(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
