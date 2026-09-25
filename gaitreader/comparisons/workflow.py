"""Official-source comparisons under the same integrated fixed-test CV protocol."""
import torch
from .models import ComparisonModel
from .data import adapt_loaders
from .training import fit_stage, evaluate
from .sources import MANIFEST
from ..utils import seed_stage, device_for, save_json


def pretrain_comparison(args, spec, original, plan, directory, resume):
    from ..pipeline import checkpoint_record
    supplied = plan["checkpoints"].get(spec["name"], {})
    if "ssl" in supplied:
        return {"vq": None, "ssl": supplied["ssl"]}
    device = device_for(args.device)
    loaders = adapt_loaders(original, args.seed)
    seed_stage(args.seed, loaders)
    model = ComparisonModel(args, device).to(device)
    stages = [s for s in model.stages if s != "downstream"]
    if not stages:
        return {"vq": None, "ssl": None}
    output = directory / "pretraining" / spec["name"]
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "args.json", vars(args))
    save_json(output / "sources.json", MANIFEST)
    # Match the reference wrapper's stage setup before pretraining.
    for stage in model.stages:
        model.set_stage(stage)
    for stage in stages:
        fit_stage(model, loaders, device, args, output, stage, resume)
    return {"vq": None, "ssl": checkpoint_record(output / "best_ssl.pt")}


def train_comparison_fold(args, checkpoints, original, device, output, resume):
    loaders = adapt_loaders(original, args.seed)
    seed_stage(args.seed, loaders)
    model = ComparisonModel(args, device).to(device)
    if checkpoints["ssl"]:
        state = torch.load(checkpoints["ssl"]["path"], map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
    seed_stage(args.seed, loaders)
    model.classifier[-1].reset_parameters()
    save_json(output / "sources.json", MANIFEST)
    fit_stage(model, loaders, device, args, output, "downstream", resume)
    save_json(output / "parameters.json", {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    result = evaluate(model, loaders, device, args, output)
    result.update(checkpoints=checkpoints, downstream_checkpoint=str(output / "best_downstream.pt"))
    save_json(output / "evaluation.json", result)
