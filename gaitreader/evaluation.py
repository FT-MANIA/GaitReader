"""Shared subject-stratified folds; fixed internal/external cohorts."""
from collections import Counter
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from .data.builders import LabeledSubjectSubset, seed_dataloader_worker
from .data.collate import collate_kinematic_subjects
from .utils import read_json, save_json

class SubjectPool(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.labels = torch.cat([dataset.labels for dataset in datasets])
        self.trace_info = [trace for dataset in datasets for trace in dataset.trace_info]

def subject_rows(dataset, group_map):
    return [{"index": index, "subject_id": str(trace["subject_id"]),
             "group_id": group_map.get(str(trace["subject_id"]), str(trace["subject_id"])),
             "disease_label": int(dataset.labels[index])}
            for index, trace in enumerate(dataset.trace_info)]

def prepare_folds(plan, loaders, directory):
    # Unwrap the original label subset, keeping every post-QC training subject.
    pool = SubjectPool([loaders["dev_data"].dataset.dataset, loaders["dev_validation_data"].dataset])
    mapping = plan["subject_group_mapping"]
    cohorts = {"pool": subject_rows(pool, mapping),
               "internal_test": subject_rows(loaders["dev_test_data"].dataset, mapping),
               "external_test": subject_rows(loaders["ext_test_data"].dataset, mapping)}
    groups = {name: {row["group_id"] for row in rows} for name, rows in cohorts.items()}
    for first, second in (("pool", "internal_test"), ("pool", "external_test"), ("internal_test", "external_test")):
        overlap = groups[first] & groups[second]
        if overlap:
            raise ValueError(f"Subject leakage between {first} and {second}: {sorted(overlap)}")
    path = directory / "fold_manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest["fold_seed"] != plan["fold_seed"] or len(manifest["folds"]) != 5:
            raise ValueError("The imported manifest must use the requested seed and five folds.")
        if manifest["cohorts"] != cohorts:
            raise ValueError("Processed subject IDs/labels/order changed; cannot resume the saved folds.")
        if manifest["standardization"] != loaders["dev_validation_data"].dataset.standardizer.summary():
            raise ValueError("Normalization differs from the shared reference; cannot compare these representations.")
        for fold in manifest["folds"]:
            fold["labeled_subsets"] = make_labeled_subsets(cohorts["pool"], fold["train_indices"], plan["label_fractions"], plan["label_subset_seed"])
        save_json(path, manifest)
        return pool, manifest
    # Stratify UNIQUE subjects, then put ALL their records into the same fold.
    # This gives 439/110 (or 440/109) with 549 unique subjects, unlike record splitting.
    labels_by_group = {}
    for row in cohorts["pool"]:
        group, label = row["group_id"], row["disease_label"]
        if group in labels_by_group and labels_by_group[group] != label:
            raise ValueError(f"Conflicting disease labels for subject group {group}.")
        labels_by_group[group] = label
    subject_ids = np.array(sorted(labels_by_group))
    labels = np.array([labels_by_group[group] for group in subject_ids])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=plan["fold_seed"])
    folds = []
    for number, (train, validation) in enumerate(splitter.split(subject_ids, labels), 1):
        fold = {"fold": number}
        for name, indices in (("train", train), ("validation", validation)):
            selected_groups = set(subject_ids[indices])
            records = [row["index"] for row in cohorts["pool"] if row["group_id"] in selected_groups]
            fold[f"{name}_indices"] = records
            fold[f"{name}_subjects"] = len(selected_groups)
            fold[f"{name}_records"] = len(records)
            fold[f"{name}_class_counts"] = torch.bincount(pool.labels[records], minlength=3).tolist()
        folds.append(fold)
    for fold in folds:
        fold["labeled_subsets"] = make_labeled_subsets(
            cohorts["pool"], fold["train_indices"], plan["label_fractions"], plan["label_subset_seed"])
    manifest = {"cohorts": cohorts, "fold_seed": plan["fold_seed"], "folds": folds,
                "standardization": loaders["dev_validation_data"].dataset.standardizer.summary()}
    save_json(path, manifest)
    return pool, manifest

def make_labeled_subsets(rows, train_indices, fractions, seed):
    """One class-wise subject permutation; all fractions take nested prefixes."""
    train_rows = [rows[index] for index in train_indices]
    labels_by_group = {row["group_id"]: row["disease_label"] for row in train_rows}
    generator = torch.Generator().manual_seed(seed)
    permutations = {}
    for label in sorted(set(labels_by_group.values())):
        groups = sorted(group for group, value in labels_by_group.items() if value == label)
        order = torch.randperm(len(groups), generator=generator).tolist()
        permutations[label] = [groups[index] for index in order]
    subsets = {}
    for fraction in fractions:
        selected = {group for groups in permutations.values()
                    for group in groups[:max(1, round(len(groups) * fraction))]}
        selected_rows = [row for row in train_rows if row["group_id"] in selected]
        subsets[str(fraction)] = {
            "requested_fraction": fraction, "subset_seed": seed,
            "full_training_subjects": len(labels_by_group), "selected_subjects": len(selected),
            "selected_records": len(selected_rows),
            "class_counts": [sum(labels_by_group[group] == label for group in selected) for label in range(3)],
            "indices_in_pool": [row["index"] for row in selected_rows], "subjects": selected_rows,
        }
    return subsets

def fold_loaders(pool, fold, original, args):
    subset = fold["labeled_subsets"][str(args.downstream_label_fraction)]
    datasets = {"dev_data": LabeledSubjectSubset(pool, subset["indices_in_pool"]),
                "dev_validation_data": LabeledSubjectSubset(pool, fold["validation_indices"]),
                "dev_test_data": original["dev_test_data"].dataset,
                "ext_test_data": original["ext_test_data"].dataset}
    return {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "dev_data",
                            num_workers=args.num_workers, collate_fn=collate_kinematic_subjects,
                            worker_init_fn=seed_dataloader_worker,
                            generator=torch.Generator().manual_seed(args.downstream_training_seed))
            for name, dataset in datasets.items()}

def summarize(plan, directory):
    summary = {"protocol": plan["protocol"], "n_splits": 5,
               "note": "Mean/sample SD across fold-trained models on the SAME fixed tests; not independent test cohorts, not an ensemble. Never select the best fold by test score.",
               "methods": {}}
    for method in plan["methods"]:
        files = sorted((directory / method["name"]).glob("fold_*/evaluation.json"))
        results = [read_json(path) for path in files]
        entry = {"completed_folds": len(results), "complete": len(results) == 5}
        for partition in ("internal_test", "external_test"):
            metrics = {}
            if results:
                for key in ("accuracy", "macro_f1", "macro_auroc", "loss"):
                    values = [row[partition][key] for row in results]
                    metrics[key] = {"values": values, "mean": float(np.mean(values)),
                                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
                for key in ("per_class_precision", "per_class_recall"):
                    metrics[key] = {}
                    for label in ("Healthy", "ACLD", "KOA"):
                        values = [row[partition][key][label] for row in results]
                        metrics[key][label] = {"values": values, "mean": float(np.mean(values)),
                                              "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
                metrics["confusion_matrices"] = [row[partition]["confusion_matrix"] for row in results]
            entry[partition] = metrics
        summary["methods"][method["name"]] = entry
    save_json(directory / "summary.json", summary)
    save_brief_summary(summary, directory)

def save_brief_summary(summary, directory):
    """Four mean metrics per method, using the currently completed folds."""
    brief = {
        name: {
            f"{split}_{metric}_mean": (
                entry[f"{split}_test"][metric]["mean"]
                if entry["completed_folds"] else None
            )
            for split in ("internal", "external")
            for metric in ("accuracy", "macro_f1")
        }
        for name, entry in summary["methods"].items()
    }
    save_json(directory / "summary_brief.json", brief)
