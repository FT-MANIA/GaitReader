"""Preserve the main pipeline's subjects, splits, QC, labels and normalization."""
from bisect import bisect_right

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, RandomSampler
from gaitreader.data.repository import collate_kinematic_subjects
from gaitreader.data.builders import seed_dataloader_worker
from gaitreader.data.batch import build_language_batch


class BenchmarkSubjects(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = dataset.labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        source = self.dataset
        # CV folds subset a concatenated original train+validation pool. Resolve
        # the raw recording from the same leaf dataset as the gait-cycle sample.
        while isinstance(source, (Subset, ConcatDataset)):
            if isinstance(source, Subset):
                index = int(source.indices[index])
                source = source.dataset
            else:
                part = bisect_right(source.cumulative_sizes, index)
                index -= source.cumulative_sizes[part - 1] if part else 0
                source = source.datasets[part]
        raw = source.raw_data[index].reshape(2, 6, -1)
        raw = raw.index_select(1, source.standardizer.MODEL_DOF_ORDER)
        sample['benchmark_signal'] = source.standardizer.transform(raw).transpose(1, 2)
        return sample


def collate_subjects(samples):
    batch = collate_kinematic_subjects(samples)
    batch['benchmark_signal'] = torch.stack([sample['benchmark_signal'] for sample in samples])
    return batch


def adapt_loaders(loaders, seed):
    return {name: DataLoader(BenchmarkSubjects(loader.dataset), batch_size=loader.batch_size,
                            shuffle=isinstance(loader.sampler, RandomSampler),
                            drop_last=name == 'ssl_data',
                            num_workers=loader.num_workers, collate_fn=collate_subjects,
                            worker_init_fn=seed_dataloader_worker,
                            generator=torch.Generator().manual_seed(seed))
            for name, loader in loaders.items()}


def prepare_batch(raw, device, args):
    batch = build_language_batch(raw, sampling_rate_hz=args.sampling_rate_hz,
                                 recording_length=args.recording_length)
    batch['signal'] = raw['benchmark_signal']
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}
