"""Hierarchical healthy-reference deviations for downstream diagnosis."""

from __future__ import annotations

import torch
from torch import nn


class HierarchicalGaitDeviationEncoder(nn.Module):
    """Aggregate signed healthy-reference shifts from words to subjects."""

    def __init__(
        self,
        *,
        word_dim: int,
        dof_hidden_dim: int,
        dropout: float,
        std_floor: float,
    ) -> None:
        super().__init__()
        self.std_floor = std_floor
        self.register_buffer(
            "reference_mean", torch.zeros(2, 6, word_dim)
        )
        self.register_buffer(
            "reference_std", torch.ones(2, 6, word_dim)
        )
        self.dof_projection = nn.Sequential(
            nn.LayerNorm(word_dim + 5),
            nn.Linear(word_dim + 5, dof_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.side_projection = nn.Sequential(
            nn.LayerNorm(6 * dof_hidden_dim + word_dim + 4),
            nn.Linear(6 * dof_hidden_dim + word_dim + 4, word_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(word_dim * 2, word_dim),
        )
        self.subject_projection = nn.Sequential(
            nn.LayerNorm(word_dim * 5 + 4),
            nn.Linear(word_dim * 5 + 4, word_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(word_dim * 2, word_dim),
        )

    def set_reference(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        """Store side- and DOF-specific healthy token statistics."""
        self.reference_mean.copy_(mean)
        self.reference_std.copy_(std.clamp_min(self.std_floor))

    def forward(
        self, tokens: torch.Tensor, word_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return signed directions and non-negative magnitudes at all levels."""
        valid = word_mask[..., None, None].to(tokens.dtype)
        reference_mean = self.reference_mean[None, :, None]
        reference_std = self.reference_std[None, :, None]
        word_direction = (
            (tokens - reference_mean) / reference_std
        ) * valid
        word_magnitude = word_direction.square().mean(dim=-1).sqrt()

        word_weights = word_mask[..., None].to(tokens.dtype)
        word_count = word_weights.sum(dim=2).clamp_min(1.0)
        dof_direction = (
            (word_direction * word_weights[..., None]).sum(dim=2)
            / word_count[..., None]
        )
        dof_magnitude_mean = (
            (word_magnitude * word_weights).sum(dim=2) / word_count
        )
        dof_magnitude_rms = (
            (word_magnitude.square() * word_weights).sum(dim=2)
            / word_count
        ).sqrt()
        centered_magnitude = (
            word_magnitude - dof_magnitude_mean[:, :, None]
        )
        dof_magnitude_std = (
            (centered_magnitude.square() * word_weights).sum(dim=2)
            / word_count
        ).sqrt()
        dof_magnitude_max = (
            word_magnitude * word_weights
        ).amax(dim=2)
        dof_direction_strength = dof_direction.square().mean(dim=-1).sqrt()
        dof_features = torch.cat(
            [
                dof_direction,
                dof_magnitude_mean[..., None],
                dof_magnitude_rms[..., None],
                dof_magnitude_std[..., None],
                dof_magnitude_max[..., None],
                dof_direction_strength[..., None],
            ],
            dim=-1,
        )
        dof_embedding = self.dof_projection(dof_features)

        side_direction = dof_direction.mean(dim=2)
        side_magnitude_mean = dof_magnitude_mean.mean(
            dim=2, keepdim=True
        )
        side_magnitude_rms = dof_magnitude_rms.square().mean(
            dim=2, keepdim=True
        ).sqrt()
        side_magnitude_max = dof_magnitude_max.max(
            dim=2, keepdim=True
        ).values
        side_direction_strength = side_direction.square().mean(
            dim=-1, keepdim=True
        ).sqrt()
        side_features = torch.cat(
            [
                dof_embedding.flatten(start_dim=2),
                side_direction,
                side_magnitude_mean,
                side_magnitude_rms,
                side_magnitude_max,
                side_direction_strength,
            ],
            dim=-1,
        )
        side_embedding = self.side_projection(side_features)

        side_shared = side_embedding.mean(dim=1)
        side_difference = side_embedding[:, 0] - side_embedding[:, 1]
        side_maximum = side_embedding.maximum(
            side_embedding.flip(dims=(1,))
        )[:, 0]
        subject_direction = side_direction.mean(dim=1)
        bilateral_direction = side_direction[:, 0] - side_direction[:, 1]
        subject_magnitude_mean = side_magnitude_mean.mean(dim=1)
        subject_magnitude_rms = side_magnitude_rms.square().mean(
            dim=1
        ).sqrt()
        subject_magnitude_max = side_magnitude_max.max(dim=1).values
        bilateral_magnitude_gap = (
            side_magnitude_mean[:, 0] - side_magnitude_mean[:, 1]
        )
        subject_features = torch.cat(
            [
                side_shared,
                side_difference.abs(),
                side_maximum,
                subject_direction,
                bilateral_direction.abs(),
                subject_magnitude_mean,
                subject_magnitude_rms,
                subject_magnitude_max,
                bilateral_magnitude_gap.abs(),
            ],
            dim=-1,
        )
        subject_embedding = self.subject_projection(subject_features)
        return {
            "word_deviation_direction": word_direction,
            "word_deviation_magnitude": word_magnitude,
            "dof_deviation_direction": dof_direction,
            "dof_deviation_magnitude_mean": dof_magnitude_mean,
            "dof_deviation_magnitude_rms": dof_magnitude_rms,
            "dof_deviation_magnitude_std": dof_magnitude_std,
            "dof_deviation_magnitude_max": dof_magnitude_max,
            "dof_direction_strength": dof_direction_strength,
            "dof_embedding": dof_embedding,
            "side_deviation_direction": side_direction,
            "side_deviation_magnitude_mean": side_magnitude_mean,
            "side_deviation_magnitude_rms": side_magnitude_rms,
            "side_deviation_magnitude_max": side_magnitude_max,
            "side_direction_strength": side_direction_strength,
            "side_embedding": side_embedding,
            "subject_deviation_direction": subject_direction,
            "subject_deviation_magnitude_mean": subject_magnitude_mean,
            "subject_deviation_magnitude_rms": subject_magnitude_rms,
            "subject_deviation_magnitude_max": subject_magnitude_max,
            "bilateral_deviation_direction": bilateral_direction,
            "bilateral_deviation_magnitude_gap": (
                bilateral_magnitude_gap
            ),
            "subject_embedding": subject_embedding,
        }


__all__ = ["HierarchicalGaitDeviationEncoder"]
