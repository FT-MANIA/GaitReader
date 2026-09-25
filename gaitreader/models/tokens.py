"""Independent per-DOF waveform encoders and shape/attribute factorization."""
from __future__ import annotations
import torch
from torch import nn


class GaitPatchEmbedding(nn.Module):
    """Map one 100-point DOF waveform to one gait-word patch."""

    def __init__(
        self,
        word_dim: int,
        hidden_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                1,
                hidden_channels,
                kernel_size=9,
                stride=4,
                padding=4,
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels,
                output_channels,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(output_channels, word_dim)

    def forward(self, words: torch.Tensor) -> torch.Tensor:
        leading = words.shape[:-1]
        flattened = words.reshape(-1, 1, words.shape[-1])
        embedded = self.network(flattened).squeeze(-1)
        return self.projection(embedded).reshape(*leading, -1)


class GaitShapeEncoder(nn.Module):
    """Encode every DOF gait word independently from waveform shape."""

    def __init__(
        self,
        *,
        word_dim: int,
        patch_hidden_channels: int,
        patch_output_channels: int,
    ) -> None:
        super().__init__()
        self.word_dim = word_dim
        self.patch_embeddings = nn.ModuleList(
            [
                GaitPatchEmbedding(
                    word_dim,
                    patch_hidden_channels,
                    patch_output_channels,
                )
                for _ in range(6)
            ]
        )

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = torch.stack(
            [
                encoder(words[..., dof, :])
                for dof, encoder in enumerate(self.patch_embeddings)
            ],
            dim=3,
        )
        token_weights = word_mask[..., None, None].to(tokens.dtype)
        tokens = tokens * token_weights
        side_embedding = tokens.sum(dim=(2, 3))
        denominator = word_mask.sum(
            dim=2, keepdim=True
        ).to(tokens.dtype)
        side_embedding = side_embedding / (denominator * 6)
        subject_embedding = side_embedding.mean(dim=1)
        return {
            "tokens": tokens,
            "side_embedding": side_embedding,
            "subject_embedding": subject_embedding,
            "cls_embedding": subject_embedding,
        }


def split_shape(words: torch.Tensor, scale_floor: float):
    with torch.autocast(device_type=words.device.type, enabled=False):
        values = words.float()
        mean = values.mean(dim=-1, keepdim=True)
        scale = values.std(dim=-1, keepdim=True, correction=0).clamp_min(scale_floor)
        shape = (values - mean) / scale
        attributes = torch.cat((mean, scale.log()), dim=-1)
    return shape, attributes, mean, scale


class ShapeAttributeVQEncoder(nn.Module):
    """Raw words -> shape-only code-space features plus explicit attributes."""

    def __init__(self, encoder, scale_floor, separate_shape=True):
        super().__init__()
        self.encoder = encoder
        self.scale_floor = scale_floor
        self.separate_shape = separate_shape

    def forward(self, words, word_mask, timing):
        shape, attributes, mean, scale = split_shape(words, self.scale_floor)
        encoded = self.encoder(shape if self.separate_shape else words, word_mask, timing)
        valid = word_mask[..., None, None]
        return {**encoded, "shape_words": shape * valid,
                "word_attributes": attributes * valid,
                "word_mean": mean * valid, "word_scale": scale * valid}


class ShapeAttributePatchEmbedding(nn.Module):
    """Separate shape CNN and mean/log-scale projection before SSL masking."""

    def __init__(self, embedding, word_dim, scale_floor, *,
                 use_shape=True, use_mean=True, use_scale=True):
        super().__init__()
        self.shape_embedding = embedding
        self.attribute_embedding = nn.Linear(2, word_dim)
        self.scale_floor = scale_floor
        self.use_shape = use_shape
        self.use_mean = use_mean
        self.use_scale = use_scale

    def forward(self, words):
        shape, attributes, _, _ = split_shape(words, self.scale_floor)
        tokens = self.shape_embedding(shape)
        if not self.use_shape:
            tokens = torch.zeros_like(tokens)
        if self.use_mean or self.use_scale:
            if not (self.use_mean and self.use_scale):
                attributes = attributes.clone()
                if not self.use_mean:
                    attributes[..., 0] = 0
                if not self.use_scale:
                    attributes[..., 1] = 0
            tokens = tokens + self.attribute_embedding(attributes)
        return tokens
