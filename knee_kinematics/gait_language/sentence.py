"""Cycle-, DOF-, and bilateral-axis modeling for gait sentences."""

from __future__ import annotations

import torch
from torch import nn

from .vq import DOFWordEncoder


class BilateralCrossAttention(nn.Module):
    """Content-based cross-attention for asynchronously recorded legs."""

    def __init__(
        self, dim: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.output_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return updated queries, aligned context, and attention weights."""
        batch_size, query_count, dim = query.shape
        context_count = context.shape[1]
        q = self.query(self.query_norm(query)).reshape(
            batch_size, query_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        normalized_context = self.context_norm(context)
        k = self.key(normalized_context).reshape(
            batch_size, context_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(normalized_context).reshape(
            batch_size, context_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        key_padding = ~context_mask[:, None, None, :]
        logits = logits.masked_fill(
            key_padding, torch.finfo(logits.dtype).min
        )
        weights = logits.softmax(dim=-1).masked_fill(key_padding, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        attended = torch.matmul(self.attention_dropout(weights), v)
        attended = attended.transpose(1, 2).reshape(
            batch_size, query_count, dim
        )
        attended = attended * query_mask[..., None]
        output = query + self.output(attended)
        output = output + self.ffn(self.output_norm(output))
        output = output * query_mask[..., None]
        return output, attended, weights


class GaitSentenceBlock(nn.Module):
    """Apply cycle, DOF, and bilateral attention in sequence."""

    def __init__(
        self, dim: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(
            dim,
            num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dof = nn.TransformerEncoderLayer(
            dim,
            num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.left_from_right = BilateralCrossAttention(
            dim, num_heads, dropout
        )
        self.right_from_left = BilateralCrossAttention(
            dim, num_heads, dropout
        )

    def _temporal_axis(
        self, tokens: torch.Tensor, word_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sides, words, dofs, dim = tokens.shape
        sequences = tokens.permute(0, 1, 3, 2, 4).reshape(
            batch_size * sides * dofs, words, dim
        )
        masks = word_mask[:, :, None, :].expand(
            batch_size, sides, dofs, words
        ).reshape(batch_size * sides * dofs, words)
        valid_sequences = masks.any(dim=-1)
        output = torch.zeros_like(sequences)
        output[valid_sequences] = self.temporal(
            sequences[valid_sequences],
            src_key_padding_mask=~masks[valid_sequences],
        )
        return output.reshape(
            batch_size, sides, dofs, words, dim
        ).permute(0, 1, 3, 2, 4)

    def _dof_axis(
        self, tokens: torch.Tensor, word_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sides, words, dofs, dim = tokens.shape
        cycles = tokens.reshape(batch_size * sides * words, dofs, dim)
        valid_cycles = word_mask.reshape(-1)
        output = torch.zeros_like(cycles)
        output[valid_cycles] = self.dof(cycles[valid_cycles])
        return output.reshape(batch_size, sides, words, dofs, dim)

    def forward(
        self,
        tokens: torch.Tensor,
        word_mask: torch.Tensor,
        *,
        use_bilateral_context: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return updated sentence and final bilateral alignment tensors."""
        tokens = self._temporal_axis(tokens, word_mask)
        tokens = self._dof_axis(tokens, word_mask)
        if not use_bilateral_context:
            return (
                tokens,
                tokens[:, 1],
                tokens[:, 0],
                tokens.new_zeros(1),
                tokens.new_zeros(1),
            )
        batch_size, _, words, dofs, dim = tokens.shape
        left = tokens[:, 0].permute(0, 2, 1, 3).reshape(
            batch_size * dofs, words, dim
        )
        right = tokens[:, 1].permute(0, 2, 1, 3).reshape(
            batch_size * dofs, words, dim
        )
        left_mask = word_mask[:, 0, None, :].expand(
            batch_size, dofs, words
        ).reshape(batch_size * dofs, words)
        right_mask = word_mask[:, 1, None, :].expand(
            batch_size, dofs, words
        ).reshape(batch_size * dofs, words)
        left_updated, right_context, left_weights = self.left_from_right(
            left,
            right,
            left_mask,
            right_mask,
        )
        right_updated, left_context, right_weights = self.right_from_left(
            right,
            left,
            right_mask,
            left_mask,
        )
        left_updated = left_updated.reshape(
            batch_size, dofs, words, dim
        ).permute(0, 2, 1, 3)
        right_updated = right_updated.reshape(
            batch_size, dofs, words, dim
        ).permute(0, 2, 1, 3)
        right_context = right_context.reshape(
            batch_size, dofs, words, dim
        ).permute(0, 2, 1, 3)
        left_context = left_context.reshape(
            batch_size, dofs, words, dim
        ).permute(0, 2, 1, 3)
        tokens = torch.stack([left_updated, right_updated], dim=1)
        return (
            tokens,
            right_context,
            left_context,
            left_weights,
            right_weights,
        )


class GaitSentenceEncoder(nn.Module):
    """Encode morphology words into rhythm, DOF, and bilateral features."""

    def __init__(
        self,
        word_encoder: DOFWordEncoder,
        *,
        word_dim: int,
        max_words: int,
        depth: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.word_encoder = word_encoder
        self.word_dim = word_dim
        self.max_words = max_words
        self.mask_token = nn.Parameter(torch.empty(6, word_dim))
        self.dof_embedding = nn.Parameter(torch.empty(6, word_dim))
        self.side_embedding = nn.Parameter(torch.empty(2, word_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(max_words, word_dim)
        )
        self.continuous_timing_embedding = nn.Sequential(
            nn.Linear(1, word_dim),
            nn.GELU(),
            nn.Linear(word_dim, word_dim),
        )
        self.duration_embedding = nn.Sequential(
            nn.Linear(1, word_dim),
            nn.GELU(),
            nn.Linear(word_dim, word_dim),
        )
        self.interval_embedding = nn.Sequential(
            nn.Linear(1, word_dim),
            nn.GELU(),
            nn.Linear(word_dim, word_dim),
        )
        self.quality_embedding = nn.Sequential(
            nn.Linear(1, word_dim),
            nn.GELU(),
            nn.Linear(word_dim, word_dim),
        )
        self.timing_mask_embedding = nn.Parameter(
            torch.empty(3, word_dim)
        )
        self.blocks = nn.ModuleList(
            [
                GaitSentenceBlock(word_dim, num_heads, dropout)
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(word_dim)
        self.shared_projection = nn.Sequential(
            nn.Linear(word_dim * 2, word_dim), nn.GELU()
        )
        self.directional_projection = nn.Linear(
            word_dim, word_dim, bias=False
        )
        self.absolute_projection = nn.Sequential(
            nn.Linear(word_dim * 2, word_dim), nn.GELU()
        )
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.dof_embedding, std=0.02)
        nn.init.normal_(self.side_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.timing_mask_embedding, std=0.02)

    @staticmethod
    def _pool_side(
        tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        weights = mask[:, :, None, None].to(tokens.dtype)
        denominator = (
            weights.sum(dim=(1, 2)) * tokens.shape[2]
        ).clamp_min(1.0)
        return (tokens * weights).sum(dim=(1, 2)) / denominator

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
        *,
        masked_positions: torch.Tensor | None = None,
        masked_timing_positions: torch.Tensor | None = None,
        use_bilateral_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return contextual tokens and swap-structured bilateral features."""
        word_count = words.shape[2]
        if word_count > self.max_words:
            raise ValueError(
                f"sentence has {word_count} cycles but max_words={self.max_words}"
            )
        tokens = self.word_encoder(words)
        if masked_positions is not None:
            mask_tokens = self.mask_token[None, None, None]
            tokens = torch.where(
                masked_positions[..., None], mask_tokens, tokens
            )
        continuous_timing = self.continuous_timing_embedding(
            timing[..., 1:2]
        )
        duration = self.duration_embedding(timing[..., 0:1])
        interval = self.interval_embedding(timing[..., 2:3])
        quality = self.quality_embedding(timing[..., 3:4])
        if masked_timing_positions is not None:
            timing_mask = masked_timing_positions[..., None]
            continuous_timing = torch.where(
                timing_mask,
                self.timing_mask_embedding[0],
                continuous_timing,
            )
            duration = torch.where(
                timing_mask,
                self.timing_mask_embedding[1],
                duration,
            )
            interval = torch.where(
                timing_mask,
                self.timing_mask_embedding[2],
                interval,
            )
        tokens = tokens + self.dof_embedding[None, None, None]
        tokens = tokens + self.side_embedding[None, :, None, None]
        tokens = tokens + self.position_embedding[
            None, None, :word_count, None
        ]
        tokens = tokens + continuous_timing[:, :, :, None]
        tokens = tokens + duration[:, :, :, None]
        tokens = tokens + interval[:, :, :, None]
        tokens = tokens + quality[:, :, :, None]
        tokens = tokens * word_mask[..., None, None]
        right_context = tokens[:, 1]
        left_context = tokens[:, 0]
        left_weights = tokens.new_zeros(1)
        right_weights = tokens.new_zeros(1)
        for block in self.blocks:
            (
                tokens,
                right_context,
                left_context,
                left_weights,
                right_weights,
            ) = block(
                tokens,
                word_mask,
                use_bilateral_context=use_bilateral_context,
            )
        tokens = self.output_norm(tokens)
        tokens = tokens * word_mask[..., None, None]
        left = self._pool_side(tokens[:, 0], word_mask[:, 0])
        right = self._pool_side(tokens[:, 1], word_mask[:, 1])
        shared = self.shared_projection(
            torch.cat([left + right, left * right], dim=-1)
        )
        directional = self.directional_projection(left - right)
        absolute = self.absolute_projection(
            torch.cat([(left - right).abs(), left * right], dim=-1)
        )
        left_difference = tokens[:, 0] - right_context
        right_difference = tokens[:, 1] - left_context
        return {
            "tokens": tokens,
            "left_embedding": left,
            "right_embedding": right,
            "shared_embedding": shared,
            "directional_difference": directional,
            "absolute_difference": absolute,
            "left_difference_map": left_difference,
            "right_difference_map": right_difference,
            "left_cross_attention": left_weights,
            "right_cross_attention": right_weights,
        }


__all__ = [
    "BilateralCrossAttention",
    "GaitSentenceBlock",
    "GaitSentenceEncoder",
]
