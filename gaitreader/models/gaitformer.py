"""GaitFormer: DOF-specific shape/attribute tokens and bilateral context."""
from __future__ import annotations
import torch
from torch import nn
from .tokens import GaitPatchEmbedding


class WithinSideEncoder(nn.Module):
    """Attend jointly over all cycle and DOF patches within each side."""

    def __init__(
        self,
        *,
        word_dim: int,
        patch_hidden_channels: int,
        patch_output_channels: int,
        max_words: int,
        depth: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        use_cls_token: bool = False,
        use_dof_embedding: bool = True,
        use_cycle_embedding: bool = True,
        use_duration_embedding: bool = True,
        use_interval_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.word_dim = word_dim
        self.max_words = max_words
        self.use_cls_token = use_cls_token
        self.use_dof_embedding = use_dof_embedding
        self.use_cycle_embedding = use_cycle_embedding
        self.use_duration_embedding = use_duration_embedding
        self.use_interval_embedding = use_interval_embedding
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
        self.mask_token = nn.Parameter(torch.empty(word_dim))
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.empty(word_dim))
        self.dof_embedding = nn.Parameter(torch.empty(6, word_dim))
        self.cycle_embedding = nn.Parameter(
            torch.empty(max_words, word_dim)
        )
        self.timing_projection = nn.Sequential(
            nn.Linear(2, word_dim),
            nn.GELU(),
            nn.Linear(word_dim, word_dim),
        )
        layer = nn.TransformerEncoderLayer(
            word_dim,
            num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(word_dim),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.mask_token, std=0.02)
        if use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)
        # Preserve the published initialization stream after removing the unused timing-mask parameter.
        nn.init.normal_(torch.empty(word_dim), std=0.02)
        nn.init.normal_(self.dof_embedding, std=0.02)
        nn.init.normal_(self.cycle_embedding, std=0.02)

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
        *,
        masked_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, sides, word_count, dofs, _ = words.shape
        tokens = torch.stack(
            [
                encoder(words[..., dof, :])
                for dof, encoder in enumerate(self.patch_embeddings)
            ],
            dim=3,
        )
        if masked_positions is not None:
            tokens = torch.where(
                masked_positions[..., None], self.mask_token, tokens
            )
        if self.use_dof_embedding:
            tokens = tokens + self.dof_embedding[None, None, None]
        if self.use_cycle_embedding:
            tokens = tokens + self.cycle_embedding[
                None, None, :word_count, None
            ]
        if self.use_duration_embedding or self.use_interval_embedding:
            timing_input = timing[..., (0, 2)]
            if not self.use_duration_embedding:
                timing_input[..., 0] = 0
            if not self.use_interval_embedding:
                timing_input[..., 1] = 0
            timing_embedding = self.timing_projection(timing_input)
            tokens = tokens + timing_embedding[..., None, :]

        sequence = tokens.reshape(
            batch_size * sides, word_count * dofs, self.word_dim
        )
        sequence_mask = word_mask[..., None].expand(
            batch_size, sides, word_count, dofs
        ).reshape(batch_size * sides, word_count * dofs)
        if self.use_cls_token:
            cls_tokens = self.cls_token[None, None].expand(
                batch_size * sides, 1, self.word_dim
            )
            sequence = torch.cat((cls_tokens, sequence), dim=1)
            sequence_mask = torch.cat(
                (
                    torch.ones(
                        batch_size * sides,
                        1,
                        dtype=torch.bool,
                        device=word_mask.device,
                    ),
                    sequence_mask,
                ),
                dim=1,
            )
        sequence = self.transformer(
            sequence,
            src_key_padding_mask=~sequence_mask,
        )
        if self.use_cls_token:
            side_embedding = sequence[:, 0].reshape(
                batch_size, sides, self.word_dim
            )
            sequence = sequence[:, 1:]
        tokens = sequence.reshape(
            batch_size, sides, word_count, dofs, self.word_dim
        )
        tokens = tokens * word_mask[..., None, None]

        if not self.use_cls_token:
            token_weights = word_mask[..., None, None].to(tokens.dtype)
            side_embedding = (tokens * token_weights).sum(dim=(2, 3))
            denominator = word_mask.sum(
                dim=2, keepdim=True
            ).to(tokens.dtype)
            side_embedding = side_embedding / (denominator * dofs)
        subject_embedding = side_embedding.mean(dim=1)
        return {
            "tokens": tokens,
            "side_embedding": side_embedding,
            "subject_embedding": subject_embedding,
            "cls_embedding": subject_embedding,
        }


class GaitCrossAttentionBlock(nn.Module):
    """Exchange contextual information between two side sequences."""

    def __init__(
        self,
        word_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(word_dim)
        self.context_norm = nn.LayerNorm(word_dim)
        self.cross_attention = nn.MultiheadAttention(
            word_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(word_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(word_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, word_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        normalized_context = self.context_norm(context)
        attended, _ = self.cross_attention(
            normalized_query,
            normalized_context,
            normalized_context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        output = query + self.attention_dropout(attended)
        return output + self.feedforward(
            self.feedforward_norm(output)
        )


class GaitFormer(nn.Module):
    """Apply within-side joint attention and bilateral cross-attention."""

    def __init__(
        self,
        *,
        word_dim: int,
        patch_hidden_channels: int,
        patch_output_channels: int,
        max_words: int,
        depth: int,
        bilateral_depth: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        use_cls_token: bool,
        use_dof_embedding: bool = True,
        use_cycle_embedding: bool = True,
        use_duration_embedding: bool = True,
        use_interval_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.word_dim = word_dim
        self.max_words = max_words
        self.within_side_encoder = WithinSideEncoder(
            word_dim=word_dim,
            patch_hidden_channels=patch_hidden_channels,
            patch_output_channels=patch_output_channels,
            max_words=max_words,
            depth=depth,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            use_cls_token=use_cls_token,
            use_dof_embedding=use_dof_embedding,
            use_cycle_embedding=use_cycle_embedding,
            use_duration_embedding=use_duration_embedding,
            use_interval_embedding=use_interval_embedding,
        )
        self.bilateral_blocks = nn.ModuleList(
            [
                GaitCrossAttentionBlock(
                    word_dim,
                    num_heads,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(bilateral_depth)
            ]
        )

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
        *,
        masked_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, sides, word_count, dofs, _ = words.shape
        encoded = self.within_side_encoder(
            words,
            word_mask,
            timing,
            masked_positions=masked_positions,
        )
        word_sequence = encoded["tokens"].reshape(
            batch_size, sides, word_count * dofs, self.word_dim
        )
        sequence = torch.cat(
            (encoded["side_embedding"][:, :, None], word_sequence),
            dim=2,
        )
        sequence_mask = word_mask[..., None].expand(
            batch_size, sides, word_count, dofs
        ).reshape(batch_size, sides, word_count * dofs)
        sequence_mask = torch.cat(
            (
                torch.ones(
                    batch_size,
                    sides,
                    1,
                    dtype=torch.bool,
                    device=word_mask.device,
                ),
                sequence_mask,
            ),
            dim=2,
        )
        for block in self.bilateral_blocks:
            paired_sequence = block(
                torch.cat((sequence[:, 0], sequence[:, 1]), dim=0),
                torch.cat((sequence[:, 1], sequence[:, 0]), dim=0),
                torch.cat(
                    (sequence_mask[:, 1], sequence_mask[:, 0]), dim=0
                ),
            )
            sequence = torch.stack(
                (
                    paired_sequence[:batch_size],
                    paired_sequence[batch_size:],
                ),
                dim=1,
            )

        side_embedding = sequence[:, :, 0]
        tokens = sequence[:, :, 1:].reshape(
            batch_size,
            sides,
            word_count,
            dofs,
            self.word_dim,
        )
        tokens = tokens * word_mask[..., None, None]
        subject_embedding = side_embedding.mean(dim=1)
        return {
            "tokens": tokens,
            "side_embedding": side_embedding,
            "subject_embedding": subject_embedding,
            "cls_embedding": subject_embedding,
        }
