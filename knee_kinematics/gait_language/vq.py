"""DOF-specific gait-word encoder, EMA codebook, and waveform decoders."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DOFWordEncoder(nn.Module):
    """Encode every DOF-cycle waveform into one morphology word."""

    def __init__(self, word_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.word_dim = word_dim
        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 7, stride=2, padding=3),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, word_dim, 5, stride=2, padding=2),
            nn.GroupNorm(1, word_dim),
            nn.GELU(),
            nn.Conv1d(word_dim, word_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dof_embedding = nn.Parameter(torch.empty(6, word_dim))
        adapter_dim = max(8, word_dim // 4)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(word_dim, adapter_dim),
                    nn.GELU(),
                    nn.Linear(adapter_dim, word_dim),
                )
                for _ in range(6)
            ]
        )
        nn.init.normal_(self.dof_embedding, std=0.02)

    def forward(self, words: torch.Tensor) -> torch.Tensor:
        """Encode ``[...,6,T]`` waveforms into ``[...,6,D]`` words."""
        leading = words.shape[:-2]
        time_steps = words.shape[-1]
        flat = words.reshape(-1, 6, time_steps)
        encoded = self.stem(flat.reshape(-1, 1, time_steps)).squeeze(-1)
        encoded = encoded.reshape(flat.shape[0], 6, self.word_dim)
        encoded = encoded + self.dof_embedding[None]
        encoded = torch.stack(
            [
                encoded[:, dof] + self.adapters[dof](encoded[:, dof])
                for dof in range(6)
            ],
            dim=1,
        )
        return encoded.reshape(*leading, 6, self.word_dim)


class DOFWordDecoder(nn.Module):
    """Reconstruct each DOF waveform with a shared MLP."""

    def __init__(self, word_dim: int, word_length: int) -> None:
        super().__init__()
        self.word_length = word_length
        self.dof_embedding = nn.Parameter(torch.empty(6, word_dim))
        self.decoder = nn.Sequential(
            nn.Linear(word_dim, word_dim * 2),
            nn.GELU(),
            nn.Linear(word_dim * 2, word_length),
        )
        nn.init.normal_(self.dof_embedding, std=0.02)

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor | None = None,
        timing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode ``[...,6,D]`` into standardized ``[...,6,T]``."""
        return self.decoder(words + self.dof_embedding)

    def decode_codebook(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode ``[6,K,D]`` prototypes into ``[6,K,T]`` waveforms."""
        return self.forward(embeddings.permute(1, 0, 2)).permute(1, 0, 2)


class TemporalTransformerWordDecoder(nn.Module):
    """Decode every word through Transformer phase tokens."""

    def __init__(
        self,
        word_dim: int,
        word_length: int,
        *,
        phase_tokens: int = 20,
        depth: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.word_length = word_length
        self.phase_tokens = phase_tokens
        self.patch_size = math.ceil(word_length / phase_tokens)
        self.dof_embedding = nn.Parameter(torch.empty(6, word_dim))
        self.phase_embedding = nn.Parameter(
            torch.empty(phase_tokens, word_dim)
        )
        adapter_dim = max(8, word_dim // 4)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(word_dim, adapter_dim),
                    nn.GELU(),
                    nn.Linear(adapter_dim, word_dim),
                )
                for _ in range(6)
            ]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=word_dim,
            nhead=num_heads,
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
        self.output_heads = nn.ModuleList(
            [nn.Linear(word_dim, self.patch_size) for _ in range(6)]
        )
        nn.init.normal_(self.dof_embedding, std=0.02)
        nn.init.normal_(self.phase_embedding, std=0.02)

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor | None = None,
        timing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode ``[...,6,D]`` via intra-word temporal self-attention."""
        leading = words.shape[:-2]
        flat = words.reshape(-1, 6, words.shape[-1])
        waveforms = []
        for dof in range(6):
            word = flat[:, dof] + self.dof_embedding[dof]
            word = word + self.adapters[dof](word)
            phase_tokens = word[:, None] + self.phase_embedding[None]
            phase_tokens = self.transformer(phase_tokens)
            patches = self.output_heads[dof](phase_tokens)
            waveforms.append(
                patches.flatten(start_dim=1)[:, : self.word_length]
            )
        decoded = torch.stack(waveforms, dim=1)
        return decoded.reshape(*leading, 6, self.word_length)

    def decode_codebook(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode ``[6,K,D]`` prototypes into ``[6,K,T]`` waveforms."""
        return self.forward(embeddings.permute(1, 0, 2)).permute(1, 0, 2)


class SentenceTransformerWordDecoder(nn.Module):
    """Contextually reconstruct all words in a bilateral gait sentence."""

    def __init__(
        self,
        word_dim: int,
        word_length: int,
        *,
        max_words: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.word_length = word_length
        self.side_embedding = nn.Parameter(torch.empty(2, word_dim))
        self.dof_embedding = nn.Parameter(torch.empty(6, word_dim))
        self.cycle_embedding = nn.Parameter(
            torch.empty(max_words, word_dim)
        )
        self.timing_projection = nn.Sequential(
            nn.Linear(4, word_dim),
            nn.GELU(),
            nn.Linear(word_dim, word_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=word_dim,
            nhead=num_heads,
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
        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(word_dim, word_dim),
                    nn.Tanh(),
                    nn.Linear(word_dim, word_length),
                )
                for _ in range(6)
            ]
        )
        nn.init.normal_(self.side_embedding, std=0.02)
        nn.init.normal_(self.dof_embedding, std=0.02)
        nn.init.normal_(self.cycle_embedding, std=0.02)

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode ``[B,2,W,6,D]`` with bilateral sentence context."""
        batch_size, _, word_count, _, word_dim = words.shape
        tokens = (
            words
            + self.side_embedding[None, :, None, None]
            + self.cycle_embedding[None, None, :word_count, None]
            + self.dof_embedding[None, None, None]
        )
        if timing is not None:
            tokens = tokens + self.timing_projection(timing)[..., None, :]
        valid = word_mask[..., None].expand(-1, -1, -1, 6)
        tokens = tokens.reshape(batch_size, -1, word_dim)
        valid = valid.reshape(batch_size, -1)
        contextual = self.transformer(
            tokens,
            src_key_padding_mask=~valid,
        )
        contextual = contextual * valid[..., None].to(contextual.dtype)
        contextual = contextual.reshape(
            batch_size, 2, word_count, 6, word_dim
        )
        decoded = torch.stack(
            [
                self.output_heads[dof](contextual[..., dof, :])
                for dof in range(6)
            ],
            dim=3,
        )
        return decoded

    def decode_codebook(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode prototypes in a canonical context-free token position."""
        waveforms = []
        for dof in range(6):
            tokens = (
                embeddings[dof, :, None]
                + self.side_embedding[0]
                + self.cycle_embedding[0]
                + self.dof_embedding[dof]
            )
            contextual = self.transformer(tokens)
            waveforms.append(
                self.output_heads[dof](contextual[:, 0])
            )
        return torch.stack(waveforms, dim=0)


class LocalContextResidualSentenceDecoder(nn.Module):
    """Anchor morphology locally and predict only a sentence residual."""

    def __init__(
        self,
        word_dim: int,
        word_length: int,
        *,
        max_words: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.word_length = word_length
        self.residual_scale = residual_scale
        self.local_decoder = DOFWordDecoder(word_dim, word_length)
        self.context_residual_decoder = SentenceTransformerWordDecoder(
            word_dim,
            word_length,
            max_words=max_words,
            depth=depth,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        for head in self.context_residual_decoder.output_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def reconstruct_components(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final reconstruction, local waveform, and residual."""
        local = self.local_decoder(words)
        residual = self.context_residual_decoder(
            words, word_mask, timing
        )
        reconstructed = local + self.residual_scale * residual
        return reconstructed, local, residual

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode a locally anchored waveform with contextual correction."""
        reconstructed, _, _ = self.reconstruct_components(
            words, word_mask, timing
        )
        return reconstructed

    def decode_codebook(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode prototypes with local and canonical residual branches."""
        local = self.local_decoder.decode_codebook(embeddings)
        residual = self.context_residual_decoder.decode_codebook(embeddings)
        return local + self.residual_scale * residual


class DOFCodebook(nn.Module):
    """Six independent EMA-updated vocabularies shared across both legs."""

    def __init__(
        self,
        codebook_size: int,
        word_dim: int,
        *,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.word_dim = word_dim
        self.decay = decay
        self.epsilon = epsilon
        self.dead_code_threshold = dead_code_threshold
        initial = F.normalize(
            torch.randn(6, codebook_size, word_dim), dim=-1
        )
        self.register_buffer("embedding", initial)
        self.register_buffer(
            "cluster_size", torch.zeros(6, codebook_size)
        )
        self.register_buffer("embedding_sum", initial.clone())
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _initialize(
        self, encoded: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        for dof in range(6):
            values = encoded[:, dof][valid_mask[:, dof]]
            indices = torch.randperm(values.shape[0], device=values.device)
            indices = indices.repeat(
                (self.codebook_size + indices.numel() - 1)
                // indices.numel()
            )[: self.codebook_size]
            selected = F.normalize(values[indices], dim=-1)
            self.embedding[dof].copy_(selected)
            self.embedding_sum[dof].copy_(selected)
            self.cluster_size[dof].fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(
        self,
        encoded: torch.Tensor,
        indices: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        for dof in range(6):
            valid = valid_mask[:, dof]
            values = encoded[:, dof][valid]
            assignments = F.one_hot(
                indices[:, dof][valid], self.codebook_size
            ).to(values.dtype)
            counts = assignments.sum(dim=0)
            sums = assignments.transpose(0, 1) @ values
            self.cluster_size[dof].mul_(self.decay).add_(
                counts, alpha=1.0 - self.decay
            )
            self.embedding_sum[dof].mul_(self.decay).add_(
                sums, alpha=1.0 - self.decay
            )
            total = self.cluster_size[dof].sum()
            smoothed = (
                (self.cluster_size[dof] + self.epsilon)
                / (total + self.codebook_size * self.epsilon)
                * total
            )
            updated = self.embedding_sum[dof] / smoothed[:, None]
            dead = self.cluster_size[dof] < self.dead_code_threshold
            if dead.any():
                replacements = values[
                    torch.randint(
                        values.shape[0],
                        (int(dead.sum().item()),),
                        device=values.device,
                    )
                ]
                updated[dead] = replacements
                self.embedding_sum[dof, dead] = replacements
                self.cluster_size[dof, dead] = 1.0
            self.embedding[dof].copy_(F.normalize(updated, dim=-1))

    def forward(
        self,
        encoded: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        update: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``[...,6,D]`` and return words, indices, commitment."""
        leading = encoded.shape[:-2]
        flat = encoded.reshape(-1, 6, self.word_dim)
        flat_mask = valid_mask.reshape(-1, 6)
        normalized = F.normalize(flat, dim=-1)
        if update and not bool(self.initialized.item()):
            self._initialize(normalized.detach(), flat_mask)
        embedding = self.embedding.to(dtype=normalized.dtype)
        similarity = torch.einsum(
            "ncd,ckd->nck", normalized, embedding
        )
        indices = similarity.argmax(dim=-1)
        expanded = embedding.unsqueeze(0).expand(
            flat.shape[0], -1, -1, -1
        )
        quantized = expanded.gather(
            2,
            indices[..., None, None].expand(
                -1, -1, 1, self.word_dim
            ),
        ).squeeze(2)
        if update:
            self._ema_update(normalized.detach(), indices, flat_mask)
        mask = flat_mask[..., None].to(normalized.dtype)
        commitment = ((normalized - quantized.detach()) ** 2 * mask).sum()
        commitment = commitment / (
            mask.sum().clamp_min(1.0) * self.word_dim
        )
        straight_through = normalized + (quantized - normalized).detach()
        straight_through = straight_through * mask
        return (
            straight_through.reshape(*leading, 6, self.word_dim),
            indices.reshape(*leading, 6),
            commitment,
        )

    def usage(
        self, indices: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean active-code ratio and assignment perplexity."""
        active_ratios = []
        perplexities = []
        for dof in range(6):
            values = indices[..., dof][valid_mask[..., dof]]
            counts = torch.bincount(values, minlength=self.codebook_size)
            probabilities = counts / counts.sum().clamp_min(1)
            active_ratios.append((counts > 0).float().mean())
            entropy = -(
                probabilities
                * probabilities.clamp_min(1e-12).log()
            ).sum()
            perplexities.append(entropy.exp())
        return torch.stack(active_ratios).mean(), torch.stack(perplexities).mean()


class GaitVQTokenizer(nn.Module):
    """Learn a DOF-specific discrete vocabulary by waveform reconstruction."""

    def __init__(
        self,
        *,
        word_length: int,
        word_dim: int,
        hidden_dim: int,
        codebook_size: int,
        codebook_decay: float,
        dead_code_threshold: float,
        commitment_weight: float,
        velocity_weight: float,
        decoder_type: str = "mlp",
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        decoder_feedforward_dim: int = 512,
        decoder_dropout: float = 0.1,
        decoder_phase_tokens: int = 20,
        context_residual_scale: float = 0.5,
        local_reconstruction_weight: float = 1.0,
        residual_energy_weight: float = 0.0,
        max_words: int = 32,
    ) -> None:
        super().__init__()
        if residual_energy_weight < 0.0:
            raise ValueError("residual_energy_weight must be non-negative")
        self.word_encoder = DOFWordEncoder(word_dim, hidden_dim)
        self.codebook = DOFCodebook(
            codebook_size,
            word_dim,
            decay=codebook_decay,
            dead_code_threshold=dead_code_threshold,
        )
        decoder_builders = {
            "mlp": lambda: DOFWordDecoder(word_dim, word_length),
            "temporal_transformer": lambda: TemporalTransformerWordDecoder(
                word_dim,
                word_length,
                phase_tokens=decoder_phase_tokens,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                feedforward_dim=decoder_feedforward_dim,
                dropout=decoder_dropout,
            ),
            "sentence_transformer": lambda: SentenceTransformerWordDecoder(
                word_dim,
                word_length,
                max_words=max_words,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                feedforward_dim=decoder_feedforward_dim,
                dropout=decoder_dropout,
            ),
            "local_context_sentence": (
                lambda: LocalContextResidualSentenceDecoder(
                    word_dim,
                    word_length,
                    max_words=max_words,
                    depth=decoder_depth,
                    num_heads=decoder_num_heads,
                    feedforward_dim=decoder_feedforward_dim,
                    dropout=decoder_dropout,
                    residual_scale=context_residual_scale,
                )
            ),
        }
        self.decoder_type = decoder_type
        self.word_decoder = decoder_builders[decoder_type]()
        self.commitment_weight = commitment_weight
        self.velocity_weight = velocity_weight
        self.local_reconstruction_weight = local_reconstruction_weight
        self.residual_energy_weight = residual_energy_weight

    def encode_indices(
        self, words: torch.Tensor, word_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return frozen vocabulary IDs for a bilateral sentence."""
        encoded = self.word_encoder(words)
        valid = word_mask[..., None].expand(*word_mask.shape, 6)
        _, indices, _ = self.codebook(encoded, valid, update=False)
        return indices

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode, quantize, reconstruct, and report VQ diagnostics."""
        encoded = self.word_encoder(words)
        valid = word_mask[..., None].expand(*word_mask.shape, 6)
        quantized, indices, commitment = self.codebook(
            encoded, valid, update=self.training
        )
        local_reconstruction = words.new_zeros(())
        residual_energy = words.new_zeros(())
        raw_context_residual_rms = words.new_zeros(())
        scaled_context_residual_rms = words.new_zeros(())
        local_to_final_improvement = words.new_zeros(())
        if self.decoder_type == "local_context_sentence":
            reconstructed, local_waveform, context_residual = (
                self.word_decoder.reconstruct_components(
                    quantized, word_mask, timing
                )
            )
        else:
            reconstructed = self.word_decoder(quantized, word_mask, timing)
        waveform_mask = valid[..., None].to(words.dtype)
        denominator = waveform_mask.sum().clamp_min(1.0) * words.shape[-1]
        reconstruction = (
            (reconstructed - words).square() * waveform_mask
        ).sum() / denominator
        if self.decoder_type == "local_context_sentence":
            local_reconstruction = (
                (local_waveform - words).square() * waveform_mask
            ).sum() / denominator
            raw_residual_energy = (
                (context_residual.square() * waveform_mask).sum()
                / denominator
            )
            scaled_residual = (
                self.word_decoder.residual_scale * context_residual
            )
            residual_energy = (
                (scaled_residual.square() * waveform_mask).sum()
                / denominator
            )
            raw_context_residual_rms = raw_residual_energy.sqrt()
            scaled_context_residual_rms = residual_energy.sqrt()
            local_to_final_improvement = (
                local_reconstruction - reconstruction
            )
        velocity_mask = valid[..., None].to(words.dtype)
        velocity_denominator = (
            velocity_mask.sum().clamp_min(1.0) * (words.shape[-1] - 1)
        )
        velocity = (
            (
                reconstructed.diff(dim=-1) - words.diff(dim=-1)
            ).square()
            * velocity_mask
        ).sum() / velocity_denominator
        active_ratio, perplexity = self.codebook.usage(indices, valid)
        loss = (
            reconstruction
            + self.local_reconstruction_weight * local_reconstruction
            + self.velocity_weight * velocity
            + self.commitment_weight * commitment
            + self.residual_energy_weight * residual_energy
        )
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction,
            "local_reconstruction_loss": local_reconstruction,
            "residual_energy_loss": residual_energy,
            "raw_context_residual_rms": raw_context_residual_rms,
            "scaled_context_residual_rms": scaled_context_residual_rms,
            "local_to_final_improvement": local_to_final_improvement,
            # Backward-compatible alias used by existing experiment logs.
            "context_residual_rms": raw_context_residual_rms,
            "velocity_loss": velocity,
            "commitment_loss": commitment,
            "active_code_ratio": active_ratio,
            "perplexity": perplexity,
            "indices": indices,
            "reconstructed": reconstructed,
        }

    def decode_codebook(self) -> torch.Tensor:
        """Decode all ``[6,K,D]`` vocabulary entries into waveforms."""
        embeddings = F.normalize(self.codebook.embedding, dim=-1)
        return self.word_decoder.decode_codebook(embeddings)


__all__ = [
    "DOFCodebook",
    "DOFWordDecoder",
    "DOFWordEncoder",
    "GaitVQTokenizer",
    "LocalContextResidualSentenceDecoder",
    "SentenceTransformerWordDecoder",
    "TemporalTransformerWordDecoder",
]
