"""VQ-Gait: six local shape vocabularies, EMA quantization and reconstruction."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .tokens import GaitShapeEncoder, ShapeAttributeVQEncoder, split_shape


class DOFCodebook(nn.Module):
    """Six EMA-updated healthy gait-word vocabularies."""

    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        *,
        decay: float,
        dead_code_threshold: float,
        kmeans_iterations: int,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.decay = decay
        self.dead_code_threshold = dead_code_threshold
        self.kmeans_iterations = kmeans_iterations
        self.epsilon = epsilon
        initial = F.normalize(
            torch.randn(6, codebook_size, code_dim), dim=-1
        )
        self.register_buffer("embedding", initial)
        self.register_buffer(
            "cluster_size", torch.zeros(6, codebook_size)
        )
        self.register_buffer("embedding_sum", initial.clone())
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _kmeans(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        centers = values[
            torch.randperm(values.shape[0], device=values.device)[
                : self.codebook_size
            ]
        ]
        for _ in range(self.kmeans_iterations):
            assignments = torch.cdist(values, centers).argmin(dim=-1)
            counts = torch.bincount(
                assignments, minlength=self.codebook_size
            )
            sums = values.new_zeros(self.codebook_size, self.code_dim)
            sums.index_add_(0, assignments, values)
            active = counts > 0
            centers[active] = sums[active] / counts[active, None]
        return F.normalize(centers, dim=-1)

    @torch.no_grad()
    def _initialize(
        self, encoded: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        for dof in range(6):
            values = encoded[:, dof][valid_mask[:, dof]]
            centers = self._kmeans(values)
            self.embedding[dof].copy_(centers)
            self.embedding_sum[dof].copy_(centers)
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
        leading = encoded.shape[:-2]
        flat = encoded.reshape(-1, 6, self.code_dim)
        flat_mask = valid_mask.reshape(-1, 6)
        normalized = F.normalize(flat, dim=-1)
        if update and not bool(self.initialized.item()):
            self._initialize(normalized.detach(), flat_mask)
        embedding = self.embedding.to(dtype=normalized.dtype)
        similarity = torch.einsum(
            "ncd,ckd->nck", normalized, embedding
        )
        indices = similarity.argmax(dim=-1)
        quantized = embedding.unsqueeze(0).expand(
            flat.shape[0], -1, -1, -1
        ).gather(
            2,
            indices[..., None, None].expand(-1, -1, 1, self.code_dim),
        ).squeeze(2)
        if update:
            self._ema_update(normalized.detach(), indices, flat_mask)
        weights = flat_mask[..., None].to(normalized.dtype)
        commitment = (
            (normalized - quantized.detach()).square() * weights
        ).sum() / (weights.sum() * self.code_dim)
        straight_through = normalized + (quantized - normalized).detach()
        straight_through = straight_through * weights
        return (
            straight_through.reshape(*leading, 6, self.code_dim),
            indices.reshape(*leading, 6),
            commitment,
        )

    def similarities(self, encoded: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(encoded, dim=-1)
        embedding = self.embedding.to(dtype=normalized.dtype)
        return torch.einsum("...cd,ckd->...ck", normalized, embedding)

    def usage(
        self, indices: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active_ratios = []
        perplexities = []
        for dof in range(6):
            values = indices[..., dof][valid_mask[..., dof]]
            counts = torch.bincount(values, minlength=self.codebook_size)
            probabilities = counts / counts.sum()
            active_ratios.append((counts > 0).float().mean())
            entropy = -(
                probabilities
                * probabilities.clamp_min(1e-12).log()
            ).sum()
            perplexities.append(entropy.exp())
        return torch.stack(active_ratios).mean(), torch.stack(
            perplexities
        ).mean()


class GaitShapeDecoder(nn.Module):
    """Decode each quantized DOF word without contextual information."""

    def __init__(
        self,
        *,
        code_dim: int,
        word_length: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.reconstruction_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(code_dim),
                    nn.Linear(code_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, word_length),
                )
                for _ in range(6)
            ]
        )

    def forward(
        self,
        quantized: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
    ) -> torch.Tensor:
        reconstructed = torch.stack(
            [
                head(quantized[..., dof, :])
                for dof, head in enumerate(self.reconstruction_heads)
            ],
            dim=3,
        )
        return reconstructed * word_mask[..., None, None]

    def decode_codebook(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                head(embeddings[dof])
                for dof, head in enumerate(self.reconstruction_heads)
            ],
            dim=0,
        )


class VQProjection(nn.Module):
    """Independent DOF CNNs followed by the shared code projection."""
    def __init__(self, word_dim, code_dim, hidden_channels, output_channels):
        super().__init__()
        self.backbone = GaitShapeEncoder(word_dim=word_dim,
            patch_hidden_channels=hidden_channels, patch_output_channels=output_channels)
        self.code_projection = nn.Sequential(nn.Linear(word_dim, word_dim),
                                             nn.Tanh(), nn.Linear(word_dim, code_dim))

    def forward(self, words, word_mask, timing):
        encoded = self.backbone(words, word_mask, timing)
        code_features = self.code_projection(encoded["tokens"]) * word_mask[..., None, None]
        return {**encoded, "context_tokens": encoded["tokens"], "tokens": code_features}


class VQGait(nn.Module):
    """Shape vocabulary (or raw-word ablation); codebooks update only in VQ."""
    extra_loss_keys = ("shape_reconstruction_loss", "waveform_reconstruction_loss")

    def __init__(self, args):
        super().__init__()
        self.word_length = args.word_length
        # Keep module construction order and state-dict paths of the paper run.
        self.encoder = VQProjection(args.word_dim, args.code_dim,
                                    args.vq_patch_hidden_channels, args.vq_patch_output_channels)
        self.codebook = DOFCodebook(args.codebook_size, args.code_dim,
            decay=args.codebook_decay, dead_code_threshold=args.dead_code_threshold,
            kmeans_iterations=args.codebook_kmeans_iterations)
        self.decoder = GaitShapeDecoder(code_dim=args.code_dim, word_length=args.word_length,
                                        hidden_dim=args.decoder_ff_dim)
        self.separate_shape = args.vq_v2_separate_shape
        self.encoder = ShapeAttributeVQEncoder(self.encoder, args.vq_v2_scale_floor,
                                              self.separate_shape)
        self.scale_floor = args.vq_v2_scale_floor
        self.commitment_weight = args.commitment_weight
        self.geometry_weight = args.vq_geometry_weight
        self.geometry_waveform_weight = args.vq_geometry_waveform_weight
        self.geometry_shape_temperature = args.vq_geometry_shape_temperature
        self.geometry_latent_temperature = args.vq_geometry_latent_temperature
        self.geometry_max_words = args.vq_geometry_max_words
        self.shape_loss_weight = args.vq_v2_shape_loss_weight
        self.waveform_loss_weight = args.vq_v2_waveform_loss_weight
        self.reconstruction_loss_fn = F.smooth_l1_loss

    def _standard_shape(self, decoded):
        return split_shape(decoded, self.scale_floor)[0]

    def morphology_geometry_loss(
        self,
        words: torch.Tensor,
        encoded: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Match within-DOF waveform and pre-quantization feature neighborhoods.

        Each target distance component is divided by its detached neighborhood
        mean. Differences are in normalized cycle coordinates, not seconds.
        Code vectors retain their existing EMA update; gradients reach the
        encoder through normalized, continuous code-space features.
        """
        with torch.autocast(device_type=encoded.device.type, enabled=False):
            waveforms = words.detach().float()[word_mask]
            features = encoded.float()[word_mask]
            count = waveforms.shape[0]
            # A neighborhood requires an anchor and at least one other word.
            if count < 2:
                return features.sum() * 0.0
            if count > self.geometry_max_words:
                if self.training:
                    selected = torch.randperm(count, device=words.device)[
                        : self.geometry_max_words
                    ]
                else:
                    selected = torch.linspace(
                        0, count - 1, self.geometry_max_words,
                        device=words.device,
                    ).long()
                waveforms = waveforms[selected]
                features = features[selected]
            count = waveforms.shape[0]
            neighbors = ~torch.eye(count, dtype=torch.bool, device=words.device)
            losses = []
            for dof in range(6):
                waveform = waveforms[:, dof]
                shape_distance = waveform.new_zeros((count, count - 1))
                if self.geometry_waveform_weight > 0:
                    waveform_distance = (
                        torch.cdist(waveform, waveform).square() / waveform.shape[-1]
                    )[neighbors].reshape(count, count - 1)
                    shape_distance = shape_distance + self.geometry_waveform_weight * waveform_distance / waveform_distance.mean().clamp_min(1e-8)
                target_log_probability = F.log_softmax(
                    -shape_distance / self.geometry_shape_temperature, dim=-1
                )
                normalized = F.normalize(features[:, dof], dim=-1)
                latent_distance = (
                    2.0 - 2.0 * (normalized @ normalized.T)
                )[neighbors].reshape(count, count - 1)
                predicted_log_probability = F.log_softmax(
                    -latent_distance / self.geometry_latent_temperature, dim=-1
                )
                losses.append(F.kl_div(
                    predicted_log_probability, target_log_probability,
                    reduction="batchmean", log_target=True,
                ))
            return torch.stack(losses).mean()

    def encode(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.encoder(words, word_mask, timing)

    @torch.no_grad()
    def tokenize(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(words, word_mask, timing)["tokens"]
        valid = word_mask[..., None].expand(*word_mask.shape, 6)
        quantized, indices, _ = self.codebook(
            encoded, valid, update=False
        )
        return {
            "features": encoded,
            "quantized": quantized,
            "indices": indices,
            "similarities": self.codebook.similarities(encoded),
        }

    def forward(self, words, word_mask, timing):
        output = self.encode(words, word_mask, timing)
        encoded = output["tokens"]
        valid = word_mask[..., None].expand(*word_mask.shape, 6)
        quantized, indices, commitment = self.codebook(encoded, valid, update=self.training)
        decoded = self.decoder(quantized, word_mask, timing)
        shape = self._standard_shape(decoded)
        mask = valid[..., None]
        if self.separate_shape:
            reconstructed = (output["word_mean"] + output["word_scale"] * shape) * mask
        else:
            # No attribute bypass: the quantized representation must reconstruct
            # offset/scale as well as shape. Keep shape supervision standardized.
            reconstructed = decoded * mask
        denominator = mask.sum() * words.shape[-1]
        shape_loss = words.new_zeros(())
        waveform_loss = words.new_zeros(())
        if self.shape_loss_weight > 0:
            shape_loss = (self.reconstruction_loss_fn(shape, output["shape_words"], reduction="none") * mask).sum() / denominator
        if self.waveform_loss_weight > 0:
            waveform_loss = (self.reconstruction_loss_fn(reconstructed, words, reduction="none") * mask).sum() / denominator
        reconstruction = self.shape_loss_weight * shape_loss + self.waveform_loss_weight * waveform_loss
        geometry = self.morphology_geometry_loss(output["shape_words"], encoded, word_mask) if self.geometry_weight else reconstruction.new_zeros(())
        active, perplexity = self.codebook.usage(indices, valid)
        return {**{key: value for key, value in output.items() if key != "tokens"},
                "loss": reconstruction + self.commitment_weight * commitment + self.geometry_weight * geometry,
                "reconstruction_loss": reconstruction, "shape_reconstruction_loss": shape_loss,
                "waveform_reconstruction_loss": waveform_loss, "commitment_loss": commitment,
                "geometry_loss": geometry, "active_code_ratio": active, "perplexity": perplexity,
                "encoded": encoded, "quantized": quantized, "indices": indices,
                "reconstructed": reconstructed, "reconstructed_shape": shape * mask}

    def decode_codebook(self):
        # SSL always combines standardized templates with predicted attributes,
        # including when VQ was trained to decode raw rather than shape words.
        embeddings = F.normalize(self.codebook.embedding, dim=-1)
        return self._standard_shape(self.decoder.decode_codebook(embeddings))
