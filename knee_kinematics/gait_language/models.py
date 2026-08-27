"""Masked gait-language SSL and downstream disease models."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .downstream import HierarchicalGaitDeviationEncoder
from .sentence import GaitSentenceEncoder
from .vq import GaitVQTokenizer


class GaitLanguageSSLModel(nn.Module):
    """Learn rhythm, cross-DOF, and bilateral conditional representations."""

    def __init__(
        self,
        target_tokenizer: GaitVQTokenizer,
        *,
        word_dim: int,
        codebook_size: int,
        max_words: int,
        depth: int,
        num_heads: int,
        dropout: float,
        within_task: bool,
        cross_dof_task: bool,
        rhythm_task: bool,
        bilateral_context_task: bool,
        contralateral_task: bool,
        bilateral_pair_task: bool,
        swap_task: bool,
        word_mask_ratio: float,
        bilateral_mask_ratio: float,
        contralateral_mask_ratio: float,
        rhythm_mask_ratio: float,
        span_length: int,
        within_weight: float,
        cross_dof_weight: float,
        rhythm_weight: float,
        duration_prediction_weight: float,
        interval_prediction_weight: float,
        bilateral_weight: float,
        contralateral_weight: float,
        bilateral_pair_weight: float,
        swap_weight: float,
        conditional_code_top_k: int,
        conditional_soft_target_temperature: float,
        cross_dof_hard_code_weight: float,
        cross_dof_soft_code_weight: float,
        cross_dof_prototype_weight: float,
        contralateral_hard_code_weight: float,
        contralateral_soft_code_weight: float,
        contralateral_prototype_weight: float,
    ) -> None:
        super().__init__()
        self.target_tokenizer = target_tokenizer
        for parameter in self.target_tokenizer.parameters():
            parameter.requires_grad = False
        self.sentence_encoder = GaitSentenceEncoder(
            copy.deepcopy(target_tokenizer.word_encoder),
            word_dim=word_dim,
            max_words=max_words,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.prediction_heads = nn.ModuleList(
            [nn.Linear(word_dim, codebook_size) for _ in range(6)]
        )
        self.prototype_heads = nn.ModuleList(
            [nn.Linear(word_dim, word_dim) for _ in range(6)]
        )
        self.rhythm_head = nn.Sequential(
            nn.LayerNorm(word_dim),
            nn.Linear(word_dim, word_dim),
            nn.GELU(),
            nn.Linear(word_dim, 2),
        )
        self.bilateral_pair_head = nn.Sequential(
            nn.LayerNorm(word_dim * 2),
            nn.Linear(word_dim * 2, word_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(word_dim, 1),
        )
        self.tasks = {
            "within": within_task,
            "cross_dof": cross_dof_task,
            "rhythm": rhythm_task,
            "bilateral": bilateral_context_task,
            "contralateral": contralateral_task,
            "bilateral_pair": bilateral_pair_task,
            "swap": swap_task,
        }
        self.word_mask_ratio = word_mask_ratio
        self.bilateral_mask_ratio = bilateral_mask_ratio
        self.contralateral_mask_ratio = contralateral_mask_ratio
        self.rhythm_mask_ratio = rhythm_mask_ratio
        self.span_length = span_length
        self.conditional_code_top_k = conditional_code_top_k
        self.conditional_soft_target_temperature = (
            conditional_soft_target_temperature
        )
        self.cross_dof_component_weights = {
            "hard": cross_dof_hard_code_weight,
            "soft": cross_dof_soft_code_weight,
            "prototype": cross_dof_prototype_weight,
        }
        self.contralateral_component_weights = {
            "hard": contralateral_hard_code_weight,
            "soft": contralateral_soft_code_weight,
            "prototype": contralateral_prototype_weight,
        }
        self.loss_weights = {
            "within": within_weight,
            "cross_dof": cross_dof_weight,
            "rhythm": rhythm_weight,
            "duration": duration_prediction_weight,
            "interval": interval_prediction_weight,
            "bilateral": bilateral_weight,
            "contralateral": contralateral_weight,
            "bilateral_pair": bilateral_pair_weight,
            "swap": swap_weight,
        }

    @staticmethod
    def _ensure_one(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            first = valid.reshape(-1).nonzero(as_tuple=False)[0, 0]
            mask = mask.reshape(-1)
            mask[first] = True
            mask = mask.reshape_as(valid)
        return mask

    def _span_mask(
        self, seed_mask: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        expanded = seed_mask.clone()
        for offset in range(1, self.span_length):
            shifted = torch.zeros_like(seed_mask)
            shifted[:, :, offset:] = seed_mask[:, :, :-offset]
            expanded |= shifted
        return expanded & valid

    def _predict(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.stack(
            [
                self.prediction_heads[dof](tokens[..., dof, :])
                for dof in range(6)
            ],
            dim=3,
        )
        prototypes = torch.stack(
            [
                self.prototype_heads[dof](tokens[..., dof, :])
                for dof in range(6)
            ],
            dim=3,
        )
        return logits, prototypes

    @staticmethod
    def _masked_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_logits = logits[mask]
        selected_targets = targets[mask]
        loss = F.cross_entropy(selected_logits, selected_targets)
        accuracy = (
            selected_logits.argmax(dim=-1) == selected_targets
        ).float().mean()
        return loss, accuracy

    def _relaxed_masked_loss(
        self,
        logits: torch.Tensor,
        predicted_prototypes: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        component_weights: dict[str, float],
    ) -> dict[str, torch.Tensor]:
        selected_logits = logits[mask]
        selected_predictions = predicted_prototypes[mask]
        selected_targets = targets[mask]
        dof_indices = torch.arange(6, device=targets.device).reshape(
            1, 1, 1, 6
        ).expand_as(targets)[mask]
        codebooks = F.normalize(
            self.target_tokenizer.codebook.embedding.detach(), dim=-1
        )[dof_indices]
        row_indices = torch.arange(
            selected_targets.shape[0], device=targets.device
        )
        target_prototypes = codebooks[row_indices, selected_targets]
        similarities = torch.einsum(
            "nd,nkd->nk", target_prototypes, codebooks
        )
        neighbor_similarity, neighbor_indices = similarities.topk(
            self.conditional_code_top_k, dim=-1
        )
        neighbor_weights = F.softmax(
            neighbor_similarity
            / self.conditional_soft_target_temperature,
            dim=-1,
        )
        log_probabilities = F.log_softmax(selected_logits, dim=-1)
        soft_loss = -(
            neighbor_weights
            * log_probabilities.gather(-1, neighbor_indices)
        ).sum(dim=-1).mean()
        hard_loss = F.cross_entropy(selected_logits, selected_targets)
        prototype_loss = (
            1.0
            - F.cosine_similarity(
                selected_predictions, target_prototypes, dim=-1
            )
        ).mean()
        exact_accuracy = (
            selected_logits.argmax(dim=-1) == selected_targets
        ).float().mean()
        prediction_top_k = selected_logits.topk(
            self.conditional_code_top_k, dim=-1
        ).indices
        top_k_accuracy = prediction_top_k.eq(
            selected_targets[:, None]
        ).any(dim=-1).float().mean()
        loss = (
            component_weights["hard"] * hard_loss
            + component_weights["soft"] * soft_loss
            + component_weights["prototype"] * prototype_loss
        )
        return {
            "loss": loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
            "prototype_loss": prototype_loss,
            "accuracy": exact_accuracy,
            "topk_accuracy": top_k_accuracy,
        }

    def _encode_task(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
        input_mask: torch.Tensor,
        *,
        use_bilateral_context: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        encoded = self.sentence_encoder(
            words,
            word_mask,
            timing,
            masked_positions=input_mask,
            use_bilateral_context=use_bilateral_context,
        )
        logits, prototypes = self._predict(encoded["tokens"])
        return encoded, logits, prototypes

    def _pair_logits(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat(
            [(left - right).abs(), left * right], dim=-1
        )
        return self.bilateral_pair_head(features).squeeze(-1)

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
        *,
        mask_generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Execute SSL objectives with an optional deterministic mask stream."""
        with torch.no_grad():
            targets = self.target_tokenizer.encode_indices(words, word_mask)
        valid = word_mask[..., None].expand(*word_mask.shape, 6)
        zero = words.new_zeros(())
        base_encoded = self.sentence_encoder(words, word_mask, timing)

        within_loss = zero
        within_accuracy = zero
        if self.tasks["within"]:
            within_targets = (
                torch.rand(
                    valid.shape,
                    device=words.device,
                    generator=mask_generator,
                )
                < self.word_mask_ratio
            ) & valid
            within_targets = self._ensure_one(within_targets, valid)
            within_input = self._span_mask(within_targets, valid)
            _, within_logits, _ = self._encode_task(
                words, word_mask, timing, within_input
            )
            within_loss, within_accuracy = self._masked_loss(
                within_logits, targets, within_targets
            )

        dof_loss = zero
        dof_hard_loss = zero
        dof_soft_loss = zero
        dof_prototype_loss = zero
        dof_accuracy = zero
        dof_topk_accuracy = zero
        if self.tasks["cross_dof"]:
            target_dofs = torch.randint(
                0,
                6,
                word_mask.shape[:2],
                device=words.device,
                generator=mask_generator,
            )
            dof_selector = F.one_hot(target_dofs, 6).bool()
            whole_dof_mask = (
                dof_selector[:, :, None, :].expand_as(valid) & valid
            )
            _, dof_logits, dof_prototypes = self._encode_task(
                words, word_mask, timing, whole_dof_mask
            )
            dof_result = self._relaxed_masked_loss(
                dof_logits,
                dof_prototypes,
                targets,
                whole_dof_mask,
                self.cross_dof_component_weights,
            )
            dof_loss = dof_result["loss"]
            dof_hard_loss = dof_result["hard_loss"]
            dof_soft_loss = dof_result["soft_loss"]
            dof_prototype_loss = dof_result["prototype_loss"]
            dof_accuracy = dof_result["accuracy"]
            dof_topk_accuracy = dof_result["topk_accuracy"]

        rhythm_loss = zero
        duration_loss = zero
        interval_loss = zero
        duration_mae = zero
        interval_mae = zero
        if self.tasks["rhythm"]:
            rhythm_targets = (
                torch.rand(
                    word_mask.shape,
                    device=words.device,
                    generator=mask_generator,
                )
                < self.rhythm_mask_ratio
            ) & word_mask
            rhythm_targets = self._ensure_one(
                rhythm_targets, word_mask
            )
            rhythm_encoded = self.sentence_encoder(
                words,
                word_mask,
                timing,
                masked_timing_positions=rhythm_targets,
            )
            rhythm_prediction = self.rhythm_head(
                rhythm_encoded["tokens"].mean(dim=3)
            )
            selected_prediction = rhythm_prediction[rhythm_targets]
            selected_timing = timing[rhythm_targets]
            duration_loss = F.smooth_l1_loss(
                selected_prediction[:, 0], selected_timing[:, 0]
            )
            interval_loss = F.smooth_l1_loss(
                selected_prediction[:, 1], selected_timing[:, 2]
            )
            duration_mae = (
                selected_prediction[:, 0] - selected_timing[:, 0]
            ).abs().mean()
            interval_mae = (
                selected_prediction[:, 1] - selected_timing[:, 2]
            ).abs().mean()
            rhythm_loss = (
                self.loss_weights["duration"] * duration_loss
                + self.loss_weights["interval"] * interval_loss
            )

        bilateral_loss = zero
        bilateral_accuracy = zero
        bilateral_surprise = words.new_zeros(targets.shape)
        target_sides = torch.randint(
            0,
            2,
            (words.shape[0],),
            device=words.device,
            generator=mask_generator,
        )
        side_selector = F.one_hot(target_sides, 2).bool()
        bilateral_valid = (
            side_selector[:, :, None, None].expand_as(valid) & valid
        )
        if self.tasks["bilateral"]:
            bilateral_targets = (
                torch.rand(
                    valid.shape,
                    device=words.device,
                    generator=mask_generator,
                )
                < self.bilateral_mask_ratio
            ) & bilateral_valid
            bilateral_targets = self._ensure_one(
                bilateral_targets, bilateral_valid
            )
            bilateral_input = self._span_mask(bilateral_targets, valid)
            _, bilateral_logits, _ = self._encode_task(
                words,
                word_mask,
                timing,
                bilateral_input,
                use_bilateral_context=True,
            )
            bilateral_loss, bilateral_accuracy = self._masked_loss(
                bilateral_logits, targets, bilateral_targets
            )
            bilateral_probability = bilateral_logits.log_softmax(dim=-1)
            bilateral_surprise = -bilateral_probability.gather(
                -1, targets[..., None]
            ).squeeze(-1)
            bilateral_surprise = bilateral_surprise * bilateral_targets

        contralateral_loss = zero
        contralateral_hard_loss = zero
        contralateral_soft_loss = zero
        contralateral_prototype_loss = zero
        contralateral_accuracy = zero
        contralateral_topk_accuracy = zero
        if self.tasks["contralateral"]:
            contralateral_targets = (
                torch.rand(
                    valid.shape,
                    device=words.device,
                    generator=mask_generator,
                )
                < self.contralateral_mask_ratio
            ) & bilateral_valid
            contralateral_targets = self._ensure_one(
                contralateral_targets, bilateral_valid
            )
            contralateral_input = bilateral_valid
            _, contralateral_logits, contralateral_prototypes = (
                self._encode_task(
                    words,
                    word_mask,
                    timing,
                    contralateral_input,
                    use_bilateral_context=True,
                )
            )
            contralateral_result = self._relaxed_masked_loss(
                contralateral_logits,
                contralateral_prototypes,
                targets,
                contralateral_targets,
                self.contralateral_component_weights,
            )
            contralateral_loss = contralateral_result["loss"]
            contralateral_hard_loss = contralateral_result["hard_loss"]
            contralateral_soft_loss = contralateral_result["soft_loss"]
            contralateral_prototype_loss = contralateral_result[
                "prototype_loss"
            ]
            contralateral_accuracy = contralateral_result["accuracy"]
            contralateral_topk_accuracy = contralateral_result[
                "topk_accuracy"
            ]

        bilateral_pair_loss = zero
        bilateral_pair_accuracy = zero
        if self.tasks["bilateral_pair"]:
            left = base_encoded["left_embedding"]
            right = base_encoded["right_embedding"]
            negative_shift = int(
                torch.randint(
                    1,
                    words.shape[0],
                    (),
                    device=words.device,
                    generator=mask_generator,
                ).item()
            )
            positive_logits = self._pair_logits(left, right)
            negative_logits = self._pair_logits(
                left, right.roll(negative_shift, dims=0)
            )
            pair_logits = torch.cat([positive_logits, negative_logits])
            pair_targets = torch.cat(
                [
                    torch.ones_like(positive_logits),
                    torch.zeros_like(negative_logits),
                ]
            )
            bilateral_pair_loss = F.binary_cross_entropy_with_logits(
                pair_logits, pair_targets
            )
            bilateral_pair_accuracy = (
                (pair_logits >= 0.0) == pair_targets.bool()
            ).float().mean()

        swap_loss = zero
        if self.tasks["swap"]:
            swapped = self.sentence_encoder(
                words.flip(1),
                word_mask.flip(1),
                timing.flip(1),
            )
            swap_loss = (
                F.mse_loss(
                    base_encoded["shared_embedding"],
                    swapped["shared_embedding"],
                )
                + F.mse_loss(
                    base_encoded["absolute_difference"],
                    swapped["absolute_difference"],
                )
                + F.mse_loss(
                    base_encoded["directional_difference"],
                    -swapped["directional_difference"],
                )
            )

        total = (
            self.loss_weights["within"] * within_loss
            + self.loss_weights["cross_dof"] * dof_loss
            + self.loss_weights["rhythm"] * rhythm_loss
            + self.loss_weights["bilateral"] * bilateral_loss
            + self.loss_weights["contralateral"] * contralateral_loss
            + self.loss_weights["bilateral_pair"] * bilateral_pair_loss
            + self.loss_weights["swap"] * swap_loss
        )
        return {
            "loss": total,
            "within_loss": within_loss,
            "cross_dof_loss": dof_loss,
            "cross_dof_hard_loss": dof_hard_loss,
            "cross_dof_soft_loss": dof_soft_loss,
            "cross_dof_prototype_loss": dof_prototype_loss,
            "rhythm_loss": rhythm_loss,
            "duration_loss": duration_loss,
            "interval_loss": interval_loss,
            "duration_mae": duration_mae,
            "interval_mae": interval_mae,
            "bilateral_loss": bilateral_loss,
            "contralateral_loss": contralateral_loss,
            "contralateral_hard_loss": contralateral_hard_loss,
            "contralateral_soft_loss": contralateral_soft_loss,
            "contralateral_prototype_loss": (
                contralateral_prototype_loss
            ),
            "bilateral_pair_loss": bilateral_pair_loss,
            "swap_loss": swap_loss,
            "within_accuracy": within_accuracy,
            "cross_dof_accuracy": dof_accuracy,
            "cross_dof_topk_accuracy": dof_topk_accuracy,
            "bilateral_accuracy": bilateral_accuracy,
            "contralateral_accuracy": contralateral_accuracy,
            "contralateral_topk_accuracy": contralateral_topk_accuracy,
            "bilateral_pair_accuracy": bilateral_pair_accuracy,
            "bilateral_surprise": bilateral_surprise,
            "shared_embedding": base_encoded["shared_embedding"],
            "directional_difference": base_encoded[
                "directional_difference"
            ],
            "absolute_difference": base_encoded[
                "absolute_difference"
            ],
            "left_difference_map": base_encoded[
                "left_difference_map"
            ],
            "right_difference_map": base_encoded[
                "right_difference_map"
            ],
        }


class GaitLanguageDownstreamModel(nn.Module):
    """Classify disease from hierarchical healthy-reference deviations."""

    def __init__(
        self,
        sentence_encoder: GaitSentenceEncoder,
        *,
        word_dim: int,
        num_classes: int = 3,
        dropout: float = 0.2,
        deviation_dof_dim: int = 64,
        deviation_std_floor: float = 0.05,
    ) -> None:
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.deviation_encoder = HierarchicalGaitDeviationEncoder(
            word_dim=word_dim,
            dof_hidden_dim=deviation_dof_dim,
            dropout=dropout,
            std_floor=deviation_std_floor,
        )
        self.disease_head = nn.Sequential(
            nn.LayerNorm(word_dim),
            nn.Linear(word_dim, word_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(word_dim, num_classes),
        )
        self.affected_side_head = nn.Sequential(
            nn.LayerNorm(word_dim + 1),
            nn.Linear(word_dim + 1, word_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(word_dim // 2, 2),
        )

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return supervised logits and interpretable deviation levels."""
        encoded = self.sentence_encoder(words, word_mask, timing)
        deviation = self.deviation_encoder(encoded["tokens"], word_mask)
        affected_side_features = torch.cat(
            [
                deviation["bilateral_deviation_direction"],
                deviation["bilateral_deviation_magnitude_gap"],
            ],
            dim=-1,
        )
        return {
            **encoded,
            **deviation,
            "disease_logits": self.disease_head(
                deviation["subject_embedding"]
            ),
            "affected_side_logits": self.affected_side_head(
                affected_side_features
            ),
        }


__all__ = ["GaitLanguageDownstreamModel", "GaitLanguageSSLModel"]
