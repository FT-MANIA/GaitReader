"""Paper SSL objectives and the subject-level linear classification head."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .gaitformer import GaitFormer
from .tokens import split_shape

MaskGenerator = torch.Generator | dict[str, torch.Generator] | None


def _prediction_heads(word_dim, output_dim):
    # Auxiliary heads must not advance the encoder/dropout RNG stream.
    with torch.random.fork_rng(devices=[]):
        return nn.ModuleList([nn.Linear(word_dim, output_dim) for _ in range(6)])


class MaskedGaitReconstruction(nn.Module):
    """One shared mask/backbone pass for code, attributes and vocabulary waveform."""
    metric_keys = ("loss", "masked_top1_loss", "masked_top1_accuracy",
                   "target_active_code_ratio", "target_perplexity")

    def __init__(self, tokenizer, encoder, args):
        super().__init__()
        # A live VQ model can retain gradients from its final training batch.
        # Freezing alone does not clear them when transitioning to SSL.
        tokenizer.zero_grad(set_to_none=True)
        self.tokenizer = tokenizer.requires_grad_(False).eval()
        self.encoder = encoder
        self.random_mask_ratio = args.random_mask_ratio
        self.masked_top1_weight = args.masked_top1_weight
        self.attribute_weight = args.ssl_attribute_weight
        self.attribute_mean_weight = args.ssl_attribute_mean_weight
        self.attribute_log_scale_weight = args.ssl_attribute_log_scale_weight
        self.code_waveform_weight = args.ssl_code_waveform_weight
        self.code_waveform_temperature = args.ssl_code_waveform_temperature
        self.attribute_active = self.attribute_weight > 0 or self.code_waveform_weight > 0
        self.code_prediction_heads = _prediction_heads(encoder.word_dim, args.codebook_size)
        self.extra_metric_keys = ()
        if self.attribute_active:
            self.attribute_prediction_heads = _prediction_heads(encoder.word_dim, 2)
            self.extra_metric_keys = ("attribute_loss", "attribute_mean_loss", "attribute_log_scale_loss")
        if self.code_waveform_weight > 0:
            self.register_buffer("shape_templates", None, persistent=False)
            self.extra_metric_keys += ("code_waveform_loss", "code_waveform_top1_mse")

    def _load_from_state_dict(self, *args, **kwargs):
        # Derived templates must follow the tokenizer restored from a checkpoint.
        if self.code_waveform_weight > 0:
            self.shape_templates = None
        super()._load_from_state_dict(*args, **kwargs)

    def _attribute_reconstruction(self, hidden, logits, words, positions):
        prediction = torch.stack([head(hidden[..., dof, :])
                                  for dof, head in enumerate(self.attribute_prediction_heads)], dim=3)
        with torch.autocast(device_type=words.device.type, enabled=False):
            _, target_attributes, _, _ = split_shape(words, self.tokenizer.scale_floor)
            predicted = prediction[positions].float()
            target = target_attributes[positions]
            mean_loss = F.smooth_l1_loss(predicted[:, 0], target[:, 0])
            log_scale_loss = F.smooth_l1_loss(predicted[:, 1], target[:, 1])
            attribute_loss = self.attribute_mean_weight * mean_loss + self.attribute_log_scale_weight * log_scale_loss
            metrics = {"attribute_loss": attribute_loss, "attribute_mean_loss": mean_loss,
                       "attribute_log_scale_loss": log_scale_loss}
            auxiliary_loss = self.attribute_weight * attribute_loss
            if self.code_waveform_weight > 0:
                if self.shape_templates is None:
                    with torch.no_grad():
                        self.shape_templates = self.tokenizer.decode_codebook().float().detach()
                # [6,K,L]; each masked word uses only its own DOF's templates.
                probability = (logits.float() / self.code_waveform_temperature).softmax(-1)
                shape = torch.einsum("bswck,ckl->bswcl", probability, self.shape_templates)[positions]
                reconstructed = predicted[:, :1] + predicted[:, 1:].exp() * shape
                waveform_loss = F.mse_loss(reconstructed, words[positions].float())
                auxiliary_loss = auxiliary_loss + self.code_waveform_weight * waveform_loss
                with torch.no_grad():
                    dof = torch.arange(6, device=words.device).expand_as(positions)[positions]
                    ids = logits[positions].argmax(-1)
                    hard_shape = self.shape_templates[dof, ids]
                    hard_waveform = predicted[:, :1] + predicted[:, 1:].exp() * hard_shape
                    hard_mse = F.mse_loss(hard_waveform, words[positions].float())
                metrics.update(code_waveform_loss=waveform_loss, code_waveform_top1_mse=hard_mse)
        return auxiliary_loss, metrics

    def train(self, mode: bool = True) -> MaskedGaitReconstruction:
        super().train(mode)
        self.tokenizer.eval()
        return self

    def _random_mask_positions(
        self,
        word_mask: torch.Tensor,
        generator: MaskGenerator,
    ) -> torch.Tensor:
        generator = self._task_generator(generator, "masked")
        batch_size, sides, word_count = word_mask.shape
        valid = word_mask[..., None].expand(
            batch_size, sides, word_count, 6
        )
        random_mask = (
            torch.rand(
                valid.shape,
                device=word_mask.device,
                generator=generator,
            )
            < self.random_mask_ratio
        )
        return random_mask & valid

    @staticmethod
    def _task_generator(generator, task):
        return generator[task] if isinstance(generator, dict) else generator

    def _code_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                head(tokens[..., dof, :])
                for dof, head in enumerate(self.code_prediction_heads)
            ],
            dim=3,
        )

    def forward(self, words, word_mask, timing, *, mask_generator=None):
        with torch.no_grad():
            targets = self.tokenizer.tokenize(words, word_mask, timing)
        valid = word_mask[..., None].expand(*word_mask.shape, 6)
        active, perplexity = self.tokenizer.codebook.usage(targets["indices"], valid)
        positions = self._random_mask_positions(word_mask, mask_generator)
        hidden = self.encoder(words, word_mask, timing, masked_positions=positions)["tokens"]
        logits = self._code_logits(hidden)
        code_loss = F.cross_entropy(logits[positions].float(), targets["indices"][positions])
        total = self.masked_top1_weight * code_loss
        metrics = {}
        if self.attribute_active:
            auxiliary, metrics = self._attribute_reconstruction(hidden, logits, words, positions)
            total = total + auxiliary
        accuracy = logits[positions].argmax(-1).eq(targets["indices"][positions]).float().mean()
        return {"loss": total, "masked_top1_loss": code_loss, "masked_top1_accuracy": accuracy,
                "target_active_code_ratio": active, "target_perplexity": perplexity, **metrics}


class WaveformPretraining(nn.Module):
    """Vocabulary-free ablation: direct masked raw waveform MSE."""
    metric_keys = ("loss", "waveform_loss")

    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder
        self.random_mask_ratio = args.random_mask_ratio
        self.waveform_heads = _prediction_heads(encoder.word_dim, args.word_length)

    _random_mask_positions = MaskedGaitReconstruction._random_mask_positions
    _task_generator = staticmethod(MaskedGaitReconstruction._task_generator)

    def forward(self, words, word_mask, timing, *, mask_generator=None):
        positions = self._random_mask_positions(word_mask, mask_generator)
        hidden = self.encoder(words, word_mask, timing, masked_positions=positions)["tokens"]
        prediction = torch.stack([head(hidden[..., d, :])
                                  for d, head in enumerate(self.waveform_heads)], dim=3)
        loss = F.mse_loss(prediction[positions].float(), words[positions].float())
        return {"loss": loss, "waveform_loss": loss}


class GaitClassifier(nn.Module):
    """Classify the subject from the pretrained gait CLS embedding."""

    def __init__(
        self,
        encoder: GaitFormer,
        *,
        fine_tune_encoder: bool,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.fine_tune_encoder = fine_tune_encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = fine_tune_encoder
        self.classifier = nn.Linear(encoder.word_dim, num_classes)

    def train(self, mode: bool = True) -> GaitClassifier:
        super().train(mode)
        if not self.fine_tune_encoder:
            self.encoder.eval()
        return self

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        timing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.fine_tune_encoder:
            encoded = self.encoder(words, word_mask, timing)
        else:
            with torch.no_grad():
                encoded = self.encoder(words, word_mask, timing)
        return {
            **encoded,
            "disease_logits": self.classifier(
                encoded["cls_embedding"]
            ),
        }
