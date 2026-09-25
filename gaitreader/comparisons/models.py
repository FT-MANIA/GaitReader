"""Subject-level wrapper and training-stage dispatch."""
import torch
from torch import nn
from .supervised import TimesNet, PatchTST, ITransformer
from .contrastive import HierarchicalContrastive, TSTCC
from .quantized import VQShape, HeartLang

MODELS = ('timesnet', 'patchtst', 'itransformer',
          'ts_tcc', 'ts2vec', 'trep', 'vqshape', 'heartlang')


class ComparisonModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.name = args.comparison_model
        if self.name in ('ts2vec', 'trep'):
            self.backbone = HierarchicalContrastive(args, self.name)
        elif self.name == 'ts_tcc':
            self.backbone = TSTCC(args, device)
        else:
            self.backbone = dict(timesnet=TimesNet,
                                 patchtst=PatchTST, itransformer=ITransformer,
                                 vqshape=VQShape, heartlang=HeartLang)[self.name](args)
        self.classifier = nn.Sequential(nn.Dropout(args.benchmark_dropout),
                                         nn.Linear(self.backbone.output_dim * 2, 3))
        self.linear_probe = args.comparison_encoder_mode == 'linear_probe'
        self.stages = ['downstream']
        if self.name in ('ts_tcc', 'ts2vec', 'trep', 'vqshape') or (self.name == 'patchtst' and args.patchtst_pretrain):
            self.stages.insert(0, 'ssl')
        if self.name == 'heartlang':
            self.stages = ['vq', 'ssl', 'downstream']

    def set_stage(self, stage):
        self.stage = stage
        self.requires_grad_(True)
        if self.name == 'heartlang':
            if stage != 'vq':
                self.backbone.freeze_tokenizer()
            else:
                self.backbone.encoder.requires_grad_(False)
                self.backbone.lm_head.requires_grad_(False)
                self.backbone.tokenizer.codebook.requires_grad_(False)
        if stage == 'downstream':
            if self.name in ('ts2vec', 'trep'):
                self.backbone.finish_pretraining()
                self.backbone.js_head.requires_grad_(False) if self.name == 'trep' else None
                self.backbone.pred_head.requires_grad_(False) if self.name == 'trep' else None
            if self.name == 'vqshape':
                self.backbone.finish_pretraining()
                self.backbone.model.decoder.requires_grad_(False)
                self.backbone.model.shape_decoder.requires_grad_(False)
            if self.name == 'ts_tcc':
                self.backbone.temporal.requires_grad_(False)
            if self.name == 'patchtst':
                self.backbone.reconstruction.requires_grad_(False)
            if self.name == 'heartlang':
                self.backbone.lm_head.requires_grad_(False)
            if self.linear_probe:
                self.backbone.requires_grad_(False)
        else:
            self.classifier.requires_grad_(False)
        if self.name in ('ts2vec', 'trep'):
            self.backbone.average.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'stage', None) == 'downstream' and self.linear_probe:
            self.backbone.eval()
        if self.name == 'heartlang' and getattr(self, 'stage', None) != 'vq':
            self.backbone.tokenizer.eval()
        return self

    def forward(self, batch):
        if self.name == 'heartlang':
            h = self.backbone.features(batch)
        else:
            h = self.backbone(batch['signal'].flatten(0, 1))
        return self.classifier(h.reshape(batch['signal'].shape[0], -1))

    def pretrain_loss(self, batch, stage):
        if self.name == 'heartlang':
            return self.backbone.loss(batch, stage)
        # Sample one side per subject; paired legs must not become contrastive negatives.
        signal = batch['signal']
        sides = torch.randint(2, (signal.shape[0],), device=signal.device)
        x = signal[torch.arange(signal.shape[0], device=signal.device), sides]
        return self.backbone.pretrain_loss(x)

    def after_update(self):
        if self.stage == 'ssl' and self.name in ('ts2vec', 'trep'):
            self.backbone.after_update()
