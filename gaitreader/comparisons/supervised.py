"""Official backbones; only subject pooling/classification is dataset-specific."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .sources import modules, verify_source


class TimesNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        source, = modules('Time-Series-Library', 'models.TimesNet')
        config = SimpleNamespace(task_name='classification', seq_len=args.recording_length,
                                 label_len=0, pred_len=0, top_k=args.timesnet_top_k,
                                 num_kernels=args.timesnet_kernels, enc_in=6, num_class=3,
                                 d_model=args.timesnet_dim, d_ff=args.timesnet_dim * 2,
                                 e_layers=args.benchmark_depth, embed='timeF', freq='s',
                                 dropout=args.benchmark_dropout)
        self.backbone = source.Model(config)
        # Keep the original flattened classification representation, not forecasting normalization.
        self.backbone.projection = nn.Identity()
        self.output_dim = args.recording_length * args.timesnet_dim

    def forward(self, x):
        return self.backbone(x, x.new_ones(x.shape[:2]), None, None)


class ITransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        source, = modules('iTransformer', 'model.iTransformer')
        config = SimpleNamespace(seq_len=args.recording_length, pred_len=1,
                                 output_attention=False, use_norm=args.itransformer_instance_norm,
                                 d_model=args.benchmark_dim, d_ff=args.benchmark_dim * 2,
                                 n_heads=args.benchmark_heads, e_layers=args.benchmark_depth,
                                 factor=1, activation='gelu', embed='timeF', freq='s',
                                 class_strategy='projection', dropout=args.benchmark_dropout)
        self.backbone = source.Model(config)
        self.backbone.projector = nn.Identity()
        self.output_dim = 6 * args.benchmark_dim

    def forward(self, x):
        if self.backbone.use_norm:
            x = (x - x.mean(1, keepdim=True).detach()) / (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        h, _ = self.backbone.encoder(self.backbone.enc_embedding(x, None), attn_mask=None)
        return h.flatten(1)


class PatchTST(nn.Module):
    def __init__(self, args):
        super().__init__()
        source, = modules('PatchTST', 'src.models.patchTST', subdir='PatchTST_self_supervised')
        self.patch_len, self.stride = args.benchmark_patch_length, args.patchtst_stride
        self.backbone = source.PatchTSTEncoder(
            6, num_patch=(args.recording_length - self.patch_len) // self.stride + 1,
            patch_len=self.patch_len, n_layers=args.benchmark_depth, d_model=args.benchmark_dim,
            n_heads=args.benchmark_heads, d_ff=args.benchmark_dim * 2, dropout=args.benchmark_dropout)
        self.output_dim = 6 * args.benchmark_dim
        self.reconstruction = source.PretrainHead(args.benchmark_dim, self.patch_len, args.benchmark_dropout)
        self.mask_ratio = args.benchmark_mask_ratio

    def patches(self, x):
        return x.unfold(1, self.patch_len, self.stride)

    def forward(self, x):
        # Same last-patch representation as the official ClassificationHead.
        return self.backbone(self.patches(x))[..., -1].flatten(1)

    def pretrain_loss(self, x):
        patches = self.patches(x)
        b, p, c, _ = patches.shape
        keep = int(p * (1 - self.mask_ratio))
        order = torch.rand(b, p, c, device=x.device).argsort(1).argsort(1)
        masked = order >= keep
        prediction = self.reconstruction(self.backbone(patches.masked_fill(masked[..., None], 0)))
        return (prediction - patches).square().mean(-1)[masked].mean()
