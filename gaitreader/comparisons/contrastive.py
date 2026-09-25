"""Original TS2Vec/T-Rep losses and TS-TCC temporal/contextual objectives."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .sources import modules


class HierarchicalContrastive(nn.Module):
    def __init__(self, args, method):
        super().__init__()
        self.method = method
        repo = 'TS2Vec' if method == 'ts2vec' else 'T-Rep'
        encoder, losses = modules(repo, 'models.encoder', 'models.losses')
        options = dict(input_dims=6, output_dims=args.benchmark_dim,
                       hidden_dims=args.contrastive_hidden, depth=args.contrastive_depth)
        if method == 'trep':
            options.update(time_embedding='t2v_sin', time_embedding_dim=args.trep_time_dim)
            heads, = modules(repo, 'models.task_heads')
            self.js_head = heads.TembedDivPredHead(args.benchmark_dim, 1, hidden_features=128)
            self.pred_head = heads.TembedCondPredHead(args.benchmark_dim + args.trep_time_dim,
                                                      args.benchmark_dim, hidden_features=[64, 128])
        self.encoder = encoder.TSEncoder(**options)
        self.average = torch.optim.swa_utils.AveragedModel(self.encoder)
        self.average.requires_grad_(False)
        self.average.update_parameters(self.encoder)
        self.loss_function = losses.hierarchical_contrastive_loss
        self.output_dim = args.benchmark_dim
        self.temporal_unit = args.contrastive_temporal_unit

    def forward(self, x):
        # Classification does not randomly mask the supervised input.
        if self.method == 'trep':
            time = torch.arange(x.shape[1], device=x.device).float()[None, :, None].expand(x.shape[0], -1, -1)
            h, _ = self.encoder(x.clone(), time, mask='all_true')
        else:
            h = self.encoder(x.clone(), mask='all_true')
        return h.amax(1)

    def pretrain_loss(self, x):
        length = x.shape[1]
        overlap = np.random.randint(2 ** (self.temporal_unit + 1), length + 1)
        left = np.random.randint(length - overlap + 1)
        right = left + overlap
        extra_left = np.random.randint(left + 1)
        extra_right = np.random.randint(right, length + 1)
        offset = torch.from_numpy(np.random.randint(-extra_left, length - extra_right + 1,
                                                    size=x.shape[0])).to(x.device)
        def view(start, end):
            indices = offset[:, None] + torch.arange(start, end, device=x.device)[None]
            data = x[torch.arange(x.shape[0], device=x.device)[:, None], indices]
            if self.method == 'trep':
                return self.encoder(data, indices.float()[..., None], mask='binomial')
            return self.encoder(data, mask='binomial'), None
        h1, t1 = view(extra_left, right)
        h2, t2 = view(left, extra_right)
        h1, h2 = h1[:, -overlap:], h2[:, :overlap]
        if self.method == 'ts2vec':
            return self.loss_function(h1, h2, temporal_unit=self.temporal_unit)
        return self.loss_function(h1, h2, t1[:, -overlap:], t2[:, :overlap], self.js_head, self.pred_head,
                                  weights=dict(instance_contrast=.25, temporal_contrast=.25,
                                               tembed_jsd_pred=.25, tembed_cond_pred=.25),
                                  temporal_unit=self.temporal_unit)

    def after_update(self):
        self.average.update_parameters(self.encoder)

    def finish_pretraining(self):
        self.encoder.load_state_dict(self.average.module.state_dict())


class TSTCC(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        backbone, temporal, loss = modules('TS-TCC', 'models.model', 'models.TC', 'models.loss')
        config = SimpleNamespace(input_channels=6, kernel_size=8, stride=1,
                                 dropout=args.benchmark_dropout, final_out_channels=args.benchmark_dim,
                                 features_len=1, num_classes=3,
                                 TC=SimpleNamespace(hidden_dim=args.benchmark_dim, timesteps=args.tstcc_timesteps))
        self.encoder = backbone.base_Model(config)
        self.encoder.logits = nn.Identity()
        self.temporal = temporal.TC(config, device)
        self.temporal.lsoftmax = nn.LogSoftmax(dim=1)
        self.loss_class = loss.NTXentLoss
        length = args.recording_length
        for _ in range(3):
            length = (length + 1) // 2 + 1
        self.output_dim = length * args.benchmark_dim
        self.scale_sigma, self.jitter_sigma = args.tstcc_scale_sigma, args.tstcc_jitter_sigma
        self.max_segments = args.tstcc_max_segments

    def forward(self, x):
        return self.encoder(x.transpose(1, 2))[1].flatten(1)

    def pretrain_loss(self, x):
        x = x.transpose(1, 2)
        # Keep the original weak/strong augmentation families. One common permutation
        # is applied to ALL six DOFs, fixing upstream's pat[0,warp] channel duplication.
        weak = x * (2 + self.scale_sigma * torch.randn(x.shape[0], 1, x.shape[-1], device=x.device))
        strong = torch.empty_like(x)
        for i in range(x.shape[0]):
            count = np.random.randint(1, self.max_segments)
            splits = np.sort(np.random.choice(x.shape[-1] - 2, count - 1, replace=False))
            segments = np.split(np.arange(x.shape[-1]), splits)
            order = np.concatenate([segments[j] for j in np.random.permutation(count)])
            strong[i] = x[i, :, torch.as_tensor(order, device=x.device)]
        strong = strong + self.jitter_sigma * torch.randn_like(strong)
        z1 = F.normalize(self.encoder(weak)[1], dim=1)
        z2 = F.normalize(self.encoder(strong)[1], dim=1)
        l1, c1 = self.temporal(z1, z2)
        l2, c2 = self.temporal(z2, z1)
        return l1 + l2 + .7 * self.loss_class(x.device, x.shape[0], .2, True)(c1, c2)
