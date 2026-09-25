"""VQShape and HeartLang gait adaptations using official model components."""
import torch
from torch import nn
from torch.nn import functional as F
from .sources import modules


class VQShape(nn.Module):
    def __init__(self, args):
        super().__init__()
        source, = modules('VQShape', 'vqshape.model')
        self.model = source.VQShape(
            dim_embedding=args.vqshape_dim, patch_size=args.vqshape_patch_length,
            num_patch=args.recording_length // args.vqshape_patch_length,
            num_enc_head=args.benchmark_heads, num_enc_layer=args.benchmark_depth,
            num_tokenizer_head=args.benchmark_heads, num_tokenizer_layer=args.benchmark_depth,
            num_dec_head=args.benchmark_heads, num_dec_layer=args.benchmark_depth,
            num_token=args.vqshape_tokens, len_s=args.vqshape_shape_length,
            len_input=args.recording_length, s_smooth_factor=11,
            num_code=args.benchmark_codebook_size, dim_code=args.vqshape_code_dim,
            lambda_commit=1., lambda_entropy=1., mask_ratio=args.vqshape_mask_ratio)
        self.output_dim = 6 * args.vqshape_tokens * (args.vqshape_code_dim + 4)
        self.loss_weights = dict(ts_loss=args.vqshape_waveform_weight, vq_loss=1.,
                                 shape_loss=args.vqshape_shape_weight, dist_loss=args.vqshape_disentangle_weight)

    def forward(self, x):
        # Preserve DOF identity instead of mixing all six code histograms together.
        signal = x.transpose(1, 2).reshape(-1, x.shape[1])
        normalized = (signal - signal.mean(-1, keepdim=True)) / (signal.var(-1, keepdim=True) + 1e-5).sqrt()
        h = self.model.tokenizer(self.model.encoder(normalized))
        z, tl, mean, scale = self.model.attr_decoder(h)
        quantized, _, _ = self.model.codebook(z)
        start = tl[..., :1] * (1 - self.model.min_shape_len)
        length = tl[..., 1:] * (1 - start) + self.model.min_shape_len
        # Equivalent to upstream tokenize()['token'], without unused reconstruction decoders.
        return torch.cat([quantized, start, length, mean, scale], -1).reshape(x.shape[0], -1)

    def pretrain_loss(self, x):
        _, losses = self.model(x.transpose(1, 2).reshape(-1, x.shape[1]), mode='pretrain')
        return sum(weight * losses[key].mean() for key, weight in self.loss_weights.items())

    def finish_pretraining(self):
        # The healthy dictionary is fixed during supervised fine-tuning.
        self.model.codebook.requires_grad_(False)
        self.model.attr_encoder.requires_grad_(False)


class HeartBackbone(nn.Module):
    """Original TokenEmbedding/Transformer with a sliced variable-length position table.

    Grouping sentences by valid length avoids introducing padded tokens into the
    upstream circular token convolution or its attention (which has no padding mask).
    """
    def __init__(self, args, *, decoder=False):
        super().__init__()
        source, = modules('HeartLang', 'backbone_vqhbr')
        self.core = source.VqhbrBackbone(
            seq_len=args.heartlang_max_cycles * 6, time_window=args.word_length,
            embed_dim=args.benchmark_dim, depth=args.heartlang_decoder_depth if decoder else args.benchmark_depth,
            heads=args.benchmark_heads, mlp_dim=args.benchmark_dim * 2,
            dim_head=args.benchmark_dim // args.benchmark_heads,
            dropout=args.benchmark_dropout, emb_dropout=args.benchmark_dropout,
            code_dim=args.heartlang_code_dim, Encoder=not decoder)
        self.core.tem_embed.embed = nn.Embedding(args.heartlang_max_cycles, args.benchmark_dim)

    def forward(self, words, mask=None):
        core = self.core
        b, n, _ = words.shape
        h = core.token_embed(words)
        if mask is not None:
            h = torch.where(mask[..., None], core.mask_token, h)
        channel = torch.arange(n, device=h.device).remainder(6)[None].expand(b, -1)
        cycle = torch.arange(n, device=h.device).div(6, rounding_mode='floor')[None].expand(b, -1)
        h = core.tem_embed(core.spa_embed(h, channel), cycle)
        h = torch.cat([core.cls_token[None, None].expand(b, 1, -1), h], 1)
        return core.norm_layer(core.transformer(core.dropout(h + core.pos_embed[:, :n + 1])))


class HeartTokenizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        source, = modules('HeartLang', 'utils.norm_ema_quantizer')
        self.encoder = HeartBackbone(args)
        self.projection = nn.Sequential(nn.Linear(args.benchmark_dim, args.benchmark_dim), nn.Tanh(),
                                         nn.Linear(args.benchmark_dim, args.heartlang_code_dim))
        self.codebook = source.NormEMAVectorQuantizer(args.benchmark_codebook_size,
                                                     args.heartlang_code_dim, beta=1., decay=.99,
                                                     kmeans_init=True)
        self.decoder = HeartBackbone(args, decoder=True)
        self.reconstruction = nn.Sequential(nn.Linear(args.benchmark_dim, args.benchmark_dim), nn.Tanh(),
                                             nn.Linear(args.benchmark_dim, args.word_length))

    def forward(self, words):
        feature = self.projection(self.encoder(words)[:, 1:])
        quantized, commitment, ids = self.codebook(feature)
        reconstruction = self.reconstruction(self.decoder(quantized)[:, 1:])
        return F.mse_loss(reconstruction, words) + commitment

    @torch.no_grad()
    def target_ids(self, words):
        # Compute IDs directly: upstream quantizer updates usage statistics even in eval().
        feature = F.normalize(self.projection(self.encoder(words)[:, 1:]), dim=-1)
        codes = self.codebook.embedding.weight
        distance = feature.square().sum(-1, keepdim=True) + codes.square().sum(-1) - 2 * feature @ codes.T
        return distance.argmin(-1)


class HeartLang(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = HeartTokenizer(args)
        # Fresh context encoder; NOT initialized from the VQ encoder.
        self.encoder = HeartBackbone(args)
        self.lm_head = nn.Linear(args.benchmark_dim, args.benchmark_codebook_size)
        self.mask_ratio = args.heartlang_mask_ratio
        self.output_dim = args.benchmark_dim

    def freeze_tokenizer(self):
        self.tokenizer.requires_grad_(False)
        self.tokenizer.eval()

    def groups(self, batch):
        words = batch['words'].flatten(0, 1)
        mask = batch['word_mask'].flatten(0, 1)
        counts = mask.sum(-1)
        for count in counts.unique(sorted=True):
            indices = (counts == count).nonzero().flatten()
            selected = words[indices][mask[indices]].reshape(len(indices), int(count) * 6, words.shape[-1])
            yield indices, selected

    def features(self, batch):
        indices, representations = [], []
        for index, words in self.groups(batch):
            indices.append(index)
            representations.append(self.encoder(words)[:, 0])
        return torch.cat(representations)[torch.cat(indices).argsort()]

    def loss(self, batch, stage):
        self.tokenizer.eval() if stage == 'ssl' else self.tokenizer.train(self.training)
        total = batch['words'].new_zeros(())
        for indices, words in self.groups(batch):
            if stage == 'vq':
                loss = self.tokenizer(words)
            else:
                ids = self.tokenizer.target_ids(words)
                b, n, _ = words.shape
                masked = torch.rand(b, n, device=words.device).argsort(1).argsort(1) >= int(n * (1 - self.mask_ratio))
                logits = self.lm_head(self.encoder(words, masked)[:, 1:])
                loss = F.cross_entropy(logits[masked], ids[masked])
            total = total + len(indices) * loss
        return total / (batch['words'].shape[0] * 2)
