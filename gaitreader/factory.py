"""Model construction order matches the archived paper implementation."""
from torch import nn
from .models import GaitFormer, VQGait, MaskedGaitReconstruction, WaveformPretraining
from .models.tokens import ShapeAttributePatchEmbedding

def build_encoder(args):
    encoder = GaitFormer(word_dim=args.word_dim, patch_hidden_channels=args.patch_hidden_channels,
        patch_output_channels=args.patch_output_channels, max_words=args.max_words,
        depth=args.encoder_depth, bilateral_depth=args.bilateral_depth,
        num_heads=args.encoder_heads, feedforward_dim=args.encoder_ff_dim,
        dropout=args.dropout, use_cls_token=True,
        use_dof_embedding=args.ssl_use_dof_embedding,
        use_cycle_embedding=args.ssl_use_cycle_embedding,
        use_duration_embedding=args.ssl_use_duration_embedding,
        use_interval_embedding=args.ssl_use_interval_embedding)
    encoder.within_side_encoder.patch_embeddings = nn.ModuleList([
        ShapeAttributePatchEmbedding(embedding, args.word_dim, args.vq_v2_scale_floor,
            use_shape=args.ssl_use_shape_embedding, use_mean=args.ssl_use_mean_embedding,
            use_scale=args.ssl_use_scale_embedding)
        for embedding in encoder.within_side_encoder.patch_embeddings])
    return encoder

def build_ssl(args, tokenizer):
    encoder = build_encoder(args)
    if args.ssl_objective == "waveform":
        return WaveformPretraining(encoder, args)
    return MaskedGaitReconstruction(tokenizer, encoder, args)
