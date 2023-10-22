import torch
from timm.models.vision_transformer import Block, PatchEmbed
import torch.nn as nn

class OriginalDecoder(nn.Module):
    def __init__(self, decoder_embed_dim=512, decoder_num_heads=16, mlp_ratio=4,
                 qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, decoder_depth=8):
        super(OriginalDecoder, self).__init__()
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])

    def forward(self, x):
        for blk in self.decoder_blocks:
            x = blk(x)
        return x
