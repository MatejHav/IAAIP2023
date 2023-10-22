from timm.models.vision_transformer import Block
import torch.nn as nn

class OriginalEncoder(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, depth=1):
        super(OriginalEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
