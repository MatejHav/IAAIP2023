import numpy as np
import torch
import math
import warnings

warnings.filterwarnings("ignore")

from torch import nn
from torchvision.models import get_model
from torchvision.transforms import Resize
from models.models_mae import *
from main import get_dataloader


class ViTT(nn.Module):

    def __init__(self, d_model, out_dim, nhead, device, dropout=0.25, num_decoder_layers=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = out_dim
        self.vit = mae_vit_base_patch16()
        state_dict = torch.load('./models/checkpoints/mae_pretrain_vit_base.pth')['model']
        for key in state_dict:
            self.vit.state_dict()[key] = state_dict[key]
            self.vit.state_dict()[key].requires_grad = False
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, dim_feedforward=256, batch_first=True)
        decoder_norm = nn.LayerNorm(d_model, *args, **kwargs)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.down_sampler = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=2),
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=2),
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)
        )
        self.up_sampler = nn.Sequential(
            nn.UpsamplingBilinear2d(size=out_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x: torch.Tensor):
        # MAE outputs the reconstructed image
        target = self.vit(x)
        target = self.down_sampler(target).mean(dim=1)
        B, H, W = target.shape
        target = target.view(1, B, H * W)
        memory = target[0, 0].view(1, 1, H * W)
        target_mask = torch.tril(torch.ones(B, B, device=x.device)).expand(48, B, B)
        # Pass segments through the decoder
        result = self.decoder(target, memory, tgt_mask=target_mask).view(B, 1, 48, 48)
        result = self.up_sampler(result).view(B, *self.out)
        return self.sigmoid(result)


if __name__ == '__main__':
    import os

    os.chdir("../")
    dataloader = get_dataloader('train', batch_size=8, subset=10)
    vitt = ViTT(d_model=768, out_dim=(576, 576), nhead=16, device=None)
    for batch, targets, masks, idx in dataloader:
        predictions = vitt(batch)
