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
from models.positional_encoder.pe import PositionalEncoding


class ViTT(nn.Module):

    def __init__(self, d_model, out_dim, nhead, device, dropout=0.1, num_decoder_layers=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = out_dim
        self.vit = mae_vit_base_patch16()
        state_dict = torch.load('./models/checkpoints/mae_pretrain_vit_base.pth')['model']
        for key in state_dict:
            self.vit.state_dict()[key] = state_dict[key]
            self.vit.state_dict()[key].requires_grad = False
        self.pe = PositionalEncoding(d_model=d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, dim_feedforward=d_model, batch_first=True)
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
        self.resize = Resize((48, 48))
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x: torch.Tensor, targets: torch.Tensor):
        # MAE outputs the reconstructed image
        memory = self.vit(x)
        memory = self.down_sampler(memory).mean(dim=1)
        B, H, W = memory.shape
        memory = memory.view(B, 1, H * W)
        memory = self.pe(memory).view(B, H * W)
        # Initial target is unknown
        if self.training:
            target = torch.zeros(1, H * W, device=x.device)
            targets = self.resize(targets).view(B, H * W)
            targets = torch.concat((target, targets), dim=0)[:-1]
            # Pass segments through the decoder
            memory_mask = torch.Tensor([[1 if j <= i else 0 for j in range(B)] for i in range(B)]).to(x.device)
            result = []
            for i in range(B):
                target = self.decoder(targets[i].view(1, H * W), memory,
                                      memory_mask=memory_mask[i].view(1, B)).view(1, H * W)
                result.append(target.view(1, H, W))
            result = torch.stack(result)
        else:
            target = torch.zeros(1, H * W, device=x.device)
            # Pass segments through the decoder
            memory_mask = torch.Tensor([[1 if j <= i else 0 for j in range(B)] for i in range(B)]).to(x.device)
            result = []
            for i in range(B):
                target = self.decoder(target, memory,
                                      memory_mask=memory_mask[i].view(1, B)).view(1, H * W)
                result.append(target.view(1, H, W))
            result = torch.stack(result)
        result = self.up_sampler(result).view(B, *self.out)
        return self.sigmoid(result)


if __name__ == '__main__':
    import os

    os.chdir("../")
    dataloader = get_dataloader('train', batch_size=8, subset=10)
    vitt = ViTT(d_model=768, out_dim=(576, 576), nhead=16, device=None)
    for batch, targets, masks, idx in dataloader:
        predictions = vitt(batch)
