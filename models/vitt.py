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

    def __init__(self, d_model, out_dim, nhead, device, dropout=0.25, num_decoder_layers=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = out_dim
        self.vit = mae_vit_base_patch16()
        state_dict = torch.load('./models/checkpoints/mae_pretrain_vit_base.pth')['model']
        for key in state_dict:
            self.vit.state_dict()[key] = state_dict[key]
            self.vit.state_dict()[key].requires_grad = False
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, dim_feedforward=256)
        decoder_norm = nn.LayerNorm(d_model, *args, **kwargs)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.linear_reshape = nn.Sequential(
            nn.Linear(197*d_model, 512),
            nn.Linear(512, math.prod(out_dim))
        )
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x: torch.Tensor):
        B, _, _, _ = x.shape
        target = self.vit(x)
        result = torch.zeros(1, *target.shape[1:], device=self.device)
        # Pass segments through the decoder
        result = self.decoder(target, result).view(B, np.prod(target.shape[1:]))
        # Result of shape (B, 768), reshape it using linear layers
        result = self.linear_reshape(result).view(B, *self.out)
        return self.sigmoid(result)


if __name__ == '__main__':
    import os

    os.chdir("../")
    dataloader = get_dataloader('train', batch_size=8, subset=10)
    vitt = ViTT(d_model=768, out_dim=(576, 576), nhead=16, device=None)
    for batch, targets, masks, idx in dataloader:
        predictions = vitt(batch)
