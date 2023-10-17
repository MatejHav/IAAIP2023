import torch
import math
import warnings

warnings.filterwarnings("ignore")

from torch import nn
from torchvision.models import get_model
from torchvision.transforms import Resize

from main import get_dataloader


class ViTT(nn.Module):

    def __init__(self, d_model, out_dim, nhead, device, dropout=0.1, num_decoder_layers=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = out_dim
        # self.vit = get_model('ViT_B_32')
        # self.vit.heads = nn.Sequential()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout, dim_feedforward=2048)
        decoder_norm = nn.LayerNorm(d_model, *args, **kwargs)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.linear_reshape = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.Linear(512, math.prod(out_dim))
        )
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x: torch.Tensor):
        B, _ = x.shape
        target = self.dropout(x)
        target = target.view(1, *target.shape)
        memory = torch.zeros(*target.shape, device=self.device)
        # Pass segments through the decoder
        result = self.decoder(target, memory)
        # Result of shape (B, 768), reshape it using linear layers

        result = self.linear_reshape(result).view(B, *self.out)

        # Sigmoid last two values representing tarting and ending x coordinates
        # result = torch.concat((result[:, :, :3], self.sigmoid(result[:, :, 3:])), dim=2)

        return self.sigmoid(result)


if __name__ == '__main__':
    import os

    os.chdir("../")
    dataloader = get_dataloader('train', batch_size=8, subset=10)
    vitt = ViTT(d_model=768, out_dim=(320, 800), nhead=16, device=None)
    for batch, targets, masks, idx in dataloader:
        predictions = vitt(batch)
