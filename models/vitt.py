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
        self.transforms = Resize(size=(224, 224))
        self.out = out_dim
        self.vit = get_model('ViT_B_32')
        self.vit.heads = nn.Sequential()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        decoder_norm = nn.LayerNorm(d_model, *args, **kwargs)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.linear_reshape = nn.Sequential(
            nn.Linear(768, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, math.prod(out_dim))
        )
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x: torch.Tensor):
        B, _, _, _ = x.shape
        # Resize inputs into correct shape
        x = self.transforms(x)
        # Pass the image through the vision transformer
        with torch.no_grad():
            target = self.vit(x)
        target = self.linear_reshape(target)
        memory = torch.zeros(1, *target.shape[1:], device=self.device)
        # Pass segments through the decoder
        result = self.decoder(target, memory).view(B, *self.out)
        # Result of shape (B, 768), reshape it using linear layers

        # Sigmoid last two values representing tarting and ending x coordinates
        result = torch.concat((result[:, :, :4], self.sigmoid(result[:, :, 4:])), dim=2)

        return result


if __name__ == '__main__':
    import os

    os.chdir("../")
    dataloader = get_dataloader('train', batch_size=8, subset=10)
    vitt = ViTT(d_model=768, out_dim=(320, 800), nhead=16, device=None)
    for batch, targets, masks, idx in dataloader:
        predictions = vitt(batch)
