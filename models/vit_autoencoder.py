import torch
import torch.nn as nn
from torchvision.models import get_model
from torchvision.transforms import Resize

class ViTAutoencoder(nn.Module):
    def __init__(self, image_size=224, hidden_dim=1024):
        super(ViTAutoencoder, self).__init__()

        self.resize = Resize(size=image_size)

        # Encoder: ViT without the head
        self.vit = get_model('vit_l_32')
        self.vit.heads = nn.Identity()  # Removing the head
        for param in self.vit.parameters():
            param.requires_grad = False

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, hidden_dim * 2),  # out_dim is * 2 but could be other values
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, image_size * image_size * 3),  # Assuming 3 channels (RGB)
            nn.Sigmoid()  # Ensuring output values are between [0, 1]
        )

    def forward(self, x):
        # Transform
        x = self.resize(x)

        # Encoding
        z = self.vit(x)

        # Decoding
        x_recon = self.decoder(z)
        x_recon = x_recon.view(x.size(0), 3, x.size(2), x.size(3))  # Reshape back to [B, C, H, W]

        return x_recon


