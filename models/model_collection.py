import torch.nn
import torchvision.transforms

from models.vitt import ViTT
from models.vision_transformer_with_pytorch import ViTAutoencoder

from models.models_mae import *

class Mean(torch.nn.Module):

    def forward(self, x):
        return x.mean(dim=1)

def get_vitt(device):
    model = ViTT(d_model=768, out_dim=(224, 224), nhead=8, device=device)
    return torch.nn.Identity(), model

def get_vit_autoencoder(device):
    vit_autoencoder = ViTAutoencoder()
    # state_dict = torch.load('./models/checkpoints/VitAutoEncoder.model')
    # vit_autoencoder.load_state_dict(state_dict)
    state_dict = torch.load('./models/checkpoints/vitt/model_1698322627_vitt_9.model')
    model = torch.nn.Sequential(vit_autoencoder, Mean())
    model.load_state_dict(state_dict)
    return torch.nn.Identity(), model