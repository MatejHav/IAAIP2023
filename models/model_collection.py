import torch.nn
import torchvision.transforms

from models.vitt import ViTT
from models.vision_transformer_with_pytorch import ViTAutoencoder

class Mean(torch.nn.Module):

    def forward(self, x):
        return x.mean(dim=1)

def get_vitt(device):
    vitt = ViTT(d_model=200, out_dim=(576, 576), nhead=5, device=device)
    vitt.to(device)
    transforms = torchvision.transforms.Resize(size=(576, 576))
    backbone = torch.nn.Sequential(transforms, torch.load('./models/backbone/encoder.model'))
    return backbone, vitt

def get_vit_autoencoder(device):
    vit_autoencoder = ViTAutoencoder()
    state_dict = torch.load('./models/checkpoints/VitAutoEncoder.model')
    vit_autoencoder.load_state_dict(state_dict)
    model = torch.nn.Sequential(vit_autoencoder, Mean())
    return torch.nn.Identity(), model