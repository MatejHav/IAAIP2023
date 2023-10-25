import torch.nn
import torchvision.transforms

from models.vitt import ViTT


def get_vitt(device):
    vitt = ViTT(d_model=200, out_dim=(576, 576), nhead=5, device=device)
    vitt.to(device)
    transforms = torchvision.transforms.Resize(size=(576, 576))
    backbone = torch.nn.Sequential(transforms, torch.load('./models/backbone/encoder.model'))
    return backbone, vitt
