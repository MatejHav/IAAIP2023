import torch.nn
import torchvision.transforms

from models.vitt import ViTT
from models.vit_autoencoder import ViTAutoencoder

from models.models_mae import *

def get_vitt(device):
    model = ViTT(d_model=2304, out_dim=(224, 224), nhead=1, device=device)
    state_dict = torch.load('./models/checkpoints/vitt/model_1699126061_vitt_6.model')
    model.load_state_dict(state_dict)
    return model

def get_vit_autoencoder(device):
    model = ViTAutoencoder()
    # state_dict = torch.load('./models/checkpoints/VitAutoEncoder.model')
    # vit_autoencoder.load_state_dict(state_dict)
    state_dict = torch.load('./models/checkpoints/vitt/model_1698322627_vitt_9.model')
    model.load_state_dict(state_dict)
    return model