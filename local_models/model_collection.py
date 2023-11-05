import torch.nn
import torchvision.transforms

from local_models.vitt import ViTT_detr_mem
from local_models.vit_autoencoder import ViTAutoencoder

from local_models.models_mae import *

def get_vitt(device):
    model = ViTT_detr_mem(d_model=2304, out_dim=(224, 224), nhead=1, device=device)
    # state_dict = torch.load('./local_models/checkpoints/vitt/model_1698863414_vitt_45.model')
    # model.load_state_dict(state_dict)
    return model

def get_vit_autoencoder(device):
    vit_autoencoder = ViTAutoencoder()
    # state_dict = torch.load('./local_models/checkpoints/VitAutoEncoder.model')
    # vit_autoencoder.load_state_dict(state_dict)
    state_dict = torch.load('./local_models/checkpoints/vitt/model_1698322627_vitt_9.model')
    model = torch.nn.Sequential(vit_autoencoder, Mean())
    model.load_state_dict(state_dict)
    return model