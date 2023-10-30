import torch
import torch.nn as nn
import torchvision.models.vision_transformer as vits

import models.models_mae as mae

class MAEFeatureExtraactor(nn.Module):
    def __init__(self, image_size=576, hidden_dim=200):
        super(MAEFeatureExtraactor, self).__init__()

        self.vit = mae.mae_vit_base_patch16()
        state_dict = state_dict = torch.load('./models/checkpoints/mae_pretrain_vit_base.pth')['model']
        for key in state_dict:
            self.vit.state_dict()[key] = state_dict[key]
            self.vit.state_dict()[key].requires_grad = False
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), # out_dim is * 2 but could be other values
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, image_size * image_size * 3),  # Assuming 3 channels (RGB)
            nn.Sigmoid()  # Ensuring output values are between [0, 1]
        )
        
    def forward(self, x):
        # put it forward through MAE
        x = self.vit(x)
        
        return x