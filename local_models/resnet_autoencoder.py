import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision.models.resnet as resnet

class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super(ResNetAutoencoder, self).__init__()

        # Encoder, use ResNet18
        # NOTE: is this maybe wrong/bad to initialize with default weights? 
        # should maybe initialize with random? also does this train? 
        self.encoder = nn.Sequential(*list(resnet.resnet34(weights='ResNet34_Weights.DEFAULT').children())[:-1])

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # size: 2x2
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # size: 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # size: 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),   # size: 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),   # size: 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # size: 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),   # size: 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # size: 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    # size: 512x512
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=(1, 2), padding=1), # size: 512x768
            nn.Sigmoid()  # To ensure the output values are in [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)



        return x