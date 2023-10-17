import torch
import torch.nn as nn
import torchvision.models.vision_transformer as vits

class ViTAutoencoder(nn.Module):
    def __init__(self, image_size=576, hidden_dim=200):
        super(ViTAutoencoder, self).__init__()

        # Encoder: ViT without the head
        self.vit = vits.VisionTransformer(
            image_size=image_size,
            patch_size=16,
            num_layers=2,
            num_heads=2,
            hidden_dim=hidden_dim,
            mlp_dim=64,
            dropout=0.1,
            attention_dropout=0.1,
            num_classes=0
        )
        self.vit.heads = nn.Identity()  # Removing the head
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), # out_dim is * 2 but could be other values
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, image_size * image_size * 3),  # Assuming 3 channels (RGB)
            nn.Sigmoid()  # Ensuring output values are between [0, 1]
        )

        # check if model is in training mode

        if self.eval:
            self.vit = torch.load('./models/checkpoints/mask/encoder.model', map_location=torch.device('cpu'))

            print(f"LOADING MODEL FROM CHECKPOINT IN TEST MODE")

            # freeze weights
            for param in self.vit.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        # Encoding
        z = self.vit(x) 
        
        # Decoding
        x_recon = self.decoder(z)
        x_recon = x_recon.view(x.size(0), 3, x.size(2), x.size(3))  # Reshape back to [B, C, H, W]
        
        return x_recon
