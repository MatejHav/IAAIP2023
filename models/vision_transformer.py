import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, feature_dim, output_shape):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(feature_dim, 256 * 20 * 50)  # Assuming feature_dim is 768 and the intermediate shape is [256, 20, 50]
        self.deconv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 20, 50)
        x = self.deconv_layers(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=3, img_size=(320, 800), patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(VisionTransformer, self).__init__()


        # Extract the height and width from img_size
        height, width = img_size

        assert height % patch_size == 0 and width % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.num_patches = (height // patch_size) * (width // patch_size)
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(in_channels * patch_size**2, dim)
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.transformer, num_layers=depth)

        self.to_cls_token = nn.Identity()

        self.decoder = Decoder(dim, img_size) # Add this line

    def forward(self, x):
        x = self.patch_image(x)
        cls_tokens = self.cls_token(x)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        # Use the decoder to reconstruct the image
        x = self.decoder(x[:, 0])
        
        return x

    def patch_image(self, x):
        batch_size, channels, height, width = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, channels * self.patch_size * self.patch_size)
        x = self.patch_to_embedding(x)
        return x

    def cls_token(self, x):
        return torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device)