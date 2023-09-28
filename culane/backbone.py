import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

class ResNet18Backbone(nn.Module):
    def __init__(self, d_model: int = 512):
        super(ResNet18Backbone, self).__init__()
        
        # Load the pre-trained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Remove the fully connected layer (classifier) at the end
        self.features = nn.Sequential(*list(self.resnet18.children())[:-2])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the ResNet-18 backbone
        return self.features(x)
    
class ResNet34Backbone(nn.Module):
    def __init__(self):
        super(ResNet34Backbone, self).__init__()
        
        # Load the pre-trained ResNet-34 model
        self.resnet34 = models.resnet34(pretrained=True)
        
        # Remove the fully connected layer (classifier) at the end
        self.features = nn.Sequential(*list(self.resnet34.children())[:-2])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the ResNet-34 backbone
        return self.features(x)
    
class ResNet50Backbone(nn.Module):
    def __init__(self):
        super(ResNet50Backbone, self).__init__()
        
        # Load the pre-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Remove the fully connected layer (classifier) at the end
        self.features = nn.Sequential(*list(self.resnet50.children())[:-2])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the ResNet-50 backbone
        return self.features(x)

if __name__ == "__main__":
    # Example usage:
    # Create an instance of the ResNet-18 backbone
    backbone = ResNet18Backbone()
    
    # Load an example input image
    input_image = torch.randn(32, 3, 224, 224)  # Batch size of 1, RGB image with size 224x224
    
    # Forward pass through the backbone to extract features
    features = backbone(input_image)
    
    # Check the shape of the extracted features
    print("Shape of extracted features:", features.shape)