from torch import nn
import torch
from torchvision.models import get_model


class Backbone(nn.Module):

    def __init__(self, model_name: str, device):
        """
        The backbone segments an input frame into smaller segments, or set of feature tensors, that should represent
        the frame in latent space. The backbone uses a pretrained model defined as input.
        :param model_name: The name of the pretrained model which can be found in torch hub under pytorch/vision
        """
        super().__init__()
        self.model = get_model(model_name)
        self.model.to(device)

        # Remove the fully connected layer (classifier) at the end
        self.features = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the input through the pretrained model defined at initialization.

        :param x: Frame of the video loaded by the dataloader
        :return: Set of feature tensors
        """
        return self.model(x)
