from torch import nn
import torch
from torchvision.models import get_model


class Backbone(nn.Module):

    def __init__(self, model_name: str):
        """
        The backbone segments an input frame into smaller segments, or set of feature tensors, that should represent
        the frame in latent space. The backbone uses a pretrained model defined as input.
        :param model_name: The name of the pretrained model which can be found in torch hub under pytorch/vision
        """
        super().__init__()
        # Input is (320, 800) so we want to split on every 10th index
        self.sizes = (10, 10)
        if model_name == 'grid':
            self.grid = True
            return

        # Remove the fully connected layer (classifier) at the end
        self.grid = False
        self.model = nn.Sequential(*list(get_model(model_name).children())[:-2])
        #
        # # Freeze training of the backbone
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the input through the pretrained model defined at initialization.

        :param x: Frame of the video loaded by the dataloader
        :return: Set of feature tensors
        """
        if self.grid:
            res = []
            for img in x:
                x_x_split = torch.stack(torch.split(img, self.sizes[0], dim=-1))
                x_y_split = torch.stack(torch.split(x_x_split, self.sizes[1], dim=-1))
                res.append(x_y_split.view(3, 32 * 80, 10, 10))
            return torch.stack(res)
        return self.model(x)


if __name__ == '__main__':
    image_batch = torch.Tensor(
        [[[[i * 800 + j for j in range(800)] for i in range(320)] for _ in range(3)] for b in range(30)])
    backbone = Backbone('grid')
    out = backbone(image_batch)
    print(out[0, 0, 0, :, :])
