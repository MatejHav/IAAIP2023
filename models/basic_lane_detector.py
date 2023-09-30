from torch import nn
import torch

from models.positional_encoder.positional_encoder import PositionalEncoder
from models.transformer.basic_transformer import BasicTransformer


class BasicLaneDetector(nn.Module):

    def __init__(self, backbone: nn.Module, pe: PositionalEncoder, transformer: BasicTransformer, device):
        """
        Detects lanes in input images using a basic transformer architecture.

        :param backbone: Backbone that splits one frame into segments.
        :param pe: Positional Encoder that is applied to encode position into the segments
        :param transformer: Transformer used to detect lanes in segments
        """
        super().__init__()
        self.backbone = backbone
        self.pe = pe
        self.transformer = transformer
        self.shape_corrector = nn.Sequential(nn.Linear(512*49, 15_000, device=device), nn.Sigmoid(), nn.Linear(15_000, 64*160*3, device=device))
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the entire model.

        :param x: Input batch. Size is (30, width, height)
        :return: Outputs for the input frames
        """

        # Turn all the frames into segments. Shape(batch_size, segment_number, segment_x, segment_y)
        batch_of_segments = self.backbone(x)

        batch_of_segments = torch.randn(30, 512, 7, 7)
        batch_of_segments = batch_of_segments.to(self.device)

        positionally_encoded_segments = self.pe.forward(batch_of_segments)
        # Flatten everything after 2nd dim
        positionally_encoded_segments = torch.flatten(positionally_encoded_segments, start_dim=2)
        # Wanted output
        target = torch.randn(30, 512, 49)
        target = target.to(self.device)

        target = self.transformer(positionally_encoded_segments, target)
        target = torch.flatten(target, start_dim=1)
        target = self.shape_corrector(target)
        target = torch.reshape(target, (target.size(0), 64, 160, 3))
        return target
