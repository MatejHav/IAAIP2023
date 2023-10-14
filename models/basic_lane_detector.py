from torch import nn
import torch
import gc

from models.positional_encoder.positional_encoder import PositionalEncoder
from models.transformer.basic_transformer import BasicTransformer


class BasicLaneDetector(nn.Module):

    def __init__(self, pe: PositionalEncoder, transformer: BasicTransformer, device):
        """
        Detects lanes in input images using a basic transformer architecture.

        :param backbone: Backbone that splits one frame into segments.
        :param pe: Positional Encoder that is applied to encode position into the segments
        :param transformer: Transformer used to detect lanes in segments
        """
        super().__init__()
        self.pe = pe
        self.transformer = transformer
        self.shape_corrector = nn.Sequential(nn.Linear(512*250, 2048), nn.Sigmoid(), nn.Linear(2048, 64*160*3))
        self.device = device

    def forward(self, batch_of_segments: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the entire model.

        :param batch_of_segments: Input batch. Size is (30, width, height)
        :return: Outputs for the input frames
        """

        positionally_encoded_segments = self.pe(batch_of_segments)
        # Flatten everything after 2nd dim
        positionally_encoded_segments = torch.flatten(positionally_encoded_segments, start_dim=2)
        # Wanted output
        target = torch.randn(batch_of_segments.shape[0], 512, batch_of_segments.shape[-2] * batch_of_segments.shape[-1], dtype=torch.float32)
        target = target.to(self.device)

        target = self.transformer(positionally_encoded_segments, target)
        target = torch.flatten(target, start_dim=1)
        target = self.shape_corrector(target)
        target = torch.reshape(target, (target.size(0), 64, 160, 3))
        return target
