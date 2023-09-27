from torch import nn
import torch
import numpy as np
from backbone.backbone import Backbone
from positional_encoder.positional_encoder import PositionalEncoding
from transformer.basic_transformer import BasicTransformer


class BasicLaneDetector(nn.Module):

    def __init__(self, backbone: Backbone, pe: PositionalEncoding, transformer: BasicTransformer):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the entire model.

        :param x: TODO should this be the entire video or a batch of frame from 1 video
        or batch of frames from different videos
        :return: Outputs for the input frames
        """
        pass


if __name__ == "__main__":
    backbone = Backbone("resnet50")
    pe = PositionalEncoding((8, 8), 0.2, 8*7*7)
    # TODO d_model should be tuple
    transformer = BasicTransformer(8*8)
    lane_detector = BasicLaneDetector(backbone, pe, transformer)
    total = 0
    for param in lane_detector.parameters():
        total += np.prod(param.data.shape)
    print(f"Total parameters registered: {total}")