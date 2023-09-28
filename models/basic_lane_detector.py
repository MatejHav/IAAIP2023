from torch import nn
import torch
import numpy as np

import culane.backbone
from backbone.backbone import Backbone
from positional_encoder.positional_encoder import PositionalEncoding
from transformer.basic_transformer import BasicTransformer
from culane import backbone


class BasicLaneDetector(nn.Module):

    def __init__(self, backbone: nn.Module, pe: PositionalEncoding, transformer: BasicTransformer):
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
        or batch of frames from different videos. Lastly agreed size: (32, width, height)
        :return: Outputs for the input frames
        """

        resnet18_backbone = culane.backbone.ResNet18Backbone()

        batch_of_segments = resnet18_backbone.forward(x)

        batch_of_segments = torch.randn(32, 512, 8, 8)  #just for testing puposes

        positionally_encoded_segments = pe.forward(batch_of_segments)

        basic_transformer = BasicTransformer(d_model=64)

        reshaped_segments = positionally_encoded_segments.view(32, 512, -1)
        reshaped_segments = reshaped_segments.permute(1, 0, 2)
        print(reshaped_segments.shape)
        target = torch.randn(512, 32, 64)

        decoder_output = basic_transformer.forward(reshaped_segments, target)

        return decoder_output


if __name__ == "__main__":
    # backbone = Backbone("resnet50")
    backbone = backbone.ResNet18Backbone
    pe = PositionalEncoding((8,8), 0.2, 512) # number of segments when using ResNet18 is 512 per image and dimensions are actually 7x7 (why is the tuple like this (8,)?
    # TODO d_model should be tuple
    transformer = BasicTransformer(8*8)
    lane_detector = BasicLaneDetector(backbone, pe, transformer)
    total = 0
    print(lane_detector.forward(torch.randn(32, 3, 224, 224)).shape)
    # for param in lane_detector.parameters():
    #     total += np.prod(param.data.shape)
    # print(f"Total parameters registered: {total}")