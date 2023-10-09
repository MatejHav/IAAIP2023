from torch import nn
import torch
import gc

from models.positional_encoder.positional_encoder import PositionalEncoder
from models.transformer.basic_transformer import BasicTransformer


class MaskPredictor(nn.Module):

    def __init__(self, pe: PositionalEncoder, transformer: torch.nn.Transformer, device):
        """
        Detects lanes in input images using a basic transformer architecture.

        :param backbone: Backbone that splits one frame into segments.
        :param pe: Positional Encoder that is applied to encode position into the segments
        :param transformer: Transformer used to detect lanes in segments
        """
        super().__init__()
        self.pe = pe
        self.transformer = transformer
        self.shape_corrector = nn.Sequential(nn.Linear(512*250, 512),
                                             nn.Linear(512, 320*800))
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the entire model.

        :param batch_of_segments: Input batch. Size is (batch_size, num_segments, x, y)
        :return: Outputs for the input frames
        """
        # print(f'Before shape correction {x.std(dim=0).mean()}')
        x = x.flatten(start_dim=1)
        x = self.shape_corrector(x)
        x = x.view(x.shape[0], 16*40, 20, 20)
        # print(f'After shape correction {x.std(dim=0).mean()}')
        positionally_encoded_segments = self.pe(x)
        # Flatten everything after 2nd dim
        positionally_encoded_segments = positionally_encoded_segments.flatten(start_dim=2)
        # Wanted output
        target = torch.randn(x.shape[0], 16*40, x.shape[-2] * x.shape[-1], dtype=torch.float32)
        target = target.to(self.device)
        target = self.transformer(positionally_encoded_segments, target)
        target = target.view(x.shape[0], 320, 800)
        # print(f'Result {x.std(dim=0).mean()}')
        return self.sigmoid(target)