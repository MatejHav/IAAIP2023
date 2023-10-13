from torch import nn
import torch
import gc

from models.positional_encoder.positional_encoder import PositionalEncoder
from models.transformer.basic_transformer import BasicTransformer


class MaskPredictor(nn.Module):

    def __init__(self, pe: PositionalEncoder, transformer: nn.Transformer, device):
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
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target_masks=None):
        # Reshaping and correction using shape_corrector
        x = x.flatten(start_dim=1)
        x = self.shape_corrector(x)
        x = x.view(x.size(0), 16*40, 20, 20)
        x = self.pe(x)
        seq_features = x.flatten(start_dim=2)

        seq_masks = target_masks.view(target_masks.size(0), 640, 400)  # Reshaping to match d_model
        outputs = self.transformer(seq_features, seq_masks)

        # Reshape back to your desired mask format
        masks = outputs.view(x.size(0), 320, 800)  # Reshape to get the mask format

        return self.sigmoid(masks)