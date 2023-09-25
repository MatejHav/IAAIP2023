import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerEncoderWrapper(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        """
        Initialization

        Args:
            num_layers (int): Number of encoder layers.
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multihead attention models.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(TransformerEncoderWrapper, self).__init__()

        # Create the TransformerEncoderLayer
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

        # Create the TransformerEncoder
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src):
        """
        Forward pass.

        Args:
            src (torch.Tensor): The input of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: The encoded output of shape (seq_len, batch_size, d_model).
        """
        output = self.transformer_encoder(src)
        return output
