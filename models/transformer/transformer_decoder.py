import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        """
        Initialization

        Args:
            num_layers (int): Number of decoder layers.
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multihead attention models.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(TransformerDecoderWrapper, self).__init__()

        # Create the TransformerDecoderLayer
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)

        # Create the TransformerDecoder
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)

        prev_output = 0

    def forward(self, input_data: Tensor, last_prev_output: Tensor):
        """
        Forward pass.

        Args:
            input_data (torch.Tensor): The input of shape (seq_len, batch_size, d_model), positionally encoded.
            last_prev_output (torch.Tensor): The sequence from the last layer of the encoder (required).

        Returns:
            torch.Tensor: The decoded output of shape (seq_len, batch_size, d_model).
        """

        print('last prev output', last_prev_output)
        output = self.transformer_decoder(input_data, last_prev_output)
        return output