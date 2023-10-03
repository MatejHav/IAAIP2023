import torch.nn as nn
import numpy as np
import torch
import math
import typing


class PositionalEncoder(nn.Module):

    def __init__(self, dim: tuple, dropout: float = 0.1, max_len: int = 5000, device=None):
        """
        Positional Encoding model adds the position of individual segments into the data.
        This helps the transformer or any model to add position into consideration when learning.
        :param dim: Dimensions of one segment (so without the batch size or number of segments)
        :param dropout: Probability of dropout.
        :param max_len: Maximum amount of segments in total (or maximum positions).
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        total_dim = np.prod(dim)
        div_term = torch.exp(torch.arange(0, total_dim, 2) * (-math.log(10000.0) / total_dim))
        odd_div_term = div_term
        if total_dim % 2 == 1:
            odd_div_term = torch.exp(torch.arange(0, total_dim - 1, 2) * (-math.log(10000.0) / total_dim))
        pe = torch.zeros(max_len, 1, total_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * odd_div_term)
        pe = torch.reshape(pe, (max_len, *dim))
        pe = pe.to(device)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that add the position into the tensor.
        For one batch the positions are encoded, if a new batch comes in with the exact same data,
        the same positions get encoded.

        :param x: Input tensor that will be fed into the transformer. Shape (N, <max_len, dim)
        :return: The same tensor but values are changed depending on their position
        """
        for n in range(x.shape[0]):
            x[n] += self.pe[:x.shape[1]]
        return self.dropout(x)
