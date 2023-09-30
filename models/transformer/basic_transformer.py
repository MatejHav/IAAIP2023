import numpy as np

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Transformer

from models.transformer.transformer_decoder import TransformerDecoderWrapper
from models.transformer.transformer_encoder import TransformerEncoderWrapper


class BasicTransformer(Transformer):
    def __init__(self, d_model: int, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1, device=None):
        super().__init__(d_model, nhead, batch_first=True)
        encoder_wrapper = TransformerEncoderWrapper(num_encoder_layers, d_model, nhead, dim_feedforward, dropout)
        decoder_wrapper = TransformerDecoderWrapper(num_decoder_layers, d_model, nhead, dim_feedforward, dropout)
        encoder_wrapper.to(device)
        decoder_wrapper.to(device)
        self.encoder = encoder_wrapper.transformer_encoder
        self.decoder = decoder_wrapper.transformer_decoder
        self.device = device

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # Flatten the input. The input is shape(batch_size, segment_num, width, height)
        src = torch.flatten(src, start_dim=2)  # Reshape it into (batch_size, segment_num, total)
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            print(src.size(-1), tgt.size(-1), self.d_model)
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output
