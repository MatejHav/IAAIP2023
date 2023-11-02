from typing import Union, Callable, Optional

import numpy as np
import torch
import math
import warnings

warnings.filterwarnings("ignore")

from torch import nn
from torchvision.models import get_model
from torchvision.transforms import Resize
from models.models_mae import *
from main import get_dataloader
from models.positional_encoder.pe import PositionalEncoding
from torch.nn import functional as F


class ViTT(nn.Module):

    def __init__(self, d_model, out_dim, nhead, device, dropout=0.1, num_decoder_layers=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = out_dim
        self.vit = mae_vit_base_patch16()
        state_dict = torch.load('./models/checkpoints/mae_pretrain_vit_base.pth')['model']
        for key in state_dict:
            self.vit.state_dict()[key] = state_dict[key]
            # self.vit.state_dict()[key].requires_grad = False
        self.pe = PositionalEncoding(d_model=d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, dim_feedforward=d_model, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, None)
        self.down_sampler = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=2),
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=2),
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)
        )
        self.up_sampler = nn.Sequential(
            nn.UpsamplingBilinear2d(size=out_dim)
        )
        self.resize = Resize((48, 48))
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x: torch.Tensor, targets: torch.Tensor):
        # MAE outputs the reconstructed image
        memory = self.vit(x)
        memory = self.down_sampler(memory).mean(dim=1)
        B, H, W = memory.shape
        memory = memory.view(B, 1, H * W)
        memory = self.pe(memory).view(B, H * W)
        # Initial target is unknown
        if self.training:
            target = torch.zeros(1, H * W, device=x.device)
            targets = self.resize(targets).view(B, H * W)
            # cv2.imshow('img', memory[0].view(H, W).detach().cpu().numpy())
            # cv2.waitKey(0)
            targets = torch.concat((target, targets), dim=0)[:-1]
            # Pass segments through the decoder
            memory_mask = torch.Tensor([[1 if j <= i else 0 for j in range(B)] for i in range(B)]).to(x.device)
            result = []
            for i in range(B):
                target = self.decoder(targets[i].view(1, H * W), memory,
                                      memory_mask=memory_mask[i].view(1, B)).view(1, H * W)
                result.append(target.view(1, H, W))
            result = torch.stack(result)
        else:
            target = torch.zeros(1, H * W, device=x.device)
            # Pass segments through the decoder
            memory_mask = torch.Tensor([[1 if j <= i else 0 for j in range(B)] for i in range(B)]).to(x.device)
            result = []
            for i in range(B):
                target = self.decoder(target, memory,
                                      memory_mask=memory_mask[i].view(1, B)).view(1, H * W)
                result.append(target.view(1, H, W))
            result = torch.stack(result)
        result = self.up_sampler(result).view(B, *self.out)
        return self.sigmoid(result)

class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        x = tgt
        x = x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
        x = x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
        x = x + self._ff_block(x)
        return x

        # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor],
                  is_causal: bool = False) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

        # multihead attention block
    def _mha_block(self, x: torch.Tensor, mem: torch.Tensor,
                   attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor],
                   is_causal: bool = False) -> torch.Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

        # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

if __name__ == '__main__':
    import os

    os.chdir("../")
    dataloader = get_dataloader('train', batch_size=8, subset=10)
    vitt = ViTT(d_model=768, out_dim=(576, 576), nhead=16, device=None)
    for batch, targets, masks, idx in dataloader:
        predictions = vitt(batch)
