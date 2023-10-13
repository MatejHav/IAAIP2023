import torch.nn

from models.backbone.backbone import Backbone
from models.positional_encoder.positional_encoder import PositionalEncoder
from models.transformer.basic_transformer import BasicTransformer
from models.basic_lane_detector import BasicLaneDetector
from models.mask_predictor import MaskPredictor
from models.vitt import ViTT


def get_basic_model(device):
    backbone = Backbone('resnet18')
    pe = PositionalEncoder((25, 10), 0.4, 512, device)
    transformer = BasicTransformer(d_model=25 * 10, nhead=10)
    model = BasicLaneDetector(pe, transformer, device)
    return backbone, model


def get_mask_model(device):
    backbone = Backbone('resnet34')
    pe = PositionalEncoder((20, 20), 0.2, 16 * 40, device)
    transformer = torch.nn.Transformer(d_model=20 * 20, nhead=10, dropout=0.1, num_encoder_layers=6,
                                       num_decoder_layers=6, dim_feedforward=256, batch_first=True)
    model = MaskPredictor(pe, transformer, device)
    return backbone, model


def get_vitt(device):
    vitt = ViTT(d_model=4*6, out_dim=(4, 6), nhead=4, device=device)
    vitt.to(device)
    return torch.nn.Sequential(), vitt
