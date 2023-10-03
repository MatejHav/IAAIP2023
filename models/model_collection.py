from models.backbone.backbone import Backbone
from models.positional_encoder.positional_encoder import PositionalEncoder
from models.transformer.basic_transformer import BasicTransformer
from models.basic_lane_detector import BasicLaneDetector
import numpy as np


def get_basic_model(device):
    backbone = Backbone('resnet18')
    pe = PositionalEncoder((25, 10), 0.2, 512, device)
    transformer = BasicTransformer(d_model=25 * 10, nhead=10)
    model = BasicLaneDetector(pe, transformer, device)
    return backbone, model
