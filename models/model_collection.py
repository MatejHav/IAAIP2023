from models.backbone.backbone import Backbone
from models.positional_encoder.positional_encoder import PositionalEncoder
from models.transformer.basic_transformer import BasicTransformer
from models.basic_lane_detector import BasicLaneDetector


def get_basic_model(device):
    backbone = Backbone('resnet18', device)
    pe = PositionalEncoder((7, 7), 0.2, 512, device)
    transformer = BasicTransformer(d_model=49, nhead=7, device=device)
    model = BasicLaneDetector(backbone, pe, transformer, device)
    return model
