from models.backbone.backbone import Backbone
from models.positional_encoder.positional_encoder import PositionalEncoder
from models.transformer.basic_transformer import BasicTransformer
from models.basic_lane_detector import BasicLaneDetector


def get_basic_model(device):
    backbone = Backbone('resnet18', device)
    backbone.to(device)
    pe = PositionalEncoder((25, 10), 0.2, 512, device)
    pe.to(device)
    transformer = BasicTransformer(d_model=250, nhead=10, device=device)
    transformer.to(device)
    model = BasicLaneDetector(backbone, pe, transformer, device)
    model.to(device)
    return model
