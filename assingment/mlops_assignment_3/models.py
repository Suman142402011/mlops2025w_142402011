import torch.nn as nn
from torchvision import models

RESNET_MAP = {
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152
}

def build_model(name, num_classes, pretrained=True):
    if name not in RESNET_MAP:
        raise ValueError(f"Unknown architecture: {name}")
    weights = "IMAGENET1K_V1" if pretrained else None
    model = RESNET_MAP[name](weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
