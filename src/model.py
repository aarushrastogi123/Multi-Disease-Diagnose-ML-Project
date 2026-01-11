# src/model.py
import torch.nn as nn
from torchvision import models

class RetinalNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super(RetinalNet, self).__init__()
        # use torchvision resnet50; no pretrained to avoid download if offline
        base = models.resnet50(weights=None) if not pretrained else models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = base.fc.in_features  # 2048
        base.fc = nn.Identity()  # we'll define our own head
        self.backbone = base
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out