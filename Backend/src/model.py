import torch
import torch.nn as nn
from torchvision import models

class RetinalNet(nn.Module):
    def __init__(self, num_classes=5):  # Change number based on your dataset
        super(RetinalNet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x))
