import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # add current folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add project root

from src.model import RetinalNet
import torch

print("🔍 Creating model...")
model = RetinalNet(num_classes=5, pretrained=False)
print("✅ Model created successfully!")

x = torch.randn(1, 3, 224, 224)
y = model(x)
print("Output shape:", y.shape)