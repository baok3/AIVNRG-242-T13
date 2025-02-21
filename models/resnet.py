import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()
        # Load pre-trained ResNet50 with latest weights
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify the final fully connected layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)
        
    def forward(self, x):
        return self.resnet(x)
