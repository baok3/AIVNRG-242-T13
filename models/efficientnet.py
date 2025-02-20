from utils import *
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        """
        Initialize an EfficientNet-B0 model customized for binary classification.
        
        Args:
            num_classes (int): Number of output classes (default: 2 for binary classification)
        """
        super(EfficientNet, self).__init__()
        # Load pre-trained EfficientNet-B0
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Modify the classifier for binary classification
        num_features = self.efficientnet.classifier[-1].in_features
        self.efficientnet.classifier[-1] = nn.Linear(num_features, 1)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Model output logits
        """
        return self.efficientnet(x)
