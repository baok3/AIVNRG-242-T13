from utils import *
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class CustomMobileNet(nn.Module):
    def __init__(self, num_classes=2):
        """
        Initialize a MobileNetV2 model customized for binary classification.
        
        Args:
            num_classes (int): Number of output classes (default: 2 for binary classification)
        """
        super(CustomMobileNet, self).__init__()
        # Load pre-trained MobileNetV2
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        
        # Modify the classifier for binary classification
        num_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(num_features, 1)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Model output logits
        """
        return self.mobilenet(x)
