import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        """
        Initialize a VGG16 model customized for binary classification.
        
        Args:
            num_classes (int): Number of output classes (default: 2 for binary classification)
        """
        super(VGG16, self).__init__()
        # Load pre-trained VGG16
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        
        # Modify the classifier for binary classification
        num_features = self.vgg.classifier[-1].in_features
        self.vgg.classifier[-1] = nn.Linear(num_features, 1)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Model output logits
        """
        return self.vgg(x)
