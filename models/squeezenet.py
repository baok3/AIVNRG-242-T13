import torch
import torch.nn as nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

class CustomSqueezeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomSqueezeNet, self).__init__()
        # Load pre-trained SqueezeNet 1.1 (more efficient version)
        self.squeezenet = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        
        # It uses a 1x1 convolution instead of fully connected layers
        self.squeezenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 1, kernel_size=1),  # Changed to 1 output channel
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        x = self.squeezenet(x)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, 1)