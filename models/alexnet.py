import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        # Load pre-trained AlexNet
        self.alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)
        
        num_features = self.alexnet.classifier[-1].in_features
        self.alexnet.classifier[-1] = nn.Linear(num_features, 1)
        
        # Thêm loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, x):
        return self.alexnet(x)
        
