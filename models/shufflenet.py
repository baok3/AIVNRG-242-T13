import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

class CustomShuffleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomShuffleNet, self).__init__()
        # Load pre-trained ShuffleNetV2 with x1.0 complexity
        self.shufflenet = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        
        # Get the number of features from the final classifier
        num_features = self.shufflenet.fc.in_features
        
        # Replace the final classifier with a binary classification head
        # We use a single output neuron with sigmoid activation for binary classification
        self.shufflenet.fc = nn.Linear(num_features, 1)
        
    def forward(self, x):
        return self.shufflenet(x)