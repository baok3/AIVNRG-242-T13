import torch
import torch.nn as nn
from transformers import AutoModel


class PretrainedModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(PretrainedModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        x = self.backbone(x).last_hidden_state[:, 0, :]
        x = self.fc(x)
        return x
