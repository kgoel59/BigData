import torch
import torch.nn as nn
import torch.nn.functional as F


class NN2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self._forward_block(x, self.fc1, self.bn1)
        x = self._forward_block(x, self.fc2, self.bn2)
        x = self._forward_block(x, self.fc3, self.bn3)
        x = self.fc4(x)
        return x

    def _forward_block(self, x, fc_layer, bn_layer):
        """Applies a linear layer followed by batch normalization, ReLU activation, and dropout."""
        x = fc_layer(x)
        x = bn_layer(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

