import numpy as np
import torch
from torch import nn



class MaskFeatureGenerator(nn.Module):
    def __init__(self, in_channels, hidden_dim, mask_dim):
        super(MaskFeatureGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, mask_dim, kernel_size=1) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
