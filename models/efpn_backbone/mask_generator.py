import numpy as np
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerationNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(MaskGenerationNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, out_channels, kernel_size=1)  # Output binary mask

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x3 = self.pool(F.relu(self.conv3(x2)))
        x4 = self.up(F.relu(self.conv4(x3)))
        x5 = self.up(F.relu(self.conv5(x4)))
        mask = torch.sigmoid(self.conv6(x5))  # Binary mask output
        return mask

