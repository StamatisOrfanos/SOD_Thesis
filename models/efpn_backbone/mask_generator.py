import numpy as np
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F

# class MaskHead(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(MaskHead, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
#         self.mask = nn.Conv2d(256, num_classes, kernel_size=1)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.deconv(x))
#         x = torch.sigmoid(self.mask(x))
#         return x

class MaskGenerationNet(nn.Module):
    def __init__(self, in_channels):
        super(MaskGenerationNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        mask = self.sigmoid(x)
        return mask