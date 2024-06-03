import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BoundingBoxGenerator(nn.Module):
    """
    Parameters:
        - nn (torch.nn): Pytorch network
    """
    def __init__(self, in_channels, num_classes):
        super(BoundingBoxGenerator, self).__init__()
        self.num_classes            = num_classes
        self.conv1                  = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2                  = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.class_convolution      = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        self.regression_convolution = nn.Conv2d(256, 4, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        class_scores = self.class_convolution(x)
        bounding_box = self.regression_convolution(x)
        return bounding_box, class_scores
    
