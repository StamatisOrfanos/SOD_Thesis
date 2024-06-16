import torch
from torch import nn
from torch.nn import functional as F

class MaskHandler(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        """
        Parameters:
            - in_channels (int): Number of channels 
            - hidden_dim (int): Number of hidden dimensions of the tensor we create
            - num_classes (int): Number of classes of the dataset
        """
        super(MaskHandler, self).__init__()
        self.mask_head = self._mask_head(in_channels, hidden_dim)
        self.mask_predictor = self._mask_predictor(hidden_dim, num_classes)

    def _mask_head(self, in_channels, hidden_dim):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _mask_predictor(self, in_channels, num_classes):
        return nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feature_map):
        mask_features = self.mask_head(feature_map)
        masks = self.mask_predictor(mask_features)
        return masks
