import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskFeatureExtractor(nn.Module):
    def __init__(self, conv_dim, mask_dim):
        """
        Parameters:
            - conv_dim (int): Input channel dimension of the feature map.
            - mask_dim (int): Dimensions of the mask features we want to produce.
        """
        super(MaskFeatureExtractor, self).__init__()
        self.mask_dim = mask_dim
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.mask_features.weight)

    def forward(self, feature_map):
        mask_features = self.mask_features(feature_map)
        return mask_features
