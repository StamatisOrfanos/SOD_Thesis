import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskFeatureExtractor(nn.Module):
    def __init__(self, conv_dim, mask_dim, number_feature_levels=5):
        """
        Parameters:
            - conv_dim (int): Input channel dimension of the feature map.
            - mask_dim (int): Dimensions of the mask features we want to produce.
            - number_feature_levels (int, optional): Number of feature maps we are going to be using. Defaults to 5.
        """
        super(MaskFeatureExtractor, self).__init__()
        self.mask_dim = mask_dim
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.mask_features.weight)
        self.num_feature_levels = number_feature_levels

    def forward(self, feature_maps):
        mask_features = self.mask_features(feature_maps[0])
        multi_scale_features = feature_maps[:self.num_feature_levels]
        return mask_features, multi_scale_features
