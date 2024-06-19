from torch import nn
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, get_norm

class MaskFeatureExtractor(nn.Module):
    def __init__(self, conv_dim, mask_dim, num_feature_levels=5):
        """
        Parameters:
            - conv_dim (int): 
            - mask_dim (int): 
            - num_feature_levels (int, optional): _description_. Defaults to 5.
        """
        super(MaskFeatureExtractor, self).__init__()
        self.mask_dim = mask_dim
        self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)
        weight_init.c2_xavier_fill(self.mask_features)
        self.num_feature_levels = num_feature_levels

    def forward(self, feature_maps):
        # Create mask features from the highest resolution feature map
        mask_features = self.mask_features(feature_maps[0])
        # Select the top N feature levels
        multi_scale_features = feature_maps[:self.num_feature_levels]
        return mask_features, multi_scale_features
