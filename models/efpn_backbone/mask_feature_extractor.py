import torch
import torch.nn as nn

class MaskFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(MaskFeatureExtractor, self).__init__()
        self.mask_features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_map):
        return self.mask_features(feature_map)

