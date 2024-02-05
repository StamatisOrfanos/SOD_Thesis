import torch.nn as nn
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights


# Define the Backbone class
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # Load a pre-trained EfficientNet-B7 model and remove the fully connected and pooling layer to get the feature maps
        self.resnet = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        features = self.feature_extractor(x)
        return features