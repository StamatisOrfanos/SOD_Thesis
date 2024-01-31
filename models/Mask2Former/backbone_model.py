import torch
import torch.nn as nn
from torchvision.models import resnet50

# Define the Backbone class
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # Load a pre-trained ResNet-50 model
        self.resnet = resnet50(pretrained=True)
        # Remove the fully connected layer and the pooling layer to get the feature maps
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        # Forward pass through the feature extractor
        features = self.feature_extractor(x)
        return features





# Given an input batch of images `input_images`, you would get the feature maps like this:
# feature_maps = backbone(input_images)

