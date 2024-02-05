import torch.nn as nn


class PixelDecoder(nn.Module):
    def __init__(self, input_channels, output_channels=256):
        super(PixelDecoder, self).__init__()
        # Define the upscaling layers to increase the resolution of the feature maps
        self.upscaling_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(input_channels // 2),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(input_channels // 2, input_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(input_channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(input_channels // 4, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        ])

    def forward(self, x):
        # Initialize a list to store the feature maps of each level
        feature_maps = [x]

        # Upscale the feature maps step by step
        for upscaling_layer in self.upscaling_layers:
            x = upscaling_layer(x)
            feature_maps.append(x)

        return feature_maps