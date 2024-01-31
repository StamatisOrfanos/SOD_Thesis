import torch
import torch.nn as nn


# Define the PixelDecoder class
class PixelDecoder(nn.Module):
    def __init__(self, input_channels, output_channels=256):
        super(PixelDecoder, self).__init__()
        # The input_channels should match the output feature size of the backbone
        
        # Using a series of transposed convolutions to upscale the feature maps
        self.upscaling_layers = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(output_channels, output_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(output_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(output_channels // 2, output_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(output_channels // 4),
            nn.ReLU(inplace=True),
        )
        
        # Final convolution to get to the desired number of channels
        self.final_conv = nn.Conv2d(output_channels // 4, output_channels // 4, kernel_size=1)

    def forward(self, x):
        x = self.upscaling_layers(x)
        x = self.final_conv(x)
        return x