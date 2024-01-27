import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet

class EFPN(nn.Module):
    def __init__(self):
        super(EFPN, self).__init__()
        # Load EfficientNet with pre-trained weights
        self.backbone = EfficientNet.from_pretrained('efficientnet-b7')

# Define FPN convolution layers to match channel dimensions if necessary
        self.conv_c2_prime = nn.Conv2d(32, 256, kernel_size=1)  
        self.conv_c2       = nn.Conv2d(48, 256, kernel_size=1)  
        self.conv_c3       = nn.Conv2d(80, 256, kernel_size=1)  
        self.conv_c4       = nn.Conv2d(224, 256, kernel_size=1) 
        self.conv_c5       = nn.Conv2d(640, 256, kernel_size=1) 

        # Define FPN lateral layers
        self.lateral_p5       = nn.Conv2d(640, 256, kernel_size=1)
        self.lateral_p4       = nn.Conv2d(224, 256, kernel_size=1)
        self.lateral_p3       = nn.Conv2d(80, 256, kernel_size=1) 
        self.lateral_p2       = nn.Conv2d(48, 256, kernel_size=1) 
        self.lateral_p2_prime = nn.Conv2d(32, 256, kernel_size=1) 

        # Define FPN top-down pathway
        self.top_down_p5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_p2 = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, x):
        # Pass input through EfficientNet backbone
        # Identify the layers or feature maps in EfficientNet that correspond to C2, C3, C4, C5
        c2_prime, c2, c3, c4, c5 = self.backbone_features(x)

        # Process feature maps through FPN
        p5 = self.lateral_p5(c5)
        p4 = self.lateral_p4(c4) + self.top_down_p5(p5)
        p3 = self.lateral_p3(c3) + self.top_down_p4(p4)
        p2 = self.lateral_p2(c2) + self.top_down_p3(p3)

        # FTT operations for P3' and P2' here
        p3_prime = FTTModule(p2, p3)
        p2_prime = self.top_down_p2(p3_prime) + self.conv_c2_prime(c2_prime)

        # Return the feature pyramids
        return p2_prime, p2, p3, p4, p5


    def backbone_features(self, x):
        # Get feature maps from the EfficientNet backbone
        c2_prime = self.backbone.extract_endpoints(x)['reduction_1']  # For C2'
        c2 = self.backbone.extract_endpoints(x)['reduction_2']        # For C2
        c3 = self.backbone.extract_endpoints(x)['reduction_3']        # For C3
        c4 = self.backbone.extract_endpoints(x)['reduction_4']        # For C4
        c5 = self.backbone.extract_endpoints(x)['reduction_5']        # For C5
        return c2_prime, c2, c3, c4, c5


# Define the FTT module of the Extended Feature Pyramid Network
class FTTModule(nn.Module):
    def __init__(self):
        super(FTTModule, self).__init__()
        self.content_extractor = ContentExtractor()
        self.texture_extractor = TextureExtractor()
        self.subpixel_conv = SubPixelConv()
    
    def forward(self, p2, p3):
        # Apply the content extractor to P3 and upsample the content features
        content_features = self.content_extractor(p3)
        upsampled_content = self.subpixel_conv(content_features)

        # Apply the texture extractor to P2 and Element-wise sum of the upsampled content and texture features
        texture_features = self.texture_extractor(p2)
        combined_features = upsampled_content + texture_features
    
        return combined_features


# Define the ContentExtractor to be used by the p3 feature map
class ContentExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


# Define the TextureExtractor to be used by the p2 feature map
class TextureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


# Define the ContentExtractor to be used by the p3 feature map
class SubPixelConv(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.upscale_factor = upscale_factor
    
    def forward(self, x):
        x = self.conv(x)
        x = F.pixel_shuffle(x, self.upscale_factor)
        return x
    



model = EFPN()
model.eval()  # Set the model to evaluation mode
test_input = torch.randn(1, 3, 600, 600)  # Random tensor simulating an image


with torch.no_grad():  # Disable gradient computation for testing
    output = model(test_input)
for i, feature_map in enumerate(output):
    print(f"Feature map {i}: {feature_map.shape}")
