from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

class EFPN(nn.Module):
    """
        Extended Feature Pyramid Network (EFPN) based on EfficientNet-B7 as the backbone.
           -  The model creates and enhances feature maps using EfficientNet's deep feature extraction capabilities combined with a 
              Feature Pyramid Network (FPN) structure for multi-scale feature integration. 
           -  The model uses a Feature  Texture Transfer (FTT) module to enrich feature maps with both content and texture 
              details, aiming to improve performance on instance segmentation tasks.
    """
    def __init__(self, num_classes):
        super(EFPN, self).__init__()
        # Load EfficientNet with pre-trained weights
        self.backbone = EfficientNet.from_pretrained('efficientnet-b7')

        # Initialize FTTModule
        self.ftt_model = FTT()

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
        
        # Define the masks for each feature map
        self.segmentation_heads = nn.ModuleDict({
            'p2_prime': SegmentationHead(256, num_classes),
            'p2': SegmentationHead(256, num_classes),
            'p3': SegmentationHead(256, num_classes),
            'p4': SegmentationHead(256, num_classes),
            'p5': SegmentationHead(256, num_classes),
        })


    def forward(self, image):
        # Pass input through EfficientNet backbone
        # Identify the layers or feature maps in EfficientNet that correspond to C2, C3, C4, C5
        c2_prime, c2, c3, c4, c5 = self.backbone_features(image)

        # Process feature maps through FPN
        p5 = self.lateral_p5(c5)
        upsampled_p5 = F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        
        p4 = self.lateral_p4(c4) + upsampled_p5
        upsampled_p4 = F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        
        p3 = self.lateral_p3(c3) + upsampled_p4
        upsampled_p3 = F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        
        p2 = self.lateral_p2(c2) + upsampled_p3

        # FTT operations for P3' and P2' here
        p3_prime = self.ftt_model(p2, p3)

        # Process c2_prime through its convolution layer
        c2_prime_processed = self.conv_c2_prime(c2_prime)
        upsampled_p3_prime = self.top_down_p2(p3_prime)
        p2_prime = upsampled_p3_prime + c2_prime_processed
        
        
        # Create the masks per feature map
        feature_maps = [p2_prime, p2, p3, p4, p5]
        masks = []
        
        for name, feature_map in zip(self.segmentation_heads.keys(), feature_maps):
            mask = self.segmentation_heads[name](feature_map)
            masks.append(mask)


        # Return the feature map pyramid and the corresponding masks
        return feature_maps, masks


    def backbone_features(self, image):
        # Get feature maps from the EfficientNet backbone
        endpoints = self.backbone.extract_endpoints(image)
        c2_prime  = endpoints['reduction_1']
        c2        = endpoints['reduction_2']
        c3        = endpoints['reduction_3']
        c4        = endpoints['reduction_4']
        c5        = endpoints['reduction_5']
        return c2_prime, c2, c3, c4, c5


class FTT(nn.Module):
    # Define the FTT module of the Extended Feature Pyramid Network
    def __init__(self):
        super(FTT, self).__init__()
        self.content_extractor = ContentExtractor(256, 256, num_layers=3)
        self.texture_extractor = TextureExtractor(256, 256, num_layers=3)
        self.subpixel_conv = SubPixelConv(256, 256, upscale_factor=2)
    
    def forward(self, p2, p3):
        # Apply the content extractor to P3 and upsample the content features
        content_features = self.content_extractor(p3)
        upsampled_content = self.subpixel_conv(content_features)

        # Apply the texture extractor to P2 and Element-wise sum of the upsampled content and texture features
        texture_features = self.texture_extractor(p2)
        combined_features = upsampled_content + texture_features
    
        return combined_features


class ContentExtractor(nn.Module):
    # Define the ContentExtractor to be used by the p3 feature map
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


class TextureExtractor(nn.Module):
    # Define the TextureExtractor to be used by the p2 feature map
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


class SubPixelConv(nn.Module):
    # Define the ContentExtractor to be used by the p3 feature map
    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.upscale_factor = upscale_factor
    
    def forward(self, x):
        x = self.conv(x)
        x = F.pixel_shuffle(x, self.upscale_factor)
        return x

 
class SegmentationHead(nn.Module):
    # Define the SegmentationHead to create the masks for each feature map 
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        # Upsample to match the original image size, if necessary Note: You might need to adjust the scale_factor based on the feature map's resolution
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x