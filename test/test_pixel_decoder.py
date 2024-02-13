import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torchvision import transforms
from PIL import Image
from models.Mask2Former.backbone_model import Backbone
from models.Mask2Former.pixel_decoder import PixelDecoder

# --------------------------------------------------------------------------------------------------------------------------------
# Step 1: Define the image transformation
transform = transforms.Compose([
    transforms.Resize(600),            
    transforms.ToTensor(),             
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

# Step 2: Load the image
input_image = Image.open("docs/fft.png").convert('RGB')

# Transform the image and add a batch dimension
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0) 


# Step 3: Initialize the backbone
backbone = Backbone()

# Step 4: Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device)
input_batch = input_batch.to(device)

# Step 5: Get the feature maps
with torch.no_grad(): feature_maps = backbone(input_batch)

# Print the shape of the feature maps
print("Feature maps shape:", feature_maps.shape)

# Step 6: You would use the PixelDecoder like this:
pixel_decoder = PixelDecoder(input_channels=2560)
backbone_feature_maps = feature_maps
all_feature_maps = pixel_decoder(backbone_feature_maps)


# Example usage:
pixel_decoder = PixelDecoder(input_channels=2560)
feature_maps_level1, feature_maps_level2, feature_maps_level3, feature_maps_level4 = pixel_decoder(feature_maps)


# Step 7: We are going to count the number of parameters for both the Backbone and Pixel Decoder
backbone_params = sum(param.numel() for param in backbone.parameters())
print("The total number of the Backbone parameters are: ", backbone_params)

pixel_decoder_params = sum(param.numel() for param in pixel_decoder.parameters())
print("The total number of Pixel Decoder parameters are: ", pixel_decoder_params)

print("The total parameters of the EfficientNet backbone and Pixel decoder are: ", backbone_params+pixel_decoder_params)
#  --------------------------------------------------------------------------------------------------------------------------------