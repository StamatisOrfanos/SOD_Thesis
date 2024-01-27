import torch
from efficientnet_pytorch import EfficientNet

# Create a test input tensor of size [batch_size, channels, height, width]
input_tensor = torch.rand(1, 3, 600, 600)

# Load the pre-trained EfficientNet-B7 model
model = EfficientNet.from_pretrained('efficientnet-b7')

# Put the model in eval mode and pass the input through it
model.eval()
with torch.no_grad():
    features = model.extract_endpoints(input_tensor)

# Iterate over the features and print their shapes
for key, feature in features.items():
    print(f"{key}: {feature.shape}")
