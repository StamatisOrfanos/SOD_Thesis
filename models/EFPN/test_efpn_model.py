import torch
from efficientnet_pytorch import EfficientNet
from models.EFPN.efpn_model import EFPN


def get_efficient_net_shapes():
    # Create a test input tensor of size [batch_size, channels, height, width] and load the pre-trained EfficientNet-B7 model
    input_tensor = torch.rand(1, 3, 600, 600)
    model = EfficientNet.from_pretrained('efficientnet-b7')

    # Put the model in eval mode and pass the input through it
    model.eval()
    with torch.no_grad():
        features = model.extract_endpoints(input_tensor)

    # Iterate over the features and print their shapes
    for key, feature in features.items():
        print(f"{key}: {feature.shape}")



def efficient_net_test():
    model = EFPN()
    model.eval()  # Set the model to evaluation mode
    test_input = torch.randn(1, 3, 600, 600)  # Random tensor simulating an image


    with torch.no_grad():  # Disable gradient computation for testing
        output = model(test_input)
    for i, feature_map in enumerate(output):
        print(f"Feature map {i}: {feature_map.shape}")



