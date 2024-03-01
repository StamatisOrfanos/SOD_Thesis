import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from PIL import Image
from models.efpn_backbone.efpn_model import EFPN
from torchviz import make_dot


# ------------------------------------------------------------------------------------------------------------------------------
# Step 1: Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((600, 600)),  # Resize to the input size expected by EfficientNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for EfficientNet
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

image = load_image("/Users/stamatiosorphanos/Documents/MCs_Thesis/SOD_Thesis/docs/Extended_Mask2Former/1.jpg")

# Step 2: Instantiate the model
model = EFPN(in_channels=256, hidden_dim=256, mask_dim=100)
model.eval()  # Set the model to evaluation mode


# Step 3: Count the number of parameters of the EFPN model
total_params = sum(param.numel() for param in model.parameters())
print("The total number of EFPN parameters are: ", total_params)


# Step 4: Pass the image through the model
with torch.no_grad():
    feature_maps, mask = model(image, 256)


# Step 5: Visualize the feature maps
print("\nThe feature maps created are the following:\n")
for fm in feature_maps:
    print(fm.shape)

    
# Step 6: Visualize the masks of each feature map
print("\nThe feature map mask created is the following:\n")
print(mask.shape)
    

# Step 7: Get the summary of the EFPN model
print("\n\nThe summary of the EFPN model is the following: \n", model)
# ------------------------------------------------------------------------------------------------------------------------------