import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from PIL import Image 
from torchvision import transforms
from models.extended_mask2former_model import ExtendedMask2Former

# ------------------------------------------------------------------------------------------------------------------------------------
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


# Step 2: Initialise the Mask2Former Model 
model = ExtendedMask2Former(num_classes=100, hidden_dim=256, num_queries=100, nheads=8, dim_feedforward=2048, dec_layers=1, mask_dim=256)

# Step 3: Count the number of parameters
total_params = sum(param.numel() for param in model.parameters())
print("The total number of Extended Mask2Former parameters are: ", total_params)
model.eval()  # Set the model to evaluation mode

# Step 4: Pass the image through the model
with torch.no_grad():
    output = model(image)

# Step 5: Visualize the feature maps
print("\nThe predicted classes are the following:\n")
print(output["pred_logits"])
    
# Step 6: Visualize the mask of the model
print("\nThe masks with shape: {} created are the following:\n".format(output["pred_masks"].shape))
print(output["pred_masks"])
    
# Step 7: Visualize the bounding box of the model
print("\nThe bounding box with shape: {} created is the following:\n".format(output["bounding_box"].shape))
print(output["bounding_box"])

# Step 8: Visualize the class scores
print("\nThe class scores with shape: {} created are the following:\n".format(output["class_scores"].shape))
print(output["class_scores"])

# # Step 9: Get the summary of the EFPN model
# print("\n\nThe summary of the EFPN model is the following: \n", model) 
# ------------------------------------------------------------------------------------------------------------------------------------