import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image 
from torchvision import transforms
from extended_mask2former_model import ExtendedMask2Former
from torchviz import make_dot




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

hidden_dim = 256
num_classes = 100
# Step 4: Get the summary of the Mask2Former model
# generate a model architecture visualization
y = model(image, hidden_dim)
prediction =  y["pred_logits"]
make_dot(prediction.mean(), params=dict(ExtendedMask2Former(num_classes).named_parameters()), show_attrs=True, show_saved=True).render("Extended_Mask2Former", format="png")

# ------------------------------------------------------------------------------------------------------------------------------------