import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mask2former_detector.mask2former_model import Mask2Former


# ------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Initialise the Mask2Former Model 
model = Mask2Former(3, 100, 128, 100, 4, 258, 12, 2)

# Step 2: Count the number of parameters
total_params = sum(param.numel() for param in model.parameters())
print("The total number of Mask2Former parameters are: ", total_params)


# Step 3: Get the summary of the Mask2Former model
print("\n\nThe summary of the Mask2Former model is the following: \n", model)
# ------------------------------------------------------------------------------------------------------------------------------------