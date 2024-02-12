import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Mask2Former.mask2former_model import Mask2Former

model = Mask2Former(3, 100, 128, 100, 3, 258, 12, 2)

total_params = sum(
	param.numel() for param in model.parameters()
)



print(total_params)