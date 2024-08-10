import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from models.efpn_backbone.efpn_model import EFPN
import torch
from models.efpn_backbone.bounding_box  import *
from models.efpn_backbone.anchors import *
from src.data_set_up import *
from torch.utils.data import DataLoader
from torchvision import transforms


# ------------------------------------------------------------------------------------------------------------------------------
# Step 1: Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((600, 600)),  # Resize to the input size expected by EfficientNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for EfficientNet
    ])
    return transform(image).unsqueeze(0)  # type: ignore # Add batch dimension

image = load_image("/Users/stamatiosorphanos/Documents/MCs_Thesis/SOD_Thesis/docs/Extended_Mask2Former/1.jpg")

# Step 2: Instantiate the model
model = EFPN(in_channels=256, hidden_dim=256,  num_classes=10, num_anchors=10)
model.eval()  # Set the model to evaluation mode


# Step 3: Count the number of parameters of the EFPN model
total_params = sum(param.numel() for param in model.parameters())
print("The total number of EFPN parameters are: ", total_params)


# Step 4: Pass the image through the model
with torch.no_grad():
    feature_maps, masks, bounding_box, class_scores = model(image)


# Step 5: Visualize the feature maps
print("\nThe feature maps created are the following:\n")
for fm in feature_maps:
    print(fm.shape)

    
# Step 6: Visualize the class scores of each feature map
print("\nThe class scores are the following:\n")
print(class_scores.size())

    
# Step 7: Visualize the masks of each feature map
print("\nThe masks of the feature maps created are the following:\n")
for mask in masks:
    print(mask.shape)

    
# Step 7: Visualize the bounding box
print("\nThe bounding box with shape: {} created is the following:\n".format(bounding_box.shape))


# Step 9: Get the summary of the EFPN model
print("\n\nThe summary of the EFPN model is the following: \n", model) 
# ------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------
# Create a mock Detection Loss to ensure that the EFPN branch works as expected
class DetectionLoss(nn.Module):
    def __init__(self, anchors, iou_threshold=0.5):
        super(DetectionLoss, self).__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)  # Preload anchors as a tensor
        self.iou_threshold = iou_threshold
        self.regression_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.classification_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_labels):
        device = pred_bboxes.device
        self.anchors = self.anchors.to(device)
        
        # Slice predicted bounding boxes and scores to match the number of anchors
        pred_bboxes = pred_bboxes[:, :self.anchors.size(0), :]
        pred_scores = pred_scores[:, :self.anchors.size(0), :]

        # Match ground truth boxes and generate regression targets
        matched_gt_boxes, anchor_max_idx = match_anchors_to_ground_truth_boxes(self.anchors, gt_bboxes, self.iou_threshold)
        regression_targets = encode_bounding_boxes(matched_gt_boxes, self.anchors).to(device)
        regression_targets = regression_targets.unsqueeze(0).repeat(pred_bboxes.size(0), 1, 1)

        # Compute regression loss
        regression_loss = self.regression_loss_fn(pred_bboxes, regression_targets)

        # Classification targets need to align with the number of predictions per class
        classification_targets = gt_labels[anchor_max_idx].to(device)
        classification_targets = classification_targets.unsqueeze(0).repeat(pred_scores.size(0), 1)

        # Flatten the scores and targets to calculate classification loss
        classification_loss = self.classification_loss_fn(
            pred_scores.reshape(-1, pred_scores.size(-1)),
            classification_targets.reshape(-1) 
        )

        total_loss = regression_loss + classification_loss
        
        return total_loss



# Get some dummy mean and standard deviation for the training, load the train data and transform to match the input of the model.
mean = [0.485, 0.456, 0.406]
standard_deviation = [0.229, 0.224, 0.225]

data_transform = {
    "train": transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=standard_deviation)
    ])
}

train_dataset = SOD_Data("data/uav_sod_data/train"  + "/images", "data/uav_sod_data/train" + "/annotations", data_transform["train"])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))



# Define model, loss function, and optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = EFPN(in_channels=3, hidden_dim=256, num_classes=20, num_anchors=9).to(device)
anchors = Anchors.generate_anchors([(38, 38), (19, 19), (10, 10), (5, 5)], scales=[32, 64, 128, 256], aspect_ratios=[0.5, 1, 2])
loss_fn = DetectionLoss(anchors=anchors).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Train for one epoch to ensure that the model can be trained as expected
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = torch.stack(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt_bboxes = targets[0]['boxes'].to(device)
        gt_labels = targets[0]['labels'].to(device)
        feature_maps, masks, pred_bboxes, pred_scores = model(images)
        loss = loss_fn(pred_bboxes, pred_scores, gt_bboxes, gt_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
# ------------------------------------------------------------------------------------------------------------------------------
