# Import libraries
import pandas as pd
import os, json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.data_set_up import SOD_Data
from models.extended_mask2former_model import ExtendedMask2Former
from models.efpn_backbone.anchors import Anchors
from src.helpers import train, validate, test


# Import data paths
map_path = "src/code_map.json"
data_info_path = "src/data_info/uav_data_preprocessing.json"
base_dir = "data/uav_sod_data/"


# Set device we are going to load the model and the data
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Load the classes of the UAV-SOD Drone dataset
map = open(map_path)
data = json.load(map)
classes = data["UAV_SOD_DRONE"]["CATEGORY_ID_TO_NAME"]
map.close() 

# The number of classes plus the background
number_classes = len(classes) + 1


# Load the mean and standard deviation for the train data
map = open(data_info_path)
data = json.load(map)
mean = data["uav_data"]["mean"]
standard_deviation = data["uav_data"]["std"]
map.close() 


# Define train, test and validation path
train_path = os.path.join(base_dir, "train")
test_path = os.path.join(base_dir, "test")
validation_path = os.path.join(base_dir, "validation")


def sod_collate_fn(batch):
    images = [item[0] for item in batch] 
    targets = [item[1] for item in batch]

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    return images, targets


# Data transform function
data_transform = {
    "train": transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=standard_deviation)]),

    "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=standard_deviation)]), 
            
    "validation": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=standard_deviation)]) 
}


# Dataset and DataLoader
train_dataset      = SOD_Data(train_path +"/images", train_path + "/annotations", data_transform["train"])
test_dataset       = SOD_Data(test_path + "/images", test_path  + "/annotations", data_transform["test"])
validation_dataset = SOD_Data(validation_path + "/images", validation_path + "/annotations", data_transform["validation"])

train_loader      = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=sod_collate_fn, pin_memory=True)
test_loader       = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=sod_collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, collate_fn=sod_collate_fn)


# Return a dictionary of the main statistics
bbox_stats = train_dataset.analyze_bounding_boxes()

# Get mean for width and height
mean_width = bbox_stats['mean_width']
mean_height = bbox_stats['mean_height']

# Get standard deviation for width and height
std_width = bbox_stats['std_width']
std_height = bbox_stats['std_height']

# Print statistics
print("Aspect Ratios:", sorted(set(bbox_stats['aspect_ratios'])))
print("Mean Width:", bbox_stats['mean_width'])
print("Mean Height:", bbox_stats['mean_height'])
print("Width Std Dev:", bbox_stats['std_width'])
print("Height Std Dev:", bbox_stats['std_height'])

# Based on the statistics above decide on the values of the statistics 
feature_map_shapes = [(38, 38), (19, 19), (10, 10)]

# Get all the scales
scales = [mean_width - std_width, mean_width, mean_width + std_width, mean_height - std_height, mean_height, mean_height + std_height]
scales = sorted(set([max(int(scale), 1) for scale in scales]))

# Define the aspect ratios
aspect_ratios = [0.75, 1.0, 1.25]

anchors = torch.tensor(Anchors.generate_anchors(feature_map_shapes, scales, aspect_ratios), dtype=torch.float32)

print("The number of anchors is: {}".format(anchors.size(0)))


# Initialise the ExtendedMask2Former model and load it to device
model = ExtendedMask2Former(num_classes=number_classes, num_anchors=anchors.size(0), device=device).to(device)
anchors = anchors.to(device)


# Hyperparameters selection
num_epochs = 1
learning_rate = 0.001
batch_size = 1

# Define the optimizer and the scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'AP', 'mAP'])

# Train for one epoch to ensure that the model can be trained as expected
num_epochs = 1

for epoch in range(num_epochs):
    
    model.train()
    
    for images, targets in train_loader:
        images = torch.stack(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        gt_bboxes = targets[0]['boxes'].to(device)
        gt_labels = targets[0]['labels'].to(device)
        gt_masks  = targets[0]['masks'].to(device)

        predictions = model(images)
        actual = {"boxes": gt_bboxes, "labels": gt_labels}

        loss = model.compute_loss(predictions, actual, anchors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
