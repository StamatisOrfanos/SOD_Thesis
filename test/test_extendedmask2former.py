import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from PIL import Image 
from torchvision import transforms
from models.extended_mask2former_model import ExtendedMask2Former
# Import libraries
import pandas as pd
import os, json, statistics
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.data_set_up import SOD_Data
from models.extended_mask2former_model import ExtendedMask2Former
from models.efpn_backbone.anchors import Anchors
import torch.nn.functional as F
from src.helpers import train, evaluate_model

# # ------------------------------------------------------------------------------------------------------------------------------------
# # Step 1: Load and preprocess the image
# def load_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize((600, 600)),  # Resize to the input size expected by EfficientNet
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for EfficientNet
#     ])
#     return transform(image).unsqueeze(0)  # type: ignore # Add batch dimension

# image = load_image('/Users/stamatiosorphanos/Documents/MCs_Thesis/SOD_Thesis/docs/Extended_Mask2Former/1.jpg')


# # Step 2: Initialise the Mask2Former Model 
# model = ExtendedMask2Former(num_classes=100, hidden_dim=256, num_queries=100, nheads=8, dim_feedforward=2048, dec_layers=1, mask_dim=256)

# # Step 3: Count the number of parameters
# total_params = sum(param.numel() for param in model.parameters())
# print('The total number of Extended Mask2Former parameters are: ', total_params)
# model.eval()  # Set the model to evaluation mode

# # Step 4: Pass the image through the model
# with torch.no_grad():
#     output = model(image)

# # Step 5: Visualize the feature maps
# print('\nThe predicted classes are the following:\n')
# print(output['pred_logits'])
    
# # Step 6: Visualize the mask of the model
# print('\nThe masks with shape: {} created are the following:\n'.format(output['pred_masks'].shape))
# print(output['pred_masks'])
    
# # Step 7: Visualize the bounding box of the model
# print('\nThe bounding box with shape: {} created is the following:\n'.format(output['bounding_box'].shape))
# print(output['bounding_box'])

# # Step 8: Visualize the class scores
# print('\nThe class scores with shape: {} created are the following:\n'.format(output['class_scores'].shape))
# print(output['class_scores'])

# # # Step 9: Get the summary of the EFPN model
# # print('\n\nThe summary of the EFPN model is the following: \n', model) 
# # ------------------------------------------------------------------------------------------------------------------------------------


# Import data paths
map_path = 'src/code_map.json'
data_info_path = 'src/data_info/uav_data_preprocessing.json'
base_dir = 'data/uav_sod_data/'


# Set device we are going to load the model and the data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the classes of the UAV-SOD Drone dataset
map = open(map_path)
data = json.load(map)
classes = data['UAV_SOD_DRONE']['CATEGORY_ID_TO_NAME']
map.close() 

# The number of classes plus the background
number_classes = len(classes) + 1


# Load the mean and standard deviation for the train data
map = open(data_info_path)
data = json.load(map)
mean = data['uav_data']['mean']
standard_deviation = data['uav_data']['std']
map.close() 


# Define train, test and validation path
train_path = os.path.join(base_dir, 'train')
test_path = os.path.join(base_dir, 'test')
validation_path = os.path.join(base_dir, 'validation')


# Data transform function
data_transform = {
    'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=standard_deviation)]),

    'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=standard_deviation)]), 
            
    'validation': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=standard_deviation)]) 
}


train_dataset      = SOD_Data(train_path +'/images', train_path + '/annotations', data_transform['train'])
test_dataset       = SOD_Data(test_path + '/images', test_path  + '/annotations', data_transform['test'])
validation_dataset = SOD_Data(validation_path + '/images', validation_path + '/annotations', data_transform['validation'])

train_loader      = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader       = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
validation_loader = DataLoader(validation_dataset, batch_size=3, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


# Return a dictionary of the main statistics
bbox_stats = train_dataset.analyze_bounding_boxes()

# Get mean for width and height
mean_width = bbox_stats['mean_width']
mean_height = bbox_stats['mean_height']

# Get standard deviation for width and height
std_width = bbox_stats['std_width']
std_height = bbox_stats['std_height']


# Based on the statistics above decide on the values of the statistics 
feature_map_shapes = [(19, 19)]

# Get all the scales
scales = [32]

# Define the aspect ratios
aspect_ratios = [0.5, 1.0]
anchors = torch.tensor(Anchors.generate_anchors(feature_map_shapes, scales, aspect_ratios), dtype=torch.float32)

# Initialise the ExtendedMask2Former model and load it to device
model = ExtendedMask2Former(num_classes=number_classes, num_anchors=anchors.size(0), device=device).to(device)
anchors = anchors.to(device)

# Hyperparameters selection
num_epochs = 1
learning_rate = 0.001


# Define the optimizer and the scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

metrics_df = pd.DataFrame(columns=['epoch', 'loss', 'precision', 'recall', 'mAP', 'mAPCOCO'])

# Train for one epoch to ensure that the model can be trained as expected
num_epochs = 1

# train(model, train_loader, device, anchors, optimizer, 1)
evaluate_model(model, test_loader, device, anchors)