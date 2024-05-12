from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class SODDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.annotations_dir = os.path.join(root_dir, split, 'annotations')
        self.images = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        annotation_path = os.path.join(self.annotations_dir, img_name.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        boxes, masks = self.parse_annotations(annotation_path)

        if self.transforms:
            image, boxes, masks = self.transforms(image, boxes, masks)

        return image, boxes, masks

    def parse_annotations(self, annotation_path):
        with open(annotation_path, 'r') as file:
            data = file.read().strip().split(',')
            boxes = torch.tensor([float(x) for x in data[:4]]).reshape(-1, 4)
            mask_points = eval('[' + ','.join(data[5:]) + ']')  # This assumes the format [(x1, y1), (x2, y2), ...]
            masks = self.points_to_mask(mask_points)
        return boxes, masks

    def points_to_mask(self, points):
        # Implementation to convert points to a binary mask of shape [600, 600]
        mask = torch.zeros((600, 600))
        # Assuming points form a polygon
        # Use a library like PIL or OpenCV to draw the polygon on the mask
        return mask
    

from torchvision.transforms import functional as F

def transform(image, boxes, mask):
    # Resize, to tensor, etc.
    image = F.to_tensor(image)
    return image, boxes, mask



import torch.nn.functional as F

def smooth_l1_loss(input, target):
    return F.smooth_l1_loss(input, target, reduction='none')



def cross_entropy_loss(logits, targets):
    return F.cross_entropy(logits, targets, reduction='none')


def binary_cross_entropy_with_logits(input, target):
    return F.binary_cross_entropy_with_logits(input, target, reduction='none')

def dice_loss(pred, target, smooth=1):
    pred = pred.sigmoid()
    numerator = 2 * (pred * target).sum((1, 2, 3))
    denominator = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return 1 - (numerator + smooth) / (denominator + smooth)


def compute_loss(class_logits, bbox_preds, mask_preds, class_targets, bbox_targets, mask_targets):
    loss_bbox = smooth_l1_loss(bbox_preds, bbox_targets).mean()
    loss_class = cross_entropy_loss(class_logits, class_targets).mean()
    loss_mask = binary_cross_entropy_with_logits(mask_preds, mask_targets).mean() + dice_loss(mask_preds, mask_targets).mean()

    # Weights for each loss component could be tuned based on validation performance
    total_loss = loss_bbox + loss_class + loss_mask
    return total_loss



def compute_losses(predictions, targets):
    # Extract predictions
    class_logits = predictions['pred_logits']
    bbox_preds = predictions['bounding_box']
    mask_preds = predictions['pred_masks']

    # Extract targets
    class_targets = targets['labels']
    bbox_targets = targets['boxes']
    mask_targets = targets['masks']

    # Calculate losses
    loss_bbox = smooth_l1_loss(bbox_preds, bbox_targets).mean()
    loss_class = cross_entropy_loss(class_logits, class_targets).mean()
    loss_mask = binary_cross_entropy_with_logits(mask_preds, mask_targets).mean() + dice_loss(mask_preds, mask_targets).mean()

    total_loss = loss_bbox + loss_class + loss_mask
    return total_loss


import torch
from torch.optim import Adam

# Assuming 'model' is your combined EFPN and Mask2Former model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = Adam(model.parameters(), lr=1e-4)



from tqdm import tqdm

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, boxes, masks in tqdm(data_loader, desc="Training"):
        # Move data to the appropriate device
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        masks = [m.to(device) for m in masks]

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images, masks, boxes)

        # Prepare targets in the same structure as predictions
        targets = {
            'labels': labels,  # Need definition
            'boxes': boxes,
            'masks': masks
        }

        # Compute loss
        loss = compute_losses(predictions, targets)
        running_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    average_loss = running_loss / len(data_loader)
    print(f"Average loss: {average_loss:.5f}")

# Assuming you have a DataLoader `train_loader`
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch(model, train_loader, optimizer, device)
