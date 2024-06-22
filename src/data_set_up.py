import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image, ImageDraw

class SOD_Data(Dataset):
    """
    Args:
        - images_directory (string): Path of the directory containing the images 
        - annotations_directory (string): Path of the directory containing the annotations
        - transform (pytorch.transform, optional): transform function for the images of the dataset
    """
    def __init__(self, images_directory, annotations_directory, transform):
        self.image_dir = images_directory
        self.annotation_dir = annotations_directory
        self.image_files = [f for f in os.listdir(images_directory) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Image handling
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # Annotations Handling
        annotation_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
        annotation_path = os.path.join(self.annotation_dir, annotation_name)
        
        boxes, labels, masks_list  = [], [], []
        
        with open(annotation_path, 'r') as f:
            for line in f:
                bbox_class_part = line.split("[")[0].split(",")
                x_min, y_min, x_max, y_max = bbox_class_part[0:4]
                class_code = int(bbox_class_part[4])
                box = [int(x_min), int(y_min), int(x_max), int(y_max)]
                
                masks_part = eval("[" + line.split("[")[1])
                masks = self.create_binary_mask((600, 600), masks_part)
                mask_resized = cv2.resize(masks, (300,300), interpolation=cv2.INTER_NEAREST)
                
                boxes.append(box)
                labels.append(class_code)
                masks_list.append(mask_resized)

        
        boxes  = torch.as_tensor(boxes, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack([torch.tensor(mask, dtype=torch.uint8) for mask in masks_list])

        target = {'boxes': boxes, 'labels': labels, 'masks': masks }

        if self.transform: image = self.transform(image)
        
        return image, target
    

    def create_binary_mask(self, original_size, polygons):
            """
            Parameters:
                - original_size (tuple): Tuple of the original size of the image (height, width)
                - polygons (list): List of tuples where each tuple is a point
            """
            mask = np.zeros(original_size, dtype=np.uint8)
            for polygon in polygons:
                if polygon:
                    polygon = ()
                    polygon = np.array(polygon, dtype=np.int32)
                    if polygon.shape[0] >= 3:
                        mask = Image.fromarray(mask)
                        mask.polygon(polygon, fill=1, outline=1)
                        mask = np.array(mask)
            return mask

