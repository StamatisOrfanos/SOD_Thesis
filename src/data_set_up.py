import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL.Image import Image

class SOD_Data(Dataset):
    
    def __init__(self, images_directory, annotations_directory, transform):
        """
        Args:
            - images_directory (string): Path of the directory containing the images 
            - annotations_directory (string): Path of the directory containing the annotations
            - transform (pytorch.transform, optional): t
        """
        self.image_dir = images_directory
        self.annotation_dir = annotations_directory
        self.image_files = [f for f in os.listdir(images_directory) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        annotation_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
        annotation_path = os.path.join(self.annotation_dir, annotation_name)
        
        boxes  = []
        labels = []
        masks  = []
        
        with open(annotation_path, 'r') as f:
            for line in f:            
                bbox_class_part = line.split("[")[0].split(",")
                x_min, y_min, x_max, y_max = bbox_class_part[0:4]
                class_code = int(bbox_class_part[4])
                box = [int(x_min), int(y_min), int(x_max), int(y_max)]
                
                masks_part = eval("[" + line.split("[")[1])
                masks_part = [tuple(map(int, point.split())) for point in masks_part]
                mask = np.zeros((600, 600), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(masks_part)], 1)
                
                boxes.append(box)
                labels.append(class_code)
                masks.append(mask)

        
        boxes  = torch.as_tensor(boxes, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks  = torch.stack(masks)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }

        if self.transform: image = self.transform(image)
        
        return image, target
    