import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchvision.transforms as transforms


class SOD_Data(Dataset):
    
    def __init__(self, images_directory, annotations_directory, transform, mean, standard_deviation, data_type):
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
        self.images = sorted(os.listdir(images_directory))
        self.annotations = sorted(os.listdir(annotations_directory))

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
                masks = create_binary_mask((600, 600), masks_part)
                
                boxes.append(box)
                labels.append(class_code)
                masks.append(masks)

        
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
    
    


def create_binary_mask(image_size, polygons):
    """
    Parameters:
        - image_size: tuple of (height, width)
        - polygons: list of lists of tuples [(x_1, y_1), ...., (x_n, y_n)]
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    for polygon in polygons:
        polygon = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask



def data_transform(data_type, mean , standard_deviation):
    data_transform = {
        "train": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
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
    
    
