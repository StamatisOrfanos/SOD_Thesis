import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class SOD_Data(Dataset):
    
    def __init__(self, images_directory, annotations_directory, transform):
        """
        Args:
            - images_directory (string): Path of the directory containing the images 
            - annotations_directory (string): Path of the directory containing the annotations
            - transform (pytorch.transform, optional): t
        """
        self.images_dir = images_directory
        self.annotations_dir = annotations_directory 
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_directory) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file      = self.image_files[idx] 
        image_path      = os.path.join(self.images_dir, image_file)
        annotation_path = os.path.join(self.annotations_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(image_path).convert("RGB")
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        
        boxes  = []
        labels = []
        masks  = []
        
        for line in lines:
            parts = line.strip().split(',')
            x_min, y_min, x_max, y_max, class_code = map(int, parts[:5])
            box = [x_min, y_min, x_max, y_max]
            mask_points = eval(parts[5])
            mask = self.create_mask_from_points(mask_points, image.size)

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

        if self.transform:
            image = self.transform(image)
        
        return image, target

    def create_mask_from_points(self, points, image_size):
        mask = Image.new('L', image_size, 0)
        points = [(int(x), int(y)) for x, y in points]
        return torch.as_tensor(np.array(mask), dtype=torch.uint8)