import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw

class SOD_Data(Dataset):
    def __init__(self, images_directory, annotations_directory, transform=None):
        self.image_dir = images_directory
        self.annotation_dir = annotations_directory
        self.image_files = [f for f in os.listdir(images_directory) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        annotation_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
        annotation_path = os.path.join(self.annotation_dir, annotation_name)

        boxes, labels, masks_list = [], [], []

        with open(annotation_path, 'r') as f:
            for line in f:
                bbox_class_part = line.split("[")[0].split(",")
                x_min, y_min, x_max, y_max = map(int, bbox_class_part[0:4])
                class_code = int(bbox_class_part[4])
                masks_part = eval("[" + line.split("[")[1])

                box = [x_min, y_min, x_max, y_max]
                mask = self.create_binary_mask((600, 600), masks_part)

                boxes.append(box)
                labels.append(class_code)
                masks_list.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack([torch.tensor(mask, dtype=torch.uint8) for mask in masks_list])

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def create_binary_mask(self, image_size, polygons):
        mask = Image.new('L', image_size, 0)
        for polygon in polygons:
            if polygon:
                ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
        return np.array(mask)