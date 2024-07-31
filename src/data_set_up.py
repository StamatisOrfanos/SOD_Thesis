import os
from collections import defaultdict
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
    def __init__(self, images_directory, annotations_directory, transform, target_size=300, max_annotations=100):
        self.image_dir = images_directory
        self.annotation_dir = annotations_directory
        self.image_files = [f for f in os.listdir(images_directory) if f.endswith(('.jpg', '.png'))]
        self.transform = transform
        self.target_size = target_size
        self.max_annotations = max_annotations
        self.valid_image_files = self.filter_images_with_annotations(self.image_files)

    def __len__(self):
        return len(self.valid_image_files)

    def __getitem__(self, idx):
        # Image handling
        image_name = self.valid_image_files[idx]
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
                box = self.resize_box(box)
                
                masks_part = eval("[" + line.split("[")[1])
                masks = self.create_binary_mask((600, 600), masks_part)
                masks = self.resize_mask(masks, (self.target_size, self.target_size))
                
                boxes.append(box)
                labels.append(class_code)
                masks_list.append(masks)

        
        # boxes  = torch.as_tensor(boxes, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32) / self.target_size
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack([torch.tensor(mask, dtype=torch.uint8) for mask in masks_list])

        target = {'boxes': boxes, 'labels': labels, 'masks': masks }

        if self.transform: image = self.transform(image)
        
        return image, target
    
    
    def resize_box(self, box, original_size=(600,600)):
        """
        Parameters:
            - box (list<int>): List of integers, representing the coordinates of the (x_min, y_min) and (x_max, y_max)
            - original_size (tuple): Tuple of the original image size (600,600)
        """
        x_min, y_min, x_max, y_max = box
        orig_w, orig_h = original_size
        new_w, new_h = self.target_size, self.target_size
        x_min = int(x_min * new_w / orig_w)
        x_max = int(x_max * new_w / orig_w)
        y_min = int(y_min * new_h / orig_h)
        y_max = int(y_max * new_h / orig_h)
        return [x_min, y_min, x_max, y_max]
    

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
                    mask.polygon(polygon, fill=1, outline=1) # type: ignore
                    mask = np.array(mask)
        return mask
        
        
   
    def resize_mask(self, mask, target_size):
        """
        Parameters:
            - mask (np.array): This is the binary mask that we create for the image
            - target_size (tuple): Tuple of the image size we want to resize to
        """
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(target_size, Image.NEAREST) # type: ignore
        return np.array(mask_img)
    

    def filter_images_with_annotations(self, image_files):
        """
        Parameters:
            - image_files (list<File>): Initial list of images before the filtering
        """
        valid_image_files = []
        for image_name in image_files:
            annotation_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
            annotation_path = os.path.join(self.annotation_dir, annotation_name)
            with open(annotation_path, 'r') as f:
                annotation_lines = f.readlines()
                if len(annotation_lines) <= self.max_annotations:
                    valid_image_files.append(image_name)
        return valid_image_files
    
    
    def analyze_bounding_boxes(self):
        """
            Analyze the bounding box sizes and return statistics.
        """
        box_widths = []
        box_heights = []
        aspect_ratios = defaultdict(int)

        for image_name in self.image_files:
            annotation_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
            annotation_path = os.path.join(self.annotation_dir, annotation_name)
            with open(annotation_path, 'r') as f:
                for line in f:
                    bbox_class_part = line.split("[")[0].split(",")
                    x_min, y_min, x_max, y_max = map(int, bbox_class_part[0:4])
                    width = x_max - x_min
                    height = y_max - y_min
                    box_widths.append(width)
                    box_heights.append(height)
                    aspect_ratios[width / height] += 1

        return {
            'box_widths': box_widths,
            'box_heights': box_heights,
            'aspect_ratios': aspect_ratios,
            'mean_width': np.mean(box_widths),
            'mean_height': np.mean(box_heights),
            'std_width': np.std(box_widths),
            'std_height': np.std(box_heights),
        }
