import torch
from PIL import Image, ImageDraw
import numpy as np
import io, boto3, os
from torch.utils.data import Dataset
from collections import defaultdict

class S3Dataset(Dataset):
    def __init__(self, images_directory, annotations_directory, transform, bucket_name, target_size=300, max_annotations=100):
        self.transform = transform
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.target_size = target_size
        self.max_annotations = max_annotations

        # Ensure directories end with '/'
        self.images_directory = images_directory.rstrip('/') + '/'
        self.annotations_directory = annotations_directory.rstrip('/') + '/'

        # List files using S3 client
        self.image_files = self.list_s3_files(self.images_directory)
        self.annotation_files = self.list_s3_files(self.annotations_directory)

        # Filters images to those that have corresponding annotations
        self.valid_image_files = self.filter_images_with_annotations(self.image_files)

    def __len__(self):
        return len(self.valid_image_files)

    def list_s3_files(self, prefix):
        """List files in an S3 bucket under a specific prefix."""
        paginator = self.s3_client.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

        files = []
        for page in response_iterator:
            if 'Contents' in page:
                files.extend([
                    content['Key'] for content in page['Contents']
                    if content['Key'].endswith(('.jpg', '.png', '.jpeg', '.txt'))
                ])
        return files

    def __getitem__(self, idx):
        # Image handling
        image_path = self.valid_image_files[idx]
        image = self.load_s3_image(image_path)

        # Annotations Handling
        annotation_filename = os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt')
        annotation_path = self.annotations_directory + annotation_filename
        boxes, labels, masks = self.load_s3_annotation(annotation_path)

        boxes = torch.as_tensor(boxes, dtype=torch.float32) / self.target_size
        labels = torch.as_tensor(labels, dtype=torch.int64)
        if masks:
            masks = torch.stack(masks)
        else:
            masks = torch.zeros((0, self.target_size, self.target_size), dtype=torch.uint8)
        masks = self.pad_masks(masks, self.max_annotations)
        mask_labels = self.pad_mask_classes(labels, self.max_annotations)
        target = {'boxes': boxes, 'labels': labels, 'masks': masks, "mask_labels": mask_labels}

        if self.transform:
            image = self.transform(image)

        return image, target

    def load_s3_image(self, path):
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=path)
        return Image.open(io.BytesIO(obj['Body'].read())).convert('RGB')

    def load_s3_annotation(self, key):
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        lines = obj['Body'].read().decode('utf-8').strip().split('\n')
        boxes, labels, masks_list = [], [], []

        for line in lines:
            bbox_class_part = line.split("[")[0].split(",")
            x_min, y_min, x_max, y_max = map(int, bbox_class_part[0:4])
            class_code = int(bbox_class_part[4])
            box = [x_min, y_min, x_max, y_max]
            box = self.resize_box(box)

            masks_part = eval("[" + line.split("[")[1])
            mask = self.create_and_resize_mask((600, 600), (self.target_size, self.target_size), masks_part)

            boxes.append(box)
            labels.append(class_code)
            masks_list.append(mask)

        return boxes, labels, masks_list

    def filter_images_with_annotations(self, image_files):
        valid_image_files = []

        for image_name in image_files:
            annotation_name = os.path.basename(image_name).replace('.jpg', '.txt').replace('.png', '.txt')
            annotation_path = self.annotations_directory + annotation_name

            # Try to get the annotation file from S3
            try:
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=annotation_path)
                annotation_lines = obj['Body'].read().decode('utf-8').strip().split('\n')
                if len(annotation_lines) <= self.max_annotations:
                    valid_image_files.append(image_name)
            except self.s3_client.exceptions.NoSuchKey:
                # Annotation file does not exist
                continue

        return valid_image_files

    
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
    
    def create_and_resize_mask(self, original_size, target_size, polygons):
        """
        Parameters:
            original_size (tuple): Original dimensions of the image (height, width).
            target_size (tuple): Target dimensions to which the mask will be resized.
            polygons (list of lists of tuples): Each list of tuples represents polygon vertices.
        """
        mask = Image.new('L', original_size, 0)
        draw = ImageDraw.Draw(mask)
            
        if len(polygons) >= 3:
            draw.polygon(polygons, outline=1, fill=1)
        else:
            print("Polygon with insufficient points:", polygons)

        mask = mask.resize(target_size, Image.NEAREST) # type: ignore
        mask_array = np.array(mask)
        return torch.tensor(mask_array, dtype=torch.uint8)
    
    def pad_masks(self, masks, max_pad=100):
        """
            Pad or truncate the mask tensor to have a fixed number of masks
        """
        padded_masks = torch.zeros((max_pad, masks.shape[1], masks.shape[2]), dtype=masks.dtype)
        actual_masks = min(max_pad, masks.shape[0])
        padded_masks[:actual_masks] = masks[:actual_masks]
        return padded_masks


    def pad_mask_classes(self, labels, max_pad=100):
        """
            Pad or truncate the labels to have a fixed number of classes
        """
        padded_labels = torch.full((max_pad,), -1, dtype=torch.int64)
        actual_labels = min(max_pad, len(labels))
        padded_labels[:actual_labels] = labels[:actual_labels]
        return padded_labels
    
    
    def analyze_bounding_boxes(self):
        """
        Analyze the bounding box sizes and return statistics.
        """
        box_widths = []
        box_heights = []
        aspect_ratios = defaultdict(int)

        for image_name in self.image_files:
            # Construct the annotation file name
            annotation_name = os.path.basename(image_name).replace('.jpg', '.txt').replace('.png', '.txt')
            annotation_path = self.annotations_directory + annotation_name
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=annotation_path)
            annotation_lines = obj['Body'].read().decode('utf-8').strip().split('\n')

            for line in annotation_lines:
                bbox_class_part = line.split("[")[0].split(",")
                x_min, y_min, x_max, y_max = map(int, bbox_class_part[0:4])
                width = x_max - x_min
                height = y_max - y_min
                if height == 0: height = 1  # Avoid division by zero
                box_widths.append(width)
                box_heights.append(height)
                aspect_ratio = width / height
                aspect_ratios[aspect_ratio] += 1

        # Compute statistics
        if box_widths and box_heights:
            mean_width = np.mean(box_widths)
            mean_height = np.mean(box_heights)
            std_width = np.std(box_widths)
            std_height = np.std(box_heights)
        else:
            mean_width = mean_height = std_width = std_height = 0

        return {
            'box_widths': box_widths,
            'box_heights': box_heights,
            'aspect_ratios': aspect_ratios,
            'mean_width': mean_width,
            'mean_height': mean_height,
            'std_width': std_width,
            'std_height': std_height,
        }
