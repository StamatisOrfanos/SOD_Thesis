from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision.transforms import functional as F


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
    


def transform(image, boxes, mask):
    # Resize, to tensor, etc.
    image = F.to_tensor(image)
    return image, boxes, mask


