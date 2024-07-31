import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class BoundingBoxGenerator(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super(BoundingBoxGenerator, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.class_convolution = nn.Conv2d(256, num_anchors * num_classes, kernel_size=1)
        self.regression_convolution = nn.Conv2d(256, num_anchors * 4, kernel_size=1)  # Output 4 coordinates per anchor

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        class_scores = self.class_convolution(x)
        bbox_offsets = self.regression_convolution(x)

        batch_size = x.size(0)
        bbox_offsets = bbox_offsets.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        class_scores = class_scores.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)

        return bbox_offsets, class_scores


def match_anchors_to_ground_truth_boxes(anchors, gt_boxes, iou_threshold=0.5):
    ious = compute_iou(anchors, gt_boxes)
    anchor_max_iou, anchor_max_idx = ious.max(dim=1)
    matched_gt_boxes = gt_boxes[anchor_max_idx]
    matched_gt_boxes[anchor_max_iou < iou_threshold] = -1

    # Ensure matched_gt_boxes has the same shape as anchors
    if matched_gt_boxes.size(0) != anchors.size(0):
        raise ValueError("Mismatch in the number of anchors and matched ground truth boxes")

    return matched_gt_boxes, anchor_max_idx

def compute_iou(anchors, gt_boxes):
    anchors = anchors.clone().detach().float()
    gt_boxes = gt_boxes.clone().detach().float()

    area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    area_gt_boxes = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    lt = torch.max(anchors[:, None, :2], gt_boxes[:, :2])
    rb = torch.min(anchors[:, None, 2:], gt_boxes[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area_anchors[:, None] + area_gt_boxes - inter
    iou = inter / union

    return iou


def encode_bounding_boxes(matched_gt_boxes, anchors):
    """
    Encode ground truth bounding boxes with respect to anchor boxes.
    """
    dx_min = matched_gt_boxes[:, 0] - anchors[:, 0]
    dy_min = matched_gt_boxes[:, 1] - anchors[:, 1]
    dx_max = matched_gt_boxes[:, 2] - anchors[:, 2]
    dy_max = matched_gt_boxes[:, 3] - anchors[:, 3]

    return torch.stack([dx_min, dy_min, dx_max, dy_max], dim=-1)
