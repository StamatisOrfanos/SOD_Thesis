import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BoundingBoxGenerator(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BoundingBoxGenerator, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.class_convolution = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        self.regression_convolution = nn.Conv2d(256, 4, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        class_scores = self.class_convolution(x)
        bounding_box = self.regression_convolution(x)
        return bounding_box, class_scores


def match_anchors_to_gt_boxes(anchors, gt_boxes, iou_threshold=0.5):
    ious = compute_iou(anchors, gt_boxes)
    anchor_max_iou, anchor_max_idx = ious.max(dim=1)
    matched_gt_boxes = gt_boxes[anchor_max_idx]
    matched_gt_boxes[anchor_max_iou < iou_threshold] = -1
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
    gt_cx = (matched_gt_boxes[:, 0] + matched_gt_boxes[:, 2]) / 2
    gt_cy = (matched_gt_boxes[:, 1] + matched_gt_boxes[:, 3]) / 2
    gt_w = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
    gt_h = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]

    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]

    # Avoid division by zero and log of zero
    eps = 1e-6
    gt_w = torch.clamp(gt_w, min=eps)
    gt_h = torch.clamp(gt_h, min=eps)
    anchor_w = torch.clamp(anchor_w, min=eps)
    anchor_h = torch.clamp(anchor_h, min=eps)

    dx = (gt_cx - anchor_cx) / anchor_w
    dy = (gt_cy - anchor_cy) / anchor_h
    dw = torch.log(gt_w / anchor_w)
    dh = torch.log(gt_h / anchor_h)

    return torch.stack([dx, dy, dw, dh], dim=-1)
