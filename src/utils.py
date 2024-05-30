import numpy as np
import torch

    
def generate_anchors(self, feature_map_shapes, scales, aspect_ratios):
    anchors = []
    for shape in feature_map_shapes:
        for scale in scales:
            for ratio in aspect_ratios:
                # Compute anchor box dimensions
                anchor_width = scale * np.sqrt(ratio)
                anchor_height = scale / np.sqrt(ratio)
                for y in range(shape[0]):
                    for x in range(shape[1]):
                        cx = (x + 0.5) / shape[1]
                        cy = (y + 0.5) / shape[0]
                        # Convert to (x_min, y_min, x_max, y_max)
                        x_min = cx - anchor_width / 2
                        y_min = cy - anchor_height / 2
                        x_max = cx + anchor_width / 2
                        y_max = cy + anchor_height / 2
                        anchors.append([x_min, y_min, x_max, y_max])
    return np.array(anchors)


def calculate_iou(anchor, ground_truth_boxes):
    """
    Parameters:
        - anchor (list): List of the anchor coordinates for (x_min, y_min, x_max, y_max)
        - ground_truth_boxes (list): List of the bounding box coordinates for (x_min, y_min, x_max, y_max)
    """
    # Anchor box corners
    anchor_x1, anchor_y1, anchor_x2, anchor_y2 = anchor
    # Ground truth box corners
    ground_truth_xmin  = ground_truth_boxes[:, 0]
    ground_truth_ymin  = ground_truth_boxes[:, 1]
    ground_truth__xmax = ground_truth_boxes[:, 2]
    ground_truth_ymax  = ground_truth_boxes[:, 3]
    inter_x1 = torch.max(anchor_x1, ground_truth_xmin)
    inter_y1 = torch.max(anchor_y1, ground_truth_ymin)
    inter_x2 = torch.min(anchor_x2, ground_truth__xmax)
    inter_y2 = torch.min(anchor_y2, ground_truth_ymax)
    inter_area        = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    anchor_area       = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    ground_truth_area = (ground_truth__xmax - ground_truth_xmin) * (ground_truth_ymax - ground_truth_ymin)
    union_area = anchor_area + ground_truth_area - inter_area
    iou        = inter_area / union_area
    return iou


    
    
def match_anchors_to_ground_truth(anchors, gt_boxes, gt_labels, num_classes):
    num_anchors = anchors.shape[0]
    num_gt_boxes = gt_boxes.shape[0]
    # Initialize matched labels and boxes
    matched_labels = torch.zeros((num_anchors,), dtype=torch.long)
    matched_boxes = torch.zeros((num_anchors, 4))
    # Compute IoUs
    ious = torch.zeros((num_anchors, num_gt_boxes))
    for i, anchor in enumerate(anchors):
        ious[i, :] = calculate_iou(anchor, gt_boxes)
    # Find the best match for each anchor
    max_ious, max_indices = ious.max(dim=1)
    for i in range(num_anchors):
        if max_ious[i] > 0.5:  # IoU threshold
            matched_labels[i] = gt_labels[max_indices[i]]
            matched_boxes[i, :] = gt_boxes[max_indices[i], :]
        else:
            matched_labels[i] = num_classes - 1  # Background class
    return matched_boxes, matched_labels
