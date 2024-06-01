import numpy as np
import torch    


def calculate_iou(anchor, ground_truth_boxes):
    """
    Parameters:
        - anchor (list): List of the anchor coordinates for (x_min, y_min, x_max, y_max)
        - ground_truth_boxes (list): List of the bounding box coordinates for (x_min, y_min, x_max, y_max)
    """
    if not isinstance(anchor, torch.Tensor): anchor = torch.tensor(anchor, dtype=torch.float32)
    if not isinstance(ground_truth_boxes, torch.Tensor): ground_truth_boxes = torch.tensor(ground_truth_boxes, dtype=torch.float32)
    
    anchor_x1, anchor_y1, anchor_x2, anchor_y2 = anchor
    
    # Ground truth box corners and the intersections
    ground_truth_xmin  = ground_truth_boxes[0]
    ground_truth_ymin  = ground_truth_boxes[1]
    ground_truth_xmax = ground_truth_boxes[2]
    ground_truth_ymax  = ground_truth_boxes[3]
    
    inter_x1 = torch.max(anchor_x1, ground_truth_xmin)
    inter_y1 = torch.max(anchor_y1, ground_truth_ymin)
    inter_x2 = torch.min(anchor_x2, ground_truth_xmax)
    inter_y2 = torch.min(anchor_y2, ground_truth_ymax)
    
    # Compute the intersection, anchor and actual bounding box area to compute the iou
    intersection_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    anchor_area       = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    ground_truth_area = (ground_truth_xmax - ground_truth_xmin) * (ground_truth_ymax - ground_truth_ymin)
    
    union_area = anchor_area + ground_truth_area - intersection_area
    iou        = intersection_area / union_area
    return iou


def match_anchors_to_ground_truth(anchors, ground_truth_boxes, ground_truth_labels, num_classes):
    """
    Parameters:
        - anchors (array): Array of the produced anchors 
        - ground_truth_boxes (array): Array of the bounding boxes.
        - ground_truth_labels (array): Array of the labels
        - num_classes (int): Number of classes of the dataset 
    """
    num_anchors = anchors.shape[0]
    num_gt_boxes = ground_truth_boxes.shape[0]
    
    # Initialize matched labels and boxes
    matched_labels = torch.zeros((num_anchors,), dtype=torch.long)
    matched_boxes = torch.zeros((num_anchors, 4))
    
    # Compute IoUs
    ious = torch.zeros((num_anchors, num_gt_boxes))
    for i, anchor in enumerate(anchors):
        ious[i, :] = calculate_iou(anchor, ground_truth_boxes)
    
    # Find the best match for each anchor
    max_ious, max_indices = ious.max(dim=1)
    
    for i in range(num_anchors):
        # Intersection over Union threshold of 50%
        if max_ious[i] > 0.5: 
            matched_labels[i] = ground_truth_labels[max_indices[i]]
            matched_boxes[i, :] = ground_truth_boxes[max_indices[i], :]
        else:
            matched_labels[i] = num_classes - 1 
            
    
    return matched_boxes, matched_labels


def calculate_precision_recall(predicted_boxes, pred_scores, ground_truth_boxes, iou_threshold=0.5):
    """    
    Parameters:
        - predicted_boxes (array): predicted bounding boxes, (x_min, y_min, x_max, y_max)
        - pred_scores (array): predicted confidence scores
        - ground_truth_boxes (array): ground truth bounding boxes, (x_min, y_min, x_max, y_max)
        - iou_threshold (float): IoU threshold for a positive match
    """
    sorted_indices = np.argsort(-pred_scores)
    predicted_boxes = predicted_boxes[sorted_indices]
    ground_truth_boxes = list(ground_truth_boxes)
    
    tp = np.zeros(len(predicted_boxes))
    fp = np.zeros(len(predicted_boxes))
    
    for i, predicted_box in enumerate(predicted_boxes):
        if len(ground_truth_boxes) == 0:
            fp[i] = 1
            continue
        
        ious = np.array([calculate_iou(predicted_box, ground_truth_box) for ground_truth_box in ground_truth_boxes])
        
        max_iou_idx = np.argmax(ious)
        if ious[max_iou_idx] >= iou_threshold:
            tp[i] = 1
            ground_truth_boxes.pop(max_iou_idx)
        else:
            fp[i] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Handle case when there are no ground truth boxes
    num_ground_truth_boxes = len(ground_truth_boxes) + np.sum(tp)
    if num_ground_truth_boxes == 0:
        recall = tp_cumsum / (num_ground_truth_boxes + 1e-10)
    else:
        recall = tp_cumsum / num_ground_truth_boxes
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    return precision, recall


def calculate_ap(precision, recall):
    """Calculate Average Precision (AP) from precision and recall values."""
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap



def calculate_map(pred_boxes_list, pred_scores_list, gt_boxes_list, num_classes, iou_threshold=0.5):
    """    
    Parameters:
        - pred_boxes_list (list): list of predicted bounding boxes for each class
        - pred_scores;lkjhgf_list (list): list of predicted confidence scores for each class
        - gt_boxes_list (list): list of ground truth bounding boxes for each class
        - num_classes (int): number of classes
        - iou_threshold (float): IoU threshold for a positive match
    """
    ap_sum = 0
    for cls in range(num_classes):
        pred_boxes = np.array(pred_boxes_list[cls])
        pred_scores = np.array(pred_scores_list[cls])
        gt_boxes = np.array(gt_boxes_list[cls])
        
        precision, recall = calculate_precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        ap = calculate_ap(precision, recall)
        ap_sum += ap
    
    return ap_sum / num_classes



pred_boxes_list = [[[0.1, 0.1, 0.4, 0.4]], [[0.2, 0.2, 0.5, 0.5]]]
pred_scores_list = [[0.7], [0.8]]
gt_boxes_list = [[[0.2, 0.2, 0.35, 0.35]], [[0.2, 0.2, 0.5, 0.5]]]
num_classes = 2

map_value = calculate_map(pred_boxes_list, pred_scores_list, gt_boxes_list, num_classes)
print(f"mAP: {map_value:.2f}")
