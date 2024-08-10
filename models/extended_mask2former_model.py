from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
from models.efpn_backbone.efpn_model import EFPN
from models.mask2former_detector.mask2former_model import Mask2Former
from models.efpn_backbone.bounding_box import encode_bounding_boxes, match_anchors_to_ground_truth_boxes



class ExtendedMask2Former(nn.Module):
    """
    Extended Mask Transformer Model integrating EFPN as the backbone and feature mask generator, with Mask2Former for instance segmentation tasks.
    The model uses the Enhanced Feature Pyramid Network (EFPN) to extract multi-scale feature maps and generate corresponding mask features 
    from images. These features are then used by the Mask2Former model to perform instance segmentation, identifying and delineating each object 
    instance in the input image.

    Parameters:
        efpn (EFPN): The Enhanced Feature Pyramid Network model used as the backbone for feature and mask feature extraction.
        mask2former (Mask2Former): The Mask2Former model used for predicting object instances and their masks based on the features provided by EFPN.
    """
    def __init__(self, num_classes, num_anchors, device, hidden_dim=256, num_queries=300, nheads=8, dim_feedforward=2048, dec_layers=1, mask_dim=256):
        super(ExtendedMask2Former, self).__init__()
        self.device = device              
        self.efpn        = EFPN(hidden_dim, hidden_dim, num_classes, num_anchors)
        self.mask2former = Mask2Former(hidden_dim, num_classes, hidden_dim, num_queries, nheads, dim_feedforward, dec_layers, mask_dim)
        
        # Define loss functions
        self.bounding_box_loss = nn.SmoothL1Loss(reduction="mean")
        self.class_loss        = nn.CrossEntropyLoss(reduction="mean")
        self.mask_loss         = nn.BCEWithLogitsLoss()


    def forward(self, image):
        image.to(self.device)
        feature_maps, masks, bounding_box, class_scores = self.efpn(image)    
        feature_maps = feature_maps.to(self.device)
        masks = masks.to(self.device)
        class_scores = class_scores.to(self.device)
        output = self.mask2former(feature_maps, masks, bounding_box, class_scores)
        output = output.to(self.device)
        return output
        

    def decode_boxes(self, predicted_offsets, anchors):
        """
        Decode predicted bounding box offsets with respect to anchor boxes.
        """
        pred_boxes = torch.zeros_like(predicted_offsets)

        pred_boxes[:, 0] = predicted_offsets[:, 0] + anchors[:, 0]
        pred_boxes[:, 1] = predicted_offsets[:, 1] + anchors[:, 1]
        pred_boxes[:, 2] = predicted_offsets[:, 2] + anchors[:, 2]
        pred_boxes[:, 3] = predicted_offsets[:, 3] + anchors[:, 3]

        return pred_boxes
    
    
    def compute_cost_matrix(self, predicted_masks, ground_truth_masks):
        """    
        Parameters:
            - predicted_masks (torch.Tensor): Predicted masks of shape (num_queries, H, W).
            - ground_truth_masks (torch.Tensor): Ground truth masks of shape (num_objects, H, W).
        """
        num_queries, H, W = predicted_masks.shape
        num_objects       = ground_truth_masks.shape[0]
        
        # Convert masks to float for distance computation
        predicted_masks    = predicted_masks.float()
        ground_truth_masks = ground_truth_masks.float()
        
        # Flatten masks for ease of computation and compute the cost matrix based on binary cross entropy loss
        pred_masks_flat = predicted_masks.view(num_queries, -1)
        gt_masks_flat   = ground_truth_masks.view(num_objects, -1)
        cost_matrix     = torch.cdist(pred_masks_flat, gt_masks_flat, p=1)
    
        return cost_matrix


    def hungarian_matching(self, predicted_masks, ground_truth_masks):
        """    
        Parameters:
            - predicted_masks (torch.Tensor): Predicted masks of shape (num_queries, H, W).
            - ground_truth_masks (torch.Tensor): Ground truth masks of shape (num_objects, H, W).    
        """
        cost_matrix = self.compute_cost_matrix(predicted_masks, ground_truth_masks)    
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
        matched_indices = list(zip(pred_indices, gt_indices))
        
        return matched_indices
    
       
    def compute_loss(self, predictions, targets, anchors):
        """
        Refactored loss computation to better align with the DetectionLoss class approach.
        """
        device = predictions.device
        anchors = anchors.to(device)
        
        # Assume predictions dictionary is structured with 'pred_logits', 'pred_masks', and 'pred_boxes'
        predicted_classes = predictions['pred_logits']
        predicted_masks = predictions['pred_masks']
        predicted_bboxes = predictions['bounding_box']

      

        ground_truth_labels = target['labels'].to(device)
        ground_truth_masks = target['masks'].to(device)
        ground_truth_boxes = target['boxes'].to(device)
        
        # Align predictions with ground truths via anchor matching
        matched_gt_boxes, anchor_max_idx = match_anchors_to_ground_truth_boxes(anchors, ground_truth_boxes)
        
        # Filter predictions that correspond to matched anchors
        matched_pred_bboxes = predicted_bboxes[i][anchor_max_idx]
        matched_pred_classes = predicted_classes[i][anchor_max_idx]
        matched_pred_masks = predicted_masks[i][anchor_max_idx]
        
        # Calculate regression and classification losses for matched anchors
        regression_targets = encode_bounding_boxes(matched_gt_boxes, anchors[anchor_max_idx]).to(device)
        total_bbox_loss += self.bounding_box_loss(matched_pred_bboxes, regression_targets)
        
        classification_targets = ground_truth_labels[anchor_max_idx]
        total_class_loss += self.class_loss(matched_pred_classes, classification_targets)
        # # Assuming ground truth masks are available and need to be matched similarly
        # if 'masks' in target:
        #     ground_truth_masks = target['masks'][anchor_max_idx].to(device)
        #     total_mask_loss += self.mask_loss(matched_pred_masks, ground_truth_masks)

        total_loss = total_class_loss + total_bbox_loss + total_mask_loss
        return total_loss
