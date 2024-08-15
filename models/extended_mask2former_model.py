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
        predicted_classes = predictions['pred_logits'][:, :self.anchors.size(0), :]
        predicted_masks = predictions['pred_masks'][:, :self.anchors.size(0), :]
        predicted_bboxes = predictions['bounding_box'][:, :self.anchors.size(0), :]  

        ground_truth_labels = targets['labels'].to(device)
        ground_truth_masks = targets['masks'].to(device)
        ground_truth_boxes = targets['boxes'].to(device)
        
        # Match ground truth boxes and generate regression targets
        matched_gt_boxes, anchor_max_idx = match_anchors_to_ground_truth_boxes(anchors, ground_truth_boxes)
        regression_targets = encode_bounding_boxes(matched_gt_boxes, anchors).to(device)
        regression_targets = regression_targets.unsqueeze(0).repeat(predicted_bboxes.size(0), 1, 1)

        # Compute regression loss
        regression_loss = self.bounding_box_loss(predicted_bboxes, regression_targets)

        # Classification targets need to align with the number of predictions per class
        classification_targets = ground_truth_labels[anchor_max_idx].to(device)
        classification_targets = classification_targets.unsqueeze(0).repeat(predicted_classes.size(0), 1)

        # Flatten the scores and targets to calculate classification loss
        classification_loss = self.classification_loss_fn(
            predicted_classes.reshape(-1, predicted_classes.size(-1)),
            classification_targets.reshape(-1) 
        )

        total_loss = regression_loss + classification_loss
        
        return total_loss
