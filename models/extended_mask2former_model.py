import torch
from torch import nn
from models.efpn_backbone.efpn_model import EFPN
from models.mask2former_detector.mask2former_model import Mask2Former



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
    def __init__(self, num_classes, hidden_dim=256, num_queries=100, nheads=8, dim_feedforward=2048, dec_layers=1, mask_dim=256):
        super(ExtendedMask2Former, self).__init__()              
        self.efpn        = EFPN(hidden_dim, hidden_dim, mask_dim, num_classes)
        self.mask2former = Mask2Former(hidden_dim, num_classes, hidden_dim, num_queries, nheads, dim_feedforward, dec_layers, mask_dim)
        
        # Define loss functions
        self.bounding_box_loss = nn.SmoothL1Loss()
        self.class_loss        = nn.CrossEntropyLoss()
        self.mask_loss         = nn.BCEWithLogitsLoss()
        
        
    def forward(self, image):
        feature_maps, masks, bounding_box, class_scores = self.efpn(image)
        output = self.mask2former(feature_maps, masks, bounding_box, class_scores)
        return output
    
    def compute_loss(self, predictions, targets, class_weight=1.0, bounding_box_weight=1.0, mask_weight=1.0):
        """        
        Parameters:
            - predictions (dict): A dictionary containing the model's predictions with keys 'pred_logits', 'pred_masks', and 'bounding_box'.
            - targets (dict): A dictionary containing the ground truth with keys 'labels', 'boxes', and 'masks'.
            - class_weight (float): A float value that signifies the significance of the classes in the loss function 
            - bounding_box_weight (float):  A float value that signifies the significance of the bounding boxes in the loss function
            - mask_weight (float):  A float value that signifies the significance of the masks in the loss
        """
        predicted_logits = predictions['pred_logits']
        predicted_masks  = predictions['pred_masks']
        predicted_boxes  = predictions['bounding_box']
        
        total_class_loss = 0
        total_bbox_loss  = 0
        total_mask_loss  = 0
        
        for i, target in enumerate(targets):
            target_labels = target['labels']
            target_masks  = target['masks']
            target_boxes  = target['boxes']
            
            # Match the shape of predicted_logits with target_labels
            num_objects = target_labels.shape[0]
            pred_logits_resized = predicted_logits[i, :num_objects]
            
            # Compute classification loss, bounding box loss, and mask loss for each target
            total_class_loss += self.class_loss(pred_logits_resized, target_labels) * class_weight
            total_bbox_loss  += self.bounding_box_loss(predicted_boxes[i], target_boxes) * bounding_box_weight
            total_mask_loss  += self.mask_loss(predicted_masks[i], target_masks) * mask_weight
        
        # Combine the losses
        total_loss = total_class_loss + total_bbox_loss + total_mask_loss
        
        return total_loss  
    