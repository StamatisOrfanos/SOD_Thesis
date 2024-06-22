import torch
from torch import nn
from torch.nn import functional as F
from models.efpn_backbone.efpn_model import EFPN
from models.mask2former_detector.mask2former_model import Mask2Former
from models.efpn_backbone.bounding_box import encode_bounding_boxes, match_anchors_to_gt_boxes



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
        self.efpn        = EFPN(hidden_dim, hidden_dim, num_classes)
        self.mask2former = Mask2Former(hidden_dim, num_classes, hidden_dim, num_queries, nheads, dim_feedforward, dec_layers, mask_dim)
        
        # Define loss functions
        self.bounding_box_loss = nn.SmoothL1Loss()
        self.class_loss        = nn.CrossEntropyLoss()
        self.mask_loss         = nn.BCEWithLogitsLoss()

        
    def forward(self, image):
        feature_maps, masks, bounding_box, class_scores = self.efpn(image)    
        output = self.mask2former(feature_maps, masks, bounding_box, class_scores)
        return output
    
    def decode_boxes(self, predicted_offsets, anchors):
        anchors = anchors.to(predicted_offsets.device)
        pred_boxes = torch.zeros_like(predicted_offsets)
        pred_boxes[:, 0] = anchors[:, 0] + predicted_offsets[:, 0] * (anchors[:, 2] - anchors[:, 0])
        pred_boxes[:, 1] = anchors[:, 1] + predicted_offsets[:, 1] * (anchors[:, 3] - anchors[:, 1])
        pred_boxes[:, 2] = anchors[:, 2] * torch.exp(predicted_offsets[:, 2])
        pred_boxes[:, 3] = anchors[:, 3] * torch.exp(predicted_offsets[:, 3])
        return pred_boxes
    
       
    def compute_loss(self, predictions, targets, anchors, class_weight=1.0, bounding_box_weight=1.0, mask_weight=1.0):
        predicted_logits = predictions['pred_logits']
        predicted_masks = predictions['pred_masks']
        predicted_bounding_boxes = predictions['bounding_box']
        
        print("The predicted masks shape is: {} and are of type: {}".format(predicted_masks.size(), type(predicted_masks)))
        
        
        total_class_loss = 0
        total_bbox_loss = 0
        total_mask_loss = 0

        for i, target in enumerate(targets):
            target_labels = target['labels']
            target_masks = target['masks']
            target_boxes = target['boxes']
            
            print("The target masks shape is: {} and are of type: {}".format(target_masks.size(), type(target_masks)))
            # # Match the shape of predicted_logits with target_labels
            # num_objects = target_labels.shape[0]
            # pred_logits_resized = predicted_logits[i, :num_objects]
            
            # # Decode the predicted bounding box offsets using the anchors
            # matched_gt_boxes, _ = match_anchors_to_gt_boxes(anchors, target_boxes)
            # encoded_gt_boxes = encode_bounding_boxes(matched_gt_boxes, anchors)
            # num_anchors = anchors.shape[0]          
            # predicted_boxes_resized = self.decode_boxes(predicted_bounding_boxes[i].view(-1, 4)[:num_anchors], anchors)

           # Resize target masks to match predicted masks size
            target_masks_resized = F.interpolate(target_masks.unsqueeze(1).float(), size=(predicted_masks.shape[2], predicted_masks.shape[3]), mode='bilinear', align_corners=False)
            target_masks_resized = target_masks_resized.squeeze(1)  # Remove the channel dimension added by unsqueeze
            
            print("The target_masks_resized shape is: {} and are of type: {}".format(target_masks_resized.size(), type(target_masks_resized)))
            
            predicted_masks_resized = predicted_masks[i].float()
            print("The predicted_masks_resized shape is: {} and are of type: {}".format(predicted_masks_resized.size(), type(predicted_masks_resized)))
            
            # Ensure the dimensions match
            if predicted_masks_resized.shape != target_masks_resized.shape:
                raise ValueError(f"Shape mismatch: predicted_masks_resized {predicted_masks_resized.shape}, target_masks_resized {target_masks_resized.shape}")
            
            
            # total_class_loss += self.class_loss(pred_logits_resized, target_labels) * class_weight
            # total_bbox_loss += self.bounding_box_loss(predicted_boxes_resized, encoded_gt_boxes) * bounding_box_weight
            total_mask_loss += self.mask_loss(predicted_masks_resized, target_masks_resized) * mask_weight

        total_loss = total_class_loss + total_bbox_loss + total_mask_loss

        return total_loss
    