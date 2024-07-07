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
    def __init__(self, num_classes, hidden_dim=256, num_queries=300, nheads=8, dim_feedforward=2048, dec_layers=1, mask_dim=256):
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
        """
        Parameters:
            - predicted_offsets (torch.Tensor): Tensor of shape (N, 4) representing the predicted offsets for each anchor box. [dx, dy, dw, dh].
            - anchors (torch.Tensor): Tensor of shape (N, 4) representing anchor boxes with values [x_min, y_min, x_max, y_max].
        """
        anchors    = anchors.to(predicted_offsets.device)
        pred_boxes = torch.zeros_like(predicted_offsets)

        anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
        anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
        anchor_w  = anchors[:, 2] - anchors[:, 0]
        anchor_h  = anchors[:, 3] - anchors[:, 1]

        pred_boxes[:, 0] = anchor_cx + predicted_offsets[:, 0] * anchor_w - (torch.exp(predicted_offsets[:, 2]) * anchor_w) / 2
        pred_boxes[:, 1] = anchor_cy + predicted_offsets[:, 1] * anchor_h - (torch.exp(predicted_offsets[:, 3]) * anchor_h) / 2
        pred_boxes[:, 2] = anchor_cx + predicted_offsets[:, 0] * anchor_w + (torch.exp(predicted_offsets[:, 2]) * anchor_w) / 2
        pred_boxes[:, 3] = anchor_cy + predicted_offsets[:, 1] * anchor_h + (torch.exp(predicted_offsets[:, 3]) * anchor_h) / 2
        
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
    
       
    def compute_loss(self, predictions, targets, anchors, class_weight=1.0, bounding_box_weight=1.0, mask_weight=1.0):
        """
        Parameters:
            - predictions (torch.Tensor): Dictionary containing the predicted logits, masks, and bounding boxes.
                                    - 'pred_logits': Tensor of shape (N, num_classes) with predicted class scores.
                                    - 'pred_masks': Tensor of shape (N, H, W) with predicted masks.
                                    - 'bounding_box': Tensor of shape (N, 4) with predicted bounding box coordinates.

            - targets (torch.Tensor): List of dictionaries, each containing ground truth labels, masks, and boxes for each image.
                                    - 'labels': Tensor of shape (num_objects,) with ground truth class labels.
                                    - 'masks': Tensor of shape (num_objects, H, W) with ground truth masks.
                                    - 'boxes': Tensor of shape (num_objects, 4) with ground truth bounding box coordinates.

            - anchors (torch.Tensor): Tensor of shape (num_anchors, 4) with anchor box coordinates.
            - class_weight (float, optional): Weight for the classification loss. Defaults to 1.0.
            - bounding_box_weight (float, optional): Weight for the bounding box loss. Defaults to 1.0.
            - mask_weight (float, optional): Weight for the mask loss. Defaults to 1.0.
        """
        predicted_logits = predictions['pred_logits']
        predicted_masks = predictions['pred_masks']
        predicted_bounding_boxes = predictions['bounding_box']
                
        total_class_loss = 0
        total_bbox_loss = 0
        total_mask_loss = 0
    

        for i, target in enumerate(targets):
            target_labels = target['labels']
            target_masks = target['masks']
            target_boxes = target['boxes']
                        
            # Match the shape of predicted_logits with target_labels
            num_objects = target_labels.shape[0]
            pred_logits_resized = predicted_logits[i, :num_objects]
            pred_logits_resized = pred_logits_resized[:len(target_labels)]
            
            
            # Decode the predicted bounding box offsets using the anchors
            matched_gt_boxes, _ = match_anchors_to_ground_truth_boxes(anchors, target_boxes)
            encoded_gt_boxes = encode_bounding_boxes(matched_gt_boxes, anchors)
            num_anchors = anchors.shape[0]
            predicted_boxes_resized = self.decode_boxes(predicted_bounding_boxes[i].view(-1, 4)[:num_anchors], anchors)
            
            # # Perform Hungarian matching to align predicted and ground truth masks
            # matched_indices = self.hungarian_matching(predicted_masks[i], target_masks)
            # # Select matched masks for computing the loss
            # matched_predicted_masks = []
            # matched_ground_truth_masks = []
            # matched_ground_truth_labels = []
            # for prediction_idx, ground_truth_idx in matched_indices:
            #     matched_predicted_masks.append(predicted_masks[i, prediction_idx])
            #     matched_ground_truth_masks.append(target_masks[ground_truth_idx])
            #     matched_ground_truth_labels.append(target_labels[ground_truth_idx])
            # matched_predicted_masks = torch.stack(matched_predicted_masks)
            # matched_ground_truth_masks = torch.stack(matched_ground_truth_masks).float()
            # matched_ground_truth_labels = torch.tensor(matched_ground_truth_labels, dtype=torch.int64)
            
            
            total_class_loss += self.class_loss(pred_logits_resized, target_labels) * class_weight
            total_bbox_loss  += self.bounding_box_loss(predicted_boxes_resized, encoded_gt_boxes) * bounding_box_weight
            # total_mask_loss += self.mask_loss(matched_predicted_masks, matched_ground_truth_masks) * mask_weight
            
            
            print("Class loss:{}, bounding box loss:{} and mask loss:{}".format(total_class_loss, total_bbox_loss, total_mask_loss))


        total_loss = total_class_loss + total_bbox_loss + total_mask_loss

        return total_loss
