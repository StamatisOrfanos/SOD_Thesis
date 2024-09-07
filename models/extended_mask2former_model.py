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
        efpn (EFPN): The Enhanced Feature Pyramid Network model used as the backbone for feature extraction and bounding box training.
        mask2former (Mask2Former): The Mask2Former model used for predicting object instances and their masks based on the features provided by EFPN.
    """
    def __init__(self, num_classes, num_anchors, device, hidden_dim=256, num_queries=100, nheads=16, dim_feedforward=2048, dec_layers=1, mask_dim=100):
        super(ExtendedMask2Former, self).__init__()
        self.device = device
        self.efpn        = EFPN(hidden_dim, num_classes, num_anchors)
        self.mask2former = Mask2Former(hidden_dim, num_classes, hidden_dim, num_queries, nheads, dim_feedforward, dec_layers, mask_dim)
        self.num_anchors = num_anchors
        
        # Define loss functions
        self.bounding_box_loss = nn.SmoothL1Loss(reduction="mean")
        self.class_loss        = nn.CrossEntropyLoss(ignore_index=-1)
        self.mask_loss         = nn.BCEWithLogitsLoss()


    def forward(self, image, masks):
        # Get the feature maps, bounding boxes and the classes from the EFPN
        image.to(self.device)
        feature_maps, bounding_box, class_scores = self.efpn(image)   
        
        # Bring all the data to the correct device
        masks  = masks.to(self.device)
        
        # Get the output of the Mask2Former model that contains the masks, classes
        output = self.mask2former(feature_maps, masks, bounding_box, class_scores)
        
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
    

    def calculate_iou(self, predicted_masks, ground_truth_masks):
        """
        Parameters:
            predicted_masks (tensor): Predicted masks, shaped [batch_size, num_queries, height, width]
            ground_truth_masks (tensor): Ground truth masks, shaped [batch_size, num_queries, height, width]
        """
        # Ensure masks are boolean
        predicted_masks = (torch.sigmoid(predicted_masks) > 0.5)
        ground_truth_masks = ground_truth_masks.bool()

        # Initialize IoU matrix
        batch_size, num_queries, _, _ = predicted_masks.shape
        iou_matrix = torch.zeros((batch_size, num_queries, num_queries), device=predicted_masks.device)

        for b in range(batch_size):
            for i in range(num_queries):
                for j in range(num_queries):
                    intersection = (predicted_masks[b, i] & ground_truth_masks[b, j]).float().sum()
                    union = (predicted_masks[b, i] | ground_truth_masks[b, j]).float().sum()
                    if union > 0:
                        iou_matrix[b, i, j] = intersection / union

        return 1 - iou_matrix  # Return the IoU costs as 1 - IoU
    
    
    def hungarian_loss(self, predicted_labels, predicted_masks, ground_truth_classes, ground_truth_masks):
        # Calculate the cost matrices with expected tensor format of [batch_size, num_queries, num_queries]
        iou_costs = self.calculate_iou(predicted_masks, ground_truth_masks)

        # Calculate class costs using softmax and negative log likelihood
        pred_classes_softmax = torch.softmax(predicted_labels, dim=-1)
        batch_size, num_queries, _ = pred_classes_softmax.shape
        class_costs = torch.full((batch_size, num_queries, num_queries), float('inf'), device=self.device)

        # Building the class cost matrix for each element in the batch
        for b in range(batch_size):
            for i in range(num_queries):
                for j in range(num_queries):
                    # The values with -1 are padding and are ignored
                    if ground_truth_classes[b, j] != -1:  
                        class_costs[b, i, j] = -torch.log(pred_classes_softmax[b, i, ground_truth_classes[b, j]] + 1e-6)  # Added epsilon to avoid log(0)

        # Ensure that costs are finite by clamping and handling inf/nan
        iou_costs = torch.clamp(iou_costs, min=0, max=10)
        class_costs = torch.clamp(class_costs, min=0, max=10)
        combined_costs = iou_costs + class_costs
        
        # Set all inf or nan values to a large finite value
        combined_costs[~torch.isfinite(combined_costs)] = 1e5  
        
        # Apply Hungarian matching
        mask_losses, class_losses = [], []
        
        for idx in range(batch_size):
            row_indices, column_indices = linear_sum_assignment(combined_costs[idx].cpu().detach().numpy())
            # Ensure mask data types are float for BCE loss
            predicted_mask_selected = predicted_masks[idx, row_indices].float()
            ground_truth_mask_selected = ground_truth_masks[idx, column_indices].float()
            mask_losses.append(self.mask_loss(predicted_mask_selected, ground_truth_mask_selected).mean())
            class_losses.append(self.class_loss(predicted_labels[idx, row_indices], ground_truth_classes[idx, column_indices]))

        # Calculate mean loss over the batch if matches were found
        mask_loss  = torch.stack(mask_losses).mean()  if mask_losses  else torch.tensor(0.0, device=self.device)
        class_loss = torch.stack(class_losses).mean() if class_losses else torch.tensor(0.0, device=self.device)

        return mask_loss, class_loss


    def class_confidence_predictor(self, bounding_box_labels, mask_labels):
        """        
        Parameters:
            - bounding_box_labels (tensor): Class logits from the bounding box branch [batch_size, num_queries, num_classes]
            - mask_labels (tensor): Class logits from the mask branch [batch_size, num_queries, num_classes]
        """
        # Softmax to convert logits to probabilities        
        box_probabilities = torch.softmax(bounding_box_labels, dim=-1)
        mask_probabilities = torch.softmax(mask_labels, dim=-1)
        
        # Combine by taking the maximum across probabilities from both predictions
        combined_probabilities = torch.maximum(box_probabilities, mask_probabilities)
        
        return combined_probabilities


    
       
    def compute_loss(self, predictions, targets, anchors, mask_weight=1.0, bounding_box_weight=1.0, class_weight=0.5):
        """
        Refactored loss computation to better align with the DetectionLoss class approach.

        Parameters:
            - predictions (dictionary): Dictionary containing the model predictions for bounding boxes, classes and masks
            - targets (dictionary): Dictionary containing the ground truth for bounding boxes, classes and masks
            - anchors (tensor): Tensor of all the anchors created
        """
        device = self.device
        anchors = anchors.to(device)
        num_objects = targets['mask_labels'].shape[1]
        
        # Extract predictions
        predicted_bounding_boxes = predictions['bounding_box'][:, :self.num_anchors, :]  
        predicted_classes_boxes = predictions['class_scores'][:, :self.num_anchors, :]
        predicted_classes_masks = predictions['pred_mask_labels'][:, :num_objects, :]
        predicted_masks = predictions['pred_masks'][:, :num_objects, :]


        # Ground truth
        ground_truth_labels = targets['labels'].to(device)
        ground_truth_boxes = targets['boxes'].to(device)
        ground_truth_masks = targets['masks'].to(device)
        ground_truth_masks_labels = targets['mask_labels'].to(device)
        
        
        # Match ground truth boxes and generate regression targets
        matched_gt_boxes, anchor_max_idx = match_anchors_to_ground_truth_boxes(anchors, ground_truth_boxes)
        regression_targets = encode_bounding_boxes(matched_gt_boxes, anchors).to(device)
        regression_targets = regression_targets.unsqueeze(0).repeat(predicted_bounding_boxes.size(0), 1, 1)

        # - Compute bounding box loss -
        bounding_box_loss = self.bounding_box_loss(predicted_bounding_boxes, regression_targets)

        # Classification targets need to align with the number of predictions per class and flatten the scores and targets to calculate bounding classification loss
        classification_targets = ground_truth_labels[anchor_max_idx].to(device)
        classification_targets = classification_targets.unsqueeze(0).repeat(predicted_classes_boxes.size(0), 1)
        bounding_box_class_loss = self.class_loss(predicted_classes_boxes.reshape(-1, predicted_classes_boxes.size(-1)), classification_targets.reshape(-1))
        
        # - Compute mask and masks class loss with hungarian matching -
        mask_loss, mask_class_loss = self.hungarian_loss(predicted_classes_masks, predicted_masks, ground_truth_masks_labels, ground_truth_masks)
        
        # - Compute final class prediction through confidence vote -
        final_class_loss = self.class_loss(self.class_confidence_predictor(predicted_classes_boxes, predicted_classes_masks))
        
        print("\n\n\n The class loss from the bounding box is of type: {}, size:{} and values:{}".format(type(bounding_box_class_loss),bounding_box_class_loss.size()))
        print("\n The class loss from the masks is of type: {}, size:{} and values:{}".format(type(mask_class_loss),mask_class_loss.size()))
        print("\n The class loss from the combinations is of type: {}, size:{} and values:{}\n\n\n".format(type(final_class_loss),final_class_loss.size()))
 

        total_loss = mask_weight * mask_loss + bounding_box_weight * bounding_box_loss + class_weight * final_class_loss
        
        return total_loss
    
