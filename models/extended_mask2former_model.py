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
            predicted_masks (tensor): _description_
            ground_truth_masks (tensor): _description_
        """
        intersection = (predicted_masks & ground_truth_masks).float().sum((2, 3))
        union = (predicted_masks | ground_truth_masks).float().sum((2, 3))
        iou = intersection / union
        return 1 - iou
    
    
    def hungarian_loss(self, pred_classes, pred_masks, gt_classes, gt_masks):
        # Calculate the cost matrices
        iou_costs = self.calculate_iou(pred_masks, gt_masks)  # [batch_size, num_queries, num_queries]
        class_costs = -torch.log_softmax(pred_classes, dim=-1)[:, :, gt_classes]  # [batch_size, num_queries, num_queries]

        combined_costs = iou_costs + class_costs  # Combine costs
        
        # Apply Hungarian matching
        batch_size = pred_classes.size(0)
        mask_losses, class_losses = [], []
        
        for idx in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(combined_costs[idx].cpu().detach().numpy())
            mask_losses.append(self.mask_loss(pred_masks[idx, row_ind], gt_masks[idx, col_ind]).mean())
            class_losses.append(self.class_loss(pred_classes[idx, row_ind], gt_classes[idx, col_ind]))

        # Calculate mean loss over the batch
        mask_loss = torch.stack(mask_losses).mean()
        class_loss = torch.stack(class_losses).mean()

        return mask_loss, class_loss
    


    def class_confidence_predictor(self, bounding_box_classes, mask_classes):
        """        
        Parameters:
            - bounding_box_classes (tensor): Class logits from the bounding box branch [batch_size, num_queries, num_classes]
            - mask_classes (tensor): Class logits from the mask branch [batch_size, num_queries, num_classes]
        """
        # Softmax to convert logits to probabilities
        box_probabilities = torch.softmax(bounding_box_classes, dim=-1)
        mask_probabilities = torch.softmax(mask_classes, dim=-1)
        
        # Combine by taking the maximum across probabilities from both predictions
        combined_probabilities = torch.max(box_probabilities, mask_probabilities, dim=0) # type: ignore
        
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
    
