from sklearn.metrics import precision_recall_curve, auc
import torch
import torch.nn as nn
from torch.nn import L1Loss as l1_loss
from torch.nn import BCELoss as binary_cross_entropy_loss
from torch.nn import CrossEntropyLoss as cross_entropy_loss
from torchmetrics.classification import Dice as dice_loss
import torch.nn.functional as F


class metrics():
    def __init__(self,):
        super(self).__init__()
        
        
    def calculate_data_mean_std(data_loader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for data, _ in data_loader:
            channels_sum += torch.mean(data, dim=[0,2,3])
            channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
            num_batches += 1

        mean = channels_sum / num_batches
        standard_deviation = (channels_squared_sum / num_batches - mean**2)**0.5
        
        return mean, standard_deviation
    

    def calculate_loss(outputs_efpn, outputs_mask2former, targets, lambda_fg_bg=1, lambda_ce=5.0, lambda_dice=5.0, lambda_cls=2.0):
        # Foreground-Background-Balanced Loss for EFPN output
        loss_global = l1_loss(outputs_efpn['global'], targets['global'])
        loss_position = l1_loss(outputs_efpn['position'], targets['position'])  # Assuming targets are appropriately defined
        loss_efpn = loss_global + lambda_fg_bg * loss_position
        
        # Mask Loss for Mask2Former output
        loss_mask_ce = binary_cross_entropy_loss(outputs_mask2former['pred_masks'], targets['masks'])  # Implement sampling as needed
        loss_mask_dice = dice_loss(outputs_mask2former['pred_masks'], targets['masks'])  # Implement sampling as needed
        loss_mask = lambda_ce * loss_mask_ce + lambda_dice * loss_mask_dice
        
        # Classification Loss for Mask2Former output
        loss_cls = cross_entropy_loss(outputs_mask2former['pred_logits'], targets['labels'])
        
        # Combine losses
        total_loss = loss_efpn + loss_mask + lambda_cls * loss_cls
        
        return total_loss
    
    
    def average_precision(self, y_true, y_scores):
        """Calculate the average precision for a class."""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        # Calculate area under the curve to get average precision
        return auc(recall, precision)

    def mean_average_precision(self, y_trues, y_scores):
        """Calculate the mean average precision across all classes and return mean average precision across all classes.
        
        Parameters:
            y_trues: A list of arrays, where each array contains the true binary labels for a class.
            y_scores: A list of arrays, where each array contains the predicted scores for a class.
        """
        aps = []
        for y_true, y_score in zip(y_trues, y_scores):
            aps.append(self.average_precision(y_true, y_score))
        return sum(aps) / len(aps)


    def intersection_over_union(self, actual_bounding_box, predicted_bounding_box):
        """_summary_

        Args:
            boxA (_type_): _description_
            boxB (_type_): _description_
        """
        xA = max(actual_bounding_box[0], predicted_bounding_box[0])
        yA = max(actual_bounding_box[1], predicted_bounding_box[1])
        xB = min(actual_bounding_box[0] + actual_bounding_box[2], predicted_bounding_box[0] + predicted_bounding_box[2])
        yB = min(actual_bounding_box[1] + actual_bounding_box[3], predicted_bounding_box[1] + predicted_bounding_box[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (actual_bounding_box[2] + 1) * (actual_bounding_box[3] + 1)
        boxBArea = (predicted_bounding_box[2] + 1) * (predicted_bounding_box[3] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    
    def calculate_loss(predictions, targets):
        # Assuming `predictions` and `targets` include both masks and bounding boxes
        mask_loss = F.binary_cross_entropy_with_logits(predictions['masks'], targets['masks'])
        bbox_loss = F.smooth_l1_loss(predictions['bounding_boxes'], targets['bounding_boxes'])
        
        # Combine the losses, potentially with different weights
        loss = mask_loss + bbox_loss
        return loss