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
    def __init__(self, num_classes, hidden_dim=256, num_queries=100, nheads=8, dim_feedforward=2048, dec_layers=1, mask_dim=100):
        super(ExtendedMask2Former, self).__init__()              
        self.efpn = EFPN(hidden_dim, hidden_dim, mask_dim)
        self.mask2former = Mask2Former(hidden_dim, num_classes, hidden_dim, num_queries, nheads, dim_feedforward, dec_layers, mask_dim)
        
        
    def forward(self, image, hidden_dim):
        feature_maps, masks, bounding_box = self.efpn(image, hidden_dim)
        output = self.mask2former(feature_maps, masks, bounding_box)
        return output
    
    