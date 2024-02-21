import torch
from torch import nn
from models.efpn_backbone.efpn_model import EFPN
from models.mask2former_detector.mask2former_model import Mask2Former



class ExtendedMask2Former(nn.Module):
    """ The class ExtendedMask2Former is our implementation of the Maks2Former model as a detector and the 
        EFPN model as a backbone and a feature map creator.

    Args:
        num_classes()     : 
        hidden_dim()      :256, 
        num_queries()     : 100, 
        nheads()          : 8, 
        dim_feedforward() : , 
        dec_layers()      :=1, 
        mask_dim          :()=256
    """
    def __init__(self, num_classes, hidden_dim=256, num_queries=100, nheads=8, dim_feedforward=2048, dec_layers=1, num_masks=100, mask_dim=256):
        super(ExtendedMask2Former, self).__init__()
                
        self.backbone = EFPN()
        self.decoder = Mask2Former(hidden_dim, num_classes, hidden_dim, num_queries, nheads, dim_feedforward, dec_layers, mask_dim)
        self.mask_features = nn.Parameter(torch.randn(num_masks, mask_dim))

        
        
    def forward(self, image):
        feature_maps = self.backbone(image)
        mask_features_list = [self.mask_features for _ in range(len(feature_maps))]
        output = self.decoder(feature_maps, mask_features_list)
        return output
    
    