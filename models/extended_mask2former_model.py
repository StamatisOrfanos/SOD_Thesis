import torch
from torch import nn
from efpn_backbone.efpn_model import EFPN
from  mask2former_detector.mask2former_model import Mask2Former

# Assuming EFPN and MultiScaleMaskedTransformerDecoder are defined as per your previous message

class ExtendedMask2Former(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=100, nheads=8, dim_feedforward=2048, dec_layers=6, mask_dim=256):
        super(ExtendedMask2Former, self).__init__()
        
        # image
        # in_channels: int, num_classes: int, hidden_dim: int, num_queries: int, nheads: int, dim_feedforward: int, dec_layers: int, mask_dim: int)
        
        self.backbone = EFPN()
        # Initialize the MultiScaleMaskedTransformerDecoder
        self.decoder = Mask2Former(
            in_channels=in_channels, 
            num_classes=num_classes, 
            hidden_dim=hidden_dim, 
            num_queries=num_queries, 
            nheads=nheads, 
            dim_feedforward=dim_feedforward, 
            dec_layers=dec_layers, 
            mask_dim=mask_dim
        )
        
        
    def forward(self, ):
        print("Hello")
        

