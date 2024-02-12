import torch
from torch import nn
from Efpn.efpn_model import EFPN
from Mask2Former import mask2former_model



class Extended_Mask2Former(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int, num_queries: int, nheads: int, dim_feedforward: int, dec_layers: int, mask_dim: int):
        """
        Parameters:
            in_channels (int): Number of channels in the input feature maps.
            num_classes (int): Number of classes for classification (including background).
            hidden_dim (int): Dimension of the feature vectors used in the Transformer.
            num_queries (int): Number of query slots used in the Transformer decoder.
            nheads (int): Number of attention heads in the multi-head attention layers.
            dim_feedforward (int): Dimensionality of the hidden layers in the feedforward network.
            dec_layers (int): Number of Transformer encoder layers to use in the decoder.
            mask_dim (int): Dimension of the mask feature vectors.
        """
        super().__init__()
        self.backbone = EFPN()
        self.transformer_decoder_layers = mask2former_model()
        
