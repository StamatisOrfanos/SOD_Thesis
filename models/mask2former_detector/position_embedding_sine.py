# Copyright (c) Facebook, Inc. and its affiliates.
# # Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    Parameters:
    - num_pos_feats (int): The number of positional features to generate. Half of these features
      will be sine-encoded and the other half will be cosine-encoded.
    - temperature (int, optional): A scaling factor used in the positional encoding formula.
      It is usually a large value like 10000.
    - normalize (bool, optional): Whether to normalize the positional encodings.
    - scale (float, optional): An optional scaling factor for the positional encodings.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        
        if scale is not None and normalize is False: raise ValueError("normalize should be True if scale is passed")
        if scale is None: scale = 2 * math.pi
        
        self.scale = scale



    def forward(self, x, mask=None):
        
        # If no mask is provided, create a mask of zeros (i.e., no masking)
        if mask is None: mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        
        # The not_mask is the inverse of the mask as the cumsum computes the cumulative sum of elements across a dimension.
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        
        # Optionally normalize the positional encodings and scale them
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale


        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Divide the position by the adjusted dimensions to get the phase for the sine and cosine
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Compute the sine and cosine for each position in pos_x and pos_y then interleave them in the last dimension
        pos_x = torch.stack( (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4 ).flatten(3)
        pos_y = torch.stack( (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4 ).flatten(3)
        
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos