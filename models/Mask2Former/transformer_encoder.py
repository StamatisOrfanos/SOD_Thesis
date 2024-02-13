# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from torch import nn
from models.Mask2Former.attention_layers import MaskedAttentionLayer, SelfAttentionLayer
from models.Mask2Former.ffn_layer import FFNLayer

import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="relu"):
        super().__init__()
        self.self_attention = SelfAttentionLayer(d_model, nhead, dropout=dropout, activation=activation)        
        self.mask_attention = MaskedAttentionLayer(d_model, nhead, dropout=dropout, activation=activation)
        self.ffn            = FFNLayer(d_model, dim_feedforward, dropout=dropout, activation=activation)


    def forward(self, src, level_index, attention_mask, query_embed, position):        
        # Apply mask-attention using the current level's source feature maps and positional encodings.
        output = self.mask_attention(output, src[level_index], attention_mask, None, position[level_index], query_embed)

        # Apply self-attention.
        output = self.self_attention(output, None, None, query_embed)
            
        # Apply feedforward network layer.
        output = self.ffn(output)
        
        return output    